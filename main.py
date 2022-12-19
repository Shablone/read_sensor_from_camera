# %%
#imports and consts

from onvif import ONVIFCamera
import os
import time
import cv2
import numpy as np
import logging
from math import atan2, cos, sin, sqrt, pi, degrees, radians
from influxdb import InfluxDBClient
from datetime import datetime

log = logging.getLogger("sensors")
log.setLevel(logging.INFO)

crops = {
    "4":{}, #goto maxes-Presets for calibration
    "5":{}, #goto maxes-Presets for calibration
    "1": {  #Camera-Preset-Number, in which sensor is found
        "Boiler_L1": {
            "crop": [727,727+87,140,140+88], #x-min, x-max, y-min,y-max crop of the picture taken which represents sensor
            "mapping": [-135,131,0,120], #value mapping: degrees of sensor to °C: from-min, from-max, to-min, to-max
            "threshold": [50, cv2.THRESH_BINARY | cv2.THRESH_OTSU], #cv contour detection settings
            "center": [44,44],  #position of sensor center in cropped image
        },
        "Boiler_L2": {
            "crop": [776,776+75,583,583+69],
            "mapping": [-133,124,0,120],
            "threshold": [50, cv2.THRESH_BINARY | cv2.THRESH_OTSU],
            "center": [39,33],
        }
    },
    "2": {
        "WW": {
            "crop": [1032,1032+59,537,537+93],
            "mapping": [-152,113,0,120],
            "threshold": [50, cv2.THRESH_BINARY],
            "center": [27,46],
        }
    }
}

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
ip = '192.168.0.101'
port_onvif = 2020
port_rtsp = 554

user = '###' #set manually in camera control
pw = '###'  #set manually in camera control
path_wsdl = './wsdl'


# %%
#functions

def move_camera(PresetToken="1"):
    global curPreset
    curPreset = PresetToken
    moverequest = ptz_service.create_type('GotoPreset')
    moverequest.ProfileToken = media_profile.token
    moverequest.PresetToken = PresetToken
    ptz_service.GotoPreset(moverequest)


def get_pic():
    cap = cv2.VideoCapture(f"rtsp://{user}:{pw}@{ip}:{port_rtsp}/stream1")
    if (cap.isOpened()):
        ret, frame = cap.read()
        cap.release()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

def get_contour_orientation(contour,center):
    ## pca

    # Construct a buffer used by the pca analysis
    sz = len(contour)
    data_pts = np.empty((sz, 2), dtype=np.float64)
    for i in range(data_pts.shape[0]):
        data_pts[i,0] = contour[i,0,0]
        data_pts[i,1] = contour[i,0,1]
    
    # Perform PCA analysis
    mean = np.empty((0))
    mean, eigenvectors, eigenvalues = cv2.PCACompute2(data_pts, mean)
    PCA_angle_rad = atan2(eigenvectors[0,1], eigenvectors[0,0]) # orientation in radians
    PCA_angle = (int(np.rad2deg(PCA_angle_rad)) + 90)

    distance_metric = lambda x: (x[0][0] - center[0])**2 + (x[0][1] - center[1])**2
    extrem_point = max(contour, key=distance_metric) 
    log.debug(f"{extrem_point=}")

    # Check if eigenvector aligns with angle to center of contour. If not flip angle
    log.debug(f"{PCA_angle=}")
    if PCA_angle>80 or PCA_angle < -80 :
        extrem_point_angle = (int(np.rad2deg(
            atan2(extrem_point[0][0] - center[0] , center[1] - extrem_point[0][1] )
            )))
        if extrem_point_angle> 135:
            extrem_point_angle -= 360
        elif extrem_point_angle < -135:
            extrem_point_angle += 360
        log.debug(f"{extrem_point_angle=}")
        if np.abs(extrem_point_angle - PCA_angle) > 150:
            if PCA_angle> 0: 
                PCA_angle -=180
            else:
                PCA_angle +=180 
        log.debug(f"{PCA_angle=}")
    return PCA_angle

def find_nearest_idx(array, value):
    array = np.asarray(array)
    return (np.abs(array - value)).argmin()

def readSensor(img, mapping, threshold,center):
    #returns temperature
    #returns -2 if no contour touches center
    #returns -3 if no contour is in area-tolerance
    #returns -4 if temperatue is not plausible


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, threshold[0], 255, threshold[1])
    contours, _ = cv2.findContours(bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    log.debug(f"{center=}")

    valid_contours = contours    

    #reduce to valid contours by checking if area capsulates center-point or a little around
    contours_copy = valid_contours
    valid_contours = []    
    for contour in contours_copy:
        if (cv2.pointPolygonTest(contour, (center[0], center[1]), False ) != -1
         or cv2.pointPolygonTest(contour, (center[0] -5, center[1]), False ) != -1
         or cv2.pointPolygonTest(contour, (center[0] +5, center[1]), False ) != -1
         or cv2.pointPolygonTest(contour, (center[0], center[1]-5), False ) != -1
         or cv2.pointPolygonTest(contour, (center[0], center[1]+5), False ) != -1
         ):
            valid_contours.append(contour)
    log.debug(f"{len(valid_contours)=}")
    if len(valid_contours)==0: return -2 

    #reduce to valids by checking area-size
    contours_copy = valid_contours
    valid_contours = []
    areas_dict = {idx: val for idx, val in enumerate([cv2.contourArea(c) for c in contours_copy])}
    log.debug(f"{areas_dict=}")
    for idx, area in areas_dict.items():
        if 30 < area < 600:
            valid_contours.append(contours_copy[idx])
    log.debug(f"{len(valid_contours)=}")
    if len(valid_contours)==0: return -3  
            

    #reduce to 1 valid by the nearest centroid
    contours_copy = valid_contours
    means = [np.mean(contour - center) for contour in contours_copy]#
    selected_idx = -1
    if len(contours_copy)>1:

        diff_means = [np.mean(contour - center) for contour in contours_copy]
        selected_idx = find_nearest_idx(list(diff_means), 0)
        selected_contour = contours_copy[selected_idx]
    else:
        selected_contour = contours_copy[0]

    #debugstuff
    log.debug(f"{areas_dict=}\n{len(valid_contours)=}")
    if log.level == logging.DEBUG:
        cv2.imwrite(f"./temp/{Sensorname}.jpg", img)
        means = [np.round(np.mean(np.abs(contour - center))) for contour in contours]
        log.debug(f"{means=}")

    #log contour
    cv2.drawContours(img, [selected_contour], 0, (0, 0, 255), 2)

    pointer_angle = get_contour_orientation(selected_contour, center)

    #debugstuff
    if log.level == logging.DEBUG: cv2.imwrite(f"./temp/{Sensorname}_cont.jpg", img)   
    log.debug(f"{selected_idx=}\n{pointer_angle=}")  

    temperatur = np.interp(pointer_angle,[mapping[0],mapping[1]],[mapping[2],mapping[3]])
    if 20 < temperatur < 120:
        return temperatur
    else:
        return -4

# %%
#init stuff
mycam = ONVIFCamera(ip, port_onvif, user, pw, path_wsdl)
ptz_service = mycam.create_ptz_service()
media_service = mycam.create_media_service()
media_profile = media_service.GetProfiles()[0]
_now = datetime.now()
timestamp = _now.strftime("%Y%m%d-%H%M%S")

influx_payload = [{
        "measurement": "sensors",
        "time": _now.strftime("%Y-%m-%dT%H:%M:%S"),
        "fields": {
            #Sensorname: temp
        }
    }]

# %%
#get pictures and evaluate
for Preset, Presetval in crops.items():
    move_camera(Preset)    
    time.sleep(11) #static wait until camera movement should be finished
    if len(Presetval) == 0: continue #preset has no sensors, probably is just limitposition-attempt

    img = get_pic()
    for Sensorname, Sensorcontent in Presetval.items():
        crop = Sensorcontent["crop"]
        crop_img = img[crop[2]:crop[3],crop[0]:crop[1]]
        cv2.imwrite(f"./pics/{Preset}/{Sensorname}/{timestamp}.jpg", crop_img)
        temperatur = readSensor(crop_img, Sensorcontent["mapping"], Sensorcontent["threshold"], Sensorcontent["center"])
        influx_payload[0]["fields"][Sensorname] =round(float(temperatur),1)

        log.info(f"{Sensorname} {temperatur}")
        log.info("")

# %%
#save to influx
client = InfluxDBClient(host='localhost', port=8086, username='writer', password='Elend-3456-3-Rekrut')
client.switch_database('keller')
client.write_points(influx_payload)


# %%

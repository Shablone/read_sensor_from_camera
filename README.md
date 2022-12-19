
# Get Temperature by Camera

Program to control camera (I used tapo C210) to check various temperature sensors.

The camera gets controlled via onvif to get pictures of several views. The views will get analyzed by opencv to recognize temperature sensor values. These get saved to an influx db

Shoutout to https://automaticaddison.com/how-to-determine-the-orientation-of-an-object-using-opencv/
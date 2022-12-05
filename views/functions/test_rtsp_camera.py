import cv2
import numpy as np
import os
# os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
vcap = cv2.VideoCapture("rtsp://ad:mi@192.168.1.6:554/live")

while(True):
    ret, frame = vcap.read()
    if ret == False:
        print("Frame is empty")
        break
    else:
        cv2.imshow('VIDEO', frame)
        cv2.waitKey(1)
    vcap.release()
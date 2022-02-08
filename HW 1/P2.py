#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  2 09:04:53 2022

@author: pulkit
"""

# import the necessary packages
from collections import deque
from imutils.video import VideoStream
import numpy as np
import argparse
import cv2
import imutils
import time
import pandas as pd

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video",
	help="path to the (optional) video file")
ap.add_argument("-b", "--buffer", type=int, default=64,
	help="max buffer size")
args = vars(ap.parse_args())


redLower = (80, 25 , 10)
redUpper = (255 , 60, 255)
pts = deque(maxlen=args["buffer"])
# if a video path was not supplied, grab the reference
# to the webcam
if not args.get("video", False):
	vs = VideoStream(src=0).start()
# otherwise, grab a reference to the video file
else:
	vs = cv2.VideoCapture(args["video"])
# allow the camera or video file to warm up
time.sleep(2.0)

redLower = (100,170,80)
redUpper = (190,255,255)

Data_Features = ['x','y','time']
Data_Points = pd.DataFrame(data = None, columns = Data_Features, dtype = float)

start = time.time()

# if not args.get("video", False):
# 	camera = cv2.VideoCapture(0)

# # otherwise, grab a reference to the video file
# else:
# 	camera = cv2.VideoCapture(args["video"])



while(True):
    ret, frame = vs.read()
    if args.get("video") and not ret:
        break
    current_time = time.time() - start
    
    frame = imutils.resize(frame, width=1200)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, redLower, redUpper)
    red = cv2.bitwise_and(frame, frame, mask = mask) 
    
    cnts = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]
    center = None
    
    if len(cnts) > 0:
        c = max(cnts, key = cv2.contourArea)
        ((x,y),radius) = cv2.minEnclosingCircle(c)
        M = cv2.moments(c)
        center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
		
        if (radius < 300) & (radius > 10 ) : 
            cv2.circle(frame, (int(x), int(y)), int(radius), (0, 255, 255), 2)
            cv2.circle(frame, center, 5, (0,0,255), -1)
            Data_Points.loc[Data_Points.size/3] = [x, y, current_time]
    pts.appendleft(center)
    
    for i in range(1, len(pts)):
        if pts[i-1] is None or pts[i] is None:
            continue
        thickness = int(np.sqrt(args["buffer"] / float(i+1)) * 2.5)
        cv2.line(frame, pts[i-1], pts[i], (255,0,0), thickness)

    time.sleep(0.05)
    cv2.imshow('frame', red)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
vs.release()
cv2.destroyAllWindows()
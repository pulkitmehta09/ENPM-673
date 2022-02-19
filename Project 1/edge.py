#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 15:02:26 2022

@author: pulkit
"""

import cv2

videofile = '1tagvideo.mp4'
cam = cv2.VideoCapture(videofile)


while(True): 
    ret, frame = cam.read()
    # Check if the frame exists, if not exit 
    if not ret:
        break
   
    
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    blurred = cv2.GaussianBlur(grayscale, (3,3), 0)
    thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,5,3)
    edges = cv2.Canny(thresh,100,200,apertureSize = 3)
    
    cv2.imshow('frame', edges)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
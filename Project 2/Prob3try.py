#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:37:56 2022

@author: pulkit
"""

import cv2
import numpy as np
from utils import Modify_frame, region, classify, adjust_gamma, prepare_warped


videofile = 'challenge.mp4'                                                     # Video file
cam = cv2.VideoCapture(videofile)   

red = (0,0,255)
blue = (255,0,0)
green = (0,255,0)



C = np.array([[(0,0), (150, 400), (250,400), (400,0)]])

while(True): 
    ret, frame = cam.read()                                                     # Reading the frame from video
       
    # Check if the frame exists, if not exit 
    if not ret:
        break
    

    # hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # v = hsv[:,:,2]    
    # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(20,20))
    # cl1 = clahe.apply(v)
    # hsv[:,:,2] = cl1
    # img_back = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    # img_back = adjust_gamma(img_back, gamma = 0.5)

    
    # modified = Modify_frame(img_back)
    # roi = region(modified)

    # lower = np.array([0,100,100])
    # upper = np.array([30,255,255])
    
    # yellow_lane_roi = region(frame) 
    # yellow_lane_hsv = cv2.cvtColor(yellow_lane_roi, cv2.COLOR_BGR2HSV)
    # mask = cv2.inRange(yellow_lane_hsv, lower, upper)
    

    # height, width, _ = frame.shape
    # pts1 = np.float32([[(200, height-50), (630,425), (725,425), (width - 150, height-50)]])
    # pts2 = np.float32([[0,400],[0,0],[200,0],[200,400]])
    
    # matrix = cv2.getPerspectiveTransform(pts1, pts2)
    # warped = cv2.warpPerspective(roi, matrix, (200,400))
    # # warped = cv2.flip(warped,0)
    # yellow_lane_warped = cv2.warpPerspective(mask, matrix, (200,400))
    # warped[:,:100] = 0
    # comb = warped + yellow_lane_warped
    
    comb, matrix = prepare_warped(frame)
    
    classified = classify(comb)
    unwarp = cv2.warpPerspective(classified, np.linalg.inv(matrix), (frame.shape[1],frame.shape[0]))
    # lines, copy = make_HoughLines(comb)
    
    cv2.imshow('res', unwarp)
    # cv2.imshow('frame', frame)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    
    
    
cam.release()
cv2.destroyAllWindows()




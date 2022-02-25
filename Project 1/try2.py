#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 22 16:43:21 2022

@author: pulkit
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:43:16 2022

@author: pulkit
"""

# BLOB------------------------------------------------------------------------

# TODO argument parser and option to choose from argument or default

import cv2
import numpy as np
from functions import *
import imutils
from matplotlib import pyplot as plt
import scipy



videofile = '1tagvideo.mp4'
cam = cv2.VideoCapture(videofile)
got_fft = False

while(True): 
    ret, frame = cam.read()
    
    # Check if the frame exists, if not exit 
    if not ret:
        break
     
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    # ret,thresh = cv2.threshold(grayscale,180,255,cv2.THRESH_BINARY)
    # edges = cv2.Canny(thresh,10,100)
    
    # ----------------------------------------------------------------
    # Shi-Tomasi Corner Detector            
    # corners = cv2.goodFeaturesToTrack(edges, 4, 0.01, 100)
    # corners = np.int0(corners)
    # for i in corners:
    #     x,y = i.ravel()
    #     cv2.circle(frame, (x,y), 10, (0,0,255), -1)
    # # ----------------------------------------------------------------

            
    # # ----------------------------------------------------------------
    # Harris Corner Detector
    # dst = cv2.cornerHarris(thresh, 5, 7, 0.4)
    # frame[dst > 0.1 * dst.max()] = [0,0,255]
    
    # # ----------------------------------------------------------------
    
    # # Blob detection
    
    params = cv2.SimpleBlobDetector_Params()
    params.filterByArea = True
    params.minArea = 20
    params.maxArea = 1000000
    params
    
    
    detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(grayscale)
    im_with_keypoints = cv2.drawKeypoints(grayscale, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    blank = np.zeros((1,1)) 
    # blobs = cv2.drawKeypoints(grayscale, keypoints, blank, (0,255,255), cv2.DRAW_MATCHES_FLAGS_DEFAULT)
    
    # if not(got_fft) :
    #     fft(grayscale, frame)
    #     got_fft = True
        
        
        
        
    cv2.imshow('frame', im_with_keypoints)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cam.release()
cv2.destroyAllWindows()

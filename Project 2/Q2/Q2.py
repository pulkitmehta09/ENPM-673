#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:20:35 2022

ENPM 673
Project 2 Question2

@author: Pulkit Mehta
UID: 117551693
"""

# ---------------------------------------------------------------------------------
# IMPORTING PACKAGES
# ---------------------------------------------------------------------------------

import imutils
from utils import *


# ---------------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------------

videofile = 'whiteline.mp4'                                                     # Video file
cam = cv2.VideoCapture(videofile)   


while(True): 
    ret, frame = cam.read()                                                     # Reading the frame from video   
    
    # Check if the frame exists, if not exit 
    if not ret:
        break

    # frame = cv2.flip(frame,1)                                                 # Uncomment to flip the frame horizontally
    copy = frame.copy()                                                         # Creating a copy of the original frame
    modified = Modify_frame(frame)
    roi = region(modified)
    height, width, _ = frame.shape
 
    src = np.float32([[(10, height), (450,320), (515,320), (width, height)]])   # Source points for perspective transform
    dst = np.float32([[(0,400), (0,0), (200, 0), (200,400)]])                   # Destination points for perspective transform
              
    homography_matrix = cv2.getPerspectiveTransform(src, dst)                   # Homography matrix
    warped = cv2.warpPerspective(roi, homography_matrix,                        # warped image
                                 (200,400), flags = cv2.INTER_LINEAR)
            
    out_img = classify(warped)                                                  # Classified lanes birds-eye view image
    unwarp = cv2.warpPerspective(out_img, np.linalg.inv(homography_matrix),     # Getting classified image in original perspective
                                 (frame.shape[1],frame.shape[0]))
    
    indices = unwarp.nonzero()
    copy[indices[0],indices[1],:] = [0,0,0]
    result = cv2.bitwise_or(unwarp, copy)                                       # Resultant image
     
    # Stacking all the frames so as to show in a single window
    stacked1 = np.vstack((frame,result))
    cv2.putText(stacked1, '(2)'
                ,(20,50), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 2) 
    cv2.putText(stacked1, '(3)'
                ,(20,590), cv2.FONT_HERSHEY_DUPLEX, 1, (255,0,0), 2) 
    
    warped_bgr = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    warped_bgr_resized = imutils.resize(warped_bgr, height = stacked1.shape[0])
    
    stacked2 = np.hstack((warped_bgr_resized,stacked1))
    stacked2 = imutils.resize(stacked2, height = 540)
    cv2.putText(stacked2, '(1)'
                ,(10,25), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 1) 
        
    blank_image = np.zeros((100,stacked2.shape[1],3), np.uint8)
    blank_image[:,:,:] = [255,200,200]
    
    cv2.putText(blank_image, '(1): Birds-eye view of the region of interest consisting of lanes'
                ,(20,20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 2) 
    cv2.putText(blank_image, '(2): Original frame'
                ,(20,50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 2) 
    cv2.putText(blank_image, '(3): Detected lane markings'
                ,(20,80), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 2) 
   
    stacked = np.vstack((stacked2,blank_image))
    
    cv2.imshow('frame', stacked)
    
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    
cam.release()
cv2.destroyAllWindows()


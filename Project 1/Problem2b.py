#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 12:13:23 2022

ENPM 673
Project 1

@author: Pulkit Mehta
UID: 117551693
"""

# --------------------------------------------------------------------------------------------------------------------
# IMPORTING PACKAGES
# --------------------------------------------------------------------------------------------------------------------

import cv2
import numpy as np
from utils import *
from CornerUtils import *
from TagUtils import Find_tag_corners

# --------------------------------------------------------------------------------------------------------------------
# Problem 2(b) Placing virtual cube on the tag
# --------------------------------------------------------------------------------------------------------------------

videofile = '1tagvideo.mp4'                                                     # Video file
cam = cv2.VideoCapture(videofile)

while(True): 
    ret, frame = cam.read()                                                     # Reading the frame from video
    
    # Check if the frame exists, if not exit 
    if not ret:
        break

    tag = Find_tag_corners(frame)                                               # Finding the corners of the AR tag.

    # Checking if the retrieved tag's aspect ratio is within acceptable value so as to avoid further operations on incorrectly detected points.
    if (0.9 < get_aspect_ratio(tag) < 1.1):
        H = homography(tag, C)                                                  # Calculating Homography matrix         
        P, T = getProjectionMatrix(H)                                           # Calculating Projection and transformation matrix     
        drawCube(P,frame)                                                       # Draw the cube on the tag
    
    cv2.imshow('Virtual cube on AR Tag', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


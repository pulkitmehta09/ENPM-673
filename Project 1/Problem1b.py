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
# Problem 1(b) Detecting the AR tag ID.
# --------------------------------------------------------------------------------------------------------------------

videofile = '1tagvideo.mp4'                                                     # Video file
cam = cv2.VideoCapture(videofile)

gotID = False                                                                   # Flag to check ID found or not

while(True): 
    ret, frame = cam.read()                                                     # Reading the frame from video
    
    # Check if the frame exists, if not exit 
    if not ret:
        break
    
    tag = Find_tag_corners(frame)                                               # Finding the corners of the AR tag.

    # Tag Identification
    H = homography(tag, C)                                                      # Calculating Homography matrix                                      
    warped = warp(H, frame, I)                                                  # Warping the tag to get birds-eye view.
    warped = cv2.flip(warped,0)
    pose = get_tag_orientation(warped)                                          # Getting the orientation information of the tag.

    oriented = orientTag(pose,warped)                                           # Oriented tag(upright position) 
    center_tag = oriented[75:125,75:125]                                        # Center 2x2 grid of the correctly oriented tag.
    
    # Check if Tag ID is retrieved or not, if not, find the ID
    if not gotID:
        gotID, ID, ar = getTagID(center_tag)
        print("Tag ID: ", ID)
        print("Binary representation of tag: ", ar)
    
    cv2.imshow('Warped and oriented tag', oriented)
    cv2.imshow("Centre 2x2 grid of tag", center_tag)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


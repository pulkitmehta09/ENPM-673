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
# Problem 2(a) Superimposing testudo image on the tag
# --------------------------------------------------------------------------------------------------------------------

videofile = '1tagvideo.mp4'                                                     # Video file
cam = cv2.VideoCapture(videofile)   

testudo = cv2.imread('testudo.png')                                             # Reading the testudo image
testudo = cv2.resize(testudo, (200,200), interpolation = cv2.INTER_LINEAR)       # Resizing the testudo image

C = np.array([0,0,200,0,200,200,0,200])                                         # Defining corner points in world frame for template image.
I = (200,200)                                                                   # Defining reference dimensions

prev_pose = 0                                                                   # Variable storing the pose of the tag

while(True): 
    ret, frame = cam.read()                                                     # Reading the frame from video
    
    # Check if the frame exists, if not exit 
    if not ret:
        break
    
    tag = Find_tag_corners(frame)                                               # Finding the corners of the AR tag.
    
    # Checking if the retrieved tag's aspect ratio is within acceptable value so as to avoid further operations on incorrectly detected points.
    if (0.9 < get_aspect_ratio(tag) < 1.1):    
        H = homography(tag, C)                                                  # Calculating Homography matrix  
        warped = warp(H, frame, I)                                              # Warping the tag to get birds-eye view.
        warped = cv2.flip(warped,0)
        pose = get_tag_orientation(warped)                                      # Getting the orientation information of the tag.
        
        oriented = orientTag(pose,warped)                                       # Oriented tag(upright position) 
        center_tag = oriented[75:125,75:125]                                    # Center 2x2 grid of the correctly oriented tag.
        
        # Check whether the pose changed compared to previous frame, if yes, rotate the testudo accordingly
        if not (prev_pose == pose):
            testudo = orientTestudo(pose, testudo, prev_pose)

        prev_pose = pose                                                        # Update variable for previous pose with current value

        frame = inverse_warp(H, tag, frame, testudo)                            # (Inverse) Warping testudo on the tag
            
    cv2.imshow('Superimposed testudo on Tag', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()


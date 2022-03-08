#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  3 23:49:34 2022

ENPM 673
Project 1

@author: Pulkit Mehta
UID: 117551693
"""

import cv2
from CornerUtils import *
from utils import *
import numpy as np


def Find_tag_corners(frame):
    """
    Finds the corners of the AR tag from the frame

    Parameters
    ----------
    frame : Array
        Input frame.

    Returns
    -------
    tag : Array
        Ordered set of corners of the AR tag.

    """
    
    # ------------------------------------------------------------------------------
    # Operations on the original frame for retrieving the outer white sheet corners
    
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                         # Converting the frame to gray-scale    
    ret,thresh = cv2.threshold(grayscale,200,255,cv2.THRESH_BINARY)             # Binary thresholding
    
    # Morphological operations
    kernel = np.ones((3,3),np.uint8)
    erosion = cv2.morphologyEx(thresh, cv2.MORPH_ERODE,kernel, iterations = 1)
    dilation = cv2.morphologyEx(erosion, cv2.MORPH_DILATE,kernel, iterations = 1)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,iterations = 1)
    
    # Shi-Tomasi Corner Detector
    corners = cv2.goodFeaturesToTrack(opening, 50, 0.1, 10)
    corners = np.int0(corners)
      
    points = get_outer_corners(corners)                                              # Finding the corners of the outer white sheet.
    points = orderpts(points)                                                   # Ordering points in the order: TL, TR, BR, BL.
    # ------------------------------------------------------------------------------
    
    # ------------------------------------------------------------------------------
    # Operations on the copy of frame for better quality points for tag corner detection.
    
    # Morphological operations with suitable parameters
    kernel2 = np.ones((9,9),np.uint8)
    erosion = cv2.morphologyEx(thresh, cv2.MORPH_ERODE, kernel2, iterations = 3)
    dilation = cv2.morphologyEx(erosion, cv2.MORPH_DILATE, kernel2, iterations = 3)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel2)
    
    # Shi-Tomasi Corner Detector
    corners2 = cv2.goodFeaturesToTrack(opening, 30, 0.1, 35)
    corners2 = np.int0(corners2)
    # ---------------------------------------------------------------
    
    tag = nearest_points(points,corners2)                                         # Finding the corners of the AR tag
    tag = orderpts(tag)                                                           # Ordering tag corners in the order: TL, TR, BR, BL.
    
    return tag
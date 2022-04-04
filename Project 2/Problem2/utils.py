#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 31 22:10:31 2022

ENPM 673
Project 2 Problem 2

@author: Pulkit Mehta
UID: 117551693
"""

import cv2
import numpy as np


def Modify_frame(frame):
    """
    Obtains the thresholded image.

    Parameters
    ----------
    frame : ndArray
        Image BGR frame.

    Returns
    -------
    thresh : ndArray
        Binary image after thresholding.

    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)                              # Converting to grayscale
    blur = cv2.GaussianBlur(gray, (11,11), 0)                                   # Adding Gaussian Blur
    ret,thresh = cv2.threshold(blur,120,255,cv2.THRESH_BINARY)                  # Binary thresholding

    return thresh    
 
   
def region(frame):
    """
    Finds the desired region of interest in the image frame.

    Parameters
    ----------
    frame : ndArray
        Image frame.

    Returns
    -------
    mask : ndArray
        Frame with only the region of interest.

    """
    height, width = frame.shape  
    trapezoid = np.array([[(110, height), (450,320), (515,320), (width-50, height)]])
    # cv2.line(frame,tuple(trapezoid[0][0]), tuple(trapezoid[0][1]),(0,0,255),2)
    # cv2.line(frame,tuple(trapezoid[0][0]), tuple(trapezoid[0][3]),(0,0,255),2)
    # cv2.line(frame,tuple(trapezoid[0][1]), tuple(trapezoid[0][2]),(0,0,255),2)
    # cv2.line(frame,tuple(trapezoid[0][2]), tuple(trapezoid[0][3]),(0,0,255),2)   
    
    mask = np.zeros_like(frame)
    mask = cv2.fillPoly(mask, trapezoid, (255,255,255))
    mask = cv2.bitwise_and(frame, mask)
    
    return mask


def classify(warped):
    """
    Finds the solid and dashed lane lines and colours them accordingly.

    Parameters
    ----------
    warped : ndArray
        Image frame of the birds-eye view of the region of interest.

    Returns
    -------
    out_img : ndArray
        Image frame with lanes classified.

    """
    
    out_img = np.dstack((warped, warped, warped)) * 255                         # Creating a 3 channel BGR image from the input binary image
    
    pts1 = np.array(warped[:,:100].nonzero())                                   # Finding white pixel locations in left half of image 
    pts1_x = np.array(pts1[0])                                                  # x - coordinates of set of points in left half of image
    pts1_y = np.array(pts1[1])                                                  # y - coordinates of set of points in left half of image

    pts2 = np.array(warped[:,100:].nonzero())                                   # Finding white pixel locations in right half of image
    pts2_x = np.array(pts2[0])                                                  # x - coordinates of set of points in right half of image
    pts2_y = np.array(pts2[1]+100)                                              # y - coordinates of set of points in right half of image    
    
    # Classify points as part of solid  or dashed lane line
    if (pts1.shape[1] < pts2.shape[1]):                                         # Check number of points, more points mean its a solid lane and vice versa
        dashed_x = pts1_x
        dashed_y = pts1_y
        solid_x = pts2_x
        solid_y = pts2_y
    else:
        dashed_x = pts2_x
        dashed_y = pts2_y
        solid_x = pts1_x
        solid_y = pts1_y

    # Colour solid and dashed lanes
    out_img[dashed_x,dashed_y] = [0,0,255]
    out_img[solid_x,solid_y] = [0,255,0]

    return out_img


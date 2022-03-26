#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:37:04 2022

@author: pulkit
"""

import cv2
import numpy as np
import imutils
import glob
from matplotlib import pyplot as plt
from cv2 import VideoWriter, VideoWriter_fourcc 


def create_histogram(img):
    """
    Takes an hsv image and creates a histogram from its brightness values. 

    Parameters
    ----------
    img : ndArray
        Input HSV image.

    Returns
    -------
    histogram : ndArray
        Array representing the histogram.

    """
    values = img[:,:,2].flatten()
    histogram = np.zeros(256, dtype=int)
    for index in range(values.size):
        histogram[values[index]] += 1
    
    return histogram


def create_cumulative_sum(histogram):
    """
    Creates an array that represents the cumulative sum of the histogram.

    Parameters
    ----------
    histogram : ndArray
        Array representing the histogram.

    Returns
    -------
    cumsum : ndArray
        Array representing the cumulative sum of the histogram.

    """
    cum_sum = np.zeros(256, dtype=int)
    cum_sum[0] = histogram[0]
    for i in range(1, histogram.size):
        cum_sum[i] = cum_sum[i-1] + histogram[i]
   
    return cum_sum


def create_mapping(histogram, cum_sum, h, w):
    """
    Maps old brightness values to new ones between 0 and 255 using:
         M(i) = max(0, round((levels*cum_sum(i))/(h*w))-1),
    where, levels is the number of brightness levels(256) in image.

    Parameters
    ----------
    histogram : ndArray
        Array representing the histogram.
    cum_sum : TYPE
        Array representing the cumulative sum of the histogram.
    h : int
        Image height.
    w : int
        Image width.

    Returns
    -------
    mapping : ndArray
        Array containing new brightness levels.

    """
    
    mapping = np.zeros(256, dtype=int)
    levels = 256
    for i in range(histogram.size):
        mapping[i] = max(0, round((levels*cum_sum[i])/(h*w))-1)
    
    return mapping


def apply_mapping(img, mapping):
    """
    Apply the new brighntness levels to the original hsv image.

    Parameters
    ----------
    img : ndArray
        Input HSV image.
    mapping : ndArray
        Array containing new brightness levels.

    Returns
    -------
    new_image : ndArray
        New histogram equalized HSV image.

    """
    new_image = img.copy()
    new_image[:,:,2] = list(map(lambda a : mapping[a], img[:,:,2]))
    return new_image


def adjust_gamma(image, gamma=1.0):

	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
    return cv2.LUT(image, table)


def clipping(hist):
    N = 256
    beta = 3000
    excess = 0
    for i in range(N):
        if(hist[i] > beta):
            excess += hist[i] - beta
            hist[i] = beta
    
    m = excess / N
    for i in range(N):
        if(hist[i] < beta - m):
            hist[i] += m
            excess -= m
        elif(hist[i] < beta):
            excess += hist[i] - beta
            hist[i] = beta
            
    while (excess > 0):
        for i in range(N):
            if(excess > 0):
                if(hist[i] < beta):
                    hist[i] += 1
                    excess -= 1
    
    return hist

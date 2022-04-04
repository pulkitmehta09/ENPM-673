#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 13:37:04 2022

ENPM 673
Project 2 Problem 1

@author: Pulkit Mehta
UID: 117551693
"""


import numpy as np


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
    cum_sum : ndArrray
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
    Apply the new brightness levels to the original hsv image.

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



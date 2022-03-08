#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 15:02:26 2022

ENPM 673
Project 1

@author: Pulkit Mehta
UID: 117551693
"""

import cv2
import numpy as np
import imutils
   

img = cv2.imread('cityscape1.png')                                              # Reading the image
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                               # Converting to gray-scale
blurred = cv2.GaussianBlur(grayscale, (5,5), 5)                                 # Gaussian blurring    

# Performing Morphological operations
kernel = np.ones((3,3))             
erode = cv2.erode(blurred, (3,3))
dilate = cv2.dilate(erode,(3,3))
opening = cv2.morphologyEx(dilate,cv2.MORPH_OPEN,kernel)

# Canny Edge detection
edges = cv2.Canny(opening,10,55)

# Applying Bilateral filter
gray_filtered = cv2.bilateralFilter(grayscale,7,50,50)
 
# Canny edge detection performed for filtered image
edges_filtered = cv2.Canny(gray_filtered,10,55)

# Inverting the image
inv = ~edges
inv_filtered = ~edges_filtered

# Resizing the images
inv = imutils.resize(inv,width = 1200)
inv_filtered = imutils.resize(inv_filtered, width = 1200)

images = np.hstack((inv, inv_filtered))

cv2.imshow('Canny Edge Detection', inv)
cv2.imshow('Canny Edge Detection after Bilinear Filter', inv_filtered)

cv2.waitKey(0)
cv2.destroyAllWindows()

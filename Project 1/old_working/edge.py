#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 15:02:26 2022

@author: pulkit
"""

import cv2
import numpy as np
import imutils
   
img = cv2.imread('cityscape1.png')
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
blurred = cv2.GaussianBlur(grayscale, (5,5), 5)



kernel = np.ones((3,3))
erode = cv2.erode(blurred, (3,3))
dilate = cv2.dilate(erode,(3,3))
opening = cv2.morphologyEx(dilate,cv2.MORPH_OPEN,kernel)


# thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,3)


gray_filtered = cv2.bilateralFilter(grayscale,7,50,50)
 

edges = cv2.Canny(opening,10,55)
edges_filtered = cv2.Canny(gray_filtered,10,55)

inv = ~edges
inv_filtered = ~edges_filtered

inv = imutils.resize(inv,width = 1200)
inv_filtered = imutils.resize(inv_filtered, width = 1200)


images = np.hstack((inv, inv_filtered))

cv2.imshow('img', inv)
cv2.imshow('img1', inv_filtered)
# cv2.imshow('Frame',images)


# cv2.imwrite('cannyedges.png',inv)
cv2.waitKey(0)
cv2.destroyAllWindows()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 15:02:26 2022

@author: pulkit
"""

import cv2

   
img = cv2.imread('cityscape1.png')
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
blurred = cv2.GaussianBlur(grayscale, (1,1), 3)
erode = cv2.erode(blurred, (9,9))
dilate = cv2.dilate(erode,(9,9))
thresh = cv2.adaptiveThreshold(blurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,3)
edges = cv2.Canny(dilate,10,100)
    

cv2.imshow('frame', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
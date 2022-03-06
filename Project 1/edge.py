#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 15:02:26 2022

@author: pulkit
"""

import cv2
import numpy as np
   
img = cv2.imread('cityscape1.png')
grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
blurred = cv2.GaussianBlur(grayscale, (5,5), 5)

erode = cv2.erode(blurred, (3,3))
dilate = cv2.dilate(erode,(3,3))

thresh = cv2.adaptiveThreshold(dilate,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,17,3)

kernel = np.ones((3,3))
opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel)

edges = cv2.Canny(dilate,10,55)
inv = ~edges

cv2.imshow('img', inv)
# cv2.imwrite('cannyedges.png',inv)
cv2.waitKey(0)
cv2.destroyAllWindows()
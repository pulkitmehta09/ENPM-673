#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:05:23 2022

@author: pulkit
"""

import cv2
from HistEqUtils import *



ahe2 = cv2.imread('ahe_N2.png')
ahe12 = cv2.imread('ahe_N12.png')

IMG_W = ahe12.shape[1]
IMG_H = ahe12.shape[0]
N = 1
w = IMG_W // N
h = IMG_H // N


gray = cv2.cvtColor(ahe12, cv2.COLOR_BGR2GRAY)
smooth = cv2.GaussianBlur(ahe12,(5,5),1)
mB = cv2.medianBlur(ahe12,5)

hsv = cv2.cvtColor(ahe2, cv2.COLOR_BGR2HSV)
hsv_copy = hsv.copy()
ahe_image_hsv = np.zeros_like(ahe2)
for i in range(N):
    for j in range(N):    
        a_hist = create_histogram(hsv_copy[i * h:(i+1) * h, j * w:(j+1) * w,:])
        a_cumsum = create_cumulative_sum(a_hist)
        a_mapping = create_mapping(a_hist, a_cumsum,h ,w)
        ahe_image_hsv[i * h:(i+1) * h, j * w:(j+1) * w,:] = apply_mapping(hsv_copy[i * h:(i+1) * h, j * w:(j+1) * w,:], a_mapping)
    
ahe_image = cv2.cvtColor(ahe_image_hsv, cv2.COLOR_HSV2BGR_FULL)



cv2.imshow('ahe12',ahe_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

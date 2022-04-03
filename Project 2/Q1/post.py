#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 14:05:23 2022

@author: pulkit
"""

import cv2
from HistEqUtils import *


def adjust_gamma(image, gamma=1.0):

 	# build a lookup table mapping the pixel values [0, 255] to
 	# their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
 	# apply gamma correction using the lookup table
    return cv2.LUT(image, table)




videofile = 'swadaptivehistogrameq.mp4'

cam = cv2.VideoCapture(videofile) 


while(True): 
    ret, frame = cam.read()                                                  # Reading the frame from video   
    
    # Check if the frame exists, if not exit 
    if not ret:
        break

    frame = adjust_gamma(frame, gamma = 1.4)
    
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(500) & 0xFF == ord('q'):
        break
    
    
    
cam.release()
cv2.destroyAllWindows()













# ahe2 = cv2.imread('ahe_N2.png')
# ahe12 = cv2.imread('ahe_N12.png')

# IMG_W = ahe12.shape[1]
# IMG_H = ahe12.shape[0]
# N = 1
# w = IMG_W // N
# h = IMG_H // N


# gray = cv2.cvtColor(ahe12, cv2.COLOR_BGR2GRAY)
# smooth = cv2.GaussianBlur(ahe12,(3,3),1)
# mB = cv2.medianBlur(ahe12,3)

# hsv = cv2.cvtColor(ahe12, cv2.COLOR_BGR2HSV)
# hsv_copy = hsv.copy()
# ahe_image_hsv = np.zeros_like(ahe2)
# for i in range(N):
#     for j in range(N):    
#         a_hist = create_histogram(hsv_copy[i * h:(i+1) * h, j * w:(j+1) * w,:])
#         a_cumsum = create_cumulative_sum(a_hist)
#         a_mapping = create_mapping(a_hist, a_cumsum,h ,w)
#         ahe_image_hsv[i * h:(i+1) * h, j * w:(j+1) * w,:] = apply_mapping(hsv_copy[i * h:(i+1) * h, j * w:(j+1) * w,:], a_mapping)
    
# ahe_image = cv2.cvtColor(ahe_image_hsv, cv2.COLOR_HSV2BGR_FULL)


# res_convimage = cv2.cvtColor(ahe12, cv2.COLOR_BGR2HSV)
# hist_res = create_histogram(res_convimage)
# chist = clipping(hist_res)
# ccumsum = create_cumulative_sum(chist)
# cmapping = create_mapping(chist, ccumsum, h, w)
# capplied = apply_mapping(res_convimage, cmapping)
# cimage = cv2.cvtColor(capplied, cv2.COLOR_HSV2BGR_FULL)

# cv2.imshow('clipped', cimage)
# cv2.imshow('ahe12', ahe12)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

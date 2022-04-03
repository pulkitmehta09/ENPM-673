#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:00:28 2022

ENPM 673
Project 2 Question1

@author: Pulkit Mehta
UID: 117551693
"""

# ---------------------------------------------------------------------------------
# IMPORTING PACKAGES
# ---------------------------------------------------------------------------------

import cv2
import glob
from matplotlib import pyplot as plt
from cv2 import VideoWriter, VideoWriter_fourcc 
from HistEqUtils import *


# ---------------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------------

images = [cv2.imread(file) for file in glob.glob("adaptive_hist_data/*.png")]   # Read the given images

IMG_W = images[0].shape[1]                                                      # Frame width        
IMG_H = images[0].shape[0]                                                      # Frame height            
N = 12                                                                          # Number of tiles across the width or height of the frame
w = IMG_W // N                                                                  # Width of tile
h = IMG_H // N                                                                  # Height of tile

x = np.linspace(0,255,256)

# Defining video writers
FPS = 1                                                                       
fourcc = VideoWriter_fourcc(*'mp4v')
video = VideoWriter('./original.mp4', fourcc, float(FPS), (IMG_W, IMG_H))
he_video = VideoWriter('./histogrameq.mp4', fourcc, float(FPS), (IMG_W, IMG_H))
ahe_video = VideoWriter('./adaptivehistogrameq.mp4', fourcc, float(FPS), (IMG_W, IMG_H))


# Looping through the given images
for i in range(len(images)):
    
    if(i==0):
        print("performing histogram equalization...")
    
    img = images[i]
    video.write(img)                                                            # Video of original frames

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)                                   # Converting to HSV space
    
    # Histogram equalization
    hist = create_histogram(hsv)                                              
    cumsum = create_cumulative_sum(hist)                                      
    mapping = create_mapping(hist, cumsum, IMG_H, IMG_W)                        
    he_image_hsv = apply_mapping(hsv, mapping)
    he_image = cv2.cvtColor(he_image_hsv, cv2.COLOR_HSV2BGR_FULL)
    he_video.write(he_image)
    
    
    # Adaptive Histogram Equalization
    hsv_copy = hsv.copy()
    ahe_image_hsv = np.zeros_like(img)
    for i in range(N):
        for j in range(N):    
            a_hist = create_histogram(hsv_copy[i * h:(i+1) * h, j * w:(j+1) * w,:])
            a_cumsum = create_cumulative_sum(a_hist)
            a_mapping = create_mapping(a_hist, a_cumsum,h ,w)
            ahe_image_hsv[i * h:(i+1) * h, j * w:(j+1) * w,:] = apply_mapping(hsv_copy[i * h:(i+1) * h, j * w:(j+1) * w,:], a_mapping)
        
    ahe_image = cv2.cvtColor(ahe_image_hsv, cv2.COLOR_HSV2BGR_FULL)
    ahe_video.write(ahe_image)


video.release()    
he_video.release()
ahe_video.release()

print("Videos saved in the current working directory")

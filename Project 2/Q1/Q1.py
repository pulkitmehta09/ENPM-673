#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:00:28 2022

@author: pulkit
"""


import cv2
import numpy as np
import imutils
import glob
from matplotlib import pyplot as plt
from PIL import Image
from cv2 import VideoWriter, VideoWriter_fourcc 


images = [cv2.imread(file) for file in glob.glob("adaptive_hist_data/*.png")]

IMG_W = images[0].shape[1]
IMG_H = images[0].shape[0]
N = 12
w = IMG_W // N
h = IMG_H // N


FPS = 1                                                                       
fourcc = VideoWriter_fourcc(*'MP4V')
video = VideoWriter('./original.mp4', fourcc, float(FPS), (IMG_W, IMG_H))
video_he = VideoWriter('./histogrameq.mp4', fourcc, float(FPS), (IMG_W, IMG_H))
video_ahe = VideoWriter('./adaptivehistogrameq.mp4', fourcc, float(FPS), (IMG_W, IMG_H))


def make_histogram(img):
    """ Take an image and create a histogram from it's luma values """
    y_vals = img[:,:,2].flatten()
    histogram = np.zeros(256, dtype=int)
    for y_index in range(y_vals.size):
        histogram[y_vals[y_index]] += 1
    return histogram

def make_cumsum(histogram):
    """ Create an array that represents the cumulative sum of the histogram """
    cumsum = np.zeros(256, dtype=int)
    cumsum[0] = histogram[0]
    for i in range(1, histogram.size):
        cumsum[i] = cumsum[i-1] + histogram[i]
    return cumsum


def make_mapping(histogram, cumsum, h, w):
    """ Create a mapping s.t. each old luma value is mapped to a new
        one between 0 and 255. Mapping is created using:
         - M(i) = max(0, round((levels*cumsum(i))/(h*w))-1)
        where luma_levels is the number of levels in the image """
    mapping = np.zeros(256, dtype=int)
    levels = 256
    for i in range(histogram.size):
        mapping[i] = max(0, round((levels*cumsum[i])/(h*w))-1)
    return mapping


def apply_mapping(img, mapping):
    """ Apply the mapping to our image """
    new_image = img.copy()
    new_image[:,:,2] = list(map(lambda a : mapping[a], img[:,:,2]))
    return new_image



for i in range(len(images)):
    img = images[i]
    video.write(img)
    
    
    conv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
    # Histogram equalization
    hist = make_histogram(conv)
    cumsum = make_cumsum(hist)
    mapping = make_mapping(hist, cumsum, IMG_H, IMG_W)
    new_convimage = apply_mapping(conv, mapping)
    new_image = cv2.cvtColor(new_convimage, cv2.COLOR_HSV2BGR_FULL)
    video_he.write(new_image)
    
    # Adaptive Histogram Equalization
    ad_img = conv.copy()
    new_ad_image = np.zeros_like(img)
    for i in range(N):
        for j in range(N):    
            a_hist = make_histogram(ad_img[i * h:(i+1) * h, j * w:(j+1) * w,:])
            a_cumsum = make_cumsum(a_hist)
            a_mapping = make_mapping(a_hist, a_cumsum,h ,w)
            new_ad_image[i * h:(i+1) * h, j * w:(j+1) * w,:] = apply_mapping(ad_img[i * h:(i+1) * h, j * w:(j+1) * w,:], a_mapping)
        
    res_image = cv2.cvtColor(new_ad_image, cv2.COLOR_HSV2BGR_FULL)
    video_ahe.write(res_image)



video.release()    
video_he.release()
video_ahe.release()

    
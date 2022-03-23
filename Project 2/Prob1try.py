#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 09:04:03 2022

@author: pulkit
"""


import cv2
import numpy as np
import imutils
import glob
from matplotlib import pyplot as plt
from PIL import Image

images = [cv2.imread(file) for file in glob.glob("adaptive_hist_data/*.png")]

img = images[0]

IMG_H = img.shape[0]
IMG_W = img.shape[1]

# hist = cv2.calcHist([img],[2],None,[256],[0,256])

# color = ('b','g','r')
# # for i,col in enumerate(color):
# #     histr = cv2.calcHist([img],[i],None,[256],[0,256])
# #     plt.plot(histr,color = col)
# #     plt.xlim([0,256])
# # plt.show()

# hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
# v = hsv[:,:,2]


# # -----------------------------------
# # HISTOGRAM EQUALIZATION

# # SIMPLE
# # v_eq = cv2.equalizeHist(v)

# # ADAPTIVE
# clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize = (8,8))
# v_eq = clahe.apply(v)
# # ------------------------------------

# v_eq = v
# x, y = np.where(v > 230)
# for i in range(len(x)):
#     v_eq[x[i]][y[i]] = 180

# xn, yn = np.where(v < 30)
# for i in range(len(xn)):
#     v_eq[xn[i]][yn[i]] = 100
    
    

# # ------------------------------------
# # GETTING BGR FROM HSV
# hsv[:,:,2] = v_eq
# img_back = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
# # ------------------------------------
# # ------------------------------------
# # PLOTTING HISTOGRAM
# # plt.hist(v_eq.ravel(),256,[0,256])
# # plt.figure()
# # plt.hist(v.ravel(),256,[0,256])
# # plt.show()
# # ------------------------------------


conv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)

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

def make_mapping(histogram, cumsum):
    """ Create a mapping s.t. each old luma value is mapped to a new
        one between 0 and 255. Mapping is created using:
         - M(i) = max(0, round((levels*cumsum(i))/(h*w))-1)
        where luma_levels is the number of levels in the image """
    mapping = np.zeros(256, dtype=int)
    levels = 256
    for i in range(histogram.size):
        mapping[i] = max(0, round((levels*cumsum[i])/(IMG_H*IMG_W))-1)
    return mapping

def make_admapping(histogram, cumsum, h, w):
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

def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


gimg = adjust_gamma(img, gamma=1.4)

hist = make_histogram(conv)
cumsum = make_cumsum(hist)
mapping = make_mapping(hist, cumsum)
new_convimage = apply_mapping(conv, mapping)
new_image = cv2.cvtColor(new_convimage, cv2.COLOR_HSV2BGR_FULL)
# new_gamma = adjust_gamma(new_image, gamma=1.2)





# -------------------------ADAPTIVE----------------------------------
N = 8
w = IMG_W // N
h = IMG_H // N
# start = 0
# tile_array = np.empty((h,w,N))
ad_img = conv.copy()
new_ad_image = np.zeros_like(img)

# org_tile1 = img[1 * h : 2 * h, 1 * w : 2 * w, :]
# tile1 = ad_img[1 * h : 2 * h, 1 * w : 2 * w, :]
# histt1 = make_histogram(tile1)
# cumsumt1 = make_cumsum(histt1)
# mappingt1 = make_admapping(histt1, cumsumt1, h, w)
# new_convimaget1 = apply_mapping(tile1, mappingt1)
# new_imaget1 = cv2.cvtColor(new_convimaget1, cv2.COLOR_HSV2BGR_FULL)
# new_imaget1 = adjust_gamma(new_imaget1, gamma=2)
# x = np.linspace(0,255,256)
# plt.bar(x,histt1)



for i in range(N):
    for j in range(N):    
        a_hist = make_histogram(ad_img[i * h:(i+1) * h, j * w:(j+1) * w,:])
        a_cumsum = make_cumsum(a_hist)
        a_mapping = make_admapping(a_hist, a_cumsum,h ,w)
        new_ad_image[i * h:(i+1) * h, j * w:(j+1) * w,:] = apply_mapping(ad_img[i * h:(i+1) * h, j * w:(j+1) * w,:], a_mapping)
        # np.append(tile_array,tile)

res_image = cv2.cvtColor(new_ad_image, cv2.COLOR_HSV2BGR_FULL)
# res_image = adjust_gamma(res_image, gamma=2)
cv2.imshow('adaptive', res_image)
# ---------------------------------------------------------------------


# cv2.imshow('gamma',gimg)
cv2.imshow('org', img)
cv2.imshow('new',new_image)
# cv2.imshow('newgamma', new_gamma)

cv2.waitKey(0)
cv2.destroyAllWindows()
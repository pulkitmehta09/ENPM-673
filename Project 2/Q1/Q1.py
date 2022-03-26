#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 13:00:28 2022

@author: pulkit
"""


from HistEqUtils import *

# Read the given images
images = [cv2.imread(file) for file in glob.glob("adaptive_hist_data/*.png")]

IMG_W = images[0].shape[1]
IMG_H = images[0].shape[0]
N = 2
w = IMG_W // N
h = IMG_H // N


x = np.linspace(0,255,256)

FPS = 1                                                                       
fourcc = VideoWriter_fourcc(*'mp4v')
video = VideoWriter('./original.mp4', fourcc, float(FPS), (IMG_W, IMG_H))
he_video = VideoWriter('./histogrameq.mp4', fourcc, float(FPS), (IMG_W, IMG_H))
ahe_video = VideoWriter('./adaptivehistogrameq.mp4', fourcc, float(FPS), (IMG_W, IMG_H))



for i in range(len(images)):
    img = images[i]
    video.write(img)

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    
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
# cv2.imwrite('ahe_N2.png',ahe_image)
video.release()    
he_video.release()
ahe_video.release()


# # CLAHE

# res_convimage = cv2.cvtColor(ahe_image, cv2.COLOR_BGR2HSV)
# hist_res = make_histogram(res_convimage)
# chist = clipping(hist_res)
# ccumsum = make_cumsum(chist)
# cmapping = make_mapping(chist, ccumsum, h, w)
# capplied = apply_mapping(res_convimage, cmapping)
# cimage = cv2.cvtColor(capplied, cv2.COLOR_HSV2BGR_FULL)

# cv2.imshow('frame', cimage)
# cv2.imshow('frame1', ahe_image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
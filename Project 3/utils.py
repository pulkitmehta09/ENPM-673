#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 18:14:18 2022

@author: pulkit
"""

import cv2
import os
import numpy as np
from dataclasses import dataclass


@dataclass
class DataSet:
    dataset_name: str  
    
    def get_data(self):
        path = 'Data/' + self.dataset_name
        im0 = cv2.imread(path + '/im0.png')
        im1 = cv2.imread(path + '/im1.png')
        if (self.dataset_name == 'curule'):
            cam0 = np.array([[1758.23, 0, 977.42],[0, 1758.23, 552.15],[0,0,1]])
            cam1 = np.array([[1758.23, 0, 977.42],[0, 1758.23, 552.15],[0,0,1]])
            doffs = 0
            baseline = 88.39
            width = 1920
            height = 1080
            ndisp = 220
            vmin = 55
            vmax = 195
            ratio = 0.7
        if (self.dataset_name == 'octagon'):
            cam0 = np.array([[1742.11, 0, 804.90],[0, 1742.11, 541.22],[0,0,1]])
            cam1 = np.array([[1742.11, 0, 804.90],[0, 1742.11, 541.22],[0,0,1]])
            doffs = 0
            baseline = 221.76
            width = 1920
            height = 1080
            ndisp = 100
            vmin = 29
            vmax = 61
            ratio = 0.7
        if (self.dataset_name == 'pendulum'):
            cam0 = np.array([[1729.05, 0, -364.24],[0, 1729.05, 552.22],[0,0,1]])
            cam1 = np.array([[1729.05, 0, -364.24],[0, 1729.05, 552.22],[0,0,1]])
            doffs = 0
            baseline = 537.75
            width = 1920
            height = 1080
            ndisp = 180
            vmin = 25
            vmax = 150
            ratio = 0.65
        
        return {"im0" :im0, "im1": im1, "cam0": cam0, 
                "cam1": cam1, "doffs": doffs, "baseline": baseline, 
                "width": width, "height": height, "ndisp": ndisp,
                "vmin": vmin, "vmax": vmax, "ratio": ratio}
  
    
def resize_image(img, scale = 0.5):

    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)

    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized
          

def match_features(im0,im1, ratio):
    # Converting to grayscale
    gray1 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    
    # Using ORB Feature Descriptor
    orb = cv2.ORB_create()
    
    # Creating keypoints and descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
    
    # Keypoints shown on im1
    # cv2.imshow('keypoints on image',cv2.drawKeypoints(im1,keypoints1,None))
    
    # Brute force matching of descriptors using two best matches for every descriptor
    match = cv2.BFMatcher()
    matches = match.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test to filter and get the best matches
    good = []
    for m,n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
   
    draw_params = dict(matchColor=(0,255,0), singlePointColor=None, flags=2)
    feature_matched_image = cv2.drawMatches(im0,keypoints1,im1,keypoints2,good,None,**draw_params) 
   
    # Defining minimum match count so that stitching is performed only when the number of good matches exceed the minimum match count        
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    else:
        print("Not enough matches are found")
    
    pts1 = np.reshape(pts1, (-1,2))
    pts2 = np.reshape(pts2, (-1,2))
    
    return pts1, pts2, feature_matched_image

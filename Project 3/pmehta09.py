#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:40:55 2022

ENPM 673
Project 3

@author: Pulkit Mehta
UID: 117551693
"""

import cv2
import numpy as np
import imutils
import argparse, textwrap
from utils import *
from data import DataSet
from tqdm import tqdm
import os


# --------------------------------------------------
# MAIN
# --------------------------------------------------

# Argument Parser
ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument("--dataset", default='curule', help = textwrap.dedent('''Dataset name
Available datasets: curule, octagon, pendulum
default: curule''')) 
args = ap.parse_args()

# Reading values of the dataset
dataset_name = args.dataset
dataset = DataSet(dataset_name)
values = dataset.get_data()

# Given images
im0 = values["im0"]
im1 = values["im1"]

# Resizing images
im0 = resize_image(im0, scale=0.5)
im1 = resize_image(im1, scale=0.5)

# Retrieving image shape
h1, w1 = im0.shape[:2]
h2, w2 = im1.shape[:2]

K1 = values["cam0"]                                                             # Camera intrinsic matrix for camera 1
K2 = values["cam1"]                                                             # Camera intrinsic matrix for camera 2        
b = values["baseline"]                                                          # Baseline distance between the camera centers        
f = K1[0,0]                                                                     # Focal length of camera
vmin = values["vmin"]                                                           # Minimum disparity for visualization
vmax = values["vmax"]                                                           # Maximum disparity for visualization

# Getting matching feature points
pts1, pts2, feature_matched_image = match_features(im0, im1, 0.7)
cv2.imshow(" Features Matching", feature_matched_image)

# Getting the Fundamental matrix and inliers after performing RANSAC
F, inliers_pts1, inliers_pts2 = ransac(pts1, pts2)
inliers_pts1 = np.int32(inliers_pts1)
inliers_pts2 = np.int32(inliers_pts2)
print("Fundamental matrix: ", F)


# Getting Homography matrices for Rectification
_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(inliers_pts1), np.float32(inliers_pts2), F, imgSize=(w1,h1))
print("H1: ", H1)
print("H2: ", H2)

# Rectified images
im0_rectified = cv2.warpPerspective(im0, H1, (w1,h1))
im1_rectified = cv2.warpPerspective(im1, H2, (w2,h2))
rectified = np.hstack((im0_rectified, im1_rectified))


# Converting rectified images to gray-scale
im0_rectified_gray = cv2.cvtColor(im0_rectified, cv2.COLOR_BGR2GRAY)
im1_rectified_gray = cv2.cvtColor(im1_rectified, cv2.COLOR_BGR2GRAY)

# Essential Matrix
E = EssentialMatrix(F, K1, K2)
print("Essential matrix: ", E)


# Getting the camera pose from Essential Matrix
R, T, pts_3D = DisambiguateCameraPose(E, inliers_pts1, inliers_pts2, K1, K2)

# Computing Epipolar lines
l1 = cv2.computeCorrespondEpilines(inliers_pts2.reshape(-1,1,2), 2, F)
l2 = cv2.computeCorrespondEpilines(inliers_pts1.reshape(-1,1,2), 1, F)

# Drawing Epipolar lines
im1_epilines, _ = drawlines(im0, im1, l1, inliers_pts1, inliers_pts2)
im0_epilines, _ = drawlines(im1, im0, l2, inliers_pts2, inliers_pts1)
epilines = np.hstack((im0_epilines, im1_epilines))


# Computing Disparity map
disparity_map = get_Disparity_Map(im0_rectified_gray, im1_rectified_gray, 7, 56)

# Gray-scale Disparity map
disparity_map_gray = ((disparity_map / disparity_map.max()) * 255).astype(np.uint8)

# Color Disparity map
disparity_map_color = None
disparity_map_color = cv2.normalize(disparity_map, disparity_map_color, alpha = vmin, beta = vmax, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)
disparity_map_colored = cv2.applyColorMap(disparity_map_color, cv2.COLORMAP_PARULA)

# Computing Depth map
depth_map = get_Depth_Map(disparity_map, b, f)

# Gray-scale Depth map
depth_map_gray = ((depth_map / depth_map.max()) * 255).astype(np.uint8)

# Color Depth map
depth_map_color = cv2.applyColorMap(depth_map_gray, cv2.COLORMAP_PARULA)


cv2.imshow("rectified.png", rectified)
cv2.imshow("epilines.png", epilines)
cv2.imwrite("disparity map gray.png", disparity_map_gray)
cv2.imwrite("disparity map colored.png", disparity_map_colored)
cv2.imwrite("depth map gray.png", depth_map_gray)
cv2.imwrite("depth map color.png", depth_map_color)


cv2.waitKey(0)
cv2.destroyAllWindows()
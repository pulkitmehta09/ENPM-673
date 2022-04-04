#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 09:41:03 2022

ENPM 673
Project 2 Problem 3

@author: Pulkit Mehta
UID: 117551693
"""

# ---------------------------------------------------------------------------------
# IMPORTING PACKAGES
# ---------------------------------------------------------------------------------

#  Without history, cropped lane
import cv2
import numpy as np
import imutils
from cv2 import VideoWriter, VideoWriter_fourcc 
from utils import *


videofile = 'challenge.mp4'                                                     # Video file
cam = cv2.VideoCapture(videofile)  

yellow_HSV_th_min = np.array([0, 70, 70])
yellow_HSV_th_max = np.array([50, 255, 255])
count = 0
left_diff = []
right_diff = []
slope_hist = []

FPS = 12                                                                       
fourcc = VideoWriter_fourcc(*'mp4v')
video = VideoWriter('./Problem3result.mp4', fourcc, float(FPS), (1310, 612))

while(True): 
    
    ret, frame = cam.read()                                                     # Reading the frame from video
       
    # Check if the frame exists, if not exit 
    if not ret:
        break
    
    
    lanes_warped, matrix, lanes = prepare_warped(frame)
    lanes_warped_bgr = cv2.cvtColor(lanes_warped, cv2.COLOR_GRAY2BGR)
 
    out_img = cv2.cvtColor(lanes_warped, cv2.COLOR_GRAY2BGR)
    
    if (count == 0):
        left_fit, right_fit, _ = fit_lanes(lanes_warped, out_img)
    
    left_fit_prev = left_fit
    right_fit_prev = right_fit
    
    left_fit, right_fit, _ = fit_lanes(lanes_warped, out_img)
    ld, rd = compare_fits(left_fit_prev, right_fit_prev,left_fit, right_fit, out_img)
    left_diff.append(np.mean(ld))
    right_diff.append(np.mean(rd))    
    
    if (np.mean(ld) > 25):
        left_fit = left_fit_prev
        right_fit = right_fit    
    else:
        left_fit = left_fit


    if (np.mean(rd) > 4.5):
        right_fit = right_fit_prev
        left_fit = left_fit
    else:
        right_fit = right_fit
    
    
    left_curvature, right_curvature, avg_curvature, turn, slope = get_curvature(left_fit, right_fit, out_img.shape)
    slope_hist.append(slope)
    
    lane_lines, lane_region = draw_lane(left_fit, right_fit, out_img)
 
    unwarp = cv2.warpPerspective(lane_region, np.linalg.inv(matrix), (frame.shape[1],frame.shape[0]))
    result = cv2.bitwise_or(unwarp, frame)
    
    # cv2.putText(result, 'Curvature - Left: {:.0f} m; Right: {:.0f} m; Center: {:.0f} m'.format(left_curvature,right_curvature,avg_curvature), (100,150), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
    cv2.putText(result, turn, (100,100), cv2.FONT_HERSHEY_TRIPLEX, 1, (0,0,255), 2)
    cv2.putText(result, '(5)', (10,40), cv2.FONT_HERSHEY_TRIPLEX, 1, (255,255,255), 1)
    
    stacked1 = np.hstack((frame, lanes))
    stacked2 = np.hstack((lanes_warped_bgr,lane_lines))
    cv2.putText(stacked2, '(3)', (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255),1)
    cv2.putText(stacked2, '(4)', (210,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255),1)
    stacked1 = imutils.resize(stacked1, width = 400)
    cv2.putText(stacked1, '(1)', (10,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255),1)
    cv2.putText(stacked1, '(2)', (210,20), cv2.FONT_HERSHEY_PLAIN, 1, (255,255,255),1)
    stacked12 = np.vstack((stacked1,stacked2))
    result_stacked = imutils.resize(result, height = 512)
    stacked = np.hstack((result_stacked, stacked12))
    blank_image = np.zeros((100,stacked.shape[1],3), np.uint8)
    blank_image[:,:,:] = [255,200,200]
    cv2.putText(blank_image, '(1): Original frame, (2): Detected lane markings, (3): Warped frame, (4): Curve fitting, (5): Detected Lane region superimposed with turn prediction'
                ,(20,20), cv2.FONT_HERSHEY_DUPLEX, 0.5, (255,0,0), 2) 
    cv2.putText(blank_image, 'Curvature - Left: {:.0f} m; Right: {:.0f} m; Mean: {:.0f} m'.format(left_curvature,right_curvature,avg_curvature), (20,50), cv2.FONT_HERSHEY_DUPLEX, 0.5, (0,0,255), 2)    
    stacked_res = np.vstack((stacked,blank_image))
    
    cv2.imshow('frame', stacked_res)
    video.write(stacked_res)   
    
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break

    count += 1
    
    
cam.release()
video.release()
cv2.destroyAllWindows()

print("Video saved in the current working directory")
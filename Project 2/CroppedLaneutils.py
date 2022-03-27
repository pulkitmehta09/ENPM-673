#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 20:42:34 2022

@author: pulkit
"""

import cv2
import numpy as np

def distances(lines):
     lines = np.reshape(lines,(-1,4))
     distance_array = np.empty((lines.shape[0],1))
     for i in range(lines.shape[0]):
          distance_array[i] = np.sqrt((lines[i][2] - lines[i][0]) ** 2 + (lines[i][3] - lines[i][1]) ** 2)
         
     return distance_array


def Modify_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    ret,thresh = cv2.threshold(blur,160,255,cv2.THRESH_BINARY)             # Binary thresholding
    edges = cv2.Canny(thresh, 20, 120)

    return thresh


def region(frame):
    height, width = frame.shape[:2]
    # triangle = np.array([[(100, height), (480,300), (width, height)]])
    
    # trapezoid = np.array([[(200, height-50), (630,425), (725,425), (width - 150, height-50)]])
    # Cropped
    trapezoid = np.array([[(220, height-50), (610,440), (725,440), (width - 150, height-50)]])
    # cv2.line(frame,tuple(triangle[0][0]), tuple(triangle[0][1]),red,2)
    # cv2.line(frame,tuple(triangle[0][0]), tuple(triangle[0][2]),red,2)
    # cv2.line(frame,tuple(triangle[0][1]), tuple(triangle[0][2]),red,2)
        
    mask = np.zeros_like(frame)
    mask = cv2.fillPoly(mask, trapezoid, (255,255,255))
    mask = cv2.bitwise_and(frame, mask)
    
    return mask


def make_HoughLines(frame):
    copy = frame.copy()
    copy = cv2.cvtColor(copy,cv2.COLOR_GRAY2BGR)
    lines = cv2.HoughLinesP(frame, 2, np.pi/180, 100, minLineLength = 40, maxLineGap = 5)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(copy,(x1,y1),(x2,y2),(0,255,0),2)
    return lines, copy
    
def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)

def classify(warped):
    
    histogram = np.sum(warped[int(warped.shape[0] / 2):, :], axis=0)
    out_img = np.dstack((warped, warped, warped)) * 255
    
    midpoint = np.int32(histogram.shape[0] / 2)
    
    pts1 = np.array(warped[:,:100].nonzero())
    pts2 = np.array(warped[:,100:].nonzero())
    pts1_x = np.array(pts1[0])
    pts1_y = np.array(pts1[1])
    pts2_x = np.array(pts2[0])
    pts2_y = np.array(pts2[1]+100)
    
    if (pts1.shape[1] < pts2.shape[1]):
        dashed_x = pts1_x
        dashed_y = pts1_y
        solid_x = pts2_x
        solid_y = pts2_y
    else:
        dashed_x = pts2_x
        dashed_y = pts2_y
        solid_x = pts1_x
        solid_y = pts1_y

    out_img[dashed_x,dashed_y] = [0,0,255]
    out_img[solid_x,solid_y] = [0,255,0]

    return out_img


def prepare_warped(frame):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]    
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(20,20))
    cl1 = clahe.apply(v)
    hsv[:,:,2] = cl1
    img_back = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img_back = adjust_gamma(img_back, gamma = 0.5)

    
    modified = Modify_frame(img_back)
    roi = region(modified)

    lower = np.array([0,100,100])
    upper = np.array([30,255,255])
    
    yellow_lane_roi = region(frame) 
    yellow_lane_hsv = cv2.cvtColor(yellow_lane_roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(yellow_lane_hsv, lower, upper)
    

    height, width, _ = frame.shape
    pts1 = np.float32([[(220, height-50), (610,440), (725,440), (width - 150, height-50)]])
    pts2 = np.float32([[0,400],[0,0],[200,0],[200,400]])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(roi, matrix, (200,400))
    # warped = cv2.flip(warped,0)
    yellow_lane_warped = cv2.warpPerspective(mask, matrix, (200,400))
    warped[:,:100] = 0
    comb = warped + yellow_lane_warped
    
    return comb, matrix






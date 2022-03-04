#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 12:13:23 2022

@author: pulkit
"""

import cv2
import numpy as np
from newfunctions import *
import imutils
from matplotlib import pyplot as plt
from faltu import *


videofile = '1tagvideo.mp4'
cam = cv2.VideoCapture(videofile)
C = np.array([0,0,200,0,200,200,0,200])
gotID = False       # Flag to check ID found or not
got_fft = False

I = (200,200) 
testudo = cv2.imread('testudo.png')
testudo = cv2.resize(testudo, (200,200), interpolation= cv2.INTER_LINEAR)
testudo = imutils.rotate(testudo, 90)  
# Align testudo bottom right corner with tag's changing BR corner so as to align the testudo always  

# def filterCornersLine(corners, rows, cols):
#     [vx, vy, x, y] = cv2.fitLine(corners, cv2.DIST_HUBER, 0, 0.1, 0.1)
#     lefty = int((-x * vy / vx) + y)
#     righty = int(((cols - x) * vy / vx) + y)

#     cornerdata = []
#     tt = 0
#     for i in corners:
#         xl, yl = i.ravel()
#         # check distance to fitted line, only draw corners within certain range
#         distance = dist(0, lefty, cols - 1, righty, xl, yl)
#         if distance > 40:  ## threshold important -> make accessible
#             cornerdata.append(tt)

#         tt += 1

#     corners_final = np.delete(corners, [cornerdata], axis=0)  # delete corners to form new array

#     return corners_final 

while(True): 
    ret, frame = cam.read()
    frame2 = frame.copy()
    # Check if the frame exists, if not exit 
    if not ret:
        break
     
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    ret,thresh = cv2.threshold(grayscale,180,255,cv2.THRESH_BINARY)
    
    
    kernel = np.ones((5,5),np.uint8)
    
    
    erosion = cv2.morphologyEx(thresh, cv2.MORPH_ERODE,kernel, iterations = 1)
    dilation = cv2.morphologyEx(erosion, cv2.MORPH_DILATE,kernel, iterations = 1)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel,iterations = 1)
    
    
    # rect = cv2.morphologyEx(thresh, cv2.MORPH_GRADIENT,kernel)
    # rect = cv2.morphologyEx(rect,cv2.MORPH_CLOSE, kernel)
    # edges = cv2.Canny(rect,50,100)
    
    # TODO Fit line 
    
    # ----------------------------------------------------------------
    # # Shi-Tomasi Corner Detector
    corners = cv2.goodFeaturesToTrack(opening, 50, 0.1, 10)
    corners = np.int0(corners)
    

    # for i in corners:
    #     x,y = i.ravel()
    #     cv2.circle(frame, (x,y), 5, (255,5,5), -1)
    # ----------------------------------------------------------------
    # outer rectangle
    points = idkwhy(corners)
    points = orderpts(points)
    
    
    kernel2 = np.ones((9,9),np.uint8)
    # INSIDE RECTANGLE
    grayscale2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY) 
    # blurred = cv2.medianBlur(grayscale2,5)
    # blurred = cv2.GaussianBlur(grayscale2, (3,3),0)

    ret,thresh2 = cv2.threshold(grayscale2,200,255,cv2.THRESH_BINARY)
    erosion = cv2.morphologyEx(thresh2, cv2.MORPH_ERODE, kernel2, iterations = 3)
    dilation = cv2.morphologyEx(erosion, cv2.MORPH_DILATE, kernel2, iterations = 3)
    opening = cv2.morphologyEx(dilation, cv2.MORPH_OPEN, kernel2)
    
    # INSIDE Shi-Tomasi Corner Detector
    # rect = cv2.morphologyEx(thresh2, cv2.MORPH_OPEN,kernel, iterations=2)
    # rect = cv2.Canny(rect,50,100)
    corners2 = cv2.goodFeaturesToTrack(opening, 30, 0.1, 35)
    corners2 = np.int0(corners2)
    for i in corners2:
        x,y = i.ravel()
        cv2.circle(frame2, (x,y), 5, (255,5,5), -1)

    inrect = nearestpoint(points,corners2)
    tag = orderpts(inrect)
    cv2.circle(frame, tuple(inrect[0]), 5, (255,5,5), -1)
    cv2.circle(frame, tuple(inrect[1]), 5, (255,5,5), -1)
    cv2.circle(frame, tuple(inrect[2]), 5, (255,5,5), -1)
    cv2.circle(frame, tuple(inrect[3]), 5, (255,5,5), -1)
    
    # TAG IDENTIFICATION
    tag = orderpts(tag)
    H = homography(tag, C)
    warped = warpPerspective(H, frame, I)

    rotated = imutils.rotate(warped, 90)        # Full Tag rotated image
    center_tag = rotated[75:125,75:125]  
    if not gotID:
        gotID, ID, ar = getTagID(center_tag)
        
    H_testudo = homography(C, tag)
    frame = warpTestudo(H_testudo, I, frame, testudo)
    
    H_cube = homography(tag, C)
    
    P, T = getProjectionMatrix(H) 
    drawCube(P,frame)
        
    
    
    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()

print("Tag ID: ", ID)
print(ar)
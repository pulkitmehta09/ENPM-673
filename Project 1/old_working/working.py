#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:43:16 2022

@author: pulkit
"""

# TODO argument parser and option to choose from argument or default

import cv2
import numpy as np
from newfunctions import *
import imutils
from matplotlib import pyplot as plt

videofile = '1tagvideo.mp4'
cam = cv2.VideoCapture(videofile)

testudo = cv2.imread('testudo.png')
testudo = cv2.resize(testudo, (200,200), interpolation= cv2.INTER_LINEAR)


gotID = False       # Flag to check ID found or not
got_fft = False

I = (200,200)       # Tag dimensions

# Corner matrix -------------------------------
C = np.array([0,0,200,0,200,200,0,200])
prev_pose = 0
count = 0

while(True): 
    ret, frame = cam.read()
    
    # Check if the frame exists, if not exit 
    if not ret:
        break
     
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
    ret,thresh = cv2.threshold(grayscale,180,255,cv2.THRESH_BINARY)
   
    # ----------------------------------------------------------------
    # Contour
    contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
   
    for contour in contours:
        if (20000 < cv2.contourArea(contour) < 100000):
            # -------------------------------------
            # PolyDP
            peri = cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour,0.1 * peri,True)
            # cv2.drawContours(frame,[approx],-1,(0,255,0),3) 
            # -------------------------------------
    
     
    corners = orderpts(approx)
    
    H = homography(corners, C)
    warped = warpPerspective(H, frame, I)
    warped = cv2.flip(warped,0)
    pose = get_tag_orientation(warped)
    # print(pose)
    rotated = orientTag(pose,warped)
    # rotated = imutils.rotate(warped, 180)        # Full Tag rotated image
    center_tag = rotated[75:125,75:125]         # 2x2 grid of tag
    if not gotID:
        gotID, ID, ar = getTagID(center_tag)
        print("Tag ID: ", ID)
        print(ar)

    # pose = get_tag_orientation(rotated)
    # print(pose)
    if not (prev_pose == pose):
        testudo = orientTestudo(pose, testudo)
    
    prev_pose = pose
    H_testudo = homography(C, corners)
   
    # # ---------------------------------------------------------------- 
    # Working
    # H_testudo_inv = np.linalg.inv(H_testudo)
    # for a in range(warped.shape[1]):
    #     for b in range(warped.shape[0]):
    #         x, y, z = np.dot(H_testudo_inv,[a,b,1])
    #         frame[int(y/z)][int(x/z)] = testudo[a][b]
    # ----------------------------------------------------------------

    # Superimposing Testudo
    # frame = warpTestudo(H_testudo, I, frame, testudo)
    

    
    # -----------------------------------------------------------------
    # Fourier 
    if(count == 17)  :
        fft(grayscale)
        got_fft = True
    # ---------------------------------------------------------------   
    
    # Cube
    H_cube = homography(corners, C)
    
    P, T = getProjectionMatrix(H) 
    drawCube(P,frame)
        
    
    cv2.imshow('frame', frame)
    # cv2.imshow('rotated', rotated)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count+=1

cam.release()
cv2.destroyAllWindows()

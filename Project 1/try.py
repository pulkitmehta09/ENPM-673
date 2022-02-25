#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:43:16 2022

@author: pulkit
"""
# Original Working verion 
# TODO argument parser and option to choose from argument or default

import cv2
import numpy as np
from functions import *
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
# C = C.reshape(4,1,2)
# ---------------------------------------------
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
            cv2.drawContours(frame,[approx],-1,(0,255,0),3) 
            # -------------------------------------
    
    # ----------------------------------------------------------------

    # # ----------------------------------------------------------------
    # # Shi-Tomasi Corner Detector            
    # corners = cv2.goodFeaturesToTrack(thresh, 4, 0.9, 10)
    # corners = np.int0(corners)
    # for i in corners:
    #     x,y = i.ravel()
    #     cv2.circle(frame, (x,y), 3, (10,10,255), -1)
    # # ----------------------------------------------------------------

            
    # # ----------------------------------------------------------------
    # # Harris Corner Detector
    # dst = cv2.cornerHarris(thresh, 2, 3, 0.04)
    # frame[dst > 0.51 * dst.max()] = [0,0,255]
    
    # # ----------------------------------------------------------------
   
    
    # # ----------------------------------------------------------------
    # TODO strighten tag without hardcoding
    
    # tag = warped[50:150,50:150]
    # tag_TL = tag[0:25,0:25]
    # tag_BL = tag[75:100,0:25]
    # tag_BR = tag[75:100,75:100]
    # tag_TR = tag[0:25,75:100]
    
    # # ----------------------------------------------------------------
    
    # # ----------------------------------------------------------------
    # Drawing on tags

    # font = cv2.FONT_HERSHEY_SIMPLEX
    # fontScale = 0.1
    # color = (0,0,255)
    # thickness = 1

    # center_tag = cv2.putText(center_tag, 'TL', orgTL, font, fontScale, color, thickness)
    # center_tag = cv2.putText(center_tag, 'TR', orgTR, font, fontScale, color, thickness)
    # center_tag = cv2.putText(center_tag, 'BL', orgBL, font, fontScale, color, thickness)
    # center_tag = cv2.putText(center_tag, 'BR', orgBR, font, fontScale, color, thickness)
    
    # # ----------------------------------------------------------------
     
    corners = orderpts(approx)
    H = homography(corners, C)
    warped = warpPerspective(H, frame, I)
    # rotated = cv2.rotate(warped, cv2.ROTATE_90_COUNTERCLOCKWISE)
    # rotated = RotateTag(warped, 90)
    rotated = imutils.rotate(warped, 90)        # Full Tag rotated image
    center_tag = rotated[75:125,75:125]         # 2x2 grid of tag
       
    if not gotID:
        gotID, ID, ar = getTagID(center_tag)
 
    P, T = getProjectionMatrix(H) 
 
    # H_testudo = homography(corners, C)
    warped_testudo = warpPerspective(H_testudo, testudo, I)
    
    # -----------------------------------------------------------------
    # Fourier 
    # if(count == 17)  :
    #     fft(grayscale, frame)
    #     got_fft = True
    # ---------------------------------------------------------------    
    cv2.imshow('frame', center_tag)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count+=1

cam.release()
cv2.destroyAllWindows()
print("Tag ID: ", ID)
print(ar)
# NOT Required ----------------------------------------------
# x = res[:,0,0]
# y = res[:,1,0]
# z = res[:,2,0]
# fig = plt.figure()
# ax = fig.add_subplot(projection = '3d')
# ax.scatter(x,y,z)
# plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 17 16:43:16 2022

@author: pulkit
"""

import cv2
import numpy as np

videofile = '1tagvideo.mp4'
cam = cv2.VideoCapture(videofile)

def homography(C):
    x_c = np.array([0, 200, 200, 0])
    y_c = np.array([0, 0, 200, 200])
    x_w = np.array([C[0][0][0], C[1][0][0], C[2][0][0], C[3][0][0]])
    y_w = np.array([C[0][0][1], C[1][0][1], C[2][0][1], C[3][0][1]])

    A = np.array([[x_w[0], y_w[0], 1, 0, 0, 0, -x_c[0] * x_w[0], -x_c[0] * y_w[0], -x_c[0]],
                  [0, 0, 0, x_w[0], y_w[0], 1, -y_c[0] * x_w[0], -y_c[0] * y_w[0], -y_c[0]],
                  [x_w[1], y_w[1], 1, 0, 0, 0, -x_c[1] * x_w[1], -x_c[1] * y_w[1], -x_c[1]],
                  [0, 0, 0, x_w[1], y_w[1], 1, -y_c[1] * x_w[1], -y_c[1] * y_w[1], -y_c[1]],
                  [x_w[2], y_w[2], 1, 0, 0, 0, -x_c[2] * x_w[2], -x_c[2] * y_w[2], -x_c[2]],
                  [0, 0, 0, x_w[2], y_w[2], 1, -y_c[2] * x_w[2], -y_c[2] * y_w[2], -y_c[2]],
                  [x_w[3], y_w[3], 1, 0, 0, 0, -x_c[3] * x_w[3], -x_c[3] * y_w[3], -x_c[3]],
                  [0, 0, 0, x_w[3], y_w[3], 1, -y_c[3] * x_w[3], -y_c[3] * y_w[3], -y_c[3]]])
    
    u, s, vh = np.linalg.svd(A, full_matrices = True)
    vt = vh.transpose()
    h = vt[:,-1]
    h = h / h[-1]
    H = np.reshape(h,(3,3))
    
    return H
 
    
 
K = np.array([[1346.100595, 0, 932.1633975],[0, 1355.933136, 654.8986796],[0, 0, 1]])

    

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
    
    
    
    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
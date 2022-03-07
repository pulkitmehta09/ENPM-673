#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 26 12:13:23 2022

ENPM 673
Project 1

@author: Pulkit Mehta
UID: 117551693
"""

# ---------------------------------------------------------------------------------
# IMPORTING PACKAGES
# ---------------------------------------------------------------------------------

import cv2
from utils import *

# ---------------------------------------------------------------------------------
# Problem 1(a) Fast Fourier Transform to detect AR tag
# ---------------------------------------------------------------------------------

videofile = '1tagvideo.mp4'                                     # Video file
cam = cv2.VideoCapture(videofile)                   
got_fft = False                                                 # Flag to check if FFT is performed or not
count = 0                                                       # Counter for frame

  
while(True): 
    
    ret, frame = cam.read()
    
    # Check if the frame exists, if not exit 
    if not ret:
        break
     
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)         # Converting image to gray-scale.
    
    # Fast Fourier Transform for a specific frame 
    if(count == 17)  :
        fft(grayscale)
        got_fft = True
        print("Result of Fast Fourier Transform saved in the current working directory.")
        break

    cv2.imshow('frame', frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    count+=1

cam.release()
cv2.destroyAllWindows()


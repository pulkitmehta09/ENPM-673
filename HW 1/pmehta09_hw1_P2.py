#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Homework 1 Problem 2
# ENPM673 Spring 2022
# Section 0101

@author: Pulkit Mehta
UID: 117551693

"""
# ---------------------------------------------------------------------------------
# IMPORTING PACKAGES
# ---------------------------------------------------------------------------------

import cv2
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# ---------------------------------------------------------------------------------

def Retrievecenter(videofile):
    """
    This function detects coordinates of the center of the ball.

    Parameters
    ----------
    videofile : video file
        Input video file.

    Returns
    -------
    coor : Array
        Stored center coordinates of the detected ball every frame.

    """
    
    cam = cv2.VideoCapture(videofile)
    
    coor = np.empty((0,2),int)                                   # Array to store coordinates of center of ball.
    
    # Defining range for red colour
    redLower1 = np.array([0,50,50])
    redUpper1 = np.array([10,255,255])
    redLower2 = np.array([170,50,50])
    redUpper2 = np.array([180,255,255])

    
    while(True): 
        ret, frame = cam.read()
        
        # Check if the frame exists, if not exit 
        if not ret:
            break
        
        frame = cv2.flip(frame,0)                                # Flipping the frame in order to align it with origin of resultant matplotlib plot. 
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)             # Converting RGB image to HSV.
        # Applying mask  
        mask1 = cv2.inRange(hsv, redLower1, redUpper1)              
        mask2 = cv2.inRange(hsv, redLower2, redUpper2)
        mask = mask1 + mask2
        M = cv2.moments(mask)                                    # Calculating first order moments
        cX = int(M["m10"] / M["m00"])                            # x-coordinate of center of ball
        cY = int(M["m01"] / M["m00"])                            # y-coordinate of center of ball
        coor = np.append(coor, [[cX,cY]], axis=0)                # Appending center coordinates in array  
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cam.release()
    cv2.destroyAllWindows()
    
    return coor

def StandardLeastSquares(coor):
    """
    This function calculates a parabolic curve fit using Standard Least Squares method. 
    
    Parameters
    ----------
    coor : Array
        Input array with x and y coordinates.

    Returns
    -------
    res : Array
        Standard Least Squares fit result.

    """
    
    x = coor[:,0]                                                # x-coordinates array
    y = coor[:,1]                                                # y-coordinates array           
    
    # System of equations formed for parabolic fit, i.e., ax^2 + bx + c = y
    x_sq = np.power(x,2)
    A = np.stack((x_sq, x, np.ones((len(x)), dtype=int )), axis=1)
    A_t = A.transpose()
    A_tA = A_t.dot(A)
    A_tY = A_t.dot(y)
    x_bar = (np.linalg.inv(A_tA)).dot(A_tY)                     
    res = A.dot(x_bar)                                           # Output after applying Least squares model 
    
    return res

# ---------------------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------------------

video1 = 'ball_video1.mp4'
video2 = 'ball_video2.mp4'

# ---------------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------------

c1 = Retrievecenter(video1)                                      # Retriving centre coordinates from Video 1
c2 = Retrievecenter(video2)                                      # Retriving centre coordinates from Video 2

y1 = StandardLeastSquares(c1)                                    # Applying Least squares fit on ball trajectory from Video 1.
y2 = StandardLeastSquares(c2)                                    # Applying Least squares fit on ball trajectory from Video 2.   

fig = plt.figure()

# Plot for Video 1
plt.subplot(121)
plt.title('Video 1')
plt.xlabel('time')
plt.ylabel('position')
plt.plot(c1[:,0], c1[:,1],'bo', label = 'Detected ball center')
plt.plot(c1[:,0],y1, c='red', label = 'Least Squares')
plt.legend()

# Plot for Video 2
plt.subplot(122)
plt.title('Video 2')
plt.xlabel('time')
plt.ylabel('position')
plt.plot(c2[:,0], c2[:,1],'bo', label = 'Detected ball center')
plt.plot(c2[:,0],y2, c='red', label = 'Least Squares')
plt.legend()

plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Homework 1 Problem 2
# ENPM673 Spring 2022
# Section 0101

@author: Pulkit Mehta
UID: 117551693

"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
import imutils
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib','qt')


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
    
    coor = np.empty((0,2),int)  
    
    redLower1 = np.array([0,50,50])
    redUpper1 = np.array([10,255,255])
    
    redLower2 = np.array([170,50,50])
    redUpper2 = np.array([180,255,255])

    
    while(True): 
        ret, frame = cam.read()
        
        if not ret:
            break
        
        frame = imutils.resize(frame, width=1200)
        frame = cv2.flip(frame,0)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        mask1 = cv2.inRange(hsv, redLower1, redUpper1)
        mask2 = cv2.inRange(hsv, redLower2, redUpper2)
        mask = mask1 + mask2
        red = cv2.bitwise_and(frame, frame, mask = mask)
        M = cv2.moments(mask)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        coor = np.append(coor, [[cX,cY]], axis=0)
        # cv2.imshow('ball',red)
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
    
    x = coor[:,0]
    y = coor[:,1]
    x_sq = np.power(x,2)
    A = np.stack((x_sq, x, np.ones((len(x)), dtype=int )), axis=1)
    A_t = A.transpose()
    A_tA = A_t.dot(A)
    A_tY = A_t.dot(y)
    x_bar = (np.linalg.inv(A_tA)).dot(A_tY)
    res = A.dot(x_bar)
    
    return res

video1 = 'ball_video1.mp4'
video2 = 'ball_video2.mp4'

c1 = Retrievecenter(video1)
c2 = Retrievecenter(video2)


y1 = StandardLeastSquares(c1)
y2 = StandardLeastSquares(c2)

fig = plt.figure()

plt.subplot(121)
plt.title('Video 1')
plt.xlabel('time')
plt.ylabel('position')
plt.plot(c1[:,0], c1[:,1],'bo', label = 'Detected ball center')
plt.plot(c1[:,0],y1, c='red', label = 'Least Squares')
plt.legend()

plt.subplot(122)
plt.title('Video 2')
plt.xlabel('time')
plt.ylabel('position')
plt.plot(c2[:,0], c2[:,1],'bo', label = 'Detected ball center')
plt.plot(c2[:,0],y2, c='red', label = 'Least Squares')

plt.legend()


plt.show()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:37:56 2022

@author: pulkit
"""

import cv2
import numpy as np


videofile = 'challenge.mp4'                                                     # Video file
cam = cv2.VideoCapture(videofile)   

red = (0,0,255)
blue = (255,0,0)
green = (0,255,0)

def distances(lines):
     lines = np.reshape(lines,(-1,4))
     distance_array = np.empty((lines.shape[0],1))
     for i in range(lines.shape[0]):
          distance_array[i] = np.sqrt((lines[i][2] - lines[i][0]) ** 2 + (lines[i][3] - lines[i][1]) ** 2)
         
     return distance_array


def Modify_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    ret,thresh = cv2.threshold(blur,125,255,cv2.THRESH_BINARY)             # Binary thresholding
    edges = cv2.Canny(thresh, 20, 200)

    return thresh    
    
def region(frame):
    height, width = frame.shape
    # triangle = np.array([[(100, height), (480,300), (width, height)]])
    
    trapezoid = np.array([[(150, height-50), (625,425), (725,425), (width - 150, height-50)]])
    
    # cv2.line(frame,tuple(triangle[0][0]), tuple(triangle[0][1]),red,2)
    # cv2.line(frame,tuple(triangle[0][0]), tuple(triangle[0][2]),red,2)
    # cv2.line(frame,tuple(triangle[0][1]), tuple(triangle[0][2]),red,2)
        
    mask = np.zeros_like(frame)
    mask = cv2.fillPoly(mask, trapezoid, (255,255,255))
    mask = cv2.bitwise_and(frame, mask)
    
    return mask


def make_HoughLines(frame):
    lines = cv2.HoughLinesP(frame, 2, np.pi/180, 100, minLineLength = 40, maxLineGap = 5)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
    return frame
    


C = np.array([[(0,0), (150, 400), (250,400), (400,0)]])

while(True): 
    ret, frame = cam.read()                                                     # Reading the frame from video
       
    # Check if the frame exists, if not exit 
    if not ret:
        break
    
    copy = frame.copy()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]    
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(20,20))
    cl1 = clahe.apply(v)
    hsv[:,:,2] = cl1
    img_back = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    modified = Modify_frame(img_back)
    roi = region(modified)
    
    # lines = cv2.HoughLinesP(roi, 2, np.pi/180, 100, minLineLength = 40, maxLineGap = 5)
    # averaged_lines = average(copy, lines)
    # black_lines = display_lines(copy, averaged_lines)
    # lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
    # for i in range(lines.shape[0]):    
    #     for x1,y1,x2,y2 in lines[i]:
    #         cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
    # height, width, _ = frame.shape
    # src_pts = np.array([[(100, height), (470,300), (490,300), (width, height)]])
    # H, mask = cv2.findHomography(src_pts, C, cv2.RANSAC, 5)
    # warped = cv2.warpPerspective(roi, H, (400,200))
    # warped = cv2.perspectiveTransform(roi, H)
    # warped = cv2.flip(warped, 0)





    
    cv2.imshow('res', roi)
    # cv2.imshow('frame', warped)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    
    
    
cam.release()
cv2.destroyAllWindows()




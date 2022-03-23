#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 17 11:37:56 2022

@author: pulkit
"""

import cv2
import numpy as np


videofile = 'whiteline.mp4'                                                     # Video file
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
    ret,thresh = cv2.threshold(blur,225,255,cv2.THRESH_BINARY)             # Binary thresholding
    edges = cv2.Canny(thresh, 20, 120)

    return thresh    
    
def region(frame):
    height, width = frame.shape
    # triangle = np.array([[(100, height), (480,300), (width, height)]])
    
    trapezoid = np.array([[(100, height), (470,300), (490,300), (width, height)]])
    
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
    

# def average(image, lines):
#     left = []
#     right = []

#     if lines is not None:
#       for line in lines:
#         print(line)
#         x1, y1, x2, y2 = line.reshape(4)
#         #fit line to points, return slope and y-int
#         parameters = np.polyfit((x1, x2), (y1, y2), 1)
#         print(parameters)
#         slope = parameters[0]
#         y_int = parameters[1]
#         #lines on the right have positive slope, and lines on the left have neg slope
#         if slope < 0:
#             left.append((slope, y_int))
#         else:
#             right.append((slope, y_int))
            
#     #takes average among all the columns (column0: slope, column1: y_int)
#     if right:
#         right_avg = np.average(right, axis=0)
#         right_line = make_points(image, right_avg)
#     if left:
#         left_avg = np.average(left, axis=0)
#         #create lines based on averages calculates
#         left_line = make_points(image, left_avg)

#     return np.array([left_line, right_line])    


# def make_points(image, average):
#     print(average)
#     try:
#         slope, y_int = average
#     except TypeError:
#         slope, y_int = 0,0
#     y1 = image.shape[0]
#     #how long we want our lines to be --> 3/5 the size of the image
#     y2 = int(y1 * (3/5))
#     #determine algebraically
#     x1 = int((y1 - y_int) // slope)
#     x2 = int((y2 - y_int) // slope)
#     return np.array([x1, y1, x2, y2])

# def display_lines(image, lines):
#     lines_image = np.zeros_like(image)
#     #make sure array isn't empty
#     if lines is not None:
#         for line in lines:
#             x1, y1, x2, y2 = line
#             #draw lines on a black image
#             cv2.line(lines_image, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 10)
#     return lines_image

C = np.array([[(0,0), (150, 400), (250,400), (400,0)]])

while(True): 
    ret, frame = cam.read()                                                     # Reading the frame from video
       
    # Check if the frame exists, if not exit 
    if not ret:
        break
    
    copy = frame.copy()
    modified = Modify_frame(frame)
    roi = region(modified)
    
    # lines = cv2.HoughLinesP(roi, 2, np.pi/180, 100, minLineLength = 40, maxLineGap = 5)
    # averaged_lines = average(copy, lines)
    # black_lines = display_lines(copy, averaged_lines)
    # lanes = cv2.addWeighted(copy, 0.8, black_lines, 1, 1)
    # for i in range(lines.shape[0]):    
    #     for x1,y1,x2,y2 in lines[i]:
    #         cv2.line(frame,(x1,y1),(x2,y2),(0,0,255),2)
    height, width, _ = frame.shape
    src_pts = np.array([[(100, height), (470,300), (490,300), (width, height)]])
    H, mask = cv2.findHomography(src_pts, C, cv2.RANSAC, 5)
    # warped = cv2.warpPerspective(roi, H, (400,200))
    # warped = cv2.perspectiveTransform(roi, H)
    # warped = cv2.flip(warped, 0)









    # hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    # v = hsv[:,:,2]    
    # clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(20,20))
    # cl1 = clahe.apply(v)
    # hsv[:,:,2] = cl1
    # img_back = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


    # grayscale = cv2.cvtColor(img_back, cv2.COLOR_BGR2GRAY)                         # Converting the frame to gray-scale    
    # ret,thresh = cv2.threshold(grayscale,225,255,cv2.THRESH_BINARY)             # Binary thresholding
    
    # canny = cv2.Canny(thresh, 20, 120, apertureSize = 3)
    
    # Hough Lines
    # lines = cv2.HoughLines(thresh, 1, np.pi/180, 100)
    # for rho, theta in lines[0]:
    #     a = np.cos(theta)
    #     b = np.sin(theta)
    #     x0 = a * rho
    #     y0 = b * rho
    #     x1 = int(x0 + 1000 * (-b))
    #     y1 = int(y0 + 1000 * (a))
    #     x2 = int(x0 - 1000 * (-b))
    #     y2 = int(y0 - 1000 * (a))
    #     cv2.line(frame, (x1,y1), (x2,y2), (0,0,255),2)
    

    
    
    cv2.imshow('res', roi)
    # cv2.imshow('frame', warped)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    
    
    
cam.release()
cv2.destroyAllWindows()




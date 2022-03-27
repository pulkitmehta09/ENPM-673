#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 23 15:20:35 2022

@author: pulkit
"""

import cv2
import numpy as np
from matplotlib import pyplot as plt
import imutils

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
    blur = cv2.GaussianBlur(gray, (11,11), 0)
    ret,thresh = cv2.threshold(blur,120,255,cv2.THRESH_BINARY)             # Binary thresholding
    edges = cv2.Canny(thresh, 20, 120)

    return thresh    
    
def region(frame):
    height, width = frame.shape
    # triangle = np.array([[(100, height), (480,300), (width, height)]])    
    trapezoid = np.array([[(110, height), (450,320), (515,320), (width-50, height)]])
    
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





def sliding_windown(img_w):

    histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0)
    # Create an output image to draw on and visualize the result
    out_img = np.dstack((img_w, img_w, img_w)) * 255
    # # Find the peak of the left and right halves of the histogram
    # # These will be the starting point for the left and right lines
    midpoint = np.int32(histogram.shape[0] / 2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    # Choose the number of sliding windows
    nwindows = 9
    # Set height of windows
    window_height = np.int32(img_w.shape[0] / nwindows)
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img_w.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    # Current positions to be updated for each window
    leftx_current = leftx_base
    rightx_current = rightx_base

    # Set the width of the windows +/- margin
    margin = 20
    # Set minimum number of pixels found to recenter window
    minpix = 50
    # Create empty lists to receive left and right lane pixel indices
    left_lane_inds = []
    right_lane_inds = []

    # Step through the windows one by one
    for window in range(nwindows):
        # Identify window boundaries in x and y (and right and left)
        win_y_low = img_w.shape[0] - (window + 1) * window_height
        win_y_high = img_w.shape[0] - window * window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # Draw the windows on the visualization image
        # cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 0, 0), 2)
        # cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 0, 0), 2)
        # Identify the nonzero pixels in x and y within the window
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        # Append these indices to the lists
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # If you found > minpix pixels, recenter next window on their mean position
        if len(good_left_inds) > minpix:
            leftx_current = np.int32(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:
            rightx_current = np.int32(np.mean(nonzerox[good_right_inds]))

    # Concatenate the arrays of indices
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # Generate x and y values for plotting
    ploty = np.linspace(0, img_w.shape[0] - 1, img_w.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    
    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 255, 0]
    # plt.imshow(out_img)
    # plt.plot(left_fitx, ploty, color='yellow')
    # plt.plot(right_fitx, ploty, color='yellow')
    # plt.xlim(0, 200)
    # plt.ylim(200, 0)

    return left_fit, right_fit, out_img



while(True): 
    ret, frame = cam.read()                                                  # Reading the frame from video   
    
    # Check if the frame exists, if not exit 
    if not ret:
        break
    # frame = cv2.flip(frame, 1)
    copy = frame.copy()
    modified = Modify_frame(frame)
    roi = region(modified)
    height, width, _ = frame.shape

    # Working
    # pts1 = np.float32([[(10, height), (450,320), (510,320), (width, height)]])
    # pts2 = np.float32([[(0,200), (0,0), (200, 0), (200,200)]]) 
 
    src = np.float32([[(10, height), (450,320), (515,320), (width, height)]])
    dst = np.float32([[(0,400), (0,0), (200, 0), (200,400)]]) 
    
          
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(roi, matrix, (200,400), flags = cv2.INTER_LINEAR)
    
    
    # x = np.linspace(0,199,200)
    # lf, rf, out_img = sliding_windown(warped)
    # plt.bar(x,hist)
    
    
    out_img = classify(warped)
    unwarp = cv2.warpPerspective(out_img, np.linalg.inv(matrix), (frame.shape[1],frame.shape[0]))
    
    indices = unwarp.nonzero()
    copy[indices[0],indices[1],:] = [0,0,0]
    result = cv2.bitwise_or(unwarp, copy)
    
    # both = np.hstack((unwarp,result))
    # both = imutils.resize(both, width = 960)
    # all_frames = np.vstack((frame,both))
    # all_frames = imutils.resize(all_frames, height = 860)
    
    # wcaptions = np.vstack((all_frames,captions)) 
  
    
    comb1 = np.vstack((frame,result))
    out_img_r = imutils.resize(out_img, height = comb1.shape[0])
    
    warped_rgb = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    warped_rgb_r = imutils.resize(warped_rgb, height = comb1.shape[0])
    
    comb2 = np.hstack((warped_rgb_r,comb1))
    comb2 = imutils.resize(comb2, height = 540)
    # cv2.imshow('res', out_img)
    
    captions = np.zeros((100,comb2.shape[1],3))
    captions[:,:,:] = [0,0,0]
    # captions = cv2.cvtColor(captions, cv2.COLOR_BGR2HSV)
    cap_img = np.vstack((comb2,captions))
    
    cv2.imshow('frame', cap_img)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    
    
    
cam.release()
cv2.destroyAllWindows()


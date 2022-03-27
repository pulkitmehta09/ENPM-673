#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 26 09:41:03 2022

@author: pulkit
"""

#  Without history, cropped lane
import cv2
import numpy as np
from CroppedLaneutils import Modify_frame, region, classify, adjust_gamma, prepare_warped



videofile = 'challenge.mp4'                                                     # Video file
cam = cv2.VideoCapture(videofile)  

yellow_HSV_th_min = np.array([0, 70, 70])
yellow_HSV_th_max = np.array([50, 255, 255])
count = 0
left_diff = []
right_diff = []

def sliding_windown(img_w, out_img):

    histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0)
    # Create an output image to draw on and visualize the result
    # out_img = np.dstack((img_w, img_w, img_w)) * 255
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

    # # Generate x and y values for plotting
    # ploty = np.linspace(0, img_w.shape[0] - 1, img_w.shape[0])
    # left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    # right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    
    # # Colour lanes
    # # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255]
    # # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 255, 0]
    
    # # Curve fit on lanes
    # verts_left = np.array(list(zip(left_fitx.astype(int), ploty.astype(int))))
    # cv2.polylines(out_img, [verts_left], False, [0,0,255], 2)
    # verts_right = np.array(list(zip(right_fitx.astype(int), ploty.astype(int))))
    # cv2.polylines(out_img, [verts_right], False, [0,0,255], 2)
    
    # # Colouring Lane region 
    # pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    # pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    # pts = np.hstack((pts_left, pts_right))    
    # corners = np.array([[left_fitx, ploty], [right_fitx, ploty]], dtype = np.int32)
    # cv2.fillPoly(out_img, np.int32([pts]), color = (0,0,255,100))


    return left_fit, right_fit, out_img



def draw_lane(left_fit, right_fit, out_img):
    
    ploty = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    
    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 255, 0]
    verts_left = np.array(list(zip(left_fitx.astype(int), ploty.astype(int))))
    cv2.polylines(out_img, [verts_left], False, [0,0,255], 2)
    verts_right = np.array(list(zip(right_fitx.astype(int), ploty.astype(int))))
    cv2.polylines(out_img, [verts_right], False, [0,0,255], 2)
    # cv2.line(out_img,(int(left_fitx[0]),int(ploty[0])),(int(right_fitx[0]),int(ploty[0])),(255,0,0),2)
    # cv2.fillPoly(out_img, pts = np.int8([corners]), color = (0,0,255))
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))    
    cv2.fillPoly(out_img, np.int32([pts]), color = (0,0,255,100))
    
    avg_fit = (left_fitx + right_fitx) / 2
    avg_verts = np.array(list(zip(avg_fit.astype(int), ploty.astype(int))))
    cv2.polylines(out_img, [avg_verts], False, [0,255,0],2)
                         
    
    return out_img


def fit_from_lines(left_fit, right_fit, out_img):
    # Assume you now have a new warped binary image from the next frame of video (also called "binary_warped")
    # It's now much easier to find line pixels!
    nonzero = out_img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    margin = 100
    left_lane_inds = ((nonzerox > (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] - margin)) & (
    nonzerox < (left_fit[0] * (nonzeroy ** 2) + left_fit[1] * nonzeroy + left_fit[2] + margin)))
    right_lane_inds = (
    (nonzerox > (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] - margin)) & (
    nonzerox < (right_fit[0] * (nonzeroy ** 2) + right_fit[1] * nonzeroy + right_fit[2] + margin)))

    # Again, extract left and right line pixel positions
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]
    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    return left_fit, right_fit


def compare_fits(left_fit_prev, right_fit_prev, left_fit, right_fit):
    
    ploty = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0])
    left_fitx_prev = left_fit_prev[0] * ploty ** 2 + left_fit_prev[1] * ploty + left_fit_prev[2]
    right_fitx_prev = right_fit_prev[0] * ploty ** 2 + right_fit_prev[1] * ploty + right_fit_prev[2]
    
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    
    left_diff_x = abs(left_fitx - left_fitx_prev)
    right_diff_x = abs(right_fitx - right_fitx_prev)
    
    
    return left_diff_x, right_diff_x


while(True): 
    ret, frame = cam.read()                                                     # Reading the frame from video
       
    # Check if the frame exists, if not exit 
    if not ret:
        break
    
    # roi = region(frame)
    comb, matrix = prepare_warped(frame)
    
    # classified = classify(comb)
 
    out_img = cv2.cvtColor(comb, cv2.COLOR_GRAY2BGR)
    
    if (count == 0):
        left_fit, right_fit, _ = sliding_windown(comb, out_img)
    
    left_fit_prev = left_fit
    right_fit_prev = right_fit
    
    left_fit, right_fit, _ = sliding_windown(comb, out_img)
    ld, rd = compare_fits(left_fit_prev, right_fit_prev,left_fit, right_fit)
    left_diff.append(np.mean(ld))
    right_diff.append(np.mean(rd))
    if (np.mean(ld) > 5):
        print("large ld")
    if (np.mean(rd) > 5):
        print("large value of rd")
    
    
    if (np.mean(ld) > 25):
        left_fit = left_fit_prev
        right_fit = right_fit    
    else:
        left_fit = left_fit


    if (np.mean(rd) > 4.5):
        right_fit = right_fit_prev
        left_fit = left_fit
    else:
        right_fit = right_fit
    
    
    # left_fit, right_fit = fit_from_lines(left_fit, right_fit, comb)
    
    lane_region = draw_lane(left_fit, right_fit, out_img)
 
    unwarp = cv2.warpPerspective(lane_region, np.linalg.inv(matrix), (frame.shape[1],frame.shape[0]))
    result = cv2.bitwise_or(unwarp, frame)
    
    cv2.imshow('frame', result)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    
    count += 1
    
cam.release()
cv2.destroyAllWindows()
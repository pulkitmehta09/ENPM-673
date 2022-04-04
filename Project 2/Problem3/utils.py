#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 20:42:34 2022

ENPM 673
Project 2 Problem 3

@author: Pulkit Mehta
UID: 117551693
"""

import cv2
import numpy as np
import imutils


def Modify_frame(frame):
    """
    Obtains the thresholded image.

    Parameters
    ----------
    frame : ndArray
        Image BGR frame.

    Returns
    -------
    thresh : ndArray
        Binary image after thresholding.

    """
   
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    ret,thresh = cv2.threshold(blur,160,255,cv2.THRESH_BINARY)             # Binary thresholding
   
    return thresh


def region(frame):
    """
    Finds the desired region of interest in the image frame.

    Parameters
    ----------
    frame : ndArray
        Image frame.

    Returns
    -------
    mask : ndArray
        Frame with only the region of interest.

    """
    
    height, width = frame.shape[:2]
    trapezoid = np.array([[(220, height-50), (610,440), (725,440), (width - 150, height-50)]])
    # cv2.line(frame,tuple(trapezoid[0][0]), tuple(trapezoid[0][1]),(0,0,255),2)
    # cv2.line(frame,tuple(trapezoid[0][0]), tuple(trapezoid[0][3]),(0,0,255),2)
    # cv2.line(frame,tuple(trapezoid[0][1]), tuple(trapezoid[0][2]),(0,0,255),2)
    # cv2.line(frame,tuple(trapezoid[0][2]), tuple(trapezoid[0][3]),(0,0,255),2)
    
    mask = np.zeros_like(frame)
    mask = cv2.fillPoly(mask, trapezoid, (255,255,255))
    mask = cv2.bitwise_and(frame, mask)
    
    return mask

    
def adjust_gamma(image, gamma=1.0):
    """
    Performs power law correction(gamma correction) on the given image

    Parameters
    ----------
    image : ndArray
        Input image.
    gamma : float, optional
        Gamma value(Values lesser than 1 make the image darker and greater than 1 make it brighter). The default is 1.0.

    Returns
    -------
    ndArray
        Gamma corrected image.

    """
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
    
    return cv2.LUT(image, table)


def prepare_warped(frame):
    """
    Creates a  birds-eye view of the region of interest with lanes.

    Parameters
    ----------
    frame : ndArray
        Input video frame.

    Returns
    -------
    result : ndArray
        Image with birds-eye view of the lanes.
    matrix : ndArray
        Homography matrix.
    lanes : ndArray
        Detected lanes in original frame.

    """
    
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]    
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(20,20))
    cl1 = clahe.apply(v)
    hsv[:,:,2] = cl1
    img_back = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    img_back = adjust_gamma(img_back, gamma = 0.5)

    
    modified = Modify_frame(img_back)
    roi = region(modified)

    lower = np.array([0,100,100])
    upper = np.array([30,255,255])
    
    yellow_lane_roi = region(frame) 
    yellow_lane_hsv = cv2.cvtColor(yellow_lane_roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(yellow_lane_hsv, lower, upper)

    lanes = roi + mask
    lanes = cv2.cvtColor(lanes, cv2.COLOR_GRAY2BGR)    

    height, width, _ = frame.shape
    pts1 = np.float32([[(220, height-50), (610,440), (725,440), (width - 150, height-50)]])
    pts2 = np.float32([[0,400],[0,0],[200,0],[200,400]])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(roi, matrix, (200,400))
    # warped = cv2.flip(warped,0)
    yellow_lane_warped = cv2.warpPerspective(mask, matrix, (200,400))
    warped[:,:100] = 0
    result = warped + yellow_lane_warped
    
    return result, matrix, lanes


def fit_lanes(img_w, out_img):
    """
    Finds the polynomial fit for the lanes.

    Parameters
    ----------
    img_w : ndArray
        Image with birds-eye view of the lanes.

    Returns
    -------
    left_fit : ndArray
        Second-order fit for the left lane points.
    right_fit : ndArray
        Second-order fit for the right lane points.

    """

    histogram = np.sum(img_w[int(img_w.shape[0] / 2):, :], axis=0)              # Histogram of the warped image with lanes
    
    midpoint = np.int32(histogram.shape[0] / 2)                                 # Midpoint of histogram    
    leftx_peak = np.argmax(histogram[:midpoint])                                # Index of peak value in left half of histogram
    rightx_peak = np.argmax(histogram[midpoint:]) + midpoint                    # Index of peak value in right half of histogram

    number_of_windows = 9                                                       # Number of sliding windows
    window_height = np.int32(img_w.shape[0] / number_of_windows)                # Height of window
    # Identify the x and y positions of all nonzero pixels in the image
    nonzero = img_w.nonzero()                                                   # Coordinates of non-zero pixels in image
    nonzeroy = np.array(nonzero[0])                                             # y - coordinates of non-zero pixels
    nonzerox = np.array(nonzero[1])                                             # x - coordinates of non-zero pixels
    
    # Current positions of peak for each window
    leftx_current = leftx_peak
    rightx_current = rightx_peak

    half_width = 20                                                             # Half of width of window

    threshold = 50                                                              # Threshold for number of points
    
    left_lane_indices = []
    right_lane_indices = []

    # Iterating through the windows
    for window in range(number_of_windows):
        
        win_y_low = img_w.shape[0] - (window + 1) * window_height
        win_y_high = img_w.shape[0] - window * window_height
        win_xleft_low = leftx_current - half_width
        win_xleft_high = leftx_current + half_width
        win_xright_low = rightx_current - half_width
        win_xright_high = rightx_current + half_width
       
        # Draw the windows for visualization
        # cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (255, 0, 0), 2)
        # cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (255, 0, 0), 2)
        
        # Identify the nonzero pixels in x and y within the window
        inside_left_indices = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (
            nonzerox < win_xleft_high)).nonzero()[0]
        inside_right_indices = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
            nonzerox < win_xright_high)).nonzero()[0]
        
        # Append these indices to the lists
        left_lane_indices.append(inside_left_indices)
        right_lane_indices.append(inside_right_indices)
        
        # Recenter next window to mean position if it satisfies threshold
        if len(inside_left_indices) > threshold:
            leftx_current = np.int32(np.mean(nonzerox[inside_left_indices]))
        if len(inside_right_indices) > threshold:
            rightx_current = np.int32(np.mean(nonzerox[inside_right_indices]))

    # Concatenate the arrays of indices
    left_lane_indices = np.concatenate(left_lane_indices)
    right_lane_indices = np.concatenate(right_lane_indices)

    # Extract left and right line pixel positions
    leftx = nonzerox[left_lane_indices]
    lefty = nonzeroy[left_lane_indices]
    rightx = nonzerox[right_lane_indices]
    righty = nonzeroy[right_lane_indices]

    # Fit a second order polynomial to each
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    return left_fit, right_fit, histogram


def draw_lane(left_fit, right_fit, out_img):
    """
    Draws the lane lines and superimposes the detected lane region on the warped image.

    Parameters
    ----------
    left_fit : ndArray
        Second-order fit for the left lane points.
    right_fit : ndArray
        Second-order fit for the right lane points.
    out_img : ndArray
        Warped image.

    Returns
    -------
    out_img : ndArray
        Image with lane lines.
    lane_superimposed : ndArray
        Image with lane region.

    """
    
    lane_superimposed = out_img.copy()
    y_values = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0])
    left_fitx = left_fit[0] * y_values ** 2 + left_fit[1] * y_values + left_fit[2]
    right_fitx = right_fit[0] * y_values ** 2 + right_fit[1] * y_values + right_fit[2]
    
    # Lane lines
    verts_left = np.array(list(zip(left_fitx.astype(int), y_values.astype(int))))
    cv2.polylines(out_img, [verts_left], False, [0,0,255], 2)
    verts_right = np.array(list(zip(right_fitx.astype(int), y_values.astype(int))))
    cv2.polylines(out_img, [verts_right], False, [0,255,0], 2)
   
    # Lane region coloring
    pts_left = np.array([np.transpose(np.vstack([left_fitx, y_values]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, y_values])))])
    pts = np.hstack((pts_left, pts_right))    
    cv2.fillPoly(lane_superimposed, np.int32([pts]), color = (0,0,255,100))
    
    # Mean line
    avg_fit = (left_fitx + right_fitx) / 2
    avg_verts = np.array(list(zip(avg_fit.astype(int), y_values.astype(int))))
    cv2.polylines(out_img, [avg_verts], False, [255,200,68],2)
    
    # Arrows
    number_arrows = 10
    all_pts = avg_verts[::20]
    end_pts = all_pts[::2]
    end_pts = end_pts[::-1]
    start_pts = all_pts[1::2]
    start_pts = start_pts[::-1]
    start_pts_t = [tuple(e) for e in start_pts]
    end_pts_t = [tuple(e) for e in end_pts]
    
    for i in range(number_arrows):     
        cv2.arrowedLine(out_img, start_pts_t[i], end_pts_t[i], (255,120,0), 2, tipLength = 0.5)
        cv2.arrowedLine(lane_superimposed, start_pts_t[i], end_pts_t[i], (255,0,0), 2, tipLength = 0.5)
    
    return out_img, lane_superimposed


def compare_fits(left_fit_prev, right_fit_prev, left_fit, right_fit, out_img):
    """
    Finds the absolute difference of left and right fit points. 

    Parameters
    ----------
    left_fit_prev : ndArray
        Previously stored second-order fit for the left lane points.
    right_fit_prev : ndArray
        Previously stored second-order fit for the right lane points.
    left_fit : ndArray
        Current second-order fit for the left lane points.
    right_fit : ndArray
        Current second-order fit for the right lane points.
    out_img : ndArray
        Warped image.

    Returns
    -------
    left_diff_x : ndArray
        Absolute difference of values of fitted points for left lane.
    right_diff_x : ndArray
        Absolute difference of values of fitted points for right lane.

    """
    
    y_values = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0])
    left_fitx_prev = left_fit_prev[0] * y_values ** 2 + left_fit_prev[1] * y_values + left_fit_prev[2]
    right_fitx_prev = right_fit_prev[0] * y_values ** 2 + right_fit_prev[1] * y_values + right_fit_prev[2]
    
    left_fitx = left_fit[0] * y_values ** 2 + left_fit[1] * y_values + left_fit[2]
    right_fitx = right_fit[0] * y_values ** 2 + right_fit[1] * y_values + right_fit[2]
    
    left_diff_x = abs(left_fitx - left_fitx_prev)
    right_diff_x = abs(right_fitx - right_fitx_prev)
    
    
    return left_diff_x, right_diff_x


def get_curvature(left_fit, right_fit, img_shape):
    """
    Finds the curvature of the left and right lanes.

    Parameters
    ----------
    left_fit : ndArray
        Second-order fit for the left lane points.
    right_fit : ndArray
        Second-order fit for the right lane points.
    img_shape : tuple
        Shape of image.

    Returns
    -------
    left_curvature : double
        Radius of curvature of left lane.
    right_curvature : double
        Radius of curvature of right lane.
    avg_curvature : double
        Mean radius of curvature.
    action : String
        Predicted action.
    slope : double
        Slope of the mean of left and right lane lines.

    """
    action = None
    y_meters_per_pixel = 30 / 720
    
    y_img = img_shape[0]

    avg_fit = (left_fit + right_fit) / 2
    
    left_curvature = ((1 + (2 * left_fit[0] * y_img * y_meters_per_pixel + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curvature = ((1 + (2 * right_fit[0] * y_img * y_meters_per_pixel + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    avg_curvature = ((1 + (2 * avg_fit[0] * y_img * y_meters_per_pixel + avg_fit[1]) ** 2) ** 1.5) / np.absolute(2 * avg_fit[0])
                                                                                                                 
    slope = 2 * avg_fit[0] * y_img * y_meters_per_pixel + avg_fit[1]
    if (-0.1 < slope < 0):
        action = 'Go Straight'
    if (slope < -0.1):
        action = ' Turn Right'
    
    return left_curvature, right_curvature, avg_curvature, action, slope

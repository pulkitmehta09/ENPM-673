#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 20:49:36 2022

@author: pulkit
"""

from HistEqUtils import *


# def adjust_gamma(image, gamma=1.0):

# 	# build a lookup table mapping the pixel values [0, 255] to
# 	# their adjusted gamma values
#     invGamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** invGamma) * 255
# 		for i in np.arange(0, 256)]).astype("uint8")
# 	# apply gamma correction using the lookup table
#     return cv2.LUT(image, table)


# def clipping(hist):
#     N = 256
#     beta = 3000
#     excess = 0
#     for i in range(N):
#         if(hist[i] > beta):
#             excess += hist[i] - beta
#             hist[i] = beta
    
#     m = excess / N
#     for i in range(N):
#         if(hist[i] < beta - m):
#             hist[i] += m
#             excess -= m
#         elif(hist[i] < beta):
#             excess += hist[i] - beta
#             hist[i] = beta
            
#     while (excess > 0):
#         for i in range(N):
#             if(excess > 0):
#                 if(hist[i] < beta):
#                     hist[i] += 1
#                     excess -= 1
    
#     return hist

# Read the given images
images = [cv2.imread(file) for file in glob.glob("adaptive_hist_data/*.png")]

IMG_W = images[0].shape[1]                                                      # Frame width        
IMG_H = images[0].shape[0]                                                      # Frame height            
N = 16                                                                          # Number of tiles across the width or height of the frame
w = IMG_W // N                                                                  # Width of tile
h = IMG_H // N                                                                  # Height of tile

x = np.linspace(0,255,256)

# Defining video writers
FPS = 1                                                                       
fourcc = VideoWriter_fourcc(*'mp4v')
video = VideoWriter('./original.mp4', fourcc, float(FPS), (IMG_W, IMG_H))
he_video = VideoWriter('./histogrameq.mp4', fourcc, float(FPS), (IMG_W, IMG_H))
ahe_video = VideoWriter('./adaptivehistogrameq.mp4', fourcc, float(FPS), (IMG_W, IMG_H))
# swahe_video = VideoWriter('./swadaptivehistogrameq.mp4', fourcc, float(FPS), (IMG_W, IMG_H))


# Looping through the given images
for i in range(len(images)):
    
    if(i==0):
        print("performing histogram equalization...")
    
    img = images[i]
    video.write(img)                                                            # Video of original frames

    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)                                   # Converting to HSV space
    
    # Histogram equalization
    hist = create_histogram(hsv)
    cumsum = create_cumulative_sum(hist)
    mapping = create_mapping(hist, cumsum, IMG_H, IMG_W)
    he_image_hsv = apply_mapping(hsv, mapping)
    he_image = cv2.cvtColor(he_image_hsv, cv2.COLOR_HSV2BGR_FULL)
    he_video.write(he_image)
    
    
    # Adaptive Histogram Equalization
    hsv_copy = hsv.copy()
    ahe_image_hsv = np.zeros_like(img)
    for i in range(N):
        for j in range(N):    
            a_hist = create_histogram(hsv_copy[i * h:(i+1) * h, j * w:(j+1) * w,:])
            a_cumsum = create_cumulative_sum(a_hist)
            a_mapping = create_mapping(a_hist, a_cumsum,h ,w)
            ahe_image_hsv[i * h:(i+1) * h, j * w:(j+1) * w,:] = apply_mapping(hsv_copy[i * h:(i+1) * h, j * w:(j+1) * w,:], a_mapping)
        
    ahe_image = cv2.cvtColor(ahe_image_hsv, cv2.COLOR_HSV2BGR_FULL)
    ahe_video.write(ahe_image)

    # # Sliding window
    # shsv_copy = hsv.copy()
    # swahe_image_hsv = np.zeros_like(img)
    # for i in range(0, IMG_H, 20):
    #     for j in range(0, IMG_W, 60):
    #         sa_hist = create_histogram(shsv_copy[i:i+h, j:j+w,:])
    #         sa_cumsum = create_cumulative_sum(sa_hist)
    #         sa_mapping = create_mapping(sa_hist, sa_cumsum,h ,w)
    #         swahe_image_hsv[i:i+h, j:j+w,:] = apply_mapping(shsv_copy[i:i+h, j:j+w,:], sa_mapping)
        
    # swahe_image = cv2.cvtColor(swahe_image_hsv, cv2.COLOR_HSV2BGR_FULL)
    # swahe_video.write(swahe_image)

# cv2.imwrite('ahe_N2.png',ahe_image)
video.release()    
he_video.release()
ahe_video.release()
# swahe_video.release()

print("Videos saved in the current working directory")




# Q2
def make_HoughLines(frame):
    lines = cv2.HoughLinesP(frame, 2, np.pi/180, 100, minLineLength = 40, maxLineGap = 5)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(frame,(x1,y1),(x2,y2),(0,255,0),2)
    return frame


# Q3
"""
Created on Fri Mar 25 20:42:34 2022

ENPM 673
Project 2 Question3

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
    # cv2.line(frame,tuple(triangle[0][0]), tuple(triangle[0][1]),red,2)
    # cv2.line(frame,tuple(triangle[0][0]), tuple(triangle[0][2]),red,2)
    # cv2.line(frame,tuple(triangle[0][1]), tuple(triangle[0][2]),red,2)
        
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
    
    lane_superimposed = out_img.copy()
    ploty = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0])
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    
    # out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [0, 0, 255]
    # out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 255, 0]
    verts_left = np.array(list(zip(left_fitx.astype(int), ploty.astype(int))))
    cv2.polylines(out_img, [verts_left], False, [0,0,255], 2)
    verts_right = np.array(list(zip(right_fitx.astype(int), ploty.astype(int))))
    cv2.polylines(out_img, [verts_right], False, [0,255,0], 2)
    # cv2.line(out_img,(int(left_fitx[0]),int(ploty[0])),(int(right_fitx[0]),int(ploty[0])),(255,0,0),2)
    # cv2.fillPoly(out_img, pts = np.int8([corners]), color = (0,0,255))
    
    pts_left = np.array([np.transpose(np.vstack([left_fitx, ploty]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, ploty])))])
    pts = np.hstack((pts_left, pts_right))    
    cv2.fillPoly(lane_superimposed, np.int32([pts]), color = (0,0,255,100))
    
    avg_fit = (left_fitx + right_fitx) / 2
    avg_verts = np.array(list(zip(avg_fit.astype(int), ploty.astype(int))))
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


# def fit_from_lines(left_fit, right_fit, out_img):
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


def compare_fits(left_fit_prev, right_fit_prev, left_fit, right_fit, out_img):
    
    ploty = np.linspace(0, out_img.shape[0] - 1, out_img.shape[0])
    left_fitx_prev = left_fit_prev[0] * ploty ** 2 + left_fit_prev[1] * ploty + left_fit_prev[2]
    right_fitx_prev = right_fit_prev[0] * ploty ** 2 + right_fit_prev[1] * ploty + right_fit_prev[2]
    
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]
    
    left_diff_x = abs(left_fitx - left_fitx_prev)
    right_diff_x = abs(right_fitx - right_fitx_prev)
    
    
    return left_diff_x, right_diff_x



def get_curvature(left_fit, right_fit, img_shape):
    
    y_meters_per_pixel = 30 / 720
    x_meters_per_pixel = 3.7 / 1280
    
    y_img = img_shape[0]

    avg_fit = (left_fit + right_fit) / 2
    
    left_curvature = ((1 + (2 * left_fit[0] * y_img * y_meters_per_pixel + left_fit[1]) ** 2) ** 1.5) / np.absolute(2 * left_fit[0])
    right_curvature = ((1 + (2 * right_fit[0] * y_img * y_meters_per_pixel + right_fit[1]) ** 2) ** 1.5) / np.absolute(2 * right_fit[0])
    avg_curvature = ((1 + (2 * avg_fit[0] * y_img * y_meters_per_pixel + avg_fit[1]) ** 2) ** 1.5) / np.absolute(2 * avg_fit[0])
                                                                                                                 
    slope = 2 * avg_fit[0] * y_img * y_meters_per_pixel + avg_fit[1]
    if (-0.1 < slope < 0):
        turn = 'Go Straight'
    if (slope < -0.1):
        turn = ' Turn Right'
    
    
    return left_curvature, right_curvature, avg_curvature, turn, slope





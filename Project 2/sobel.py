#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 09:15:04 2022

@author: pulkit
"""

import cv2
import numpy as np


videofile = 'challenge.mp4'                                                     # Video file
cam = cv2.VideoCapture(videofile) 


def adjust_gamma(image, gamma=1.0):
	# build a lookup table mapping the pixel values [0, 255] to
	# their adjusted gamma values
	invGamma = 1.0 / gamma
	table = np.array([((i / 255.0) ** invGamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
	# apply gamma correction using the lookup table
	return cv2.LUT(image, table)


def apply_CLAHE(frame):
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
    v = hsv[:,:,2]    
    clahe = cv2.createCLAHE(clipLimit=1.0, tileGridSize=(20,20))
    cl1 = clahe.apply(v)
    hsv[:,:,2] = cl1
    img_back = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return img_back


def region(frame):
    height, width = frame.shape[:2]
    # triangle = np.array([[(100, height), (480,300), (width, height)]])
    
    # trapezoid = np.array([[(200, height-50), (630,425), (725,425), (width - 150, height-50)]])
    # Cropped
    trapezoid = np.array([[(220, height-50), (610,440), (725,440), (width - 150, height-50)]])
    # cv2.line(frame,tuple(triangle[0][0]), tuple(triangle[0][1]),red,2)
    # cv2.line(frame,tuple(triangle[0][0]), tuple(triangle[0][2]),red,2)
    # cv2.line(frame,tuple(triangle[0][1]), tuple(triangle[0][2]),red,2)
        
    mask = np.zeros_like(frame)
    mask = cv2.fillPoly(mask, trapezoid, (255,255,255))
    mask = cv2.bitwise_and(frame, mask)
    
    return mask


def abs_sobel(gray_img, x_dir=True, kernel_size=3, thres=(0, 255)):
    """
    Applies the sobel operator to a grayscale-like (i.e. single channel) image in either horizontal or vertical direction
    The function also computes the asbolute value of the resulting matrix and applies a binary threshold
    """
    sobel = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size) if x_dir else cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size) 
    sobel_abs = np.absolute(sobel)
    sobel_scaled = np.uint8(255 * sobel / np.max(sobel_abs))
    
    gradient_mask = np.zeros_like(sobel_scaled)
    gradient_mask[(thres[0] <= sobel_scaled) & (sobel_scaled <= thres[1])] = 1
    return gradient_mask


def mag_sobel(gray_img, kernel_size=3, thres=(0, 255)):
    """
    Computes sobel matrix in both x and y directions, merges them by computing the magnitude in both directions
    and applies a threshold value to only set pixels within the specified range
    """
    sx = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size)
    sy = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size)
    
    sxy = np.sqrt(np.square(sx) + np.square(sy))
    scaled_sxy = np.uint8(255 * sxy / np.max(sxy))
    
    sxy_binary = np.zeros_like(scaled_sxy)
    sxy_binary[(scaled_sxy >= thres[0]) & (scaled_sxy <= thres[1])] = 1
    
    return sxy_binary


def dir_sobel(gray_img, kernel_size=3, thres=(0, np.pi/2)):
    """
    Computes sobel matrix in both x and y directions, gets their absolute values to find the direction of the gradient
    and applies a threshold value to only set pixels within the specified range
    """
    sx_abs = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=kernel_size))
    sy_abs = np.absolute(cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=kernel_size))
    
    dir_sxy = np.arctan2(sx_abs, sy_abs)

    binary_output = np.zeros_like(dir_sxy)
    binary_output[(dir_sxy >= thres[0]) & (dir_sxy <= thres[1])] = 1
    
    return binary_output


def combined_sobels(sx_binary, sy_binary, sxy_magnitude_binary, gray_img, kernel_size=3, angle_thres=(0, np.pi/2)):
    sxy_direction_binary = dir_sobel(gray_img, kernel_size=kernel_size, thres=angle_thres)
    
    combined = np.zeros_like(sxy_direction_binary)
    # Sobel X returned the best output so we keep all of its results. We perform a binary and on all the other sobels    
    combined[(sx_binary == 1) | ((sy_binary == 1) & (sxy_magnitude_binary == 1) & (sxy_direction_binary == 1))] = 1
    
    return combined


while(True): 
    ret, frame = cam.read()                                                     # Reading the frame from video
       
    # Check if the frame exists, if not exit 
    if not ret:
        break
    
    img_back = apply_CLAHE(frame)
    img_back = adjust_gamma(img_back, gamma = 0.5)
    
 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (7,7), 0)
    ret,thresh = cv2.threshold(blur,160,255,cv2.THRESH_BINARY)             # Binary thresholding
    edges = cv2.Canny(thresh, 20, 120)



    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    y_lower = np.array([0,100,100])
    y_upper = np.array([30,255,255])
    
    w_lower = np.array([0,0,200])
    w_upper = np.array([255,255,255])
        
    yellow_mask = cv2.inRange(hsv, y_lower, y_upper)
    white_mask = cv2.inRange(hsv, w_lower, w_upper)

    mask = yellow_mask + white_mask    
    
    
    sobx_best = abs_sobel(gray, kernel_size=15, thres=(20, 120))
    soby_best = abs_sobel(gray, x_dir=False, kernel_size=15, thres=(20, 120))
    sobxy_best = mag_sobel(gray, kernel_size=15, thres=(80, 200))
    
    sobel_combined_best = combined_sobels(sobx_best, soby_best, sobxy_best, gray, kernel_size=15, angle_thres=(np.pi/4, np.pi/2))   

    result = sobel_combined_best + mask
    roi = region(result)

    cv2.imshow('frame', roi)
    if cv2.waitKey(50) & 0xFF == ord('q'):
        break
    
    
cam.release()
cv2.destroyAllWindows()
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 09:40:44 2022

ENPM 673
Project 1

@author: Pulkit Mehta
UID: 117551693
"""

import numpy as np
import cv2
from scipy.spatial import distance as dist
from matplotlib import pyplot as plt
import scipy
import imutils


def fft(grayscale):
    """
    Performs Fast Fourier Transform and finds the edges in the given grayscaled image. 

    Parameters
    ----------
    grayscale : Array
        Grayscaled image.


    Returns
    -------
    None.

    """
    f = scipy.fft.fft2(grayscale, axes = (0,1))                                 # Fast fourier transform of the grayscale image
    fft_shift = scipy.fft.fftshift(f)                                           # Shifts the zero-frequency(DC) component to center  
    mag_spec = 20 * np.log(np.abs(fft_shift))                                   # Magnitude Spectrum   
    
    rows, cols = grayscale.shape                                                # Extracting shape of gray-scale image.
    crow, ccol = int(rows/2), int(cols/2)                                       # center coordinates
    mask = np.ones((rows, cols), np.uint8)                                      # Initializing a mask array
    r = 100                                                                     # Radius of circular mask
    center = [crow, ccol]                                                       # Defining center point
    x, y = np.ogrid[:rows, :cols] 
    mask_area = (x - center[0])**2 + (y - center[1])**2 <= r**2                 # Defining masked area
    mask[mask_area] = 0                                                         # Assigning pixel values to zero of masked area
    
    fshift = fft_shift * mask                                                   # Perfoming multiplication in frequency domain.
    f_ishift = np.fft.ifftshift(fshift)                                         # Shifting center to top left
    img_back = np.fft.ifft2(f_ishift)                                           # Inverse Fourier Transform
    img_back = np.abs(img_back)                                                 # Absolute value of the resultant complex number.
    
    # Plotting the figures
    fig = plt.figure(1)
    plt.imshow(grayscale, cmap = 'gray')
    plt.title("Original grayscaled image")
    plt.savefig('Original gray-scaled image')
    fig2 = plt.figure(2)
    plt.imshow(mag_spec, cmap = 'gray')
    plt.title("Magnitude spectrum")
    plt.savefig('Magitude spectrum')
    fig3 = plt.figure(3)
    plt.imshow(img_back, cmap = 'gray')
    plt.title("Edge-detected image")
    plt.savefig('Edge-detected image')
    plt.show(block=False)
    

def orderpts(pts):
    """
    Orders the points in image in the following fashion: Top Left, Top Right, Bottom Right and Bottom Left.

    Parameters
    ----------
    pts : Array
        Given point coordinates.

    Returns
    -------
    Array
        Ordered array of points

    """
    pts = pts.reshape(4,2)                                                      # Reshaping point coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]
    
    leftMost = xSorted[:2, :]                                                   # Grab the left-most and right-most points from the sorted x-coordinate points
    rightMost = xSorted[2:, :]
	
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]                          # Sort the left-most coordinates according to their y-coordinates so we can grab the top-left and bottom-left points, respectively
    (tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean theorem, the point with the largest distance will be
	# our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
	
    return np.array([tl, tr, br, bl], dtype="float32")  


def homography(C, W):
    """
    Calculates the homography matrix. 

    Parameters
    ----------
    C : Array
        Coordinates in camera frame.
    W : Array
        Coordinates in World frame.

    Returns
    -------
    H : Array
        Homography matrix.

    """
    W = W.reshape(4,2)
    C = C.reshape(4,2)
    
    x_c = np.array([C[0][0], C[1][0], C[2][0], C[3][0]])
    y_c = np.array([C[0][1], C[1][1], C[2][1], C[3][1]])
    x_w = np.array([W[0][0], W[1][0], W[2][0], W[3][0]])
    y_w = np.array([W[0][1], W[1][1], W[2][1], W[3][1]])

    A = np.array([[x_w[0], y_w[0], 1, 0, 0, 0, -x_c[0] * x_w[0], -x_c[0] * y_w[0], -x_c[0]],
                  [0, 0, 0, x_w[0], y_w[0], 1, -y_c[0] * x_w[0], -y_c[0] * y_w[0], -y_c[0]],
                  [x_w[1], y_w[1], 1, 0, 0, 0, -x_c[1] * x_w[1], -x_c[1] * y_w[1], -x_c[1]],
                  [0, 0, 0, x_w[1], y_w[1], 1, -y_c[1] * x_w[1], -y_c[1] * y_w[1], -y_c[1]],
                  [x_w[2], y_w[2], 1, 0, 0, 0, -x_c[2] * x_w[2], -x_c[2] * y_w[2], -x_c[2]],
                  [0, 0, 0, x_w[2], y_w[2], 1, -y_c[2] * x_w[2], -y_c[2] * y_w[2], -y_c[2]],
                  [x_w[3], y_w[3], 1, 0, 0, 0, -x_c[3] * x_w[3], -x_c[3] * y_w[3], -x_c[3]],
                  [0, 0, 0, x_w[3], y_w[3], 1, -y_c[3] * x_w[3], -y_c[3] * y_w[3], -y_c[3]]])
    
    u, s, vh = np.linalg.svd(A, full_matrices = True)
    vt = vh.transpose()
    h = vt[:,-1]
    h = h / h[-1]
    H = np.reshape(h,(3,3))
    
    return H


def warp(H,img,dim):
    """
    Changes the perspective of image to a birds-eye view of image. 

    Parameters
    ----------
    H : Array
        Homography matrix.
    img : Array
        source image
    dim : Tuple
        size of image in world frame.

    Returns
    -------
    warped : Array
        Warped image.

    """

    warped=np.zeros((dim[0],dim[1],3),np.uint8)                                 # Create a blank image
    # Update the pixel values of the blank image according to the source image in the camera frame after performing homography.
    for a in range(dim[0]):
        for b in range(dim[1]):
            f = [a,b,1]
            f = np.reshape(f,(3,1))
            x, y, z = np.matmul(H,f)
            warped[a][b] = img[int(y/z)][int(x/z)]
    
    return warped


def warpTestudo(H,dim,frame,testudo):
    """
    Warpes the template testudo image onto the tag in frame.

    Parameters
    ----------
    H : Array
        Homography matrix.
    dim : Tuple
        Size of the template image.
    frame : Array
        Frame on which template image is superimposed.
    testudo : Array
        Template image.

    Returns
    -------
    frame : Array
        Frame on which template image is superimposed.

    """
    H_inv = np.linalg.inv(H)
    for a in range(dim[1]):
        for b in range(dim[0]):
            x, y, z = np.matmul(H_inv,[a,b,1])
            frame[int(y/z)][int(x/z)] = testudo[a][b] 
            
    return frame


def get_tag_orientation(warped):
    """
    Retrieves the orientation of the tag in the warrped image

    Parameters
    ----------
    warped : Array
        Warped image.

    Returns
    -------
    index : int
        Number indicating the orientation,i.e., location of the white square in the outer grid of tag: 
            0: Top Left, 1: Top Right, 2: Bottom Right, 3: Bottom Left.

    """
    
    # Getting the required outer cells of the grid which indicate the orientation of tag.
    TL = warped[50:75,50:75]    
    TR = warped[125:150,50:75]
    BR = warped[125:150,125:150]
    BL = warped[50:75,125:150]
    # index: TL = 0, TR = 1, BR = 2, BL = 3
    array = [TL.mean(), TR.mean(), BR.mean(), BL.mean()]
    ar = [int(C > 150) for C in array]                                          # Using a value for threshold and converting into a binary array
    index = np.argmax(ar)
            
    return index


def orientTag(pose,warped):
    """
    Rotates the warped tag accordingly to get the upright tag in each frame. 

    Parameters
    ----------
    pose : int
        Code indicating the orientation of tag.
    warped : Array
        Warped image.

    Returns
    -------
    warped : Array
        Warped image.

    """
    
    # Rotate the tag accordingly to get the resultant upright tag.
    if (pose == 0):
        warped = imutils.rotate(warped, 180)
    if (pose == 1):
        warped = imutils.rotate(warped, 90)
    if (pose == 2):
        warped = imutils.rotate(warped, 0)
    if (pose == 3):
        warped = imutils.rotate(warped, 270)
    
    return warped
    

def orientTestudo(pose,testudo):
    """
    Rotates the testudo template image accordingly to align with the tag orientation in each frame.

    Parameters
    ----------
    pose : int
        Code indicating the orientation of tag.
    testudo : Array
        Template testudo image.

    Returns
    -------
    testudo : Array
        Template testudo image aligned with the tag.

    """
    
    if (pose == 0):
        testudo = imutils.rotate(testudo, 0)
    if (pose == 1):
        testudo = imutils.rotate(testudo, 0)
    if (pose == 2):
        testudo = imutils.rotate(testudo, 0)
    if (pose == 3):
        testudo = imutils.rotate(testudo, 90)
    
    return testudo
    

def getTagID(image):
    """
    Retrieves the tag ID from the correctly oriented image.

    Parameters
    ----------
    image : Array
        Image of the center 2x2 grid of tag.

    Returns
    -------
    status : bool
        Indicating if the tag ID is found or not.
    ID : int
        Tag ID.
    ar : Array
        Binary reprersentation of the tag ID.

    """
    
    ID = -1                                                                     # Initializing tag ID
    h , w = image.shape[:-1]                                                    # Shape of the center 2x2 grid
    hh = int(h/2)                                                       
    ww = int(w/2)
    # Finding the Top Left(TL), Top Right(TR), Bottom Right(BR) and Bottom Left(BL) cells of the center grid.
    TL = image[0:hh,0:ww]
    BL = image[hh:h,0:ww]
    TR = image[0:hh,ww:w]
    BR = image[hh:h,ww:w]
    array = [TL.mean(), TR.mean(), BR.mean(), BL.mean()]                        # Array with mean values of the cells in order:TL,TR,BR,BL
    ar = [int(C > 150) for C in array]                                          # Converting into a binary array with certain threshold values.
    binary = ar[::-1]
    ID = int("".join(str(x) for x in binary), 2)                                # Converting the binary number to decimal form.
    if ID != -1:                                                                # Check if ID is retrieved or not.
        status = True               
   
    return status, ID, ar


def getProjectionMatrix(H):
    """
    Calculates the Projection matrix.

    Parameters
    ----------
    H : Array
        Homography matrix.

    Returns
    -------
    P : Array
        Projection matrix.
    T : Array
        Transformation matrix(T = [R|t] , where R is Rotation matrix and t is translation vector).

    """
    
    
    K = np.array([[1346.100595, 0, 932.1633975],                                # Camera intrinsic matrix
                  [0, 1355.933136, 654.8986796],[0, 0, 1]])                   
    K_inv = np.linalg.inv(K)
    lambda_ = 2 / (np.linalg.norm(np.matmul(K_inv,H[:,0])) + np.linalg.norm(np.matmul(K_inv,H[:,1])))
    B_tilda = lambda_ * np.matmul(K_inv, H)
    det = np.linalg.det(B_tilda)
    if(det > 0):
        B = B_tilda
    else:
        B = -B_tilda
    
    r1 = B[:,0]                                                                 # First column of rotation matrix
    r2 = B[:,1]                                                                 # Second column of rotation matrix
    r3 = np.cross(r1, r2)                                                       # Third column of rotation matrix
    t = B[:,2]                                                                  # Translation vector
    T = np.column_stack((r1, r2, r3, t))                                
    P = np.matmul(K, T)
    
    return P, T


def drawCube(P,frame):
    """
    Draws the cube on the tag.

    Parameters
    ----------
    P : Array
        Projection matrix.
    frame : Array
        Frame on which cube is drawn.

    Returns
    -------
    None.

    """
    
    s = 200                                                                     # Side of the Cube    
    corners = np.array([[0,0,0,1],[s,0,0,1],[s,s,0,1],[0,s,0,1],                # Defining corners of the cube
                        [0,0,-s,1],[s,0,-s,1],[s,s,-s,1],[0,s,-s,1]])       
    corners = corners.reshape(8,4,1)            
    res = np.empty((8,3,1))
    
    for C in range(8): 
        res[C] = np.matmul(P,corners[C])
        res[C] = res[C] / res[C][2]
     
    # Drawing lines between the corners of the  cube.
    cv2.line(frame,(res[0][0],res[0][1]),(res[1][0],res[1][1]),(0,255,0),2)
    cv2.line(frame,(res[0][0],res[0][1]),(res[3][0],res[3][1]),(0,255,0),2)
    cv2.line(frame,(res[1][0],res[1][1]),(res[2][0],res[2][1]),(0,255,0),2)
    cv2.line(frame,(res[2][0],res[2][1]),(res[3][0],res[3][1]),(0,255,0),2)
    
    cv2.line(frame,(res[4][0],res[4][1]),(res[5][0],res[5][1]),(0,0,255),2)
    cv2.line(frame,(res[4][0],res[4][1]),(res[7][0],res[7][1]),(0,0,255),2)
    cv2.line(frame,(res[5][0],res[5][1]),(res[6][0],res[6][1]),(0,0,255),2)
    cv2.line(frame,(res[6][0],res[6][1]),(res[7][0],res[7][1]),(0,0,255),2)
    
    cv2.line(frame,(res[0][0],res[0][1]),(res[4][0],res[4][1]),(255,0,0),2)
    cv2.line(frame,(res[1][0],res[1][1]),(res[5][0],res[5][1]),(255,0,0),2)
    cv2.line(frame,(res[2][0],res[2][1]),(res[6][0],res[6][1]),(255,0,0),2)
    cv2.line(frame,(res[3][0],res[3][1]),(res[7][0],res[7][1]),(255,0,0),2)
    


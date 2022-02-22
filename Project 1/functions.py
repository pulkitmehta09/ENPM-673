#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 09:40:44 2022

@author: pulkit
"""

import numpy as np
import cv2
from scipy.spatial import distance as dist


def orderpts(pts):
    pts = pts.reshape(4,2)
    xSorted = pts[np.argsort(pts[:, 0]), :]
    # grab the left-most and right-most points from the sorted
	# x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
	# now, sort the left-most coordinates according to their
	# y-coordinates so we can grab the top-left and bottom-left
	# points, respectively
    leftMost = leftMost[np.argsort(leftMost[:, 1]), :]
    (tl, bl) = leftMost
	# now that we have the top-left coordinate, use it as an
	# anchor to calculate the Euclidean distance between the
	# top-left and right-most points; by the Pythagorean
	# theorem, the point with the largest distance will be
	# our bottom-right point
    D = dist.cdist(tl[np.newaxis], rightMost, "euclidean")[0]
    (br, tr) = rightMost[np.argsort(D)[::-1], :]
	# return the coordinates in top-left, top-right,
	# bottom-right, and bottom-left order

    return np.array([tl, tr, br, bl], dtype="float32")  



def homography(C, I):
    C = C.reshape(4,2)
    x_c = np.array([0, I[0], I[0], 0])
    y_c = np.array([0, 0, I[1], I[1]])
    x_w = np.array([C[0][0], C[1][0], C[2][0], C[3][0]])
    y_w = np.array([C[0][1], C[1][1], C[2][1], C[3][1]])

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

def warpPerspective(H,img,dim):
    """
    

    Parameters
    ----------
    H : TYPE
        Homography matrix.
    img : TYPE
        source image
    dim : TYPE
        size of image.

    Returns
    -------
    warped : TYPE
        warped image.

    """
    H_inv=np.linalg.inv(H)
    warped=np.zeros((dim[0],dim[1],3),np.uint8)
    for a in range(dim[0]):
        for b in range(dim[1]):
            f = [a,b,1]
            f = np.reshape(f,(3,1))
            x, y, z = np.matmul(H_inv,f)
            warped[a][b] = img[int(y/z)][int(x/z)]
    return warped



def warpTestudo(H,img,maxHeight,maxWidth,superimg):
    H_inv=np.linalg.inv(H)
    warped = cv2.transpose(superimg)
    for a in range(maxHeight):
        for b in range(maxWidth):
            f = [a,b,1]
            f = np.reshape(f,(3,1))
            x, y, z = np.matmul(H_inv,f)
            warped[a][b] = img[int(y/z)][int(x/z)]
    return warped

# Not required
def RotateTag(warped, angle):
    center = tuple(np.array(warped.shape[1::-1]) / 2)   # coordinates of center of image
    R = cv2.getRotationMatrix2D(center, angle, 1.0)     # Rotation matrix
    rotated = cv2.warpAffine(warped, R, warped.shape[1::-1], flags = cv2.INTER_LINEAR)
    
    return rotated

# TODO later (Straighten tag without hardcoding)
def orientTag(image):
    tag = image[50:150,50:150]
    
    return True


def getTagID(image):
    ID = -1
    h , w = image.shape[:-1]
    hh = int(h/2)
    ww = int(w/2)
    TL = image[0:hh,0:ww]
    BL = image[hh:h,0:ww]
    TR = image[0:hh,ww:w]
    BR = image[hh:h,ww:w]
    ar = [TL.mean(), TR.mean(), BR.mean(), BL.mean()]
    ar = [int(i > 150) for i in ar]
    binary = ar[::-1]
    ID = int("".join(str(x) for x in binary), 2)
    if ID != -1:
        status = True
   
   # ---------------------------
   # OLD Method
   # index = np.argmin(ar)
   #  if index == 0:
   #      ID = 14
   #      status = True
   #  if index == 1:
   #      ID = 13
   #      status = True
   #  if index == 2:
   #      ID = 11
   #      status = True
   #  if index == 3:
   #      ID = 7
   #      status = True
   # -------------------------
   
    return status, ID, ar



def getProjectionMatrix(H):
    K = np.array([[1346.100595, 0, 932.1633975],[0, 1355.933136, 654.8986796],[0, 0, 1]])
    K_inv = np.linalg.inv(K)
    scale = 2 / (np.linalg.norm(np.matmul(K_inv,H[:,0])) + np.linalg.norm(np.matmul(K_inv,H[:,1])))
    B_tilda = scale * np.matmul(K_inv, H)
    det = np.linalg.det(B_tilda)
    if(det > 0):
        B = B_tilda
    else:
        B = -B_tilda
    
    r1 = scale * B[:,0]
    r2 = scale * B[:,1]
    r3 = np.cross(r1, r2)
    t = scale * B[:,2]
    T = np.column_stack((r1, r2, r3, t))
    P = np.matmul(K, T)
    
    return P, T


def drawCube(P):
    s = 200
    corners = np.array([[0,0,0,1],[s,0,0,1],[s,s,0,1],[0,s,0,1],[0,0,s,1],[s,0,s,1],[s,s,s,1],[0,s,s,1]])
    corners = corners.reshape(8,4,1)
    res = np.empty((8,3,1))
    for i in range(8): 
        res[i] = np.matmul(P,corners[i])
        
    
    return res

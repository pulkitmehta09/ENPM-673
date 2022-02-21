#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 20 09:40:44 2022

@author: pulkit
"""

import numpy as np
import cv2

def homography(C):
    x_c = np.array([0, 200, 200, 0])
    y_c = np.array([0, 0, 200, 200])
    x_w = np.array([C[0][0][0], C[1][0][0], C[2][0][0], C[3][0][0]])
    y_w = np.array([C[0][0][1], C[1][0][1], C[2][0][1], C[3][0][1]])

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

def warpPerspective(H,img,maxHeight,maxWidth):
    H_inv=np.linalg.inv(H)
    warped=np.zeros((maxHeight,maxWidth,3),np.uint8)
    for a in range(maxHeight):
        for b in range(maxWidth):
            f = [a,b,1]
            f = np.reshape(f,(3,1))
            x, y, z = np.matmul(H_inv,f)
            warped[a][b] = img[int(y/z)][int(x/z)]
    return warped


def RotateTag(warped, angle):
    center = tuple(np.array(warped.shape[1::-1]) / 2)   # coordinates of center of image
    R = cv2.getRotationMatrix2D(center, angle, 1.0)     # Rotation matrix
    rotated = cv2.warpAffine(warped, R, warped.shape[1::-1], flags = cv2.INTER_LINEAR)
    
    return rotated


def orientTag(image):
    tag = image[50:150,50:150]
    
    return True


def getTagID(image):
    TL = image[0:25,0:25]
    BL = image[25:50,0:25]
    TR = image[0:25,25:50]
    BR = image[25:50,25:50]
    ar = [TL.mean(), TR.mean(), BR.mean(), BL.mean()]
    index = np.argmin(ar)
    if index == 0:
        ID = 14
        status = True
    if index == 1:
        ID = 13
        status = True
    if index == 2:
        ID = 11
        status = True
    if index == 3:
        ID = 7
        status = True
    return status, ID



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


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 17:40:55 2022

@author: pulkit
"""

import cv2
import numpy as np
import imutils
import argparse, textwrap
from utils import match_features, resize_image, DataSet
from tqdm import tqdm


def FundamentalMatrix(pts1, pts2):
    
 
    def Normalize(pts):
    
        pts_mean = np.mean(pts, axis=0)
        tx, ty = pts_mean
        pts_centered = pts - pts_mean
        
        s = (2 ** 0.5) / (np.mean(np.sum(pts_centered ** 2, axis=1) ** 0.5))
        T = np.array([[s, 0, -s*tx], [0, s, -s*ty], [0, 0, 1]])
        pts_h = np.column_stack((pts,np.ones(len(pts))))
        pts_h_T = pts_h.T
        
        pts_norm = (np.dot(T,pts_h_T)).T
        
        return pts_norm, T
    
    pts1_norm, T1 = Normalize(pts1)
    pts2_norm, T2 = Normalize(pts2)
    
    x1 = pts1_norm[:,0]
    y1 = pts1_norm[:,1]
    x2 = pts2_norm[:,0]
    y2 = pts2_norm[:,1]
    
    n = len(x1)
    A = np.zeros((n,9))
    
  
    for i in range(n):
      # A[i] = np.array([x1[i]*x2[i], x1[i]*y2[i], x1[i], y1[i]*x2[i], y1[i]*y2[i], y1[i], x2[i], y2[i], 1])
      A[i] = np.array([x2[i]*x1[i], x2[i]*y1[i], x2[i], y2[i]*x1[i], y2[i]*y1[i], y2[i], x1[i], y1[i], 1])

    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    F = Vt.T[:,-1]
    F = np.reshape(F,(3,3))
    
    U_F, S_F, Vt_F = np.linalg.svd(F)
    
    S_F[-1] = 0
    diag_S_F = np.diag(S_F)
    
    F_res = np.dot(U_F, np.dot(diag_S_F,Vt_F))
    F_res = np.dot(T2.T, np.dot(F_res, T1))
    
    return F_res


def ransac(pts1, pts2):
    
    
    def sample_points(pts1, pts2):
    
        random_indices = np.random.choice(len(pts1), 8, replace=True)
        pts1_sample = np.zeros((8,2))
        pts2_sample = np.zeros((8,2))
        
        for idx,i in enumerate(random_indices):
            pts1_sample[idx] = pts1[i]
            pts2_sample[idx] = pts2[i]
            
        
        return pts1_sample, pts2_sample
    
    
    def calculate_error(pts1, pts2, F):

        pts1_h = np.column_stack((pts1,np.ones(len(pts1))))
        pts2_h = np.column_stack((pts2,np.ones(len(pts2))))
        E = []
        
        for (pt1_h,pt2_h) in zip(pts1_h,pts2_h):
            error = abs(np.squeeze(np.dot(pt2_h,np.dot(F,pt1_h.T))))
            E.append(error)
        
        return E

    
    M = np.inf 
    count = 0
    threshold = 0.01  
    max_inliers = 0
    p = 0.99
    
    F_best = None
    
    while (count < M):
        inlier_count = 0
        S1 = []
        S2 = []
        
        sample_1, sample_2 = sample_points(pts1, pts2)
        F = FundamentalMatrix(sample_1, sample_2)
        
        E = calculate_error(pts1, pts2, F)
        E = np.array(E)
        
        inliers = E < threshold
        inlier_count = np.sum(inliers)
        
        for i in range(len(inliers)):
            if(inliers[i] == True):
                S1.append(pts1[i])
                S2.append(pts2[i])
        
        if(inlier_count > max_inliers):
            max_inliers = inlier_count
            F_best = F
            inliers_pts1 = S1
            inliers_pts2 = S2
        
        inlier_ratio = inlier_count / len(pts1)
        if(np.log(1 - (inlier_ratio ** 8)) == 0):
            continue
        
        M = np.log(1 - p) / (np.log(1 - inlier_ratio ** 8))
        count += 1

    return F_best, inliers_pts1, inliers_pts2


def EssentialMatrix(F, K1, K2):
    E = np.dot(K2.T,np.dot(F,K1))
    U, S, V = np.linalg.svd(E)
    sigma = np.diag([1,1,0])
    E_res = np.dot(U,np.dot(sigma,V))
    
    return E_res


def EstimateCameraPose(E):
    
    U, S, Vt = np.linalg.svd(E)
    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
   
    C1 = U[:,-1]
    R1 = np.dot(U, np.dot(W, Vt))
    
    C2 = -U[:,-1]
    R2 = np.dot(U, np.dot(W, Vt))
    
    C3 = U[:,-1]
    R3 = np.dot(U, np.dot(W.T, Vt))
    
    C4 = -U[:,-1]
    R4 = np.dot(U, np.dot(W.T, Vt))
    
    if(np.linalg.det(R1) < 0):
        R1 = -R1
        C1 = -C1
    if(np.linalg.det(R2) < 0):
        R2 = -R2
        C2 = -C2
    if(np.linalg.det(R3) < 0):
        R3 = -R3
        C3 = -C3
    if(np.linalg.det(R4) < 0):
        R4 = -R4
        C4 = -C4
    
    C1 = np.reshape(C1,(3,1))
    C2 = np.reshape(C2,(3,1))
    C3 = np.reshape(C3,(3,1))
    C4 = np.reshape(C4,(3,1))    
    
    C = [C1, C2, C3, C4]
    R = [R1, R2, R3, R4]
        
    return R, C


def Triangulate(S1, S2, R1, C1, R2, C2, K1, K2):
    
    
    def ProjectionMatrix(K, R, C):
    
        
        I = np.identity(3)
        P = np.dot(K, np.dot(R, np.hstack((I, -C))))
        
        return P
    
    
    def get_crossproduct_A(P1, P2, pt1, pt2):
        
        p1, p2, p3 = P1
        p1_, p2_, p3_ = P2
        
        p1 = np.reshape(p1, (1,-1))
        p2 = np.reshape(p2, (1,-1))
        p3 = np.reshape(p3, (1,-1))
        p1_ = np.reshape(p1_, (1,-1))
        p2_ = np.reshape(p2_, (1,-1))
        p3_ = np.reshape(p3_, (1,-1))
        
        x, y = pt1
        x_, y_ = pt2
        
        A = np.vstack((y * p3 - p2, p1 - x * p3,
                       y_ * p3_ - p2_, p1_ - x_ * p3_ ))
    
        return A
    
    P1 = ProjectionMatrix(K1, R1, C1)
    P2 = ProjectionMatrix(K2, R2, C2)
    
    pts_3D = []
    
    for pt1, pt2 in zip(S1, S2):
        A = get_crossproduct_A(P1, P2, pt1, pt2)
        U, S, Vt = np.linalg.svd(A)
        pt_3D = Vt[-1]
        pt_3D = pt_3D / pt_3D[-1]
        pts_3D.append(pt_3D[:3])
        
    pts_3D = np.array(pts_3D)
    
    return pts_3D
    

def LinearTriangulation(S1, S2, R, C, K1, K2):
    
    pts_3D = []
    
    for i in range(len(C)):
        R1 = np.identity(3)
        R2 = R[i]
        C1 = np.zeros((3,1))
        C2 = C[i]
        
        pt_3D = Triangulate(S1, S2, R1, C1, R2, C2, K1, K2)
        pts_3D.append(pt_3D)
    
    return pts_3D


def DisambiguateCameraPose(E, S1, S2, K1, K2):
    
    
    def DepthConstraint(pts_3D, C, r3):
        
        positive = 0
        for pt in pts_3D:
            pt = np.reshape(pt,(-1,1))
            if(np.dot(r3,(pt - C)) > 0 and pt[2] > 0):
                positive += 1
        
        return positive

    
    R, C = EstimateCameraPose(E)
    pts_3D = LinearTriangulation(S1, S2, R, C, K1, K2)
    
    max_num_positive = 0
    optimum = 0
    
    for i in range(len(R)):
        R_ = R[i]
        C_ = C[i]
        r3 = R_[2]
        pt_3D = pts_3D[i]
        num_positive = DepthConstraint(pt_3D, C_, r3)
        
        if (num_positive > max_num_positive):
            max_num_positive = num_positive
            optimum = i
    
    R_res = R[optimum]
    C_res = C[optimum]
    pts_3D_res = pts_3D[optimum]
    
    return R_res, C_res, pts_3D_res


def drawlines(img1src, img2src, lines, pts1src, pts2src):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
   
    img1 = np.copy(img1src)
    img2 = np.copy(img2src)
    
    r, c = img1src.shape[:2]
    lines = np.reshape(lines,(-1,3))    
    
    np.random.seed(0)
    for r, pt1, pt2 in zip(lines, np.int32(pts1src), np.int32(pts2src)):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1]])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    
    return img1, img2


def compute_SSD(left_window, right_window):
    
    if(left_window.shape != right_window.shape):
        return -1
    
    ssd = np.sum(np.square(left_window - right_window))
    
    return ssd


def match_block(y, x, left_window, right_window, blockSize, searchRange):
    
    x_min = max(0, x - searchRange)
    x_max = min(right_window.shape[1], x + searchRange)
    min_ssd = None
    min_x = None
    
    for x in range(x_min, x_max):
        block_right = right_window[y: y + blockSize, x: x + blockSize]
        ssd = compute_SSD(left_window, block_right)
        
        if(min_ssd):
            if(ssd < min_ssd):
                min_ssd = ssd
                min_x = x
        else:
            min_ssd = ssd
            min_x = x
            
    return min_x


def get_Disparity_Map(left_img, right_img, blockSize, searchRange):
    
    left_img = left_img.astype(np.int32)
    right_img = right_img.astype(np.int32)
    
    if(left_img.shape != right_img.shape):
        raise Exception("Left and Right image shape mismatch")
    
    h, w = left_img.shape[:2]
    disparity_map = np.zeros((h,w))
    
    for y in tqdm(range(blockSize, h - blockSize)):
        for x in range(blockSize, w - blockSize):
            block_left = left_img[y: y + blockSize, x: x + blockSize]
            min_x = match_block(y, x, block_left, right_img, blockSize, searchRange)
            disparity_map[y, x] = abs(min_x - x)
            
    return disparity_map
    

def get_Depth_Map(disparity_map):
    
    h,w = disparity_map.shape[:2]
    depth_map = np.zeros_like(disparity_map)
    for y in range(h):
        for x in range(w):
            if(disparity_map[y, x] == 0):
                depth_map[y, x] = 0
            else:
                depth_map[y, x] = int(b * f / disparity_map[y, x])
    
    return depth_map
    

# --------------------------------------------------
# MAIN
# --------------------------------------------------

ap = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)
ap.add_argument("--dataset", default='octagon', help = textwrap.dedent('''Dataset name
Available datasets: curule, octagon, pendulum
default: curule''')) 
args = ap.parse_args()

dataset_name = args.dataset
dataset = DataSet(dataset_name)
values = dataset.get_data()


im0 = values["im0"]
im1 = values["im1"]

im0 = resize_image(im0, scale=0.5)
im1 = resize_image(im1, scale=0.5)

h1, w1 = im0.shape[:2]
h2, w2 = im1.shape[:2]

K1 = values["cam0"]
K2 = values["cam1"]
b = values["baseline"]
f = K1[0,0]
vmin = values["vmin"]
vmax = values["vmax"]


# Converting to graim0_epilines, _ = drawlines(im0, im1, l1, inliers_pts1, inliers_pts2)yscale
im0_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
im1_gray = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)  

pts1, pts2, feature_matched_image = match_features(im0, im1, 0.7)
cv2.imshow(" Features Matching", feature_matched_image)

F, inliers_pts1, inliers_pts2 = ransac(pts1, pts2)
inliers_pts1 = np.int32(inliers_pts1)
inliers_pts2 = np.int32(inliers_pts2)


# F, inliers = cv2.findFundamentalMat(pts1,pts2,cv2.FM_RANSAC)
# inliers_pts1 = pts1[inliers.ravel() == 1]
# inliers_pts2 = pts2[inliers.ravel() == 1]



E = EssentialMatrix(F, K1, K2)

R, T, pts_3D = DisambiguateCameraPose(E, inliers_pts1, inliers_pts2, K1, K2)


l1 = cv2.computeCorrespondEpilines(inliers_pts2.reshape(-1,1,2), 2, F)
im0_epilines, _ = drawlines(im0, im1, l1, inliers_pts1, inliers_pts2)
l2 = cv2.computeCorrespondEpilines(inliers_pts1.reshape(-1,1,2), 1, F)
im1_epilines, _ = drawlines(im1, im0, l2, inliers_pts2, inliers_pts1)
epilines = np.hstack((im0_epilines, im1_epilines))


_, H1, H2 = cv2.stereoRectifyUncalibrated(np.float32(inliers_pts1), np.float32(inliers_pts2), F, imgSize=(w1,h1))

im0_rectified = cv2.warpPerspective(im0, H1, (w1,h1))
im1_rectified = cv2.warpPerspective(im1, H2, (w2,h2))
rectified = np.hstack((im0_rectified, im1_rectified))

im0_rectified_gray = cv2.cvtColor(im0_rectified, cv2.COLOR_BGR2GRAY)
im1_rectified_gray = cv2.cvtColor(im1_rectified, cv2.COLOR_BGR2GRAY)

# disparity_map = get_Disparity_Map(im0_rectified_gray, im1_rectified_gray, 7, 56)
# disparity_map_gray = ((disparity_map / disparity_map.max()) * 255).astype(np.uint8)

# disparity_map_color = None
# disparity_map_color = cv2.normalize(disparity_map, disparity_map_color, alpha = vmin, beta = vmax, norm_type = cv2.NORM_MINMAX, dtype = cv2.CV_8U)

# disparity_map_colored = cv2.applyColorMap(disparity_map_color, cv2.COLORMAP_PARULA)

# depth_map = get_Depth_Map(disparity_map)
# depth_map_gray = ((depth_map / depth_map.max()) * 255).astype(np.uint8)
# depth_map_color = cv2.applyColorMap(depth_map_gray, cv2.COLORMAP_PARULA)


cv2.imshow("rectified", rectified)
cv2.imshow("epilines", epilines)
# cv2.imwrite("disparity map gray.png", disparity_map_gray)
# cv2.imwrite("disparity map colored.png", disparity_map_colored)
# cv2.imwrite("depth map gray non.png", depth_map_gray)



cv2.waitKey(0)
cv2.destroyAllWindows()
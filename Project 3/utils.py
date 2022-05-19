#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 15 18:14:18 2022

@author: pulkit
"""

import cv2
import numpy as np
from tqdm import tqdm

    
def resize_image(img, scale = 0.5):
    """
    Resizes an image upto a given scale.

    Parameters
    ----------
    img : ndArray
        Given image.
    scale : float, optional
        Scale factor. The default is 0.5.

    Returns
    -------
    resized : ndArray
        Resized image.

    """

    width = int(img.shape[1] * scale)
    height = int(img.shape[0] * scale)
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
   
    return resized
          

def match_features(im0,im1, ratio):
    """
    Finds the feature matching points from two corresponding images using ORB. 

    Parameters
    ----------
    im0 : ndArray
        Left image.
    im1 : ndArray
        Right image.
    ratio : float
        Lowe's ratio.

    Returns
    -------
    pts1 : ndArray
        Points in left image.
    pts2 : ndArray
        Points in right image.
    feature_matched_image : ndArray
        Image depicting matched features.

    """
    
    # Converting to grayscale
    gray1 = cv2.cvtColor(im0, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    
    # Using ORB Feature Descriptor
    orb = cv2.ORB_create()
    
    # Creating keypoints and descriptors
    keypoints1, descriptors1 = orb.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(gray2, None)
    
    # Keypoints shown on im1
    # cv2.imshow('keypoints on image',cv2.drawKeypoints(im1,keypoints1,None))
    
    # Brute force matching of descriptors using two best matches for every descriptor
    match = cv2.BFMatcher()
    matches = match.knnMatch(descriptors1, descriptors2, k=2)
    
    # Apply ratio test to filter and get the best matches
    good = []
    for m,n in matches:
        if m.distance < ratio * n.distance:
            good.append(m)
   
    draw_params = dict(matchColor=(0,255,0), singlePointColor=None, flags=2)
    feature_matched_image = cv2.drawMatches(im0,keypoints1,im1,keypoints2,good,None,**draw_params) 
   
    # Defining minimum match count so that stitching is performed only when the number of good matches exceed the minimum match count        
    MIN_MATCH_COUNT = 10
    if len(good) > MIN_MATCH_COUNT:
        pts1 = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
        pts2 = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    else:
        print("Not enough matches are found")
    
    pts1 = np.reshape(pts1, (-1,2))
    pts2 = np.reshape(pts2, (-1,2))
    
    return pts1, pts2, feature_matched_image


def FundamentalMatrix(pts1, pts2):
    """
    Calculates the Fundamental Matrix.

    Parameters
    ----------
    pts1 : ndArray
        Points from left image.
    pts2 : ndArray
        Points from right image.

    Returns
    -------
    ndArray
        Fundamental matrix.

    """
    
 
    def Normalize(pts):
        """
        

        Parameters
        ----------
        pts : ndArray
            Points to be normalized.

        Returns
        -------
        pts_norm : ndArray
            Normalized points.
        T : ndArray
            Transformation matrix.

        """
    
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
    
    # Constructing the A martix as per the equation Af = 0.
    for i in range(n): 
        A[i] = np.array([x2[i]*x1[i], x2[i]*y1[i], x2[i], y2[i]*x1[i], y2[i]*y1[i], y2[i], x1[i], y1[i], 1])

    # Performing SVD of A
    U, S, Vt = np.linalg.svd(A, full_matrices=True)
    F = Vt.T[:,-1]
    F = np.reshape(F,(3,3))
    
    # Performing SVD of F
    U_F, S_F, Vt_F = np.linalg.svd(F)
    
    # Rank 2 constraint
    S_F[-1] = 0
    diag_S_F = np.diag(S_F)
    
    # Reconstructing the F matrix
    F_res = np.dot(U_F, np.dot(diag_S_F,Vt_F))
    
    # Denormalizing
    F_res = np.dot(T2.T, np.dot(F_res, T1))
    
    return F_res


def ransac(pts1, pts2):
    """
    Performs Random Sample Consensus(RANSAC) for outlier rejection.

    Parameters
    ----------
    pts1 : ndArray
        Points from left image.
    pts2 : ndArray
        Points from right image.

    Returns
    -------
    F_best : ndArray
        Best estimate of Fundamental matrix.
    inliers_pts1 : ndArray
        Set of inlier points from left image.
    inliers_pts2 : ndArray
        Set of inlier points from right image.

    """
    
    
    def sample_points(pts1, pts2):
        """
        Samples pair of eight random points from feature matched points in left and right image.

        Parameters
        ----------
        pts1 : ndArray
            Points from left image.
        pts2 : ndArray
            Points from right image.

        Returns
        -------
        pts1_sample : ndArray
            Sampled points from left image.
        pts2_sample : ndArray
            Sampled points from right image.

        """
    
        random_indices = np.random.choice(len(pts1), 8, replace=True)
        pts1_sample = np.zeros((8,2))
        pts2_sample = np.zeros((8,2))
        
        for idx,i in enumerate(random_indices):
            pts1_sample[idx] = pts1[i]
            pts2_sample[idx] = pts2[i]
            
        
        return pts1_sample, pts2_sample
    
    
    def calculate_error(pts1, pts2, F):
        """
        Calculates error based on the equation E = x'Fx. 

        Parameters
        ----------
        pts1 : npArray
            Points from the left image.
        pts2 : npArray
            Points from the right image.
        F : npArray
            Fundamental matrix.

        Returns
        -------
        E : npArray
            Error.

        """

        pts1_h = np.column_stack((pts1,np.ones(len(pts1))))
        pts2_h = np.column_stack((pts2,np.ones(len(pts2))))
        E = []
        
        for (pt1_h,pt2_h) in zip(pts1_h,pts2_h):
            error = abs(np.squeeze(np.dot(pt2_h,np.dot(F,pt1_h.T))))
            E.append(error)
        
        return E
    

    M = np.inf                                                                  # Maximum number of iterations set as infinity
    count = 0                                               
    threshold = 0.01                                                            # Threshold value for error    
    max_inliers = 0                                                     
    p = 0.99                                                                    # probability
    
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
    """
    Calculates the Essential Matrix from fundamental matrix.

    Parameters
    ----------
    F : ndArray
        Fundamental matrix.
    K1 : ndArray
        Intrinsic matrix for camera 1.
    K2 : ndArray
        Intrinsic matrix for camera 2.

    Returns
    -------
    E_res : ndArray
        Essential Matrix.

    """
    E = np.dot(K2.T,np.dot(F,K1))
    U, S, V = np.linalg.svd(E)
    sigma = np.diag([1,1,0])
    E_res = np.dot(U,np.dot(sigma,V))
    
    return E_res


def EstimateCameraPose(E):
    """
    Estimates the camera pose,i.e., the Rotation and translation from the Essential matrix.

    Parameters
    ----------
    E : ndArray
        Essential matrix.

    Returns
    -------
    R : ndArray
        Rotation matrix.
    C : ndArray
        Translation matrix.

    """
    
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
    """
    Calculate point in 3D space after triangulation.

    Parameters
    ----------
    S1 : ndArray
        Set of points from left image.
    S2 : ndArray
        Set of points from right image.
    R1 : ndArray
        Rotation matrix.
    C1 : ndArray
        Translation matrix.
    R2 : ndArray
        Rotation matrix.
    C2 : ndArray
        Translation matrix.
    K1 : ndArray
        Intrinsic matric for camera 1.
    K2 : ndArray
        Intrinsic matric for camera 2.

    Returns
    -------
    ndArray
        Points in 3D space.

    """
    
    
    def ProjectionMatrix(K, R, C):
        """
        Calculates the Projection matrix.

        Parameters
        ----------
        K : ndArray
            Intrinsic matric for camera.
        R : ndArray
            Rotation matrix.
        C : ndArray
            Translation matrix.

        Returns
        -------
        P : ndArray
            Projection matrix.

        """
        
        I = np.identity(3)
        P = np.dot(K, np.dot(R, np.hstack((I, -C))))
        
        return P
    
    
    def get_crossproduct_A(P1, P2, pt1, pt2):
        """
        Calculate matrix A for the equation AX = 0 for triangulation.

        Parameters
        ----------
        P1 : ndArray
            Projection matrix 1.
        P2 : ndArray
            Projection matrix 2.
        pt1 : ndArray
            Point 1.
        pt2 : ndArray
            Point 1.

        Returns
        -------
        A : ndArray

        """
        
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
    """
    Performs Linear Triangulation.

    Parameters
    ----------
    S1 : ndArray
        Set of points from left image.
    S2 : ndArray
        Set of points from right image.
    R : ndArray
        Rotation matrix.
    C : ndArray
        Translation matrix.
    K1 : ndArray
        Intrinsic matric for camera 1.
    K2 : ndArray
        Intrinsic matric for camera 2.

    Returns
    -------
    pts_3D : ndArray
        Points in 3D space.

    """
    
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
    """
    Gets the correct Rotation and Translation matrix for decomposition of Essential matrix.

    Parameters
    ----------
    E : ndArray
        Essential matrix.
    S1 : ndArray
        Set of points from left image.
    S2 : ndArray
        Set of points from right image.
    K1 : ndArray
        Intrinsic matric for camera 1.
    K2 : ndArray
        Intrinsic matric for camera 2.

    Returns
    -------
    R_res : ndArray
        Rotation matrix.
    C_res : ndArray
        Translation matrix
    pts_3D_res : ndArray
        Points in 3D space.

    """
     
    def DepthConstraint(pts_3D, C, r3):
        """
        Applies the positive depth constraint.

        Parameters
        ----------
        pts_3D : ndArray
            Points.
        C : ndArray
            Translation matrix.
        r3 : ndArray
            Third row of Rotation matrix.

        Returns
        -------
        positive : int
            Number of points with positive depth.

        """
        
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
    """
    Draws the epipolar lines on the two images.

    Parameters
    ----------
    img1src : ndArray
        Left image.
    img2src : ndArray
        Right image.
    lines : ndArray
        Computed lines.
    pts1src : ndArray
        Inlier points of left image.
    pts2src : ndArray
        Inlier points of right image.

    Returns
    -------
    img1 : ndArray
        Left image with epipolar lines.
    img2 : ndArray
        Right image with epipolar lines.

    """
   
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



def match_block(y, x, left_window, right_image, blockSize, searchRange):
    """
    Performs block matching.

    Parameters
    ----------
    y : int
        y-coordinate of point.
    x : int
        x-coordinate of point.
    left_window : ndArray
        Window from left image.
    right_image : ndArray
        Right image.
    blockSize : int
        Dimension of block.
    searchRange : int
        search range in x direction.

    Returns
    -------
    min_x
        x-coordinate on right image.

    """
    
    
    def compute_SSD(left_window, right_window):
        """
        Computes Sum of Squared Distances(SSD).

        Parameters
        ----------
        left_window : ndArray
            Left window.
        right_window : ndArray
            Right window.

        Returns
        -------
        double
            Sum of squared distances.

        """
        
        if(left_window.shape != right_window.shape):
            return -1
        
        ssd = np.sum(np.square(left_window - right_window))
        
        return ssd
    
    
    x_min = max(0, x - searchRange)
    x_max = min(right_image.shape[1], x + searchRange)
    min_ssd = None
    min_x = None
    
    for x in range(x_min, x_max):
        block_right = right_image[y: y + blockSize, x: x + blockSize]
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
    """
    Computes the disparity map.

    Parameters
    ----------
    left_img : ndArray
        Left image.
    right_img : ndArray
        Right image.
    blockSize : int
        Dimension of block.
    searchRange : int
        search range in x-direction.

    Raises
    ------
    Exception
        if left and right image shapes mismatch.

    Returns
    -------
    disparity_map : ndArray
        Disparity map.

    """
    
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
    

def get_Depth_Map(disparity_map, b, f):
    """
    Computes the Depth map.

    Parameters
    ----------
    disparity_map : ndArray
        Disparity map.
    b : float
        Baseline distance between the two camera centers.
    f : float
        Focal length of camera.

    Returns
    -------
    depth_map : ndArray
        Depth map.

    """
    
    h,w = disparity_map.shape[:2]
    depth_map = np.zeros_like(disparity_map)
    for y in range(h):
        for x in range(w):
            if(disparity_map[y, x] == 0):
                depth_map[y, x] = 0
            else:
                depth_map[y, x] = int(b * f / disparity_map[y, x])
    
    return depth_map
    

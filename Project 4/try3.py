#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 16:17:02 2022

@author: pulkit
"""

import cv2
import numpy as np

from cv2 import VideoWriter, VideoWriter_fourcc


cap = cv2.VideoCapture('cars.mp4')

height, width = 360, 640


FPS = 24                                                                      
fourcc = VideoWriter_fourcc(*'mp4v')
video = VideoWriter('./result.mp4', fourcc, float(FPS), (width, height))

feature_params = dict(maxCorners = 500,
                      qualityLevel = 0.2,
                      minDistance = 2,
                      blockSize = 7)


def dist(a, b):
    return np.sqrt(np.power(b[0] - a[0], 2) + np.power(b[1] - a[1], 2))

def get_angle(v1, v2):
    dx = v2[0] - v1[0]
    dy = v2[1] - v2[1]
    angle = np.arctan2(dy, dx) *  180.0 / np.pi
    return angle

def norm_dist(v1, v2, sig = 15):
    theta = get_angle(v1, v2)
    
    x = v1[0] + sig * np.cos(theta)
    y = v1[1] + sig * np.sin(theta)
    
    return v1, (int(x), int(y))

def optical_flow(old_gray, frame_gray, p0):
    lk_params = dict(winSize = (15,15),
                     maxLevel = 2,
                     criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
    
    p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    if st is not None:
        good_new = p1[st==1]
        good_old = p0[st==1]
        return p1,good_new, good_old
    else:
        return None, None

def warp(frame):
    pts1 = np.float32([[(50, height), (250,0), (330,0), (width, height)]])
    pts2 = np.float32([[0,400],[0,0],[200,0],[200,400]])
    
    matrix = cv2.getPerspectiveTransform(pts1, pts2)
    warped = cv2.warpPerspective(frame, matrix, (200,400))

    return warped


def region(cmask):
    
    l1 = np.float32([[(0,375), (0,360), (45,360), (45,375)]])
    l2 = np.float32([[(45,375), (45,360), (90,360), (90,375)]])
    l3 = np.float32([[(90,375), (90,360), (140,360), (140,375)]])
    l4 = np.float32([[(145,375), (145,360), (190,360), (190,375)]])
    cv2.fillPoly(cmask, [l1], (120,0,120), cv2.LINE_AA)
    return True


ret, old_frame = cap.read()
warped_old_frame = warp(old_frame)
old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
warped_old_gray = warp(old_gray)

p0 = cv2.goodFeaturesToTrack(warped_old_gray, mask = None, **feature_params)

mask = np.zeros_like(warped_old_frame)
color = np.random.randint(0,255,(500,3))
frame_count = 0
current_angle = 0

cal_mask = np.zeros_like(old_frame[:,:,0])
cal_poly = np.array([[0,375], [0,360], [190,360], [190,375]])

l1 = np.array([[(0,375), (0,360), (45,360), (45,375)]])
l2 = np.array([[(45,375), (45,360), (90,360), (90,375)]])
l3 = np.array([[(90,375), (90,360), (140,360), (140,375)]])
l4 = np.array([[(145,375), (145,360), (190,360), (190,375)]])

cv2.fillConvexPoly(cal_mask, cal_poly, 1)

while(True):
    frame_count += 1
    ret, frame = cap.read()
    if not ret:
        break
    
    cmask = frame.copy()
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # angles = [0]
    # kps = 0
    warped_frame = warp(frame)
    warped_frame_gray = warp(frame_gray)
    p1,good_new, good_old = optical_flow(warped_old_gray, warped_frame_gray, p0)
    
    
    # cv2.fillPoly(cmask, [l1], (120,0,120), cv2.LINE_AA)
    
    # if good_new is not None:
    
    for i,(new,old) in enumerate(zip(good_new,good_old)):
        new_x, new_y = new.ravel().astype(np.int0)
        old_x, old_y = old.ravel().astype(np.int0)
        # print(new_x, new_y, old_x, old_y)
        # mask = cv2.line(mask, (int(new_x), int(new_y)), (int(old_x), int(old_y)), (0,0,255), 1)
        warped_frame = cv2.circle(warped_frame, (new_x, new_y), 5, 255, -1)
        img = cv2.add(warped_frame,mask)
        
      
    # res1 = cv2.pointPolygonTest([l1], p1, True)
    
    warped_old_gray = warped_frame_gray.copy()
    if good_new is not None:
    
        p0 = cv2.goodFeaturesToTrack(warped_old_gray, mask = None, **feature_params)

        # p0 = good_new.reshape(-1,1,2)
    else:
        print("finding new features")
        p0 = cv2.goodFeaturesToTrack(warped_old_gray, mask = None, **feature_params)
    cv2.imshow('frame', img)
    if cv2.waitKey(10) & 0xff == ord('q'):
        break

cv2.destroyAllWindows()
cap.release()
            
                
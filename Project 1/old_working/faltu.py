#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 22:24:47 2022

@author: pulkit
"""

import numpy as np

def get_Outer_Corners(corners):
    
    C = corners.reshape(-1,2)
    d1 = 0
    d2 = 0
    d3 = 0
    d4 = 0
    tl = [0,0]
    tr = [1920,0]
    br = [0,1080]
    bl = [1920,1080]

    for i in range(C.shape[0]):
        if(dist(C[i],bl) > d1):
            d1 = dist(C[i], bl)
            coor1 = C[i]
    for i in range(C.shape[0]):
        if(dist(C[i],br) > d2):
            d2 = dist(C[i], br)
            coor2 = C[i]
    for i in range(C.shape[0]):
        if(dist(C[i],tl) > d3):
            d3 = dist(C[i], tl)
            coor3 = C[i]
    for i in range(C.shape[0]):
        if(dist(C[i],tr) > d4):
            d4 = dist(C[i], tr)
            coor4 = C[i]
            
    return np.array([coor1,coor2,coor3,coor4])


def dist(p1,p2):
    d = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    return d

def line(p1, p2, x, y):
    
    f = ((p2[1] - p1[1]) * (x - p1[0])) / ( p2[0] - p1[0]) + (p1[1] + - y)
    
    return abs(f)

# def get_Inner_(points,corners):
#     C = corners.reshape(-1,2)
    
#     for i in range(C.shape[0]):
#         line(points[0],points[2],C[i][0],C[i][1])
    
#     return True


def nearestpoint(excorner, corners):
    C = corners.reshape(-1,2)
    d1 = 10000
    d2 = 10000
    d3 = 10000
    d4 = 10000
    tl = tuple(excorner[0])
    tr = tuple(excorner[1])
    br = tuple(excorner[2])
    bl = tuple(excorner[3])
    coor1 = (0,0)
    coor2 = (0,0)
    coor3 = (0,0)
    coor4 = (0,0)
    
    for i in range(C.shape[0]):
        if(dist(C[i],bl) < d1 and notonline(bl,tl,C[i]) and notonline(bl,br,C[i])):
            d1 = dist(C[i], bl)
            coor1 = C[i]
    for i in range(C.shape[0]):
        if(dist(C[i],br) < d2 and notonline(br,tr,C[i]) and notonline(br,bl,C[i])):
            d2 = dist(C[i], br)
            coor2 = C[i]
    for i in range(C.shape[0]):
        if(dist(C[i],tl) < d3 and notonline(tl,tr,C[i]) and notonline(tl,bl,C[i])):
            d3 = dist(C[i], tl)
            coor3 = C[i]
    for i in range(C.shape[0]):
        if(dist(C[i],tr) < d4 and notonline(tr,tl,C[i]) and notonline(br,tr,C[i])):
            d4 = dist(C[i], tr)
            coor4 = C[i]
    
    return np.array([coor1,coor2,coor3,coor4])


def notonline(p1,p2,p3):
    flag = False
    if (line(p1,p2,p3[0],p3[1]) > 20):
        flag = True
    return flag;

def sheetcorners(corners):
    
    a = corners.reshape(-1,2)

    x = a[:,0]
    y = a[:,1]
    
    xarg =  np.argsort(x)
    
    xmin_index = np.argmin(x)
    xmin = x[xarg][0]
    y_for_xmin = a[xmin_index][1]
    C1 = [xmin,y_for_xmin]
    
    xmax_index = np.argmax(x)
    xmax = x[xarg][-1]
    y_for_xmax = a[xmax_index][1]
    C2 = [xmax,y_for_xmax]
    
    yarg = np.argsort(y)
    
    ymin_index = np.argmin(y)
    ymin = y[yarg][0]
    x_for_ymin = a[ymin_index][0]
    C3 = [x_for_ymin,ymin]
    
    ymax_index = np.argmax(y)
    ymax = y[yarg][-1]
    x_for_ymax = a[ymax_index][0]
    C4 = [x_for_ymax,ymax]
    
    return np.array([C1,C2,C3,C4])
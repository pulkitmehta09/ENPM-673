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



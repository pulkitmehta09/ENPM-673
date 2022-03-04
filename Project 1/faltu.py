#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 22:24:47 2022

@author: pulkit
"""

import numpy as np

def idkwhy(corners):
    
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

# Fartpoint sfrom window corner and then NEAREST POINTS FROM THE RESULTING ORDERED SET OF POINTS

def dist(p1,p2):
    d = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    return d

def line(p1, p2, x, y):
    
    f = ((p2[1] - p1[1]) * (x - p1[0])) / ( p2[0] - p1[0]) + (p1[1] + - y)
    
    return abs(f)

def getinnerpts(points,corners):
    C = corners.reshape(-1,2)
    
    for i in range(C.shape[0]):
        line(points[0],points[2],C[i][0],C[i][1])
    
    return True


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
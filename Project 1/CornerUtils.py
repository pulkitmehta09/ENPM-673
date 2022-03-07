#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  2 22:24:47 2022

ENPM 673
Project 1

@author: Pulkit Mehta
UID: 117551693
"""

import numpy as np


def dist(p1,p2):
    """
    Calculates the distance between two points.

    Parameters
    ----------
    p1 : Array
        Coordinates of point 1.
    p2 : Array
        Coordinates of point 2.

    Returns
    -------
    d : double
        Distance between the two points.

    """
    
    d = ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    
    return d


def line(p1, p2, x, y):
    """
    Calculates the absolute distance of a point from the line formed by two given points. 

    Parameters
    ----------
    p1 : Array
        Coordinates of point 1 for forming line.
    p2 : Array
        Coordinates of point 2 for forming line.
    x : double
        x-coordinate of the point for which distance needs to be calculated.
    y : double
        y-coordinate of the point for which distance needs to be calculated.

    Returns
    -------
    double
        Absolute distance of point from line.

    """
    
    f = ((p2[1] - p1[1]) * (x - p1[0])) / ( p2[0] - p1[0]) + (p1[1] + - y)
    
    return abs(f)


def get_Outer_Corners(corners):
    """
    Retrieves the outer white sheet corners by finding the diagonal point at maximum distance from the corners of the image frame.

    Parameters
    ----------
    corners : Array
        Obtained corners from Corner detection.

    Returns
    -------
    Array
        Coordinates of the corners of the white sheet.

    """
    
    C = corners.reshape(-1,2)
    # Initialize distances to zero
    d1 = 0
    d2 = 0
    d3 = 0
    d4 = 0
    # Defining the frame corners.
    tl = [0,0]
    tr = [1920,0]
    br = [0,1080]
    bl = [1920,1080]

    for i in range(C.shape[0]):                                                 # Loop through every obtained corner from corner detection
        if(dist(C[i],bl) > d1):                                                 # Find the point with maximum distance and save it
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


def notonline(p1,p2,p3):
    """
    Checks is a point is within close proximity of a line created from another given set of points.

    Parameters
    ----------
    p1 : Array
        Coordinates of point 1.
    p2 : Array
        Coordinates of point 2.
    p3 : Array
        Coordinates of point 3.

    Returns
    -------
    bool
        Whether the point is near line or not.

    """
    flag = False
    if (line(p1,p2,p3[0],p3[1]) > 20):
        flag = True
    return flag;


def nearestpoint(excorner, corners):
    """
    Retrieves the point nearest to another point.
    Creates lines between adjacent points and filters out points in close proximity to the line.

    Parameters
    ----------
    excorner : Array
        Set of corners(of the outer white sheet in this case).
    corners : Array
        Extracted set of corners.

    Returns
    -------
    Array
        Coordinates of the corners of the tag.

    """
    C = corners.reshape(-1,2)                                                   # Reshaping the corners array obtained from Corner detection.
    # Distance is initialized to a certain high value.
    d1 = 10000
    d2 = 10000
    d3 = 10000
    d4 = 10000
    # Corners of white sheet 
    tl = tuple(excorner[0])                                                     # Top left corner
    tr = tuple(excorner[1])                                                     # Top right corner
    br = tuple(excorner[2])                                                     # Bottom right corner
    bl = tuple(excorner[3])                                                     # Bottom left corner            
    # Initializing resultant corner coordinates.
    coor1 = (0,0)
    coor2 = (0,0)
    coor3 = (0,0)
    coor4 = (0,0)
    
    for i in range(C.shape[0]):                                                 # Loop through every obtained corner from corner detection.                        
        if(dist(C[i],bl) < d1 and notonline(bl,tl,C[i])                         # Finding point with least distance which is
           and notonline(bl,br,C[i])):                                          # not on or near line created with adjacent points 
            d1 = dist(C[i], bl)
            coor1 = C[i]
    for i in range(C.shape[0]):
        if(dist(C[i],br) < d2 and notonline(br,tr,C[i]) 
           and notonline(br,bl,C[i])):
            d2 = dist(C[i], br)
            coor2 = C[i]
    for i in range(C.shape[0]):
        if(dist(C[i],tl) < d3 and notonline(tl,tr,C[i])
           and notonline(tl,bl,C[i])):
            d3 = dist(C[i], tl)
            coor3 = C[i]
    for i in range(C.shape[0]):
        if(dist(C[i],tr) < d4 and notonline(tr,tl,C[i]) 
           and notonline(br,tr,C[i])):
            d4 = dist(C[i], tr)
            coor4 = C[i]
    
    return np.array([coor1,coor2,coor3,coor4])


def get_outer_corners(corners):
    """
    Retrieves the corners of the outer white sheet.

    Parameters
    ----------
    corners : Array
        DESCRIPTION.

    Returns
    -------
    TYPE
        DESCRIPTION.

    """
    
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
    

def get_aspect_ratio(points):
    """
    

    Parameters
    ----------
    points : TYPE
        DESCRIPTION.

    Returns
    -------
    aspect_ratio : TYPE
        DESCRIPTION.

    """
    
    aspect_ratio = dist(points[0],points[1])/dist(points[0],points[3])
    
    return aspect_ratio
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Homework 1 Problem 1
# ENPM673 Spring 2022
# Section 0101

@author: Pulkit Mehta
UID: 117551693

"""

import math

#----------------------------------------------------------------------------------
# GIVEN DATA
# ---------------------------------------------------------------------------------
resolution = 5e6                            # Camera resolution
sensor_width = 14                           # Sensor width in mm
sensor_height = sensor_width                # Sensor height in mm
sensor_area = math.pow(sensor_width, 2)     # Calculated area of sensor in mm^2
focal_length = 25                           # Given focal length in mm
    
object_width = 50                           # Object width in mm
object_height = object_width                # Object height in mm
object_distance = 20e3                      # Distance of object from camera in mm


# ---------------------------------------------------------------------------------
# PART 1
# ---------------------------------------------------------------------------------



def FieldofView(d, f):
    """
    This function calculates the field of view of camera.

    Parameters
    ----------
    d : float
        Sensor dimension.
    f : float
        Focal length.

    Returns
    -------
    fov : float
        Field of view of camera.

    """
    
    fov = (2 * (math.atan(d/(2*f)))) * (180/math.pi)
    return fov

# ---------------------------------------------------------------------------------
# PART 2
# ---------------------------------------------------------------------------------

def NumberofPixels(object_distance):
    """
    This function calculates the number of pixels occupied by the object in image.

    Parameters
    ----------
    object_distance : float
        Distance of object from camera.

    Returns
    -------
    number_of_pixels : int
        Number of pixels occupied by object in image.

    """

    
    image_height = object_height * focal_length / object_distance
    image_width = object_width * focal_length / object_distance
    image_area = image_height * image_width
    number_of_pixels = math.floor(image_area * resolution / sensor_area)
    
    return number_of_pixels


# ---------------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------------
print("Horizontal field of view in degrees is:", FieldofView(sensor_width,focal_length))
print("Vertical field of view in degrees is:",FieldofView(sensor_height,focal_length))
print("Number of pixels occupied by the object in image:", NumberofPixels(object_distance))
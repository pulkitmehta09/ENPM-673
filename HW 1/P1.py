#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 13:12:53 2022

@author: pulkit
"""

import numpy as np
import math

resolution = 5e6
sensor_width = 14
sensor_height = sensor_width
sensor_area = math.pow(sensor_width, 2)
focal_length = 25

object_width = 50
object_height = object_width
object_distance = 20e3


def numberofPixels(object_distance):
    image_distance = (focal_length * object_distance)/(object_distance - focal_length)
    image_height = (image_distance / object_distance)*object_height
    image_width = (image_distance / object_distance)*object_width
    image_area = image_height * image_width
    number_of_pixels = image_area * resolution / sensor_area
    return number_of_pixels

def FieldofView(d, f):
    return 2 * (math.atan(d/(2*f)))

hor_fov = FieldofView(sensor_width,focal_length)
ver_fov = FieldofView(sensor_height,focal_length)

N = numberofPixels(object_distance)
print(N)
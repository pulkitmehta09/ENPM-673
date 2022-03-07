#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 11:15:09 2022

@author: pulkit
"""

import cv2
import numpy as np

img = cv2.imread('testudo.png')
img = cv2.resize(img, (200,200), interpolation= cv2.INTER_LINEAR)
# T = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]])
# tr = np.convolve(T,img)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


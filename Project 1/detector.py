#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 19 11:15:09 2022

@author: pulkit
"""

import cv2

img = cv2.imread('tag.png')
img = cv2.resize(img, (100,100), interpolation= cv2.INTER_LINEAR)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


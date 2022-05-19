#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 18 16:08:03 2022

@author: pulkit
"""

from dataclasses import dataclass
import cv2
import numpy as np


@dataclass
class DataSet:
    dataset_name: str  
    
    def get_data(self):
        path = 'Data/' + self.dataset_name
        im0 = cv2.imread(path + '/im0.png')
        im1 = cv2.imread(path + '/im1.png')
        if (self.dataset_name == 'curule'):
            cam0 = np.array([[1758.23, 0, 977.42],[0, 1758.23, 552.15],[0,0,1]])
            cam1 = np.array([[1758.23, 0, 977.42],[0, 1758.23, 552.15],[0,0,1]])
            doffs = 0
            baseline = 88.39
            width = 1920
            height = 1080
            ndisp = 220
            vmin = 55
            vmax = 195
            ratio = 0.7
        if (self.dataset_name == 'octagon'):
            cam0 = np.array([[1742.11, 0, 804.90],[0, 1742.11, 541.22],[0,0,1]])
            cam1 = np.array([[1742.11, 0, 804.90],[0, 1742.11, 541.22],[0,0,1]])
            doffs = 0
            baseline = 221.76
            width = 1920
            height = 1080
            ndisp = 100
            vmin = 29
            vmax = 61
            ratio = 0.7
        if (self.dataset_name == 'pendulum'):
            cam0 = np.array([[1729.05, 0, -364.24],[0, 1729.05, 552.22],[0,0,1]])
            cam1 = np.array([[1729.05, 0, -364.24],[0, 1729.05, 552.22],[0,0,1]])
            doffs = 0
            baseline = 537.75
            width = 1920
            height = 1080
            ndisp = 180
            vmin = 25
            vmax = 150
            ratio = 0.65
        
        return {"im0" :im0, "im1": im1, "cam0": cam0, 
                "cam1": cam1, "doffs": doffs, "baseline": baseline, 
                "width": width, "height": height, "ndisp": ndisp,
                "vmin": vmin, "vmax": vmax, "ratio": ratio}
  
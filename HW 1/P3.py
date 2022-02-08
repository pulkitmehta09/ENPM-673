#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 18:27:51 2022

@author: pulkit
"""

import numpy as np
import matplotlib.pyplot as plt
import math
import csv
import pandas as pd


def getCovarianceMatrix(x):
    A = x - x.mean(axis=0)
    cov_matrix = np.dot(A.T, A)/len(x)
    eig, eigv = np.linalg.eig(cov_matrix)
    
    return cov_matrix, eig, eigv

def StandardLeastSquares(data):
    x = data[:,0]
    y = data[:,1]
    
    A = np.stack((x, np.ones((len(x)), dtype=int )), axis=1)
    
    A_t = A.transpose()
    A_tA = A_t.dot(A)
    A_tY = A_t.dot(y)
    x_bar = (np.linalg.inv(A_tA)).dot(A_tY)
    res = A.dot(x_bar)
    
    return res

## Old Method
# def ComputeSVD(A):
#     A_t = A.T
#     AA_t = A.dot(A_t)
    
#     U_eig, U_eigv = np.linalg.eig(AA_t)
    
#     sort = U_eig.argsort()[::-1]
#     U_eigs = U_eig[sort]
#     U_eigvs = U_eigv[:,sort]
    
#     diag = np.diag((np.sqrt(U_eigs)))
#     sigma_inv = np.zeros_like(A)
#     sigma_inv[:diag.shape[0],:diag.shape[1]] = diag
    
#     VT = sigma_inv.dot(U_eigvs.T)
#     VT = VT.dot(A)
    
#     return U_eigvs, sigma_inv, VT


def ComputeSVD(A):
    A_t = A.T
    AA_t = A.dot(A_t)
    U_eig, U_eigv = np.linalg.eig(AA_t)
    
    A_tA = A_t.dot(A)
    V_eig, V_eigv = np.linalg.eig(A_tA)
    V_eigv_t = V_eigv.T
    
    temp = np.array(np.diag((np.sqrt(U_eig))))
    sigma = np.zeros_like(A)
    sigma[:temp.shape[0],:temp.shape[1]] = temp
    # H = V_eigv[:,8]
    # H = np.reshape(H,(3,3))
    return U_eigv, sigma, V_eigv_t

def TotalLeastSquares(data):
    
    U = np.vstack(((data[:,0] - data[:,0].mean(axis=0)), (data[:,1] - data[:,1].mean(axis=0)))).T
    
    UTU = np.dot(U.transpose(), U)
    
    temp = np.dot(UTU.transpose(),UTU)
    eig, eigv = np.linalg.eig(temp)
    
        
    
    
    

    return data



filename = 'ENPM673_hw1_linear_regression_dataset - Sheet1.csv'

# ages = []
# charges = []

# with open(filename, 'r') as csvfile:
#     reader = csv.reader(csvfile)
#     for row in reader:
#         ages.append(row[0])
#         charges.append(row[6])
        
df = pd.read_csv(filename)
data = df[['age','charges']].to_numpy()


cov_matrix, eig, eigv = getCovarianceMatrix(data)
eigvec1 = eigv[:,0]
eigvec2 = eigv[:,1]
origin = [data[:,0].mean(axis=0),data[:,1].mean(axis=0)]

fig = plt.figure()
plt.subplot(121)
plt.plot(data[:,0], data[:,1], 'bo')
plt.quiver(*origin, *eigvec1, color=['r'], scale=21)
plt.quiver(*origin, *eigvec2, color=['g'], scale=21)


ls = StandardLeastSquares(data)
plt.subplot(122)
plt.title('Standard Least Squares')
plt.xlabel('Age')
plt.ylabel('Cost')
plt.plot(data[:,0], data[:,1],'bo', label = 'Data')
plt.plot(data[:,0],ls, 'r', label = 'Least Squares')
plt.legend()




plt.show()


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
from IPython import get_ipython

get_ipython().run_line_magic('matplotlib','qt')


def getCovarianceMatrix(x):
    """
    This function calculates the Covariance matrix for pair of elements in a given array.

    Parameters
    ----------
    x : N-D Array
        Contains data of different categories such as age and charges.

    Returns
    -------
    cov_matrix : N-D Array
        Covariance matrix.
    eig : N-D Array
        Eigenvalues of covariance matrix.
    eigv : N-D Array
        Eigenvectors of covariance matrix.

    """
    A = x - x.mean(axis=0)
    cov_matrix = np.dot(A.T, A)/len(x)
    eig, eigv = np.linalg.eig(cov_matrix)
    
    return cov_matrix, eig, eigv

def StandardLeastSquares(data):
    """
    This function calculates a linear fit using Standard Least Squares method.

    Parameters
    ----------
    data : 2-D Array
        Input data.

    Returns
    -------
    res : 1-D Array
        Standard Least Squares fit result.

    """
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
    """
    This function calculates the Singular Value Decomposition(SVD) of a given matrix.

    Parameters
    ----------
    A : 2-D Array
        Matrix for which SVD is to be computed.

    Returns
    -------
    U_eigv : Array
        Eigenvectors of matrix U.
    sigma : Array
        Sigma Matrix.
    V_eigv_t : Array
        Eigenvectors of matrix V.

    """
    A_t = A.T
    AA_t = A.dot(A_t)
    U_eig, U_eigv = np.linalg.eig(AA_t)
    
    A_tA = A_t.dot(A)
    V_eig, V_eigv = np.linalg.eig(A_tA)
    V_eigv_t = V_eigv.T
    
    temp = np.array(np.diag((np.sqrt(U_eig))))
    sigma = np.zeros_like(A)
    sigma[:temp.shape[0],:temp.shape[1]] = temp

    return U_eigv, sigma, V_eigv_t


def TotalLeastSquares(data):
    """
    This function calculates a linear fit using Total Least Squares method.

    Parameters
    ----------
    data : 2-D Array
        Input data.

    Returns
    -------
    y : 1-D Array
        Total least squares fit result.

    """
    
    U = np.vstack(((data[:,0] - data[:,0].mean(axis=0)), (data[:,1] - data[:,1].mean(axis=0)))).T
    
    UTU = np.dot(U.transpose(), U)
    
    B = np.dot(UTU.transpose(),UTU)
    eig, eigv = np.linalg.eig(B)
    i = np.argmin(eig)
    a, b = eigv[:, i]
    d = a * data[:,0].mean(axis=0) + b
    y = []
    for i in range(len(data[:,0])):
        y.append((d - (a * data[i,0])) / b)
    
    # # Method 2
    # eig, eigv = np.linalg.eig(UTU)
    # d = eigv[0][1] * data[:,0].mean(axis=0) + eigv[1][1] * data[:,1].mean(axis=0)
    # y = []
    # for i in range(len(data[:,0])):
    #     y.append(((-eigv[0][1] * data[i,0]) - d) / (eigv[1][1]))
        
    return y


def LeastSquares(X, Y):
    """
    This function calculates the model parameter for linear standard least squares fit.

    Parameters
    ----------
    X : Array
    Y : Array

    Returns
    -------
    ls : float
        Model parameter.

    """
    
    X_tX = X.transpose().dot(X)
    X_tY = X.transpose().dot(Y)
    ls = (np.linalg.inv(X_tX)).dot(X_tY)
    return ls
    


def RANSAC(data):
    """
    This function uses RANSAC to fit a line to given data.

    Parameters
    ----------
    data : 2-D Array
        Input data..

    Returns
    -------
    res : float
        RANSAC fit result.

    """
    
   
    y_val = data[:,1]
    N = math.inf
    sample_count = 0
    p = 0.95
    optimum = None
    threshold = np.std(y_val)/3
    max_inliers = 0
    
    A = np.stack((data[:,0], np.ones((len(data[:,0])), dtype = int)), axis = 1)
    temp = np.column_stack((A, data[:,1]))
    
    while N > sample_count:
        np.random.shuffle(temp)
        choices = temp[:2,:]
        
        x = choices[:,:-1]
        y = choices[:,-1:]
        
        ls = LeastSquares(x, y)
        
        inliers = A.dot(ls)
        
        error = np.abs(y_val - inliers.T)
        num_inliers = np.count_nonzero(error < threshold)
        
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            optimum = ls
        
        outlier_prob = 1 - num_inliers / len(data)
        N = math.log(1 - p) / math.log(1 - (1 - outlier_prob)**2)
        
        sample_count += 1
    
    res = A.dot(optimum)
    # res = np.array(res)
    # res = res.reshape((len(data),1))
    return res





filename = 'ENPM673_hw1_linear_regression_dataset - Sheet1.csv'
        
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
tls = TotalLeastSquares(data)
R = RANSAC(data)
plt.subplot(122)
plt.title('Standard Least Squares')
plt.xlabel('Age')
plt.ylabel('Cost')
plt.plot(data[:,0], data[:,1],'bo', label = 'Data')
plt.plot(data[:,0],ls, 'r', label = 'Least Squares')
plt.plot(data[:,0],tls, 'b', label = 'Total Least Squares')
plt.plot(data[:,0],R, 'g', label = 'RANSAC')
plt.legend()

plt.show()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Homework 1 Problem 3
# ENPM673 Spring 2022
# Section 0101

@author: Pulkit Mehta
UID: 117551693

"""
# ---------------------------------------------------------------------------------
# IMPORTING PACKAGES
# ---------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd

# ---------------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# ---------------------------------------------------------------------------------

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
    eigenvalues : N-D Array
        Eigenvalues of covariance matrix.
    eigenvectors : N-D Array
        Eigenvectors of covariance matrix.

    """
    A = x - x.mean(axis=0)                                                        # Element wise mean subtraction 
    cov_matrix = np.dot(A.T, A)/len(x)                                            # Covariance matrix
    eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)                         # Finding eigenvalues and eigenvectors of covariance matrix 
    
    return cov_matrix, eigenvalues, eigenvectors

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
    x = data[:,0]                                                                 # Data on x-axis
    y = data[:,1]                                                                 # Data on y-axis
    
    # System of equations formed for linear fit, i.e., ax + b = y
    A = np.stack((x, np.ones((len(x)), dtype=int )), axis=1)
    A_t = A.transpose()
    A_tA = A_t.dot(A)
    A_tY = A_t.dot(y)
    x_bar = (np.linalg.inv(A_tA)).dot(A_tY)
    res = A.dot(x_bar)                                                            # Output after applying Least squares model
    
    return res



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
    
    U = np.vstack(((data[:,0] - data[:,0].mean(axis=0)), (data[:,1] - data[:,1].mean(axis=0)))).T    # Calculating U matrix
    
    UTU = np.dot(U.transpose(), U)
    B = np.dot(UTU.transpose(),UTU)                                                                  
    eig, eigv = np.linalg.eig(UTU)                                                                     # Finding eigenvalues and eigenvectors                                                                    
    i = np.argmin(eig)
    a, b = eigv[:, i]                                                                                # Model parameters corresponding to the smallest eigenvalue                       
    d = a * data[:,0].mean(axis=0) + b                                                               # d = a*x_mean + b
    y = []
    
    # Finding Total least squares solution and appending to array
    for i in range(len(data[:,0])):
        y.append((d - (a * data[i,0])) / b)
   
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
    
   
    y_val = data[:,1]                                                             # Data on y-axis
    N = math.inf                                                                  # Maximum number of iterations set to infinity 
    sample_count = 0                                                              # Sample count                      
    p = 0.95                                                                      # Probability for inlier
    optimum = None                                                                # Best fit to be calculated
    threshold = np.std(y_val)/3                                                   # Distance threshold
    max_inliers = 0                                                               # Maximum number of inliers  
    
    A = np.stack((data[:,0], np.ones((len(data[:,0])), dtype = int)), axis = 1)
    temp = np.column_stack((A, data[:,1]))
    
    while N > sample_count:
        # Randomly selecting two points from data
        np.random.shuffle(temp)
        choices = temp[:2,:]                        
        
        x = choices[:,:-1]                                                        # x-coordinates selected points  
        y = choices[:,-1:]                                                        # y-coordinates selected points                          
        
        ls = LeastSquares(x, y)                                                   # Applying least squares fit  
        
        fit = A.dot(ls)                                                           # Least squares fit  
        
        error = np.abs(y_val - fit.T)                                             # Error   
        num_inliers = np.count_nonzero(error < threshold)                         # Number of inliers
        
        if num_inliers > max_inliers:
            max_inliers = num_inliers
            optimum = ls
        
        outlier_prob = 1 - num_inliers / len(data)                                # Outlier probability
        N = math.log(1 - p) / math.log(1 - (1 - outlier_prob)**2)                 # Recalculating maximum number of iterations  
        
        sample_count += 1
    
    res = A.dot(optimum)
    
    return res



# ---------------------------------------------------------------------------------
# INPUT
# ---------------------------------------------------------------------------------

filename = 'ENPM673_hw1_linear_regression_dataset - Sheet1.csv'                   # Input csv file
 
# Creating dataframe and extracting required columns of data.       
df = pd.read_csv(filename)
data = df[['age','charges']].to_numpy()

# ---------------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------------

# ---------------------------------------------------------------------------------
# PART 1
# ---------------------------------------------------------------------------------

cov_matrix, eig, eigv = getCovarianceMatrix(data)
eigvec1 = eigv[:,0]                                                               # Eigenvector 1
eigvec2 = eigv[:,1]                                                               # Eigenvector 2          
origin = [data[:,0].mean(axis=0),data[:,1].mean(axis=0)]                          # Origin of eigenvector plot  

fig = plt.figure()
plt.title('Eigenvectors')
plt.xlabel('Age')
plt.ylabel('Cost')
plt.plot(data[:,0], data[:,1], 'co', label = 'Data')
plt.quiver(*origin, *eigvec1, color=['r'], scale=10, label = 'Eigenvector 1')
plt.quiver(*origin, *eigvec2, color=['g'], scale=10, label = 'Eigenvector 2')
plt.legend()

# ---------------------------------------------------------------------------------
# PART 2
# ---------------------------------------------------------------------------------

ls = StandardLeastSquares(data)                                                   # Least squares fit
tls = TotalLeastSquares(data)                                                     # Total least squares fit  
R = RANSAC(data)                                                                  # RANSAC fit
fig2 = plt.figure()
plt.title('Comparison of LS, TLS and RANSAC')
plt.xlabel('Age')
plt.ylabel('Cost')
plt.plot(data[:,0], data[:,1],'co', label = 'Data')
plt.plot(data[:,0],ls, 'r', label = 'Least Squares')
plt.plot(data[:,0],tls, 'g', label = 'Total Least Squares')
plt.plot(data[:,0],R, 'b', label = 'RANSAC')
plt.legend()

plt.show()


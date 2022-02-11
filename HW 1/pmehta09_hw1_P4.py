#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
# Homework 1 Problem 4
# ENPM673 Spring 2022
# Section 0101

@author: Pulkit Mehta
UID: 117551693

"""
# ---------------------------------------------------------------------------------
# IMPORTING PACKAGES
# ---------------------------------------------------------------------------------

import numpy as np

# ---------------------------------------------------------------------------------
# FUNCTION DEFINITIONS
# ---------------------------------------------------------------------------------

def ComputeSVD(A):
    """
    This function calculates the Singular Value Decomposition(SVD) of a given matrix.

    Parameters
    ----------
    A : 2-D Array
        Matrix for which SVD is to be computed.
        
    Returns
    -------       
    U_eigvs : Array
        Eigenvectors of matrix U.
    sigma : Array
        Sigma Matrix.
    H : Array
        Homography matrix.
    V_t : Array
        Eigenvectors of matrix V.

    """
    A_t = A.T
    AA_t = A.dot(A_t)
    
    # Constructing U matrix
    U_eig, U_eigv = np.linalg.eig(AA_t)
    sort = U_eig.argsort()[::-1]
    U_eigs = U_eig[sort]
    U_eigvs = U_eigv[:,sort]
    
    for i in range(len(U_eigs)):
        if U_eigs[i] <= 0:
            U_eigs[i] *= -1
    
    # Constructing V matrix
    A_tA = A_t.dot(A)
    V_eig, V_eigv = np.linalg.eig(A_tA)
    sort1 = V_eig.argsort()[::-1]
    V_eigvs = V_eigv[:,sort1]
    V_t = V_eigvs.transpose()
    
    # Contructing Sigma matrix
    temp = np.array(np.diag((np.sqrt(U_eigs))))
    sigma = np.zeros_like(A)
    sigma[:temp.shape[0],:temp.shape[1]] = temp
    
    # Homography matrix
    H = V_eigvs[:,-1:]
    H = np.reshape(H,(3,3))
    
    return U_eigvs, sigma, V_t, H




# ---------------------------------------------------------------------------------
# GIVEN DATA
# ---------------------------------------------------------------------------------

x = np.array([5,150,150,5])
y = np.array([5,5,150,150])
xp = np.array([100,200,220,100])
yp = np.array([100,80,80,200])
    
# Given A matrix
A = np.array([[-x[0],-y[0],-1,0,0,0,x[0]*xp[0],y[0]*xp[0],xp[0]],
              [0,0,0,-x[0],-y[0],-1,x[0]*yp[0],y[0]*yp[0],yp[0]],
              [-x[1],-y[1],-1,0,0,0,x[1]*xp[1],y[1]*xp[1],xp[1]],
              [0,0,0,-x[1],-y[1],-1,x[1]*yp[1],y[1]*yp[1],yp[1]],
              [-x[2],-y[2],-1,0,0,0,x[2]*xp[2],y[2]*xp[2],xp[2]],
              [0,0,0,-x[2],-y[2],-1,x[2]*yp[2],y[2]*yp[2],yp[2]],
              [-x[3],-y[3],-1,0,0,0,x[3]*xp[3],y[3]*xp[3],xp[3]],
              [0,0,0,-x[3],-y[3],-1,x[3]*yp[3],y[3]*yp[3],yp[3]]])

# ---------------------------------------------------------------------------------
# OUTPUT
# ---------------------------------------------------------------------------------

U, S, V_t, H = ComputeSVD(A)

print("Homography matrix: \n", H)
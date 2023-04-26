# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 18:51:09 2023

@author: 22097
"""

import numpy as np

A = np.linspace(1, 17, 16).reshape((4, 4))
B = A
dt = np.linalg.det(A)
n = np.size(A, axis=0)


I = np.eye(n)
L = I
U = np.zeros([n, n])
c_k = np.zeros(n).reshape((n, 1)) 
      

for j in range(n-1):
    c_k[j] = 0
    e_k = I[:, j].reshape((1, n))

    for k in range(j+1, n):
        c_k[k] = A[k][j]/A[j][j]
        
    L_k=I+np.dot(c_k,e_k)
    L=L+np.dot(c_k, e_k)
    T_k= I-np.dot(c_k,e_k)
    A = np.dot(T_k, A)

U = A

B = np.linspace(1, 17, 16).reshape((4, 4))
B_test = np.dot(L, U)
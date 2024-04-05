#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 17:05:32 2021

@author: lubos
"""

import numpy as np
import matplotlib.pyplot as plt

def solveJacobi(A, b):
    nr = A.shape[0]   # number of rows
    x = np.zeros(nr)
    x_new = np.zeros(nr)
    # repeate 1000 times
    for it in range(1000):
        for r in range(nr):
            x_new[r] = (b[r] - np.dot(A[r,:],x) + A[r,r]*x[r])/A[r,r]
        
        # copy down solution
        x[:] = x_new[:]
        return x

def solveGS(A, b, max_it):
    nr = A.shape[0]   # number of rows
    x = np.zeros(nr)
    # repeate 1000 times
    for it in range(max_it):
        for r in range(nr):
            x[r] = (b[r] - np.dot(A[r,:],x) + A[r,r]*x[r])/A[r,r]
    return x

L = 1          # domain size
ni = 16         # number of nodes
dx = L/(ni-1)  # cell spacing
b0 = -100
        
A = np.zeros((ni,ni))  # ni*ni empty matrix
b = np.zeros(ni)   # RHS vector

A[0,0] = 1      # left boundary
b[0] = 0
A[ni-1,ni-1] = 1 # right boundary
b[ni-1] = 0
for r in range(1,ni-1):  
    A[r,r-1] = 1/(dx*dx)  # coefficient for T[r-1]
    A[r,r] = -2/(dx*dx)   # coefficient for T[r]
    A[r,r+1] = 1/(dx*dx)  # coefficient for T[r+1]
    b[r] = b0             # forcing vector

#T_j = solveJacobi(A,b)

T_g = solveGS(A,b,10)

# plot solution
plt.close('all')
plt.figure(figsize=(12,5))
plt.rc('font', size=18)          # controls default text sizes
plt.rc('legend',fontsize=15)

x_n = np.linspace(0,1,ni)
plt.plot(x_n,T_g,label='G-S 10',
         color=(0.8,0.8,0.8), 
         linewidth=3, linestyle='--', 
         marker='o', markersize=6)


T_g = solveGS(A,b,25)
# plot solution
plt.plot(x_n,T_g,label='G-S 25',
         color=(0.6,0.6,0.6), 
         linewidth=3, linestyle='--', 
         marker='o', markersize=7)


T_g = solveGS(A,b,50)
# plot solution
plt.plot(x_n,T_g,label='G-S 50',
         color=(0.4,0.4,0.4), 
         linewidth=4, linestyle='--', 
         marker='o', markersize=8)

T_g = solveGS(A,b,100)
# plot solution
plt.plot(x_n,T_g,label='G-S 100',
         color=(0.2,0.2,0.2), 
         linewidth=4, linestyle='--', 
         marker='o', markersize=9)

T_g = solveGS(A,b,200)
# plot solution
plt.plot(x_n,T_g,label='G-S 200',
         color=(0.0,0.0,0.0), 
         linewidth=4, linestyle='--', 
         marker='o', markersize=10)

x = np.linspace(0,L,100)
plt.plot(x,0.5*b0*x*x-0.5*b0*x,color=(0.2,0.2,0.2),
         linewidth=3, label='Analytical')

handles, labels = plt.gca().get_legend_handles_labels()
plt.legend(handles[::-1], labels[::-1], loc='upper right')
plt.xlabel('x (m)')
plt.ylabel('T')
plt.tight_layout()
plt.savefig('convergence.pdf',dpi=300)
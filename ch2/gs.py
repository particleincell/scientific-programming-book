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
    for it in range(10000):
        for r in range(nr):
            x_new[r] = (b[r] - np.dot(A[r,:],x) + A[r,r]*x[r])/A[r,r]
        
        # copy down solution
        x[:] = x_new[:]
        
        # convergence check every 25 steps
        if it%1==0:
            r = b-np.dot(A,x)   # r = b-A*x
            s = np.dot(r,r)     # s = sum(r[i]*r[i])
            norm = np.sqrt(s/nr)
            if (norm<1e-6): 
                print("J %d: %.2g"%(it,norm))            
                break  # exit for loop
    return x

def solveGS(A, b):
    nr = A.shape[0]   # number of rows
    x = np.zeros(nr)
    for it in range(10000):
        for r in range(nr):
            x[r] = (b[r] - np.dot(A[r,:],x) + A[r,r]*x[r])/A[r,r]
            
        # convergence check every 25 steps
        if it%1==0:
            r = b-np.dot(A,x)   # r = b-A*x
            s = np.dot(r,r)     # s = sum(r[i]*r[i])
            norm = np.sqrt(s/nr)            
            if (norm<1e-6): 
                print("%d: %.2g"%(it,norm))
                break  # exit for loop          
    return x

def solveGSSOR(A, b):
    nr = A.shape[0]   # number of rows
    x = np.zeros(nr)
    for it in range(10000):
        for r in range(nr):
            x_star = (b[r] - np.dot(A[r,:],x) + A[r,r]*x[r])/A[r,r]
            x[r] = x[r] + 1.4*(x_star-x[r])
            
        # convergence check every 25 steps
        if it%1==0:
            r = b-np.dot(A,x)   # r = b-A*x
            s = np.dot(r,r)     # s = sum(r[i]*r[i])
            norm = np.sqrt(s/nr)            
            if (norm<1e-6): 
                print("%d: %.2g"%(it,norm))
                break  # exit for loop
            
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

T_j = solveJacobi(A,b)

T_g = solveGS(A,b)
T_g = solveGSSOR(A,b)


x_n = np.linspace(0,1,ni)
# plot solution
plt.close('all')

plt.plot(x_n,T_g,label='G-S',
         color=(0.0,0.0,0.0), 
         linewidth=4, linestyle='--', 
         marker='o', markersize=10)

x = np.linspace(0,L,100)
plt.plot(x,0.5*b0*x*x-0.5*b0*x,color=(0.2,0.2,0.2),
         linewidth=2, label='Analytical')

#handles, labels = plt.gca().get_legend_handles_labels()
#plt.legend(handles[::-1], labels[::-1], loc='upper right')
plt.legend()
plt.xlabel('x (m)')
plt.ylabel('T')
plt.savefig('convergence.pdf')
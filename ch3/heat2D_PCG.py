#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASTE-404 heat equation solver using PCG
Created on Mon Mar 15 16:28:36 2021

@author: lubos
"""

import numpy as np
import matplotlib.pyplot as plt
from random import random as rand
import time


ni = 20        # number of nodes in x
nj = ni       # number of nodes in y
nu = ni*nj     # total number of nodes
Lx = 1         # length in x
Ly = 1         # length in y
dx = Lx/(ni-1)  # cell spacing in x
dy = Ly/(nj-1)  # cell spacing in y

A = np.zeros((nu,nu))
b = np.zeros(nu)

def PCG(A,b):
    n,nc = np.shape(A)
    L = np.zeros_like(A)
    U = np.zeros_like(A)
    
    print("Performing L-U decomposition")
    if (A[0,0]==0):
        print("Can't continue")
        return
    L[0,0] = 1.0
    U[0,0] = A[0,0]/L[0,0]
    
    for j in range(1,n):   #j=[1,2,3,....,n-1]
        U[0,j] = A[0,j]/L[0,0]
        L[j,0] = A[j,0]/U[0,0]
    
    for i in range(1,n-1):
        s = 0
        for k in range(i): s+=L[i,k]*U[k,i]
        L[i,i] = 1.0
        U[i,i] = A[i,i] - s
        if (U[i,i]==0):
            print("Can't continue")
            return
        for j in range(i+1,n):
            s = 0
            for k in range(i):  s+=L[i,k]*U[k,j]
            U[i,j] = (1/L[i,i])*(A[i,j]-s)
            s = 0
            for k in range(i): s+=L[j,k]*U[k,i]
            L[j,i] = 1/(U[i,i])*(A[j,i]-s)
            
    s = 0
    for k in range(n): s+=L[n-1,k]*U[k,n-1]
    L[n-1,n-1] = 1
    U[n-1,n-1] = A[n-1,n-1]-s
        
    return L, U
    
# set coefficients
for j in range(nj):
    for i in range(ni):
        u = j*ni+i     #note lack of -1 since zero-based indexing
         
        # boundary conditions
        if (i==0):       # Neumann on xmin
            A[u,u] = 1;
            A[u,u+1] = -1;
            b[u] = 0.0;
        elif (i==ni-1): # Neumann on xmax
            A[u,u] = 1;
            A[u,u-1] = -1;
            b[u] = 0.0;  
        elif (j==0):    # Dirichlet on ymin
            A[u,u] = 1;
            b[u] = 0.0;
        elif (j==nj-1): # Neumann on ymax
            A[u,u] = 1;
            A[u,u-ni] = -1;
            b[u] = 0.0;
        else:
            A[u,u-ni] = 1/(dy*dy)
            A[u,u-1] = 1/(dx*dx)
            A[u,u+1] = 1/(dx*dx)
            A[u,u+ni] = 1/(dy*dy)
            A[u,u] = - 2/(dx*dx) - 2/(dy*dy)
            b[u] = 0
        
# assign random internal points
for p in range (20):
    i = 1+int((ni-2)*rand())
    j = 1+int((nj-2)*rand())
    u = j*ni+i
    A[u,:] = 0    # clear row
    A[u,u] = 1    # Dirichlet node
    b[u] = 1+9*rand() # random value
    
t1 = time.perf_counter()
T = solvePCG(A,b);   
t2 = time.perf_counter()
t_solve = t2-t1

print ("solver time %.4g seconds"%(t_solve))


T2d = np.reshape(T,(nj,ni))  # convert to 2D matrix

xs = np.linspace(0,Lx,ni)  # x-coordinates for plotting
ys = np.linspace(0,Ly,nj)  # y-coordinates for plotting

#plot countour levels and lines
fig,ax = plt.subplots()
plt.contourf(xs,ys,T2d,levels=12,cmap='hot_r')
plt.colorbar();
plt.contour(xs,ys,T2d,levels=12,colors='black')
plt.savefig('heat2d-pcg.png',dpi=300)        
        
        
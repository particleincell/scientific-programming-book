#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 17:05:32 2021

@author: lubos
"""

import numpy as np
import matplotlib.pyplot as plt

def solveGJ(A,b):      
    # make augmented matrix
    G = np.zeros((ni,ni+1))   # one extra column
    G[:,:-1] = A              # copy A into [0:ni-1,0:ni-1]
    G[:,-1] = b               # copy b into the last column
    
    # perform G-J elimination
    for r in range(1,ni-1):
        # show the elimination rule
        #print("L%d + %gL%d -> L%d"%(r,-(G[r,r-1]/G[r-1,r-1]),r-1,r))
        
        # apply rule
        G[r,:] = G[r,:] - (G[r,r-1]/G[r-1,r-1])*G[r-1,:]
        
        # display matrix 
        if (False):
            for r in range(ni):
                for c in range(ni+1):
                    print("%6.2f"%G[r,c],end='')
                print()
            print()
        
    # back substitution
    x = np.zeros(ni)
    
    # loop from r=ni-1 to r=0
    for r in range(ni-1,-1,-1):  
        s = 0  # compute sum a[r,r+1:]*T[r+1:]
        for c in range(r+1,ni):
            s += G[r,c]*x[c]
        # get value for T_gj[r]
        x[r] = (G[r,ni]-s)/G[r,r]
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

T_gj = solveGJ(A,b)

# plot solution
x_gj = np.linspace(0,L,ni)
plt.close('all')
plt.figure(figsize=(10,5))
plt.rc('font', size=20)          # controls default text sizes

x = np.linspace(0,L,100)
plt.plot(x,0.5*b0*x*x-0.5*b0*x,color=(0.2,0.2,0.2),
         linewidth=4, label='Analytical')


plt.plot(x_gj,T_gj,label='Gauss-Jordan',
         color="0.3", 
         linewidth=4, linestyle='--', 
         marker='o', markersize=10)


plt.legend()
plt.xlabel('x (m)')
plt.ylabel('T')
plt.tight_layout()

plt.savefig('gj.pdf',dpi=300)
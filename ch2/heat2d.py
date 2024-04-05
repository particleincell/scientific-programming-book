#USC ASTE-499 lesson 2 example
import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp_mat
import matplotlib.pyplot as plt
from random import random as rand
import time

ni = 80        # number of nodes in x
nj = 60        # number of nodes in y
nu = ni*nj     # total number of nodes
Lx = 2         # length in x
Ly = 1         # length in y
dx = Lx/(ni-1)  # cell spacing in x
dy = Ly/(nj-1)  # cell spacing in y
A = sp_mat.lil_matrix((nu,nu))
b = np.zeros(nu)

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
            b[u] = 1.0;
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
            b[u] = -10
        
# assign random internal points
for p in range (20):
    i = 2+int((ni-2)*rand())
    j = 2+int((nj-2)*rand())
    u = j*ni+i
    A[u,:] = 0    # clear row
    A[u,u] = 1    # Dirichlet node
    b[u] = 1+9*rand() # random value

t1 = time.perf_counter()
T = spsolve(A,b);    # sparse matrix solver
t2 = time.perf_counter()
print ("solver took %.4g seconds"%(t2-t1))

T2d = np.reshape(T,(nj,ni))  # convert to 2D matrix

xs = np.linspace(0,Lx,ni)  # x-coordinates for plotting
ys = np.linspace(0,Ly,nj)  # y-coordinates for plotting

#plot countour levels and lines
plt.figure(figsize=(12,6))
plt.contourf(xs,ys,T2d,levels=12,cmap='gray_r')
plt.colorbar();
plt.contour(xs,ys,T2d,levels=12,colors='black')
plt.savefig('heat2d.png',dpi=300)        
    
    

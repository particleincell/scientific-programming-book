#USC ASTE-499 lesson 2 example
import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp_mat
import matplotlib.pyplot as plt
from random import random as rand

ni = 14*2
nj = 20*2
nu = ni*nj

dx = 1/(ni-1)
dy = 1/(nj-1)
A = np.zeros((nu,nu))
b = np.zeros(nu)
boundary_node = np.zeros(nu)

for j in range(nj):
    for i in range(ni):
        u = j*ni+i     #note lack of -1 since zero-based indexing
         
        boundary_node[u] = True
        
        # boundary conditions
        if (i==0):             # 0 instead of 1
            A[u,u] = 1;
            A[u,u+1] = -1;
            b[u] = 0.0;            
        elif (i==ni-1):        # ni-1 istead of ni
            A[u,u] = 1;
            A[u,u-1] = -1;
            b[u] = 0.0;  
        elif (j==0):
            A[u,u] = 1;
            A[u,u+ni] = -1;
            b[u] = 0.0;
        elif (j==nj-1):
            A[u,u] = 1;
            A[u,u-ni] = -1;
            b[u] = 0.0;
        #elseif ((rx*rx+ry*ry)<(r*r))
        #    A(u,u) = 1;
        #   b(u) = 4;
        else:
            A[u,u-ni] = 1/(dy*dy)
            A[u,u-1] = 1/(dx*dx)
            A[u,u+1] = 1/(dx*dx)
            A[u,u+ni] = 1/(dy*dy)
            A[u,u] = - 2/(dx*dx) - 2/(dy*dy)
            b[u] = 0  
            boundary_node[u] = False
        
for p in range (20):
    i = 2+int((ni-2)*rand())
    j = 2+int((nj-2)*rand())
    u = j*ni+i
    A[u,:] = 0
    A[u,u] = 1
    b[u] = 1+1*rand()
    boundary_node[u] = True



# linear solver, solve A*x=b
def GS_linear(A,b):
    x = np.zeros_like(b)
    nu = b.size;
    
    # G-S iteration loop
    for k in range(1000):
        for u in range(nu):
            x_new = (b[u] - (np.dot(A[u,:],x) - A[u,u]*x[u]))/A[u,u]
            x[u] = x_new
        
    
    return x

def NR(A,b)
  x = np.zeros(ni)
  for k_nr in range(20):
      p =...
      I = np.identity(ni)
      J = A - I*p;      
      F = A*x-b(x);
     
      # use a linear solver to solve Jy=F
      #y = GS_linear(J,F) 
      y = PCG_linear(J,F)
      
      # get new solution
      x = x - y
      
      # add convergence check
      r = A*x-b(x)
      if (norm(r)<tol): break;
      
    return x
        
  

T = NR(A,b)
#x = linsolve(A,b);
T2d = np.reshape(T,(nj,ni))
plt.contourf(T2d,levels=12,cmap='hot')
plt.contour(T2d,levels=12,colors='black')

plt.colorbar;

plt.savefig('heat01.png')        
    
    

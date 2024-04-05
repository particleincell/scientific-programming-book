#USC ASTE-499 lesson 2 example
import numpy as np
import matplotlib.pyplot as plt
from random import random as rand
import time 

ni = 50        # number of nodes in x
nj = ni        # number of nodes in y
nu = ni*nj     # total number of nodes
Lx = 1         # length in x
Ly = 1         # length in y
dx = Lx/(ni-1)  # cell spacing in x
dy = Ly/(nj-1)  # cell spacing in y

b = np.zeros(nu)  # coeff for T[i,j-1]
c = np.zeros(nu)  # coeff for T[i-1,j]
d = np.zeros(nu)  # coeff for T[i,j]
e = np.zeros(nu)  # coeff for T[i+1,j]
f = np.zeros(nu)  # coeff for T[i,j+1]
y = np.zeros(nu)  # right hand side vector

def solveGSSOR(b,c,d,e,f,x,y,ni):
    nu = b.size   # number of rows / nodes
    for it in range(10000):
        for u in range(nu):
            # row-vector dot product minus diagonal term
            dot_nodiag = ((b[u]*x[u-ni] if b[u]!=0 else 0) + 
                          (c[u]*x[u-1]  if c[u]!=0 else 0) +
                          (e[u]*x[u+1]  if e[u]!=0 else 0) +
                          (f[u]*x[u+ni] if f[u]!=0 else 0))
            x_star = (y[u] - dot_nodiag)/d[u]
            x[u] = x[u] + 1.4*(x_star-x[u])
            
        # convergence check every 25 steps
        if it%25==0:
            s = 0
            for u in range(nu):
                dot = ((b[u]*x[u-ni] if b[u]!=0 else 0) + 
                       (c[u]*x[u-1]  if c[u]!=0 else 0) +
                       (d[u]*x[u]) + 
                       (e[u]*x[u+1]  if e[u]!=0 else 0) +
                       (f[u]*x[u+ni] if f[u]!=0 else 0))
                res = y[u]-dot   # r = b-A*x
                s += res*res  # sum of r^2
            norm = np.sqrt(s/nu)  
            if (norm<1e-6): 
                print("Converged in %d iterations, norm: %.2g"%(it,norm))
                break  # exit for loop            
    return x


for j in range(nj):
    for i in range(ni):
        u = j*ni+i     
         
        # boundary conditions
        if (i==0):       # Neumann on xmin
            d[u] = 1;
            e[u] = -1;
            y[u] = 0.0;
        elif (i==ni-1): # Neumann on xmax
            d[u] = 1;
            c[u] = -1;
            y[u] = 0.0;  
        elif (j==0):    # Dirichlet on ymin
            d[u] = 1;
            y[u] = 0.0;
        elif (j==nj-1): # Neumann on ymax
            d[u] = 1;
            b[u] = -1;
            y[u] = 0.0;
        else:
            b[u] = 1/(dy*dy)
            c[u] = 1/(dx*dx)
            e[u] = 1/(dx*dx)
            f[u] = 1/(dy*dy)
            d[u] = - 2/(dx*dx) - 2/(dy*dy)
            y[u] = 0
        
# assign random internal points
for p in range (20):
    i = 1+int((ni-2)*rand())
    j = 1+int((nj-2)*rand())
    u = j*ni+i
    b[u] = c[u] = d[u] = e[u] = f[u] = 0    # clear row
    d[u] = 1    # Dirichlet node
    y[u] = 1+9*rand() # random value


x = np.zeros_like(y)
t_solve = 0
for it in range(20):   # multiple iterations
    #change RHS
    y[np.where(y!=0)] += 1e-3*(-1+2*rand())
    
    t1 = time.perf_counter()
    T = solveGSSOR(b,c,d,e,f,x,y,ni);   
    t2 = time.perf_counter()
    t_solve += t2-t1

print ("GS solver took on average %.4g seconds"%(t_solve/(it+1)))

T2d = np.reshape(T,(nj,ni))  # convert to 2D matrix

xs = np.linspace(0,Lx,ni)  # x-coordinates for plotting
ys = np.linspace(0,Ly,nj)  # y-coordinates for plotting

#plot countour levels and lines
plt.figure(figsize=(12,6))
plt.contourf(xs,ys,T2d,levels=12,cmap='hot_r')
plt.colorbar();
plt.contour(xs,ys,T2d,levels=12,colors='black')
plt.savefig('heat2d.png',dpi=300)        
    
    

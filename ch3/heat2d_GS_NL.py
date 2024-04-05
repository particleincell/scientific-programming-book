#USC ASTE-404 lesson 8 NL GS example
import numpy as np
import matplotlib.pyplot as plt
import random
from random import random as rand

random.seed(0)  # fix seed to generate the same random numbers on each run

ni = 41        # number of nodes in x
nj = 31        # number of nodes in y
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
nl_coeff = np.zeros(nu) # coefficient used to disable NL term on boundaries

def solveGS_NL(b,c,d,e,f,x,y,nl_coeff,ni):
    nu = b.size   # number of rows / nodes
    for it in range(10000):
        for u in range(nu):
            # row-vector dot product minus diagonal term
            dot_nodiag = ((b[u]*x[u-ni] if b[u]!=0 else 0) + 
                          (c[u]*x[u-1]  if c[u]!=0 else 0) +
                          (e[u]*x[u+1]  if e[u]!=0 else 0) +
                          (f[u]*x[u+ni] if f[u]!=0 else 0))
            
            y_nl = y[u] + nl_coeff[u]*np.exp(-x[u]/10)
            x_star = (y_nl - dot_nodiag)/d[u]
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
                
                y_nl = y[u] + nl_coeff[u]*np.exp(-x[u]/10)
                res = y_nl - dot   # r = b-A*x
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
            nl_coeff[u] = 0
        
# assign random internal points
for p in range (20):
    i = 1+int((ni-2)*rand())
    j = 1+int((nj-2)*rand())
    u = j*ni+i
    b[u] = c[u] = d[u] = e[u] = f[u] = 0    # clear row
    d[u] = 1    # Dirichlet node
    y[u] = 1+9*rand() # random value
    nl_coeff[u] = 100


x = np.zeros_like(y)
t_solve = 0

T = solveGS_NL(b,c,d,e,f,x,y,nl_coeff,ni);   
T2d = np.reshape(T,(nj,ni))  # convert to 2D matrix

xs = np.linspace(0,Lx,ni)  # x-coordinates for plotting
ys = np.linspace(0,Ly,nj)  # y-coordinates for plotting

#plot countour levels and lines
fig, ax = plt.subplots()  
plt.contourf(xs,ys,T2d,levels=12,cmap='hot_r')
plt.colorbar();
plt.contour(xs,ys,T2d,levels=12,colors='black')
plt.savefig('heat2d.png',dpi=300)        
    
    

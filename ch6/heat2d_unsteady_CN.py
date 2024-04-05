#USC ASTE-404 FTCS example
import numpy as np
from scipy.sparse.linalg import spsolve
import scipy.sparse as sp_mat
import matplotlib.pyplot as plt
from random import random as rand
from matplotlib.animation import FuncAnimation, PillowWriter  

ni = 41
nj = 31
nu = ni*nj

dx = 1/(ni-1)
dy = 1/(nj-1)

L = sp_mat.lil_matrix((nu,nu))
b = np.zeros(nu)
T = np.ones(nu)*0
R = np.zeros(nu)

I = sp_mat.identity(nu)

# set up matrix
for j in range(nj):
    for i in range(ni):
        u = j*ni+i     #note lack of -1 since zero-based indexing
         
        # boundary conditions
        if (i==0):             # 0 instead of 1
            L[u,u] = 1;
            L[u,u+1] = -1;
            b[u] = 0.0;
        elif (i==ni-1):        # ni-1 istead of ni
            L[u,u] = 1;
            L[u,u-1] = -1;
            b[u] = 0.0;  
        elif (j==0):
            L[u,u] = 1;
            L[u,u+ni] = -1;
            b[u] = 0.0;
        elif (j==nj-1):
            L[u,u] = 1;
            L[u,u-ni] = -1;
            b[u] = 0.0;
        #elseif ((rx*rx+ry*ry)<(r*r))
        #    A(u,u) = 1;
        #   b(u) = 4;
        else:
            L[u,u-ni] = 1/(dy*dy)
            L[u,u-1] = 1/(dx*dx)
            L[u,u+1] = 1/(dx*dx)
            L[u,u+ni] = 1/(dy*dy)
            L[u,u] = - 2/(dx*dx) - 2/(dy*dy)
            b[u] = 0
    

dt = 2e-3
D = 1e-2

A = I - 0.5*dt*L
B = I + 0.5*dt*L

for p in range (20):
    i = 1+int((ni-2)*rand())
    j = 1+int((nj-2)*rand())
    u = j*ni+i
    A[u,:] = 0       # note that correction is done on A, B, not L
    A[u,u] = 1
    B[u,:] = 0
    B[u,u] = 1
    R[u] = 1+1*rand()

def heat_solver_CN():
    global T, R, B
    
    b = B*T + R
    T = spsolve(A,b)
    
    R = R*0  # turn off after the first iteration
    return np.reshape(T,(nj,ni))


fig, ax = plt.subplots()  


# main loop, march solution forward
#for k in range(1000):
#    T = (I+D*dt*A)*T
   

T2d = np.reshape(T,(nj,ni))
lev = np.linspace(0,2,10)
ax.contourf(T2d,cmap='hot_r',levels=lev)
ax.contour(T2d,colors='black',levels=lev)
        
#-------------------------------------------


def update(i): 
    T2d = heat_solver_CN()           # get new solution
    
    #plotting
    if i%10==0:
        print("Time step: %d"%i)
        lev = np.linspace(0,2,10)
        ax.clear()
        ax.contourf(T2d,cmap='hot_r',levels=lev)
        ax.contour(T2d,colors='black',levels=lev)
        plt.savefig('img/cn2d_%d.png'%i,dpi=300)        


   
#writer = PillowWriter(fps=2)  
ani = FuncAnimation(fig, update, 201, repeat=False)  
#ani.save("heat_anim.gif", writer=writer) 
#plt.savefig('heat01.png')        
#plt.show();


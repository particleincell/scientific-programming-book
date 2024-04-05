# -*- coding: utf-8 -*-
"""
FTCS Convection-Diffusion Equation solver
Fluid Modeling of Plasmas Lesson 1
https://www.particleincell.com/fluid-modeling/
"""

import numpy as np
import pylab as pl

#simulation inputs
rho = 1
u = 1
D = 0.02
phi0 = 0
phiL = 1

#set domain parameters
L = 1
ni = 81

# time step
dt = 1.0e-2

# nominal: 1e-2, ni=41, break at 2e-2

x = np.linspace(0,L,ni)

#analytical solution
x_true = np.linspace(0,L,501)
Pe = rho*u*L/D
phi_true = phi0 + ((np.exp(x_true*Pe/L)-1)/
           (np.exp(Pe)-1)*(phiL-phi0))

pl.close('all')
fig=pl.figure(figsize=(6,4))
fig.set_tight_layout('true')

#A matrix
A = np.zeros((ni,ni))
b = np.zeros(ni)

#dirichlet condition on left and right
A[0,0] = 1
A[ni-1,ni-1] = 1
b[0] = phi0
b[ni-1] = phiL

#assuming uniform spacing
dx = x[1]-x[0]

#diffusive term
AdW = -D/(dx*dx)
AdE = -D/(dx*dx)
AdP = -(AdE + AdW)

#convective term
AcE = rho*u/(2*dx)
AcW = -rho*u/(2*dx)
AcP = -(AcW+AcE)

#contribution from both terms
Aw = AdW + AcW
Ap = AdP + AcP
Ae = AdE + AcE

#set internal nodes
for i in range(1,ni-1):    
    A[i,i-1] = Aw
    A[i,i] = Ap
    A[i,i+1] = Ae

#initial values
phi = np.zeros(ni)
phi[0] = b[0]
phi[-1] = b[-1]

#iterate using forward time
max_it = 20
for it in range (max_it):
    if (it%1==0):
        c = 0.2+0.7*(1-it/max_it)
        pl.plot(x,phi,linewidth=3,color=(c,c,c))

    R = np.dot(A,phi) #A*b
    phi[1:-1] = phi[1:-1] - dt*R[1:-1]

 
#plot results
pl.plot(x_true,phi_true,linewidth = 5, 
        linestyle="dashed",color="black",label="Theory")
 
#pl.title("ni=%d, dt=%.1e"%(ni,dt),size=14)
#pl.legend(loc='upper left',fontsize=14)
pl.tick_params(axis='both',which='major',labelsize=13)
pl.grid()
pl.xlim([0.85,1.0])
pl.savefig('ad1-ftcs.pdf')

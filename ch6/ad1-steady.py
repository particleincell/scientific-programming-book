# -*- coding: utf-8 -*-
"""
Steady-State Advection-Diffusion Equation solver
to compare UDS and CDS
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
ni = 21
x = np.linspace(0,L,ni)

#analytical solution
x_true = np.linspace(0,L,501)
Pe = rho*u*L/D
phi_true = phi0 + ((np.exp(x_true*Pe/L)-1)/
           (np.exp(Pe)-1)*(phiL-phi0))

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

#upwind scheme for the convective derivative
AcE = min(rho*u,0)/dx
AcW = -max(rho*u,0)/dx
AcP = -(AcE+AcW)

#contribution from both terms
Aw = AdW + AcW
Ap = AdP + AcP
Ae = AdE + AcE

#set internal nodes
for i in range(1,ni-1):
    A[i,i-1] = Aw
    A[i,i] = Ap
    A[i,i+1] = Ae

#obtain the solution    
phi_uds = np.linalg.solve(A,b)

#repeat for CDS
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

#obtain the solution    
phi_cds = np.linalg.solve(A,b)

#plot results
pl.close('all')
fig=pl.figure(figsize=(6,4))
fig.set_tight_layout('true')
pl.plot(x_true,phi_true,linewidth = 5, 
        linestyle="solid",color="0.6",label="Theory")
pl.plot(x,phi_uds,linestyle=(0.5,(4, 1, 1, 1)),linewidth=4,label="UDS",color="0")
pl.plot(x,phi_cds,linestyle=(0, (1, 0.5)),linewidth=4,label="CDS",color="0.4")
#pl.title("ni=%d, D=%.1e"%(ni,D))
pl.tick_params(axis='both',which='major',labelsize=13)
pl.grid()
pl.legend(loc='upper left',fontsize=13)
pl.savefig('ad1-steady2.pdf')

        
    

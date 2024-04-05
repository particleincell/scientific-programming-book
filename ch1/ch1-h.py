"""
Updated version with random functions

Brieda, Wang, Martin, "Introduction to Modern Scientific Programming and Numerical Methods", CRC Press 2021
"""

import matplotlib.pylab as plt
import numpy as np
from random import random
print(random())


# global inputs 
ni = 200
dt = 0.01
gx = 0
gy = -9.81

#-------------------
# returns [x,y,vx,vy] given initial conditions
def integrate(x0,y0,vx0,vy0):          
    # allocate arrays
    t = np.zeros(ni)
    x = np.zeros(ni)
    y = np.zeros(ni)
    vx = np.zeros(ni)
    vy = np.zeros(ni)

    dt = 0.02
    t[0]  = 0
    x[0] = x0
    y[0] = y0
    vx[0] = vx0
    vy[0] = vy0

    for i in range(1,ni):    #i = [1,2,3,..,ni-1]
        t[i] = t[i-1] + dt
        x[i] = x[i-1] + vx[i-1]*dt
        y[i] = y[i-1] + vy[i-1]*dt
        vx[i] = vx[i-1] + gx*dt
        vy[i] = vy[i-1] + gy*dt
        if (y[i]<0):   # check for ground impact     
            y[i] = -y[i]  
            alpha = 0.5   
            vy[i] *= -alpha 
            vx[i] *= alpha  
            
    return x,y,vx,vy
#-------------------

plt.figure(figsize=(10,4))

# display 25 random serves
for s in range(25):
    x0 = 0 + (-1+2*random())*0.05
    y0 = 2.1 + (-1+2*random())*0.2
    vx0 = 45 + (-1+2*random())*15
    vy0 = 0 + (-1+2*random())*2
    x,y,vx,vy = integrate(x0,y0,vx0,vy0)
    plt.plot(x,y,LineWidth=1)

plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.savefig('trace.eps', format='eps')

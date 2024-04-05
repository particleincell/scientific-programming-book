"""
Updated version with NumPy

Brieda, Wang, Martin, "Introduction to Modern Scientific Programming and Numerical Methods", CRC Press 2021
"""

import matplotlib.pylab as plt
import numpy as np

# allocate arrays
ni = 100         # array size
t = np.zeros(ni)
x = np.zeros(ni)
y = np.zeros(ni)
vx = np.zeros(ni)
vy = np.zeros(ni)

#set initial values
t[0] = 0
x[0] = 0
y[0] = 2.1
vx[0] = 45
vy[0] = 0

#additional parameters
gx = 0
gy = -9.81
dt = 0.04

#initialize array index
i = 0
while (y[i]>0):    # repeat until ground impact
    i = i+1        # increment array index
    t[i] = t[i-1] + dt
    x[i] = x[i-1] + vx[i-1]*dt
    y[i] = y[i-1] + vy[i-1]*dt
    vx[i] = vx[i-1] + gx*dt
    vy[i] = vy[i-1] + gy*dt
    
print("The tennis ball hit the ground after about "
      "%.3f s at distance %.3f m"%(t[i],x[i]))

# plot x vs. y 
plt.plot(x[:i+1],y[:i+1])

#plt.savefig('trace.eps', format='eps')

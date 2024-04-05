"""
Updated version with bounce off

Brieda, Wang, Martin, "Introduction to Modern Scientific Programming and Numerical Methods", CRC Press 2021
"""

import matplotlib.pylab as plt
import numpy as np
    
# allocate arrays
ni = 200         # array size
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
dt = 0.01

for i in range(1,ni):    #i = [1,2,3,..,ni-1]
    t[i] = t[i-1] + dt
    x[i] = x[i-1] + vx[i-1]*dt
    y[i] = y[i-1] + vy[i-1]*dt
    vx[i] = vx[i-1] + gx*dt
    vy[i] = vy[i-1] + gy*dt
    if (y[i]<0):        
        y[i] = -y[i]  # flip back to above ground
        alpha = 0.5   # specify bounciness
        vy[i] *= -alpha # scale and flip y-velocity
        vx[i] *= alpha  # scale x-velocity
        
print("The tennis ball hit the ground after about "
      "%.3f s at distance %.3f m"%(t[i],x[i]))

# plot x vs. y 
plt.plot(x[:i+1],y[:i+1])

#plt.savefig('trace.eps', format='eps')

"""
Updated version that plots the trajectory

Brieda, Wang, Martin, "Introduction to Modern Scientific Programming and Numerical Methods", CRC Press 2021
"""

import matplotlib.pylab as plt

t = [0]        # initialize t
x = [0]        # initialize position
y = [2.1]
vx = [45]      # initialize velocity
vy = [0]
gx = 0         # initialize acceleration
gy = -9.81
dt = 0.04      # assign time step

while (y[-1]>0):    # repeat until ground impact
    t.append(t[-1]+dt)     # increment time
    x.append(x[-1] + vx[-1]*dt)  # increment position
    y.append(y[-1] + vy[-1]*dt)
    vx.append(vx[-1] + gx*dt) # increment velocity
    vy.append(vy[-1] + gy*dt) 

print("The tennis ball hit the ground after about "
      "%.3f s at distance %.3f m"%(t[-1],x[-1]))

# plot x vs. y using the specified styling
plt.plot(x,y,color='purple',linewidth=3,dashes=[6,2],
        marker='o',markersize=12, fillstyle='full',
        markerfacecolor='yellow',label='tennis ball')
plt.xlabel('x (m)')   # add x-axis label
plt.ylabel('y (m)')   # add y-axis label
plt.grid()            # show x-y grid
plt.legend(loc=3)     # show legend in bottom left corner

plt.savefig('trace.eps', format='eps')

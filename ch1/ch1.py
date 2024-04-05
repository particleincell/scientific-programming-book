"""
Initial version of example code to simulate a tennis ball

Brieda, Wang, Martin, "Introduction to Modern Scientific Programming and Numerical Methods", CRC Press 2021
"""
t = 0        # initialize t
x = 0        # initialize position
y = 2.1
vx = 45      # initialize velocity
vy = 0
gx = 0       # initialize acceleration
gy = -9.81
dt = 0.04    # assign time step

#show initial values
print("%.2f, %.2f, %.2f, %.2f, %.2f"%(t,x,y,vx,vy))

while (y>0):    # repeat until ground impact
    t += dt     # increment time
    x += vx*dt  # increment position
    y += vy*dt
    vx += gx*dt # increment velocity
    vy += gy*dt 
    # display current position and velocity
    print("%.2f, %.2f, %.2f, %.2f, %.2f"%(t,x,y,vx,vy))
    
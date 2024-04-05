"""
Updated version of example code to simulate a tennis ball that writes to a file

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

with open("results.csv","w") as f:
    print("%.2f, %.2f, %.2f, %.2f, %.2f"%(t,x,y,vx,vy),file=f)
    
    while (y>0):    # repeat until ground impact
        t += dt     # increment time
        x += vx*dt  # increment position
        y += vy*dt
        vx += gx*dt # increment velocity
        vy += gy*dt 
        print("%.2f, %.2f, %.2f, %.2f, %.2f"%(t,x,y,vx,vy),file=f)

print("The tennis ball hit the ground after about %.3f s "
      "at distance %.3f m"%(t,x))
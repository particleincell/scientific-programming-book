"""
Comparison with analytical prediction

Brieda, Wang, Martin, "Introduction to Modern Scientific Programming and Numerical Methods", CRC Press 2021
"""

import matplotlib.pylab as plt
import numpy as np
    
# allocate arrays
ni = 100         # array size
t = np.zeros(ni)
xf = np.zeros(ni)
yf = np.zeros(ni)
xb = np.zeros(ni)
yb = np.zeros(ni)
xm = np.zeros(ni)
ym = np.zeros(ni)
vx = np.zeros(ni)
vy = np.zeros(ni)

def test():
    print(abc)
    

abc = 5
b = 3
test()
            
    
#set initial values
t[0] = 0
xf[0] = 0
yf[0] = 2.1
xb[0] = xf[0]
yb[0] = yf[0]
xm[0] = xf[0]
ym[0] = yf[0]

vx[0] = 45
vy[0] = 0

#additional parameters
gx = 0
gy = -9.81
dt = 0.1

# integrate position
for i in range(ni-1):    
    t[i+1] = t[i] + dt
    vx[i+1] = vx[i] + gx*dt
    vy[i+1] = vy[i] + gy*dt

    #forward method    
    xf[i+1] = xf[i] + vx[i]*dt
    yf[i+1] = yf[i] + vy[i]*dt
    
    #backward method
    xb[i+1] = xb[i] + vx[i+1]*dt
    yb[i+1] = yb[i] + vy[i+1]*dt

    #midpoint method
    xm[i+1] = xm[i] + 0.5*(vx[i+1]+vx[i])*dt
    ym[i+1] = ym[i] + 0.5*(vy[i+1]+vy[i])*dt
    
    if (yf[i]<0):
        break     # exit loop if ground impact

plt.figure(figsize=(8,4))
plt.plot(xf[:i+1],yf[:i+1],'--',color='0.3',label='forward')
plt.plot(xb[:i+1],yb[:i+1],'-.',color='0.2',label='backward')
plt.plot(xm[:i+1],ym[:i+1],'-',color='0.0',label='midpoint')


#plot the analytical solution
t_th = np.linspace(0,0.654,25)
x_th = 45*t_th
y_th = 0.5*gy*t_th**2 + 2.1
plt.plot(x_th,y_th,':o', color='0.4',markersize=5,label='analytical')
plt.legend(loc=3)

plt.ylim([-0.1,2.2])
plt.xlabel('x (m)')
plt.ylabel('y (m)')

plt.savefig('trace.eps', format='eps')

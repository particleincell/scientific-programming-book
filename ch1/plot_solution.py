#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pylab as pl

a = -9.81
x0 = 0
v0 = 10

t = np.linspace(0,1,100)
x = 0.5*a*t*t+v0*t+x0
pl.plot(t,x)


#numerical integration
num_ts = 50
tn = np.linspace(0,1,num_ts)
xn = np.linspace(0,1,num_ts)
vn = np.linspace(0,1,num_ts)
xv = np.linspace(0,1,num_ts)
vv = np.linspace(0,1,num_ts)
xn[0] = x0
vn[0] = v0
xv[0] = xn[0]
vv[0] = vn[0]

dt = (tn[-1]-tn[0])/xn.size

vn[0] = v0

for i in range(xn.size-1):
    vn[i+1] = vn[i] + a*dt
    xn[i+1] = xn[i] + 0.5*(vn[i]+vn[i+1])*dt
    #xn[i+1] = xn[i] + vn[i]*dt
    
    
    at = a  #a(x[i])
    xv[i+1] = xv[i]+vv[i]*dt+0.5*at*dt*dt
    at2 = a  #this is a(x[i+1])
    vv[i+1] = vv[i] + 0.5*(at+at2)*dt
 #   xn[i+1] = xn[i] + vn[i]*dt
        
pl.plot(tn,xn,color='red',label='euler')
#pl.plot(tn,xv,color='green',label='vertlet')
pl.legend()




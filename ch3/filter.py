#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASTE-404 filtering example
Created on Mon Mar 15 16:28:36 2021

@author: lubos
"""

import numpy as np
import matplotlib.pyplot as plt

#create inputs
ns = 2000
t = np.linspace(0,5,ns)
y = np.exp(-0.5*t)+0.2*np.exp(-0.4*t)*(
        np.cos(2*np.pi*t*0.5)+0.1*np.sin(2*np.pi*t*10)+0.2*np.cos(2*np.pi*t*20))

#window filter
y2 = np.zeros_like(y)
ws = 50
for i in range(ws,len(y2)):
    i1 = i-(int)(0.5*ws)
    i2 = i+(int)(0.5*ws)
    y2[i] = np.average(y[i1:i2])
    

#RC filter
dt = t[1]-t[0]
RC = 0.05
alpha = dt / (RC + dt)
y3 = np.zeros_like(y)
y3[0] = alpha * y[0]
for i in range(1,ns):
    y3[i] = alpha*y[i] + (1-alpha)*y3[i-1]

fig,ax = plt.subplots()
plt.plot(t,y,'b-')
plt.plot(t[ws:],y2[ws:],'r',label='moving window')
plt.plot(t[:],y3[:],'g',label='RC low pass')
plt.xlabel('time (s)')
plt.legend()
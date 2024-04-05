#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt

t = np.linspace(0,6,2000)
nt = len(t)
R1 = -1+2*np.random.random(nt)
R2 = -1+2*np.random.random(nt)
R3 = -1+2*np.random.random(nt)
t_noise = t + 0.1*R1
y = 2+((1+0.2*(-1+2*np.cos(t/0.65)))*np.cos((1+0.001*R3)*t_noise/0.5*np.pi)+
     0.6*R2*np.sin(t_noise/0.04*np.pi))

plt.plot(t,y)

with open("signal.csv","w") as f:
    print("t,signal",file=f)
    for i in range(nt):
        print("%g, %g"%(t[i],y[i]),file=f)


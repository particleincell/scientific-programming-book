#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

"""
from mpl_toolkits.mplot3d import Axes3D
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt


#fun = lambda x: (x[0] - 1)**2 + (x[1] - 2.5)**2
fun = lambda x:(np.sin((x[0]+0.1)*np.pi)+(x[0]+0.2)**2-0.5*x[0]*x[1] + x[1]**2 + 1)

x,y = np.meshgrid(np.linspace(-2, 2, 40),
                  np.linspace(-2, 2, 40))
z = fun([x,y])

# show function
plt.close('all')
fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(x,y,z)

ax = fig.add_subplot(111)
ax.contour(x, y, z,cmap='gray',levels=20)
ax.tick_params(axis='both', labelsize=12)
plt.show()
plt.grid(color='gray',linestyle='--')
plt.tight_layout()

# find minimum
res = minimize(fun, (-1.3, 0), method='CG')
print ("%.3g, %.3g -> %.3g"%(res.x[0],res.x[1],res.fun))

import numpy as np
import matplotlib.pylab as pl

# function that evaluates max(x^2-2x,0)
def fun(x):
    z = x*x - 2*x
    if (z>=0):
        return z
    else:
        return 0
    
y = [0]*11  # make am array of 11 zeroes

# evaluate fun(x) for x=[-1,1]
for i in range(11):
    x = -1 + 2*i/10
    y[i] = fun(x)

# show some results    
print("f(-1)=%g, f(1)=%g"%(y[0],y[10]))


# numpy version
x = np.linspace(-1,1,11)
y = np.maximum(x*x - 2*x,0)
print("f(-1)=%g, f(1)=%g"%(y[0],y[10]))


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ASTE-404 Eigenvalue demo
@author: lubos
"""

import numpy as np

A = np.array([[0,2,0],[2,0,0],[1,0,1]])
[lam,vec] = np.linalg.eig(A)

for i in range(3):
    print("lambda=%g, vec="%(lam[i]),end='')
    print(vec[i])
    v = vec[:,i]
    print(np.dot(A,v))
    print(lam[i]*v)
    print("****")
        
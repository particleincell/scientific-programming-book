#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 17:05:32 2021

@author: lubos
"""

import numpy as np

A = np.array([[1,2,0],[0,2,1],[1,0,1]])
x = np.array([0,1,2])
b = np.dot(A,x)  # matrix-vector multiplication
x_solved = np.linalg.solve(A,b)
print(x,x_solved)
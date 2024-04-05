#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 17:29:53 2023

@author: lubos
"""

rem = 0

dN = 12.8
N_sum = 0

for k in range(100):
    N = int(dN+rem)
    N_sum += N
    rem += dN-N
    
    
print(N_sum/(k+1))
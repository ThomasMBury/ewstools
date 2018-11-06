#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 21:21:29 2018

@author: tb460

script to test functions
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------
# Test smooth_function
#-------------

from ews_std import smooth_function

# create a noisy trajectory
t=np.linspace(1,10,100)
x=0.5*5
xn=x+np.random.randn(len(t))*0.5

# put into smooth_function 
y=smooth_function(xn,band_width=0.2)

# make and display a plot
plt.plot(t,y,t,xn)
plt.ylabel('x')
plt.xlabel('t')
plt.title('smoothed function')


#-----------------
# Test variance function
#-----------------



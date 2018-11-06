#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 21:21:29 2018

@author: tb460

script to test functions in ews_std
"""

# standard libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# EWS functions
from ews_std import smooth_function
from ews_std import roll_ews_std



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
plt.title('Smoothed function')


#-----------------
# Test roll_ews_std
#-----------------

# put residuals into a series
series = pd.Series(xn-y,index = t)

# apply roll_ews_std
lags = [1,2]
rw = 0.2

df_ews = roll_ews_std(series, roll_window=rw, lag_times=lags, ews=['var'])






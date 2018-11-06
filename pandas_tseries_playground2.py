#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 23:29:41 2018

@author: tb460
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# create a noisy trajectory
t = np.linspace(0,10,110)
x = np.random.randn(len(t))*0.5

# create a Series for the state variable indexed by time
x_series = pd.Series(data=x, index=t, name='State variable')
x_series.index.rename('Time',inplace=True)

# parameters
roll_window = 20
lag_time = 1

# compute rolling statistics as series
roll_var = x_series.rolling(window=roll_window).var()
roll_var.name = 'Variance'

roll_ac = x_series.rolling(window=roll_window).apply(
        func=lambda x: pd.Series(x).autocorr(lag=lag_time))
roll_ac.name = 'Lag-1 autocorrelation'

roll_std = x_series.rolling(window=roll_window).std()
roll_std.name = 'Standard deviation'

roll_skew = x_series.rolling(window=roll_window).skew()
roll_skew.name = 'Skewness'


# put time-series and EWS into a dataframe
df=pd.concat([x_series,roll_var,roll_ac,roll_std,roll_skew], axis=1)

# cheeky plot for verification
plt.figure();
df.plot();
plt.legend(loc='best')





#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 21:21:29 2018

@author: tb460

script to test the function ews_compute

"""

# import standard libraries
import numpy as np
import pandas as pd
import time


#  import EWS functions
from ews_compute import ews_compute



# create a noisy trajectory
t=np.linspace(1,400,401)
x=0.5*5
xn=x+np.random.randn(len(t))*0.5

# put in the form of a pandas.Series
series = pd.Series(xn, index=t)


# run ews_std

# begin a timer
start = time.time() 

df_ews = ews_compute(series, 
                      roll_window=1, 
                      lag_times=[1,2,3],
                      ham_length=40,
                      ews=['var','ac','smax','cf'])

# end timer
end = time.time()

# Note: df_ews is indexed by t and has (titled) columns csp to each EWS

# a crazy plot of every metric
#df_ews.plot()

# plot some standard metrics
df_ews[['State variable','Smoothing','Variance', 'Lag-1 AC']].plot()

# plot some spectral metrics
df_ews[['Coherence factor','Smax']].dropna().plot()

## plot AIC weights
#df_ews[['AIC fold','AIC hopf', 'AIC null']].dropna().plot()

# print time taken to run ews_std
print('ews_compute took ',end-start,' seconds to run')



## Compute kendall tau values
        
# Put time values as their own series for correlation computation
time_series = pd.Series(series.index, index=series.index)
    
# Find kendall tau correlation coefficient for each EWS (column of df_ews)
ktau = df_ews.corrwith(time_series)



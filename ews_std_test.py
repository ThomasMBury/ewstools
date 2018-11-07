#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 21:21:29 2018

@author: tb460

script to test functions in ews_std
"""

# import standard libraries
import numpy as np
import pandas as pd


#  import EWS functions
from ews_std import ews_std



# create a noisy trajectory
t=np.linspace(1,10,100)
x=0.5*5
xn=x+np.random.randn(len(t))*0.5

# put in the form of a pandas.Series
series = pd.Series(xn, index=t)


# run ews_std
df_ews, ktau = ews_std(series, 
                      roll_window=0.2, 
                      lag_times=[1,2,3], 
                      ews=['var','ac','sd','skew','kurt','cv'])


# Note: df_ews is indexed by t and has (titled) columns csp to each EWS

# a crazy plot of every metric
df_ews.plot()

# choose metrics to plot
df_ews[['State variable','Smoothing','Variance', 'Lag-1 AC']].plot()


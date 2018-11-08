#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:52:31 2018

@author: Thomas Bury

script to test functions in ews_spec
"""

# import standard libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# import EWS functions
from ews_spec import pspec_welch


# create a noisy trajectory
t=np.linspace(1,10,100)
x=0.5*5
xn=x+np.random.randn(len(t))*0.5

# put in the form of a pandas.Series
series = pd.Series(xn, index=t)

# compute the power spectrum of the series
pspec=pspec_welch(series,40)

# make a plot
pspec.plot()





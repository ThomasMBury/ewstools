#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 11:52:31 2018

@author: Thomas Bury

script to test functions in ews_spec
"""

# import standard libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time


# import EWS functions
from ews_spec import pspec_welch, pspec_metrics


# create a noisy trajectory
t=np.linspace(0,100,1000)
x=0.5*5
xn=x+np.random.randn(len(t))*0.5

# put in the form of a pandas.Series
series = pd.Series(xn, index=t)

#-------------------------------
## Test pspec_welch
#--------------------------------



# compute the power spectrum of the series
yVals = series.values
dt = series.index[1]-series.index[2]
pspec=pspec_welch(yVals, dt, ham_length=40, w_cutoff=1)


# make a plot
pspec.plot()



#---------------------------
## Test spec_metrics
#---------------------------


# begin a timer
start = time.time()

# put the power spectrum into pspec_metrics
spec_ews = pspec_metrics(pspec, ews=['smax', 'cf', 'aic', 'aic_params'])


# end the timer
end = time.time()

# make a plot of fitted models

# define models to fit
def fit_fold(w,sigma,lam):
    return (sigma**2 / (2*np.pi))*(1/(w**2+lam**2))
        
def fit_hopf(w,sigma,mu,w0):
    return (sigma**2/(4*np.pi))*(1/((w+w0)**2+mu**2)+1/((w-w0)**2 +mu**2))
        
def fit_null(w,sigma):
    return sigma**2/(2*np.pi)* w**0

# plot with parameter values
w_vals = np.linspace(-max(pspec.index),max(pspec.index),100)

plt.plot(w_vals, fit_fold(w_vals, spec_ews['Params fold']['sigma'], spec_ews['Params fold']['lam']))
plt.plot(w_vals, fit_hopf(w_vals, spec_ews['Params hopf']['sigma'], spec_ews['Params hopf']['mu'], spec_ews['Params hopf']['w0']))
plt.plot(w_vals, fit_null(w_vals, spec_ews['Params null']['sigma']))




print(spec_ews)

# print time elapsed for functions to run
print('spec_metrics took ', end - start,' seconds')









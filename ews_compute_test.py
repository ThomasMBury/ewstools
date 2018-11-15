#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 21:21:29 2018

@author: tb460

Script to execute the function ews_compute on a stochastic simulation
of May's harvesting model with additive white noise.

"""


# import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as time

# import EWS function
from ews_compute import ews_compute
from ews_spec import pspec_welch, pspec_metrics


#--------------------
# Stochastic simulation of May's harvesting model
#----------------------


# Simulation parameters
dt = 1
t0 = 0
tmax = 1000
seed = 1 # random number generation seed

# Model: dx/dt = de_fun(x,t) + sigma dW(t)
def de_fun(x,r,k,h,s):
    return r*x*(1-x/k)  - h*(x**2/(s**2 + x**2))
    
    
# Model parameters
sigma = 0.02 # noise intensity
r = 1 # growth rate
k = 1 # carrying capacity
s = 0.1 # half-saturation constant of harvesting function
hl = 0.15 # initial harvesting rate
hh = 0.28 # final harvesting rate
hbif = 0.260437 # bifurcation point (computed in Mathematica)
x0 = 0.8197 # intial condition (equilibrium value computed in Mathematica)


# Initialise arrays to store data
t = np.arange(t0,tmax,dt)
x = np.zeros(len(t))
x[0] = x0


# Set up control parameter h, that increases linearly in time from hl to hh
h = pd.Series(np.linspace(hl,hh,len(t)),index=t)
# Time at which bifurcation occurs
tbif = h[h > hbif].index[1]


## Implement Euler Maryuyama for stocahstic simulation

# Set seed
np.random.seed(seed)
# Create brownian increments (s.d. sqrt(dt))
dW = np.random.normal(loc=0, scale=np.sqrt(dt), size = len(t))

# Loop over time
for i in range(len(t)-1):
    x[i+1] = x[i] + de_fun(x[i],r,k,h[i],s)*dt + sigma*dW[i]
    # make sure that state variable remains >= 0 
    if x[i+1] < 0:
        x[i+1] = 0
        
# Store data as a series indexed by time
series = pd.Series(x, index=t)



#--------------------------------------
## Compute EWS using ews_compute
#------------------------------------

start = time.time()  # begin a timer

df_ews = ews_compute(series,
                     band_width=0.1,
                     upto=tbif*1,
                     roll_window=0.25, 
                     lag_times=[1],
                     ham_length=40,
                     ews=['var','ac','smax','aic'],
                     updates=True)

end = time.time() # end timer
# Print time taken to run ews_std
print('ews_compute took ',end-start,' seconds to run')



# Make plot of EWS
fig, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(6,6))
df_ews[['State variable','Smoothing']].plot(ax=axes[0])
df_ews['Variance'].plot(ax=axes[1])
df_ews['Lag-1 AC'].plot(ax=axes[1], secondary_y=True)
df_ews['Smax'].dropna().plot(ax=axes[2])
df_ews[['AIC fold','AIC hopf','AIC null']].dropna().plot(ax=axes[3])



#---------------------------------
## Compute kendall tau values of EWS
#-------------------------------------

# Put time values as their own series for correlation computation
time_series = pd.Series(series.index, index=series.index)
    
# Find kendall tau correlation coefficient for each EWS (column of df_ews)
ktau = df_ews.corrwith(time_series)

# Print kendall tau values
print(ktau[['Variance','Lag-1 AC','Smax']])








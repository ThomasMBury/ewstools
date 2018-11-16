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

# import EWS functions
from ews_compute import ews_compute
from ews_spec import pspec_welch, pspec_metrics


#--------------------
# Stochastic simulation of May's harvesting model
#----------------------


# Simulation parameters
dt = 1
t0 = 0
tmax = 800
tburn = 50 # burn-in period
seed = 3 # random number generation seed

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


# Set up control parameter h, that increases linearly in time from hl to hh
h = pd.Series(np.linspace(hl,hh,len(t)),index=t)
# Time at which bifurcation occurs
tbif = h[h > hbif].index[1]


## Implement Euler Maryuyama for stocahstic simulation

# Set seed
np.random.seed(seed)
# Create brownian increments (s.d. sqrt(dt))
dW_burn = np.random.normal(loc=0, scale=np.sqrt(dt), size = int(tburn/dt))
dW = np.random.normal(loc=0, scale=np.sqrt(dt), size = len(t))

# Simulate burn-in period to obtain x0 (h is fixed)
for j in range(int(tburn/dt)):
    x0 = x0 + de_fun(x0,r,k,h[0],s)*dt + sigma*dW_burn[j]
    
# Initial condition post burn-in period
x[0]=x0

# Run simulation
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

# EWS parameters
rw = 0.25 # rolling window
bw = 0.1 # band width for Gaussian smoothing
ham_len = 40 # length of Hamming window for spectrum computation


start = time.time()  # begin a timer

# execute function ews_compute
df_ews = ews_compute(series,
                     band_width=bw,
                     upto=tbif*1,
                     roll_window=rw, 
                     lag_times=[1],
                     ham_length=ham_len,
                     ews=['var','ac','smax','aic'])

end = time.time() # end timer
# Print time taken to run ews_std
print('ews_compute took ',end-start,' seconds to run\n')

# Note : df_ews provides a dataframe indexed by time with each column csp. to time-series (state, residuals, EWS)

# Make plot of EWS
fig1, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(6,6))
df_ews[['State variable','Smoothing']].plot(ax=axes[0],title='Early warning signals')
df_ews['Variance'].plot(ax=axes[1],legend=True)
df_ews['Lag-1 AC'].plot(ax=axes[1], secondary_y=True,legend=True)
df_ews['Smax'].dropna().plot(ax=axes[2],legend=True)
df_ews[['AIC fold','AIC hopf','AIC null']].dropna().plot(ax=axes[3],legend=True)





#---------------------------------
## Compute kendall tau values of EWS
#-------------------------------------

# Put time values as their own series for correlation computation
time_series = pd.Series(series.index, index=series.index)
    
# Find kendall tau correlation coefficient for each EWS
ktau = pd.DataFrame([df_ews[x].corr(time_series,method='kendall') for x in df_ews.columns],index=df_ews.columns)


# Print kendall tau values
print('Kendall tau values for each metric are as follows are:\n',ktau.loc[['Variance','Lag-1 AC','Smax']])




#-------------------------------------
# Display power spectrum and fits at a given instant in time
#------------------------------------

t_pspec = 300

# Use function pspec_welch to compute the power spectrum of the residuals at a particular time
pspec=pspec_welch(df_ews.loc[t_pspec-rw*len(t):t_pspec,'Residuals'], dt, ham_length=ham_len, w_cutoff=1)

# Execute the function pspec_metrics to compute the AIC weights and fitting parameters
spec_ews = pspec_metrics(pspec, ews=['smax', 'cf', 'aic', 'aic_params'])
# Define the power spectrum models
def fit_fold(w,sigma,lam):
    return (sigma**2 / (2*np.pi))*(1/(w**2+lam**2))
        
def fit_hopf(w,sigma,mu,w0):
    return (sigma**2/(4*np.pi))*(1/((w+w0)**2+mu**2)+1/((w-w0)**2 +mu**2))
        
def fit_null(w,sigma):
    return sigma**2/(2*np.pi)* w**0


# Make plot
w_vals = np.linspace(-max(pspec.index),max(pspec.index),100)

fig2=plt.figure(2)
pspec.plot(label='Measured')
plt.plot(w_vals, fit_fold(w_vals, spec_ews['Params fold']['sigma'], spec_ews['Params fold']['lam']),label='Fold fit')
plt.plot(w_vals, fit_hopf(w_vals, spec_ews['Params hopf']['sigma'], spec_ews['Params hopf']['mu'], spec_ews['Params hopf']['w0']),label='Hopf fit')
plt.plot(w_vals, fit_null(w_vals, spec_ews['Params null']['sigma']),label='Null fit')
plt.ylabel('Power')
plt.legend()
plt.title('Power spectrum and fits at time t='+str(t_pspec))










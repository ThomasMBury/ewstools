#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 21:21:29 2018

@author: tb460

Script to execute the function ews_compute on a stochastic trajectory of
May's harvesting model and visualise the EWS.

"""


# import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time as time

# import EWS functions
from ews_compute import ews_compute


#---------------------
# Global parameters
#–--------------------

# Simulation parameters
dt = 0.01 # time step for simulation
t0 = 0
tmax = 800
tburn = 50 # burn-in period
seed = 3 # random number generation seed
sigma = 0.02 # noise intensity

# EWS parameters
dt2 = 1 # resolution of time-series for EWS computation
rw = 0.4 # rolling window
bw = 0.05 # band width for Gaussian smoothing
ham_len = 40 # length of Hamming window for spectrum computation
pspec_roll_offset = 20 # offset of rolling window when computing power spectrum



#--------------------
# Stochastic simulation of May's harvesting model
#----------------------

# Model: dx/dt = de_fun(x,t) + sigma dW(t)
def de_fun(x,r,k,h,s):
    return r*x*(1-x/k)  - h*(x**2/(s**2 + x**2))
    
    
# Model parameters

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
    x[i+1] = x[i] + de_fun(x[i],r,k,h[t[i]],s)*dt + sigma*dW[i]
    # make sure that state variable remains >= 0 
    if x[i+1] < 0:
        x[i+1] = 0
        
# Store data as a series indexed by time
series = pd.Series(x, index=t)



#--------------------------------------
## Compute EWS using ews_compute
#------------------------------------

# Filter time-series to have spacing dt2
series = series.loc[::int(dt2/dt)]

# Begin a timer
start = time.time()  

# Execute function ews_compute to obtain dictionary of EWS metrics and power spectra
ews_dic = ews_compute(series,
                     band_width=bw,
                     upto=tbif*1,
                     roll_window=rw, 
                     lag_times=[1],
                     ham_length=ham_len,
                     pspec_roll_offset=pspec_roll_offset,
                     ews=['var','ac','smax','aic'])

# The DataFrame of EWS
df_ews = ews_dic['EWS metrics']
# The DataFrame of power spectra
df_pspec = ews_dic['Power spectrum']
# The Series of Kendall tau values
ktau = ews_dic['Kendall tau']

end = time.time() # end timer
# Print time taken to run ews_std
print('\n The function ews_compute took ',end-start,' seconds to run\n')


#-----------------------------------
# Plots of EWS and power spectra
#–---------------------------------


## Grid plot of EWS
fig1, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(6,6))
df_ews[['State variable','Smoothing']].plot(ax=axes[0],title='Early warning signals')
df_ews['Variance'].plot(ax=axes[1],legend=True)
df_ews['Lag-1 AC'].plot(ax=axes[1], secondary_y=True,legend=True)
df_ews['Smax'].dropna().plot(ax=axes[2],legend=True)
df_ews[['AIC fold','AIC hopf','AIC null']].dropna().plot(ax=axes[3],legend=True)


## Grid plot of power spectra and fits
g = sns.FacetGrid(df_pspec.reset_index(), 
                  col='Time',
                  col_wrap=3,
                  sharey=False,
                  aspect=1.5,
                  size=1.8
                  )

g.map(plt.plot, 'Frequency', 'Empirical', color='k', linewidth=2)
g.map(plt.plot, 'Frequency', 'Fit fold', color='b', linestyle='dashed', linewidth=1)
g.map(plt.plot, 'Frequency', 'Fit hopf', color='r', linestyle='dashed', linewidth=1)
g.map(plt.plot, 'Frequency', 'Fit null', color='g', linestyle='dashed', linewidth=1)

# Axes properties
axes = g.axes
# Set y labels
for ax in axes[::3]:
    ax.set_ylabel('Power')
# Set y limit as max power over all time
for ax in axes:
    ax.set_ylim(top=1.05*max(df_pspec['Empirical']), bottom=0)
    ax.set_xlim(left=-np.pi, right=np.pi)



# Print kendall tau values
print('\nKendall tau values are as follows:\n', ktau)




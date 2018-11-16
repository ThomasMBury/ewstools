#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 21:21:29 2018

@author: tb460

Script demonstrating a way to use ews_compute to find EWS over multiple trajectories
and obtain a DataFrame that is indexed by both realisation number and time.

"""


# import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time as time

# import EWS functions
from ews_compute import ews_compute



#----------------------------------
# Simulate many (transient) realisations of May's harvesting model
#----------------------------------


# Simulation parameters
dt = 1
t0 = 0
tmax = 1000
tburn = 100 # burn-in period
numSims = 10
seed = 5 # random number generation seed

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




# initialise DataFrame to store all realisations
df_sims = pd.DataFrame([])

# Initialise arrays to store single time-series data
t = np.arange(t0,tmax,dt)
x = np.zeros(len(t))

# Set up control parameter h, that increases linearly in time from hl to hh
h = pd.Series(np.linspace(hl,hh,len(t)),index=t)
# Time at which bifurcation occurs
tbif = h[h > hbif].index[1]

## Implement Euler Maryuyama for stocahstic simulation


# Set seed
np.random.seed(seed)


# loop over simulations
for j in range(numSims):
    
    
    # Create brownian increments (s.d. sqrt(dt))
    dW_burn = np.random.normal(loc=0, scale=np.sqrt(dt), size = int(tburn/dt))
    dW = np.random.normal(loc=0, scale=np.sqrt(dt), size = len(t))
    
    # Run burn-in period on x0
    for i in range(int(tburn/dt)):
        x0 = x0 + de_fun(x0,r,k,h[0],s)*dt + sigma*dW_burn[i]
        
    # Initial condition post burn-in period
    x[0]=x0
    
    # Run simulation
    for i in range(len(t)-1):
        x[i+1] = x[i] + de_fun(x[i],r,k,h[i],s)*dt + sigma*dW[i]
        # make sure that state variable remains >= 0 
        if x[i+1] < 0:
            x[i+1] = 0
            
    # Store data as a Series indexed by time
    series = pd.Series(x, index=t)
    # add Series to DataFrame of realisations
    df_sims['Sim '+str(j+1)] = series







#----------------------
## Execute ews_compute for each realisation
#---------------------


# set up a list to store output dataframes from ews_compute- we will concatenate them at the end
appended_ews = []

# loop through each trajectory as an input to ews_compute
for i in range(numSims):
    df_temp = ews_compute(df_sims['Sim '+str(i+1)], 
                      roll_window=0.25, 
                      band_width=0.1,
                      lag_times=[1], 
                      ews=['var','ac','smax','aic'],
                      ham_length=20,                     
                      upto=tbif)
    # include a column in the dataframe for realisation number
    df_temp['Realisation number'] = pd.Series((i+1)*np.ones([len(t)],dtype=int),index=t)
    
    # add DataFrame to list
    appended_ews.append(df_temp)
    
    # print status every 10 realisations
    if np.remainder(i+1,2)==0:
        print('Realisation '+str(i+1)+' complete')


# concatenate EWS DataFrames - use realisation number and time as indices
df_ews = pd.concat(appended_ews).set_index('Realisation number',append=True).reorder_levels([1,0])



#------------------------
# Some plots
#-----------------------

# plot of all variance trajectories
df_ews.loc[:,'Variance'].unstack(level=0).plot(legend=False, title='Variance') # unstack puts index back as a column

# plot of all autocorrelation trajectories
df_ews.loc[:,'Lag-1 AC'].unstack(level=0).plot(legend=False, title='Lag-1 AC') 

# plot of all smax trajectories
df_ews.loc[:,'Smax'].unstack(level=0).dropna().plot(legend=False, title='Smax') # drop Nan values

# plot of all Fold AIC trajectories
df_ews.loc[:,'AIC fold'].unstack(level=0).dropna().plot(legend=False, title='AIC fold') # drop Nan values




#---------------------------
## Compute distribution of kendall tau values
#----------------------------

# make the time values their own series and use pd.corrwith






#-----------------------------------------------
## Examples of obtaining specific data / plots
#------------------------------------------------
#
#
## EWS of realisation 1 at time 2
#df_ews.loc[(1,2)] # must include () when referencing multiple indexes
#
## Varaince of realisation 7
#df_ews.loc[7,'Variance']
#
## plot of all variance trajectories
#df_ews.loc[:,'Variance'].unstack(level=0).plot() # unstack puts index back as a column
#
## plot of autocorrelation and variance for a single realisation
#df_ews.loc[3,['Variance','Lag-1 AC']].plot()




#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 20 16:41:47 2018

@author: Thomas Bury

Script to bootstrap time-series of the Ricker model and compute EWS

"""

# Import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import bootstrap module
from roll_bootstrap import roll_bootstrap

# Import EWS module
import sys
sys.path.append('../../early_warnings')
from ews_compute import ews_compute





#--------------------------------
# Global parameters
#–-----------------------------


# Simulation parameters
dt = 1 # time-step (must be 1 since discrete-time system)
t0 = 0
tmax = 200
tburn = 100 # burn-in period
numSims = 1
seed = 0 # random number generation seedaa
sigma = 0.05 # noise intensity



# Bootstrapping parameters
block_size = 10 # size of blocks used to resample time-series
bs_type = 'Stationary' # type of bootstrapping
n_samples = 4 # number of bootstrapping samples to take
roll_offset = 5 # rolling window offset


# EWS parameters
dt2 = 1 # spacing between time-series for EWS computation
rw = 0.5 # rolling window
span = 0.1 # proportion of data for Loess filtering

lags = [1,2,3] # autocorrelation lag times
ews = ['var','ac','sd','cv','skew','kurt','smax','aic','cf'] # EWS to compute
ham_length = 40 # number of data points in Hamming window
ham_offset = 0.5 # proportion of Hamming window to offset by upon each iteration
pspec_roll_offset = 20 # offset for rolling window when doing spectrum metrics




#----------------------------------
# Simulate transient realisation of Ricker model
#----------------------------------

# Model
    
# Model parameters
r = 0.75 # growth rate
k = 10 # carrying capacity
h = 0.75 # half-saturation constant of harvesting function
bl = 0 # bifurcation parameter (harvesting) low
bh = 3 # bifurcation parameter (harvesting) high
bcrit = 2.364 # bifurcation point (computed in Mathematica)
x0 = 0.8 # initial condition

def de_fun(x,r,k,f,h,xi):
    return x*np.exp(r*(1-x/k)+xi) - f*x**2/(x**2+h**2)




# Initialise arrays to store single time-series data
t = np.arange(t0,tmax,dt)
x = np.zeros(len(t))

# Set bifurcation parameter b, that increases linearly in time from bl to bh
b = pd.Series(np.linspace(bl,bh,len(t)),index=t)
# Time at which bifurcation occurs
tcrit = b[b > bcrit].index[1]

## Implement Euler Maryuyama for stocahstic simulation

# Set seed
np.random.seed(seed)

# Initialise a list to collect trajectories
list_traj_append = []

# loop over simulations
print('\nBegin simulations \n')
for j in range(numSims):
    
    
    # Create brownian increments (s.d. sqrt(dt))
    dW_burn = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = int(tburn/dt))
    dW = np.random.normal(loc=0, scale=sigma*np.sqrt(dt), size = len(t))
    
    # Run burn-in period on x0
    for i in range(int(tburn/dt)):
        x0 = de_fun(x0,r,k,bl,h,dW_burn[i])
        
    # Initial condition post burn-in period
    x[0]=x0
    
    # Run simulation
    for i in range(len(t)-1):
        x[i+1] = de_fun(x[i],r,k,b.iloc[i], h,dW[i])
        # make sure that state variable remains >= 0
        if x[i+1] < 0:
            x[i+1] = 0
            
    # Store series data in a temporary DataFrame
    data = {'Realisation number': (j+1)*np.ones(len(t)),
                'Time': t,
                'x': x}
    df_temp = pd.DataFrame(data)
    # Append to list
    list_traj_append.append(df_temp)
    
    print('Simulation '+str(j+1)+' complete')

#  Concatenate DataFrame from each realisation
df_traj = pd.concat(list_traj_append)
df_traj.set_index(['Realisation number','Time'], inplace=True)






#------------------------------
# Bootstrap time-series
#-------------------------------

df_samples = roll_bootstrap(df_traj.loc[1]['x'],
                   span = span,
                   roll_window = rw,
                   roll_offset = roll_offset,
                   upto = tcrit,
                   n_samples = n_samples,
                   bs_type = bs_type,
                   block_size = block_size)






#----------------------
## Execute ews_compute for each bootstrapped time-series
#---------------------



# Filter time-series to have time-spacing dt2
df_traj_filt = df_traj.loc[::int(dt2/dt)]

# List to store EWS DataFrames
list_df_ews = []

# Print update
print('\nBegin EWS computation\n')

# Realtime values
tVals = np.array(df_samples.index.levels[0])
# Sample values
sampleVals = np.array(df_samples.index.levels[1])




# Loop through realtimes
for t in tVals:
    
    # Loop through sample values
    for sample in sampleVals:
        
        # Compute EWS for near-stationary sample series
        series_temp = df_samples.loc[t].loc[sample]['x']
        
        ews_dic = ews_compute(series_temp,
                          roll_window = 1, 
                          band_width = 1,
                          lag_times = lags, 
                          ews = ews,
                          ham_length = ham_length,
                          ham_offset = ham_offset,
                          pspec_roll_offset = pspec_roll_offset,
                          upto=tcrit,
                          sweep=False)
        
        # The DataFrame of EWS
        df_ews_temp = ews_dic['EWS metrics']
        
        # Include columns for sample value and realtime
        df_ews_temp['Sample'] = sample
        df_ews_temp['Time'] = t

        # Drop NaN values
        df_ews_temp = df_ews_temp.dropna()        
        
        # Append list_df_ews
        list_df_ews.append(df_ews_temp)
    
    # Print update
    print('EWS for t=%.2f complete' % t)
        
# Concatenate EWS DataFrames. Index [Realtime, Sample]
df_ews = pd.concat(list_df_ews).reset_index(drop=True).set_index(['Time','Sample'])



#--------------------------------------
# Compute summary statistics of EWS
#--------------------------------------




## Plot of AIC weights

# Put DataFrame in form for Seaborn plot
data = df_ews.reset_index().melt(id_vars = 'Time',
                         value_vars = ('AIC fold','AIC hopf', 'AIC null'),
                         var_name = 'EWS',
                         value_name = 'Magnitude')
# Make plot with error bars
aic_plot = sns.relplot(x="Time", 
            y="Magnitude",
            hue="EWS", 
            kind="line", 
            data=data)
# To use median, use estimator=np.median and vary CI with e.g. ci=0.95



## Plot of Variance and Smax
# Put DataFrame in form for Seaborn plot
data = df_ews.reset_index().melt(id_vars = 'Time',
                         value_vars = ('Variance', 'Smax'),
                         var_name = 'EWS',
                         value_name = 'Magnitude')
# Make plot with error bars
smax_plot = sns.relplot(x="Time", 
            y="Magnitude",
            hue="EWS", 
            kind="line", 
            data=data)



## Plot of Autocorrelation at various lags
# Put DataFrame in form for Seaborn plot
data = df_ews.reset_index().melt(id_vars = 'Time',
                         value_vars = ('Lag-1 AC','Lag-2 AC','Lag-3 AC'),
                         var_name = 'EWS',
                         value_name = 'Magnitude')
# Make plot with error bars
smax_plot = sns.relplot(x="Time", 
            y="Magnitude",
            hue="EWS", 
            kind="line", 
            data=data)







# Quantiles to compute
quantiles = [0.05,0.25,0.5,0.75,0.95]

# DataFrame of quantiles for each EWS
df_quant = df_ews.groupby(level=0).quantile([0.25,0.5,0.75], axis=0)
# Rename and reorder index of DataFrame
df_quant.index.rename(['Time','Quantile'], inplace=True)
df_quant = df_quant.reorder_levels(['Quantile','Time']).sort_index()








#
#
#
##-------------------------
## Visualise EWS and confidence intervals
##-------------------------
#
## Realisation number to plot
#plot_num = 1
#var = 'x'
### Plot of trajectory, smoothing and EWS of var (x or y)
#fig1, axes = plt.subplots(nrows=4, ncols=1, sharex=True, figsize=(6,6))
#df_ews.loc[plot_num,var][['State variable','Smoothing']].plot(ax=axes[0],
#          title='Early warning signals for a single realisation')
#df_ews.loc[plot_num,var]['Variance'].plot(ax=axes[1],legend=True)
#df_ews.loc[plot_num,var][['Lag-1 AC','Lag-2 AC','Lag-3 AC']].plot(ax=axes[1], secondary_y=True,legend=True)
#df_ews.loc[plot_num,var]['Smax'].dropna().plot(ax=axes[2],legend=True)
#df_ews.loc[plot_num,var]['Coherence factor'].dropna().plot(ax=axes[2], secondary_y=True, legend=True)
#df_ews.loc[plot_num,var][['AIC fold','AIC hopf','AIC null']].dropna().plot(ax=axes[3],legend=True)
#
#










#
#
##------------------------------------
### Export data / figures
##-----------------------------------
#
## Export power spectrum evolution (grid plot)
#plot_pspec.savefig('figures/pspec_evol.png', dpi=200)
#
### Export the first 5 realisations to see individual behaviour
## EWS DataFrame (includes trajectories)
#df_ews.loc[:5].to_csv('data_export/'+dir_name+'/ews_singles.csv')
## Power spectrum DataFrame (only empirical values)
#df_pspec.loc[:5,'Empirical'].dropna().to_csv('data_export/'+dir_name+'/pspecs.csv',
#            header=True)
#
### Export ensemble statistics
##df_ews_means.to_csv('data_export/'+dir_name+'/ews_ensemble_mean.csv')
##df_ews_deviations.to_csv('data_export/'+dir_name+'/ews_ensemble_std.csv')
#








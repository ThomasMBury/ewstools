#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 21:21:29 2018

@author: tb460

Script demonstrating how to use ews_std to compute EWS for multiple 
trajectories and obtain a dataframe indexed by realisation number and time.


"""

# import standard libraries
import numpy as np
import pandas as pd


#  import EWS functions
from ews_std import ews_std



#-----------------------
## Import trajectories (here we just create some) 
#----------------------


# create many noisy trajectories in an array (len(t) x num_sims )
num_sims = 100
t = np.linspace(1,10,100)
xn = np.zeros([len(t),num_sims]) # initialize
for i in range(num_sims):
    xn[:,i]=1+np.random.randn(len(t))*0.5

# put into a DataFrame
df_traj = pd.DataFrame(xn, index=t)
df_traj.index.rename('Time',inplace=True)



#----------------------
## Feed trajectories into ews_std
#---------------------


# set up lists to store output data of ews_std
appended_ews = []
appended_ktau = []

# loop through each trajectory as an input to ews_std
for i in range(num_sims):
    df_temp,ktau_temp = ews_std(df_traj[i], 
                      roll_window=0.2, 
                      lag_times=[1], 
                      ews=['var','ac'])
    # add realisation number as a column for reference
    df_temp['Realisation number'] = pd.Series(int(i)*np.ones([len(t)],dtype=int),index=t)
    ktau_temp['Realisation number'] = int(i)
    
    # append data to lists
    appended_ews.append(df_temp)
    appended_ktau.append(ktau_temp)
    
    # print status every 10 realisations
    if np.remainder(i+1,10)==0:
        print('Realisation '+str(i+1)+' complete')


# concatenate EWS DataFrames - use realisation number and time as indices
df_ews = pd.concat(appended_ews).set_index('Realisation number',append=True).reorder_levels([1,0])

# concatenate Kendall tau data into DataFrame indexed by realisation number
df_ktau = pd.concat(appended_ktau, axis=1).transpose().set_index('Realisation number')




#-----------------------------------------------
## Examples of obtaining specific data / plots
#------------------------------------------------


# EWS of realisation 1 at time 2
df_ews.loc[(1,2)] # must include () when referencing multiple indexes

# Varaince of realisation 7
df_ews.loc[7,'Variance']

# kendall tau of variance for all realisations
df_ktau.loc[:,'Variance']

# plot of all variance trajectories
df_ews.loc[:,'Variance'].unstack(level=0).plot() # unstack puts index back as a column






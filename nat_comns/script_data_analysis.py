
#!/usr/bin/env python3
# -*- coding: utf-8 -*-


"""
Created on Mon Feb 11 2019

@author: Thomas Bury

Early warning signal anlaysis using bootstrapping with the Fussmann dataset

"""

# Import python libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Import ewstools
from ewstools import ewstools



# Name of directory within data_export
dir_name = 'bootstrap_test'

if not os.path.exists('../data_export/'+dir_name):
    os.makedirs('../data_export/'+dir_name)


#------------------------
# Parameters
#–-----------------------

# EWS computation parameters
span = 80 # span used for Loess filtering of time-series (number of days)
ham_length = 80 # length of Hamming window
ham_offset = 0.5 # offset of Hamming windows
w_cutoff = 1 # cutoff of higher frequencies
ews = ['var','ac','smax','aic'] # EWS to compute
lags = [1,2,10] # lag times for autocorrelation computation (lag of 10 to show decreasing AC where tau=T/2)
sweep = False # whether to sweep over initialisation parameters during AIC fitting


# Bootstrapping parameters
block_size = 40 # size of blocks used to resample time-series
bs_type = 'Circular' # type of bootstrapping
n_samples = 100 # number of bootstrapping samples to take







#------------------
## Data import and curation
#-------------------

# Import raw data
raw = pd.read_excel('../data/raw_fussmann_2000.xls',header=[1])

# Round delta column to 2d.p
raw['meandelta'] = raw['meandelta'].apply(lambda x: round(x,2))

# Rename column labels
raw.rename(columns={'meandelta':'Delta', 'day#':'Time'}, inplace=True)

## Shift day# to start at 0

# Function to take list and subtract minimum element
def zero_shift(array):
    return array-min(array)

# Unique delta values
deltaVals = raw['Delta'].unique()

# Loop through delta values
for d in deltaVals: 
    # Shift time values to start at 0
    raw.loc[ raw['Delta']==d,'Time'] = zero_shift(
            raw.loc[ raw['Delta']==d,'Time']) 

## Index dataframe by Delta and Time
raw.set_index(['Delta','Time'], inplace=True)            

# Export trajectories as a csv file
raw_traj = raw[['Chlorella','Brachionus']]
raw_traj.to_csv("../data_export/series_data.csv", index_label=['Delta','Time'])               
               
               
# Compute number of data points for each value of delta
series_lengths=pd.Series(index=deltaVals)
series_lengths.index.name= 'Delta'
for d in deltaVals:
    series_lengths.loc[d] = len(raw.loc[d])
    
# Only consider the delta values for which the corresponding trajecotories have over 25 data points
deltaValsFilt = series_lengths[ series_lengths > 1 ].index

       






#-------------------------------------
# Compute bootstrapped samples of time-series data
#–----------------------------------

# Compute bootstrapped residuals of each time-series

# Initialise list for DataFrames of bootstrapped time-series
list_df_samples = []

# Loop over dilution rates
for d in deltaValsFilt:
    # Loop over species
    for species in ['Chlorella','Brachionus']:
        
        # Time-series to work with
        series = raw_traj.loc[d][species]
        
        # Compute bootstrapped series   
        df_samples_temp = ewstools.roll_bootstrap(series,
                           span = span,
                           roll_window = 1,
                           n_samples = n_samples,
                           bs_type = bs_type,
                           block_size = block_size
                           )

        # Add column for dilution rate and species
        df_samples_temp['Species'] = species
        df_samples_temp['Dilution rate'] = d
        
        # Remove indexing
        df_samples_temp.reset_index(inplace=True)
        
        # Add DF to list
        list_df_samples.append(df_samples_temp)
        
    # Print update
    print('Bootstrap samples for d = %.2f complete' % d)
        
# Concatenate dataframes of samples
df_samples = pd.concat(list_df_samples)

# Drop the Time column (redundant as no rolling window used)
df_samples.drop('Time', axis=1, inplace=True)

# Set a sensible index
df_samples.set_index(['Dilution rate', 'Species', 'Sample', 'Wintime'], inplace=True)

# Sort the index
df_samples.sort_index(inplace=True)


        
        
        
#----------------------------------------------
# Compute EWS for each bootstrapped time-series
#-------------------------------------------------

# List to store EWS DataFrames
list_df_ews = []
# List to store power spectra DataFrames of one of the samples
list_pspec = []

# Sample values
sampleVals = np.array(df_samples.index.levels[2])


# Loop through dilution rate
for d in deltaValsFilt:
    
    # Loop through species
    for species in ['Chlorella', 'Brachionus']:
        
        # Loop through sample values
        for sample in sampleVals:
            
            # Compute EWS of sample series
            series_temp = df_samples.loc[(d, species, sample, ),'x']
            
            ews_dic = ewstools.ews_compute(series_temp,
                              roll_window = 1, 
                              smooth = 'None',
                              ews = ews,
                              lag_times = lags,
                              upto='Full',
                              sweep=sweep,
                              ham_length=ham_length,
                              ham_offset=ham_offset,
                              w_cutoff=w_cutoff)
            
            ## The DataFrame of EWS
            df_ews_temp = ews_dic['EWS metrics']
            
            # Include columns for dilution rate, species and sample number
            df_ews_temp['Dilution rate'] = d
            df_ews_temp['Species'] = species
            df_ews_temp['Sample'] = sample

            # Drop NaN values
            df_ews_temp = df_ews_temp.dropna()        
            
            # Append list_df_ews
            list_df_ews.append(df_ews_temp)
            
            
            ## The DataFrame of power spectra
            df_pspec_temp = ews_dic['Power spectrum'][['Empirical']].dropna()
            # Add columns for species and dilution rate
            df_pspec_temp['Dilution rate'] = d
            df_pspec_temp['Species'] = species
            df_pspec_temp['Sample'] = sample
            
            # Append list
            list_pspec.append(df_pspec_temp)
    
    # Print update
    print('EWS for d = %.2f complete' % d)
        
# Concatenate EWS DataFrames. Index [Dilution rate, species , Sample]
df_ews_boot = pd.concat(list_df_ews).reset_index(drop=True).set_index(['Dilution rate','Species', 'Sample'])
# Sort the index 
df_ews_boot.sort_index(inplace=True)

# Concatenate power spectrum DataFrames
df_pspec_boot = pd.concat(list_pspec).reset_index().set_index(['Dilution rate','Species','Sample','Frequency'])
# Drop the time column
df_pspec_boot.drop('Time', inplace=True, axis=1)
df_pspec_boot.sort_index(inplace=True)



#---------------------------------------
# Compute mean and confidence intervals
#–----------------------------------------


# Relevant EWS to work with
ews_export = ['Variance','Lag-1 AC','Lag-2 AC','Lag-10 AC','AIC fold',
              'AIC hopf', 'AIC null', 'Smax']


# List to store confidence intervals for each EWS
list_intervals = []

# Loop through each EWS
for i in range(len(ews_export)):
    
    # Compute mean, and confidence intervals
    series_intervals = df_ews_boot[ews_export[i]].groupby(['Dilution rate','Species']).apply(ewstools.mean_ci, alpha=0.95)
    
    # Add to the list
    list_intervals.append(series_intervals)
    
# Concatenate the series
df_intervals = pd.concat(list_intervals, axis=1)
    







#-------------------
## Some plots of EWS
#------------------


## Smax
#df_ews_boot.xs(('Chlorella'), level=('Species'))['Smax'].unstack(level='Sample').plot()
#
## AIC hopf
#df_ews_boot.xs(('Chlorella'), level=('Species'))['AIC hopf'].unstack(level='Sample').plot()

# Organise DataFrame for plotting with seaborn
df_plot = df_ews_boot.xs('Chlorella', level=1).reset_index()
# All AIC Hopf samples
g1=sns.relplot(data=df_plot, x='Dilution rate', y='AIC hopf', kind='line', hue='Sample')
# Confidence intervals of AIC Hopf
g2=sns.relplot(data=df_plot, x='Dilution rate', y='AIC hopf',kind='line')

# Plot variance
# All AIC Hopf samples
g1=sns.relplot(data=df_plot, x='Dilution rate', y='Variance', kind='line', hue='Sample')
# Confidence intervals of AIC Hopf
g2=sns.relplot(data=df_plot, x='Dilution rate', y='Variance',kind='line', ci=99)



#-------------------------------
# Power spectra visualisation
#–-----------------------------

# Limits for x-axis
xmin = -3.2
xmax = 3.2

## Chlorella
species='Chlorella'
g = sns.FacetGrid(df_pspec_boot.xs(species,level=1).reset_index(),
                  col='Dilution rate',
                  hue='Sample',
                  palette='Set1',
                  col_wrap=3,
                  sharey=False,
                  aspect=1.5,
                  height=1.8
                  )



# Plots
plt.rc('axes', titlesize=10) 
g.map(plt.plot, 'Frequency', 'Empirical', linewidth=1)




## Brachionus
species='Brachionus'
g = sns.FacetGrid(df_pspec_boot.xs(species,level=1).reset_index(),
                  col='Dilution rate',
                  hue='Sample',
                  palette='Set1',
                  col_wrap=3,
                  sharey=False,
                  aspect=1.5,
                  height=1.8
                  )





# Plots
plt.rc('axes', titlesize=10) 
g.map(plt.plot, 'Frequency', 'Empirical', linewidth=1)


## Axes properties
#axes = g.axes
## Global axes properties
#for i in range(len(axes)):
#    ax=axes[i]
#    d=deltaValsFilt[i]
#    ax.set_ylim(bottom=0, top=1.1*df_pspec_boot.xs(species,level=1)['Empirical'].loc[(slice(None),slice(None),slice(xmin,xmax))].max())
#    ax.set_xlim(left=xmin, right=xmax)
#    ax.set_xticks([-2,-1,0,1,2])
#    ax.set_title('Delta = %.2f' % deltaValsFilt[i])
#
#
## Y labels
#for ax in axes[::3]:
#    ax.set_ylabel('Power')
#    
## Specific Y limits
#for ax in axes[:4]:
#    ax.set_ylim(top=0.004)
#for ax in axes[6:9]:
#    ax.set_ylim(top=0.25)
## Assign to plot label
#pspec_plot_chlor=g








#-----------------------
# Export data for plotting in MMA
#-----------------------
    

# EWS data
df_ews_boot[ews_export].to_csv('../data_export/'+dir_name+'/ews.csv')

# EWS confidnece intervals over samples
df_intervals.to_csv('../data_export/'+dir_name+'/ews_intervals.csv')

# Export empirical pspec data for plotting in MMA
df_pspec_boot.to_csv('../data_export/'+dir_name+'/pspec.csv')











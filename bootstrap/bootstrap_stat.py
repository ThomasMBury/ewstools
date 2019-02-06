#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  5 15:35:06 2019

Function to bootstrap the rolling window outupts of a time-series

@author: tb460
"""



# Import python modules
import numpy as np
import pandas as pd





# Modules for filtering timeseries
from scipy.ndimage.filters import gaussian_filter as gf
from statsmodels.nonparametric.smoothers_lowess import lowess

# Module for bootstrapping
from arch.bootstrap import StationaryBootstrap

def segment_bootstrap(raw_series,
                      span = 0.1,
                      roll_window = 0.25,
                      upto = 'Full'):
    '''
    Smoothes raw_series and computes residuals over a rolling window.
    Bootstraps each segment and outputs samples.
    
    Inputs:
        raw_series - pandas Series indexed by time
        span (0.2) - proportion of data used for Loess filtering
        roll_windopw (0.25) - size of the rolling window (as a proportion
                     of the lenght of the data)
        upto ('Full') - if 'Full', use entire time-series, ow input time up 
            to which EWS are to be evaluated
        
    Output:
        DataFrame indexed by time, sample number and then rolling window time
        
    '''




    # Initialise a DataFrame to store EWS data - indexed by time
    df_ews = pd.DataFrame(raw_series)
    df_ews.columns = ['State variable']
    df_ews.index.rename('Time', inplace=True)
    
    
    
    
    
    #------------------------------
    ## Data detrending
    #–------------------------------
    
    
    # Select portion of data up to 'upto'
    if upto == 'Full':
        series = raw_series
    else: series = raw_series.loc[:upto]
    
    
    # Smooth the series and compute the residuals
    smooth_data = lowess(series.values, series.index.values, frac=0.5)[:,1]
    smooth_series = pd.Series(smooth_data, index=series.index)
    residuals = series.values - smooth_data
    resid_series = pd.Series(residuals, index=series.index)

    
    # Add smoothed data and residuals to the EWS DataFrame
    df_ews['Smoothing'] = smooth_series
    df_ews['Residuals'] = resid_series
    
    
    
    #---------------------------
    # Bootstrapping
    #–---------------------------
    
    
    
    # Compute the rolling window size (integer value)
    rw_size=int(np.floor(roll_window * raw_series.shape[0]))




    # Bootstrap residuals using stationary bootstrap
    block_size = 10
    bs = StationaryBootstrap(block_size, residuals)
    n_samples = 100
    
    # List of sample time-series from residuals
    list_samples = []
    
    count = 1
    for data in bs.bootstrap(n_samples):
        
        df_temp = pd.DataFrame({'sample': count, 
                                'twin': series.index.values,
                                'x': data[0][0]})
        list_samples.append(df_temp)
        count += 1
        
    # Concatenate list of samples
    df_samples = pd.concat(list_samples)
    df_samples.set_index(['sample','twin'], inplace=True)





# Function to take in stationary series and output (block) bootstrap samples
def block_bootstrap(series,
              n_samples,
              type = 'Stationary',
              block_size = 10
              ):

    '''
    Computes bootstrapped samples of series.
    
    Inputs:
        series: pandas Series indexed by time
        n_samples: # bootstrapped samples to output
        type ('Stationary'): type of bootstrapping to perform.
            Options include ['Stationary', 'Circular']
        block_size: # size of resampling blocks. Should be big enough to
            capture important frequencies in the series
            
    Ouput:
        DataFrame indexed by sample number and time
        
    
    '''

    # Set up list for sampled time-series
    list_samples = []
    
    # Stationary bootstrapping
    if type == 'Stationary':
        bs = StationaryBootstrap(block_size, series)
                
        # Count for sample number
        count = 1
        for data in bs.bootstrap(n_samples):
            
            df_temp = pd.DataFrame({'sample': count, 
                                    'time': series.index.values,
                                    'x': data[0][0]})
            list_samples.append(df_temp)
            count += 1
            
        
    

    # Concatenate list of samples
    df_samples = pd.concat(list_samples)
    df_samples.set_index(['sample','time'], inplace=True)

    
    # Output DataFrame of samples
    return df_samples

    
    




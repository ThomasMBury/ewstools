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
from statsmodels.nonparametric.smoothers_lowess import lowess

# Module for bootstrapping
from arch.bootstrap import StationaryBootstrap, CircularBlockBootstrap



#----------------------------------------------
# Block bootstrap single stationary time-series
#–--------------------------------------------

def block_bootstrap(series,
              n_samples,
              bs_type = 'Stationary',
              block_size = 10
              ):

    '''
    Computes bootstrapped samples of series.
    
    Inputs:
        series: pandas Series indexed by time
        n_samples: # bootstrapped samples to output
        bs_type ('Stationary'): type of bootstrapping to perform.
            Options include ['Stationary', 'Circular']
        block_size: # size of resampling blocks. Should be big enough to
            capture important frequencies in the series
            
    Ouput:
        DataFrame indexed by sample number and time
        
    
    '''

    # Set up list for sampled time-series
    list_samples = []
    
    # Stationary bootstrapping
    if bs_type == 'Stationary':
        bs = StationaryBootstrap(block_size, series)
                
        # Count for sample number
        count = 1
        for data in bs.bootstrap(n_samples):
            
            df_temp = pd.DataFrame({'sample': count, 
                                    'time': series.index.values,
                                    'x': data[0][0]})
            list_samples.append(df_temp)
            count += 1
            
    if bs_type == 'Circular':
        bs = CircularBlockBootstrap(block_size, series)
                
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

    
 


#------------------------------------
# Block bootstrap a non-stationary time-series over a rolling window
#-----------------------------------------
    


def roll_bootstrap(raw_series,
                   span = 0.1,
                   roll_window = 0.25,
                   roll_offset = 1,
                   upto = 'Full',
                   n_samples = 20,
                   bs_type = 'Stationary',
                   block_size = 10
                   ):
    
    
    '''
    Smoothes series and computes residuals over a rolling window.
    Bootstraps each segment and outputs samples.
    
    Inputs:
        raw_series - pandas Series indexed by time
        span (0.1) - proportion of data used for Loess filtering
        roll_windopw (0.25) - size of the rolling window (as a proportion
                     of the lenght of the data)
        roll_offset (1) - number of points to shift the rolling window
            upon each iteration (reduce to increase computation time)
        upto ('Full') - if 'Full', use entire time-series, ow input time up 
            to which EWS are to be evaluated
        n_samples: # bootstrapped samples to output
        bs_type ('Stationary'): type of bootstrapping to perform.
            Options include ['Stationary', 'Circular']
        block_size: # size of resampling blocks. Should be big enough to
            capture important frequencies in the series
        
        
    Output:
        DataFrame indexed by realtime, sample number and then rolling window time
        
    '''

    
    
    #-------------------------
    # Parameter configuration
    #–-------------------------
    
    # Compute the rolling window size (integer value)
    rw_size=int(np.floor(roll_window * raw_series.shape[0]))
    
    
    
    #------------------------------
    ## Data detrending
    #–------------------------------
    
    
    # Select portion of data up to 'upto'
    if upto == 'Full':
        series = raw_series
    else: series = raw_series.loc[:upto]
    
    
    # Smooth the series and compute the residuals
    smooth_data = lowess(series.values, series.index.values, frac=0.5)[:,1]
    residuals = series.values - smooth_data
    resid_series = pd.Series(residuals, index=series.index)



    #-------------------------------
    # Rolling window over residuals
    #-------------------------------
    
    # Number of components in the residual time-series
    num_comps = len(resid_series)
    # Make sure window offset is an integer
    roll_offset = int(roll_offset)
    
    # Initialise a list for the sample residuals at each time point
    list_samples = []
    
    # Loop through window locations shifted by roll_offset
    for k in np.arange(0, num_comps-(rw_size-1), roll_offset):
        
        # Select subset of series contained in window
        window_series = resid_series.iloc[k:k+rw_size]           
        # Asisgn the time value for the metrics (right end point of window)
        t_point = resid_series.index[k+(rw_size-1)]            
        
        # Compute bootstrap samples of residauls within rolling window
        df_samples_temp = block_bootstrap(window_series, n_samples, 
                                          bs_type, block_size)
        
        # Add column with real time (end of window)
        df_samples_temp['realtime'] = t_point
                
        # Reorganise index
        df_samples_temp.reset_index(inplace=True)
        df_samples_temp.set_index(['realtime','sample','time'], inplace=True)
        df_samples_temp.index.rename(['realtime','sample','wintime'],inplace=True)
        
        # Append the list of samples
        list_samples.append(df_samples_temp)
        
        # Print update
        if int(t_point):
            print('Bootstrap samples for window at t = %.2f complete' % (t_point))
 
    


    #-------------------------
    # Organise output DataFrame
    #–--------------------------

    
    # Concatenate list of samples
    df_samples = pd.concat(list_samples)
    
    # Output DataFrame
    return df_samples









   




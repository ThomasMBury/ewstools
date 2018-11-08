    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:11:58 2018

@author: tb460

A module containing a function to compute the standard EWS from time-series data.
"""

# import required python modules
import numpy as np
from scipy.ndimage.filters import gaussian_filter as gf
import pandas as pd

        
#--------------------------------
    
def ews_std(raw_series, 
            roll_window=0.25,
            smooth=True,
            band_width=0.2,
            ews=['var','ac','cv','skew'], 
            lag_times=[1]):
    '''
    Function to compute the standard EWS over a rolling window.
    The pandas library is used to compute rolling statistics.    
    
    Input (default)
    raw_series : pandas Series indexed by time 
    roll_windopw (0.25) : size of the rolling window (as a proportion
    of the length of the data)
    smooth (True) : if True, series data is detrended with a Gaussian kernel
    band_width (0.2) : bandwidth of Gaussian kernel
    ews (['var,'ac'] : list of strings corresponding to the desired EWS.
         Options include
             'var'   : Variance
             'ac'    : Autocorrelation
             'sd'    : Standard deviation
             'cv'    : Coefficient of variation
             'skew'  : Skewness
             'kurt'  : Kurtosis
             
    lag_times : list of integers corresponding to the desired lag times for AC
    
    Output
    DataFrame indexed by time with columns csp to each EWS
    '''


    # initialise a DataFrame to store EWS data - indexed by time
    df_ews = pd.DataFrame(raw_series)
    df_ews.columns = ['State variable']
    df_ews.index.rename('Time', inplace=True)

    ## detrend the data
    
    # compute the size of the bandwidth
    bw_size=raw_series.shape[0]*band_width   
    
    # compute smoothed data and residuals
    if smooth:
        smooth_data = gf(raw_series.values, sigma=bw_size, mode='reflect')
        smooth_series = pd.Series(smooth_data, index=raw_series.index)
        residuals = raw_series.values - smooth_data
        resid_series = pd.Series(residuals,index=raw_series.index)
    
        # add them to the DataFrame
        df_ews['Smoothing'] = smooth_series
        df_ews['Residuals'] = resid_series
        
        
    # use residuals for EWS if smooth=True, ow use raw series
    eval_series = resid_series if smooth else raw_series
    
    #-----------------
    ## compute EWS
    #-----------------
    
    # compute the size of the rolling window (this must be an integer)
    rw_size=int(np.floor(roll_window * raw_series.shape[0]))
    
    
    # compute the stabdard deviation as a Series and add to DataFrame
    if 'sd' in ews:
        roll_sd = eval_series.rolling(window=rw_size).std()
        df_ews['Standard deviation'] = roll_sd
    
    # compute the variance as a Series and add to DataFrame
    if 'var' in ews:
        roll_var = eval_series.rolling(window=rw_size).var()
        df_ews['Variance'] = roll_var
    
    # compute the autocorrelation for each lag in lag_times and add to DataFrame   
    if 'ac' in ews:
        for i in range(len(lag_times)):
            roll_ac = eval_series.rolling(window=rw_size).apply(
        func=lambda x: pd.Series(x).autocorr(lag=lag_times[i]))
            df_ews['Lag-'+str(lag_times[i])+' AC'] = roll_ac

            
    # compute the coefficient of variation and add to DataFrame
    if 'cv' in ews:
        # mean of raw_series
        roll_mean = raw_series.rolling(window=rw_size).mean()
        # standard deviation of residuals
        roll_std = eval_series.rolling(window=rw_size).std()
        # coefficient of variation
        roll_cv = roll_std.divide(roll_mean)
        df_ews['Coefficient of variation'] = roll_cv

    # compute the skewness and add to DataFrame
    if 'skew' in ews:
        roll_skew = eval_series.rolling(window=rw_size).skew()
        df_ews['Skewness'] = roll_skew

    # compute the kurtosis and add to DataFrame
    if 'kurt' in ews:
        roll_kurt = eval_series.rolling(window=rw_size).kurt()
        df_ews['Kurtosis'] = roll_kurt



#--------------------
# Compute kendall taus
#-----------------------
        
    # Put time values as their own series for correlation computation
    time_series = pd.Series(raw_series.index, index=raw_series.index)
    
    # Find kendall tau correlation coefficient for each EWS (column of df_ews)
    ktau = df_ews.corrwith(time_series)

    return df_ews, ktau
    
# update readme file with kendall tau info.

#-------------------------------------













    
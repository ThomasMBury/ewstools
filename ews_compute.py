    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:11:58 2018

@author: tb460

A module containing functions to compute the EWS from time-series data.
"""

# import required python modules
import numpy as np
from scipy.ndimage.filters import gaussian_filter as gf
import pandas as pd

# import local modules
from ews_spec import pspec_welch, pspec_metrics


        
#--------------
# roll_window
#--------------

from itertools import islice

def roll_window(seq, n=2):
    '''
    Returns a rolling window (of length n) over data from the iterable
    s -> (s0,s1,...s[n-1]), (s1,s2,...,sn), ...
    ''' 
    
    it = iter(seq)
    result = tuple(islice(it, n))
    if len(result) == n:
        yield result
    for elem in it:
        result = result[1:] + (elem,)
        yield result
        
        
        
#------------------




def ews_compute(raw_series, 
            roll_window=0.25,
            smooth=True,
            band_width=0.2,
            ews=['var','ac','cv','skew'], 
            lag_times=[1],
            ham_length=40,
            ham_offset=0.5,
            w_cutoff=1):
    '''
    Function to compute EWS from time-series data.   
    
    Input (default value)
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
             'smax'  : Peak in the power spectrum
             'cf'    : Coherence factor
             'aic'   : AIC weights
             
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
    ## compute standard EWS
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

    
    #-----------------
    ## compute spectral EWS
    #-----------------
    
   
    if 'smax' in ews or 'cf' in ews or 'aic' in ews:
       
        # make a class for rolling.apply to accept functions with multiple outputs
        from collections import deque
        class multi_output_function_class:
            def __init__(self):
                self.deque_2 = deque()
                self.deque_3 = deque()
                self.deque_4 = deque()
                self.deque_5 = deque()
                self.deque_6 = deque()
                self.deque_7 = deque()
                self.deque_8 = deque()
            
            def f1(self, window):
                # compute power spectrum of window data using function pspec_welch
                dt = eval_series.index[1]-eval_series.index[2]
                pspec = pspec_welch(window,dt,ham_length=ham_length,ham_offset=ham_offset,w_cutoff=w_cutoff)
                
                # compute the spectral metrics and put into a dataframe
                df_spec_metrics = pspec_metrics(pspec,ews)
                self.k = df_spec_metrics
                self.deque_2.append(self.k['Coherence factor'])
                self.deque_3.append(self.k['AIC fold'])
                self.deque_4.append(self.k['AIC hopf'])
                self.deque_5.append(self.k['AIC null'])
                self.deque_6.append(self.k['Params fold'])
                self.deque_7.append(self.k['Params hopf'])
                self.deque_8.append(self.k['Params null'])
                return self.k['Smax']    

            def f2(self, window):
                return self.deque_2.popleft()   
            def f3(self, window):
                return self.deque_3.popleft()
            def f4(self, window):
                return self.deque_4.popleft()
            def f5(self, window):
                return self.deque_5.popleft()
            def f6(self, window):
                return self.deque_6.popleft()
            def f7(self, window):
                return self.deque_7.popleft()
            def f8(self, window):
                return self.deque_8.popleft()            
            
            
            
        # introduce a member of class: func
        func = multi_output_function_class()
            
        # apply func over a rolling window
        output = eval_series.rolling(window=rw_size).agg(
                {'Smax':func.f1,
                 'Coherence factor':func.f2,
                 'AIC fold':func.f3,
                 'AIC hopf':func.f4,
                 'AIC null':func.f5,
                 'Params fold':func.f6,
                 'Params hopf':func.f7,
                 'Params null':func.f8})
                
        # join to main DataFrame
        df_ews = df_ews.append(output)
        
        
        
        















#--------------------
# Compute kendall taus of EWS trends
#-----------------------
        
    # Put time values as their own series for correlation computation
    time_series = pd.Series(raw_series.index, index=raw_series.index)
    
    # Find kendall tau correlation coefficient for each EWS (column of df_ews)
    ktau = df_ews.corrwith(time_series)

    return df_ews, ktau
    
# update readme file with kendall tau info.

#-------------------------------------













    
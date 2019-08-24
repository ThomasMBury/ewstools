#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 21:12:35 2019

@author: tbury


Test for eval_recon function

"""



import pytest
import numpy as np
import pandas as pd

# Import eval_recon function
import sys
sys.path.append('../ewstools')
import eval_recon






# Simulate a simple multi-variate time series
tVals = np.arange(0,10,0.1)
xVals = 5 + np.random.normal(0,1,len(tVals))
yVals = 10 + np.random.normal(0,1,len(tVals))
zVals = 0 + np.random.normal(0,1,len(tVals))

df_test = pd.DataFrame({'x':xVals,'y':yVals,'z':zVals}, index=tVals)

n = len(df_test.columns)




# Build function that computes Jac over a rolling window
# with smoothing etc. (temporary location for function)

def eval_recon_rolling(df_in,
                       roll_window=0.4,
                       smooth='Lowess',
                       span=0.1,
                       band_width=0.2,
                       upto='Full'):
    
        '''
    Compute reconstructed eigenvalues from residuals of multi-variate
    non-stationary time series
    	
    Args
    ----
    df_in: pd.DataFrame
        Time series data with variables in different columns, indexed by time
    roll_window: float
        Rolling window size as a proportion of the length of the time series 
        data.
    smooth: {'Gaussian', 'Lowess', 'None'}
        Type of detrending.
    band_width: float
        Bandwidth of Gaussian kernel. Taken as a proportion of time-series length if in (0,1), 
        otherwise taken as absolute.
    span: float
        Span of time-series data used for Lowess filtering. Taken as a 
        proportion of time-series length if in (0,1), otherwise taken as 
        absolute.
    upto: int or 'Full'
        Time up to which EWS are computed. Enter 'Full' to use
        the entire time-series. Otherwise enter a time value.

    Returns
    --------
    dict of pd.DataFrames:
        A dictionary with the following entries.
        **'EWS metrics':** A DataFrame indexed by time with columns corresopnding 
        to each EWS.
        **'Power spectrum':** A DataFrame of the measured power spectra and the best fits 
        used to give the AIC weights. Indexed by time. 
        **'Kendall tau':** A DataFrame of the Kendall tau values for each EWS metric.
    '''
    
    # Initialise a DataFrame to store EWS data - indexed by time
    df_ews = pd.DataFrame(raw_series)
    df_ews.columns = ['State variable']
    df_ews.index.rename('Time', inplace=True)
    
    # Select portion of data where EWS are evaluated (e.g only up to bifurcation)
    if upto == 'Full':
        short_series = raw_series
    else: short_series = raw_series.loc[:upto]


    #------Data detrending--------

    # Compute the absolute size of the bandwidth if it is given as a proportion
    if 0 < band_width <= 1:
        bw_size = short_series.shape[0]*band_width
    else:
        bw_size = band_width
        
    # Compute the Lowess span as a proportion if given as absolute
    if not 0 < span <= 1:
        span = span/short_series.shape[0]
    else:
        span = span
    
    
    # Compute smoothed data and residuals
    if  smooth == 'Gaussian':
        smooth_data = gf(short_series.values, sigma=bw_size, mode='reflect')
        smooth_series = pd.Series(smooth_data, index=short_series.index)
        residuals = short_series.values - smooth_data
        resid_series = pd.Series(residuals,index=short_series.index)
    
        # Add smoothed data and residuals to the EWS DataFrame
        df_ews['Smoothing'] = smooth_series
        df_ews['Residuals'] = resid_series
    
    if  smooth == 'Lowess':
        smooth_data = lowess(short_series.values, short_series.index.values, frac=span)[:,1]
        smooth_series = pd.Series(smooth_data, index=short_series.index)
        residuals = short_series.values - smooth_data
        resid_series = pd.Series(residuals, index=short_series.index)
    
        # Add smoothed data and residuals to the EWS DataFrame
        df_ews['Smoothing'] = smooth_series
        df_ews['Residuals'] = resid_series
        
    # Use the short_series EWS if smooth='None'. Otherwise use reiduals.
    eval_series = short_series if smooth == 'None' else resid_series
    
    # Compute the rolling window size (integer value)
    rw_size=int(np.floor(roll_window * raw_series.shape[0]))
    
    
    



















#------------------------------------
# Write test functions

def test_compute_autocov():

    ar_out = eval_recon.compute_autocov(df_test)
    assert type(ar_out) == np.ndarray
    assert ar_out.shape == (n,n)


def test_eval_recon():
    
    dic_out = eval_recon.eval_recon(df_test)
    jac = dic_out['Jacobian']
    evals = dic_out['Eigenvalues']
    evecs = dic_out['Eigenvectors']
    
    assert type(dic_out) == dict
    assert type(jac) == pd.DataFrame
    assert type(evals) == np.ndarray
    assert type(evecs) == np.ndarray
    assert jac.shape == (n,n)
    




































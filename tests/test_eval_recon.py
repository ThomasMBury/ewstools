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


# For detrending time-series
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.ndimage.filters import gaussian_filter as gf

def eval_recon_rolling(df_in,
                       roll_window=0.4,
                       roll_offset=1,
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
    roll_offset: int
        Offset of rolling window upon each EWS computation - make larger to save
        on computational time
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
    
    
    # Properties of df_in
    var_names = df_in.columns
    
    
    
    # Select portion of data where EWS are evaluated (e.g only up to bifurcation)
    if upto=='Full':
        df_pre = df_in.copy()
    else: df_pre = df_in.loc[:upto]


    #------Data detrending--------

    # Compute the absolute size of the bandwidth if it is given as a proportion
    if 0 < band_width <= 1:
        bw_size = df_pre.shape[0]*band_width
    else:
        bw_size = band_width
        
    # Compute the Lowess span as a proportion if given as absolute
    if not 0 < span <= 1:
        span = span/df_pre.shape[0]
    else:
        span = span
    
    
    # Compute smoothed data and residuals
    if  smooth == 'Gaussian':
        # Loop through variables
        for var in var_names:
            
            smooth_data = gf(df_pre[var].values, sigma=bw_size, mode='reflect')
            smooth_series = pd.Series(smooth_data, index=df_pre.index)
            residuals = df_pre[var].values - smooth_data
            resid_series = pd.Series(residuals,index=df_pre.index)
            # Add smoothed data and residuals to df_pre
            df_pre[var+'_s'] = smooth_series
            df_pre[var+'_r'] = resid_series
    
    if  smooth == 'Lowess':
        # Loop through variables
        for var in var_names:
            
            smooth_data = lowess(df_pre[var].values, df_pre.index.values, frac=span)[:,1]
            smooth_series = pd.Series(smooth_data, index=df_pre.index)
            residuals = df_pre[var].values - smooth_data
            resid_series = pd.Series(residuals, index=df_pre.index)
            # Add smoothed data and residuals to df_pre
            df_pre[var+'_s'] = smooth_series
            df_pre[var+'_r'] = resid_series
    
    # Compute the rolling window size (integer value)
    rw_size=int(np.floor(roll_window * df_in.shape[0]))
    
    

    # Set up a rolling window

    # Number of components in the residual time-series
    num_comps = len(df_pre)
    # Rolling window offset (can make larger to save on computation time)
    roll_offset = int(roll_offset)
    
    # Initialise a list of dictionaries containing eval data
    list_evaldata = []
    
    # Loop through window locations shifted by roll_offset
    for k in np.arange(0, num_comps-(rw_size-1), roll_offset):
        
        # Select subset of residuals contained in window
        df_window = df_pre[[var+'_r' for var in var_names]].iloc[k:k+rw_size]
        # Asisgn the time value for the metrics (right end point of window)
        t_point = df_pre.index[k+(rw_size-1)]            
        
        # Do eigenvalue reconstruction on residuals
        dic_eval_recon = eval_recon.eval_recon(df_window)
        # Add time component
        dic_eval_recon['Time'] = t_point
        # Add them to list
        list_evaldata.append(dic_eval_recon)
        
    # Create dataframe from list of dicts of eval data
    df_evaldata = pd.DataFrame(list_evaldata)
    df_evaldata.set_index('Time',inplace=True)
        
    # Create output dataframe that merges all useful info
    df_out = pd.concat([df_in, 
                        df_pre[[var+'_r' for var in var_names]+[var+'_s' for var in var_names]], 
                        df_evaldata],axis=1)

    return df_out



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
    




































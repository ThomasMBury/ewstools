#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:47:31 2019

@author: tbury

Functions to reconstruct eigenvalues from a df of time series data
See Williamson (2015) for theory and implementation details

"""

import numpy as np
import pandas as pd


# For detrending time-series
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.ndimage.filters import gaussian_filter as gf


#------------------------
##  Function to compute lag-1 autocovariance matrix

def compute_autocov(df_in):
    '''
    Computes the autocovariance (lag-1) matrix of n 
    time series provided in df_in.
    
    Using the definition
        phi_ij = < X_i(t+1) X_j(t) >
    for each element of the autocovariance matrix phi.
    
    Input:
        df_in: DataFrame with n columns indexed by time
    Ouptut:
        np.array of autocovariance matrix
    '''
    
    # Obtain column names of df_in
    col_names = df_in.columns
    # Number of variables
    n = len(col_names)
    
    
    # Define function to compute autocovariance of two columns
    def autocov_cols(a,b):
        '''
        Computes autocovariance of two columns (can be the same)
        Note that this does not commute (a<->b) in general
        Input:
            a,b: Series indexed by time
        Output:
            float: autocovariance between the columns
        '''
        
        # Shift the column of a by 1
        a_shift = a.shift(1)
        
        # Put into a dataframe
        df_temp = pd.concat([a_shift,b], axis=1)
        
        # Compute covariance of columns a and b_shift
        cov = df_temp.cov().iloc[0,1]
        
        # Output
        return cov
            
        
    # Compute elements of autocovariance matrix
    list_elements = []
    
    for i in range(n):
        for j in range(n):
            a = df_in[col_names[i]]
            b = df_in[col_names[j]]
            # Compute autocovaraince between cols
            autocov = autocov_cols(a,b)
            # Append to list of elements
            list_elements.append(autocov)
    
    # Create autocovariance matrix from list of elements
    ar_autocov = np.array(list_elements).reshape(n,n)

    # Output
    return ar_autocov




#---------------------------------------
## Function to do Jacobian and eval reconstruction 


def eval_recon(df_in):
    '''
    Constructs estimate of Jacobian matrix from stationary time-series data
    and outputs the eigenvalues
    Input:
        df_in: DataFrame with two columns indexed by time
    Output:
    	dictionary consisting of
    		- 'Eigenvalues': np.array of eigenvalues
    		- 'Eigenvectors': np.array of eigenvectors
    		- 'Jacobian': pd.DataFrame of Jacobian entries
    '''
    
    # Compute autocovaraince matrix from columns
    ar_autocov = compute_autocov(df_in)
    
    # Compute the covariance matrix (built in function)
    ar_cov = df_in.cov()
    
    # Estimate of Jacobian (formula in Williamson (2015))
    # Requires computation of an inverse matrix
    jac = np.matmul(ar_autocov, np.linalg.inv(ar_cov))
    # Write the Jacobian as a df for output (so we have col lables)
    df_jac = pd.DataFrame(jac, columns = df_in.columns, index=df_in.columns)
      
    # Compute eigenvalues and eigenvectors
    evals, evecs = np.linalg.eig(jac)

    # Dictionary of data output
    dic_out = {'Eigenvalues':evals, 
               'Eigenvectors':evecs,
               'Jacobian':df_jac}
    
    return dic_out



#-----------------------------
# Function to do Jacobian reconstruction over a rolling window



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
        # Loop through variabless
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
        dic_eval_recon = eval_recon(df_window)
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
























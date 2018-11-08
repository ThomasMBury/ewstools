    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:11:58 2018

@author: tb460

A module containing functions to compute spectral EWS from time-series data.
"""

# import required python modules
import numpy as np
from scipy import signal
import pandas as pd

        
#--------------------------------
## pspec_welch
#------------------------------

def pspec_welch(series,
                ham_length,
                ham_offset=0.5,
                w_cutoff=1,
                scaling='spectrum'):


    '''
    Function to compute the power spectrum of *series* using Welch's method.
    This involves computing the periodogram with overlapping Hamming windows.
    
    Input (default)
    series : pandas Series indexed by time
    ham_length (40) : number of data points in the Hamming window
    ham_offset (0.5) : proportion of ham_length to use as an offset for each
                       Hamming window.
    w_cutoff (1) : proportion of maximum frequency with which to cutoff higher
                   frequencies.
    scaling ('spectrum') : selects between computing the power spectrum 
                           ('spectrum') and the power spectral density 
                           ('density') which is normalised.
                 
    Output
    Pandas series of power values indexed by frequency
    
    '''

    ## Assign properties of *series* to parameters
    
    # increment in time between data points (assuming uniform)
    dt = series.index[1] - series.index[0]
    # compute the sampling frequency 
    fs = 1/dt
    # number of data points
    num_points = series.shape[0]
    # if ham_length given as a proportion - compute number of data points in ham_length
    if 0 < ham_length <= 1:
        ham_length = num_points * ham_length
    # compute number of points in offset
    ham_offset_points = int(ham_offset*ham_length)
        
    ## compute the periodogram using Welch's method (scipy.signal function)
    pspec_raw = signal.welch(series.values,
                               fs,
                               nperseg=ham_length,
                               noverlap=ham_offset_points,
                               return_onesided=False,
                               scaling=scaling)
    
    # put into a pandas series and index by frequency (scaled by 2*pi)
    pspec_series = pd.Series(pspec_raw[1], index=2*np.pi*pspec_raw[0], name='Power spectrum')
    pspec_series.index.name = 'Frequency'
    
    # sort into ascending frequency
    pspec_series.sort_index(inplace=True)
    
    # append power spectrum with first value (by symmetry)
    pspec_series.at[-min(pspec_series.index)] = pspec_series.iat[0]
    
    # impose cutoff frequency
    wmax = w_cutoff*max(pspec_series.index) # cutoff frequency
    pspec_output = pspec_series[-wmax:wmax] # subset of power spectrum
    
    
    return pspec_output




#--------------------------
## pspec_metrics
#-------------------------



def pspec_metrics(pspec,
                  ews = ['coher_factor','smax','aic']):


    '''
    Function to compute the metrics associated with *pspec* that can be
    used as EWS.
    
    Input (default)
    pspec : power spectrum in the form of a Series indexed by frequency
    ews ( ['coher_factor', 'smax', 'aic'] ) : array of strings corresponding 
    to the EWS to be computed. Options include
        'coher_factor' : coherence factor
        'smax' : peak in the power spectrum
        'aic' : Hopf, Fold and Null AIC weights
        
                 
    Output
    Series of spectral metrics 
    
    '''



    













    
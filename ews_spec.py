    #!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:11:58 2018

@author: tb460

A module containing functions to compute spectral EWS from time-series data.
"""

# import required python modules
import numpy as np
from scipy.ndimage.filters import gaussian_filter as gf
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
    perio_raw = signal.welch(series.values,
                               fs,
                               nperseg=ham_length,
                               noverlap=ham_offset_points,
                               return_onesided=False,
                               scaling=scaling)
    
    # put into a pandas series and index by frequency (scaled by 2*pi)
    perio_series = pd.Series(perio_raw[1], index=2*np.pi*perio_raw[0], name='Power spectrum')
    perio_series.index.name = 'Frequency'
    
    # sort into ascending frequency
    perio_series.sort_index(inplace=True)
    
    # append power spectrum with first value (by symmetry)
    perio_series.at[-min(perio_series.index)] = perio_series.iat[0]
    
    # impose cutoff frequency
    wmax = w_cutoff*max(perio_series.index) # cutoff frequency
    perio_output = perio_series[-wmax:wmax] # subset of power spectrum
        
    
    return perio_output












    
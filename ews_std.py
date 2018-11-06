#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:11:58 2018

@author: tb460

A module containing functionns to cmopute EWS from time-series data.
"""

# import required python modules
import numpy as np
from scipy.ndimage.filters import gaussian_filter as gf
import pandas as pd

#–----------------------------

def smooth_function(x,band_width=0.2):
    '''
    function to smooth data using a Gaussian filter
    
    Input
    x : the input signal (nx1 array)
    band_width (0.2) : bandwidth of the smoothing kernel (as a proportion
    of the length of the data)
    
    Output
    detrended signal (nx1 array)
    '''
    
    # compute the size of the bandwidth 
    bw_size=np.size(x)*band_width
    
    # use pre-built gaussian filter function
    output=gf(x,sigma=bw_size, mode='reflect')
    
    # return output
    return output


#--------------------------------
    
def roll_variance(tseries,roll_window=0.25):
    '''
    Function to compute the variance of a trajecotry over a rolling window.
    We use pandas to compute rolling statistics.    
    
    Input
    tseries : input time-series ((t1,x1),(t2,x2),...,(tn,xn))
    roll_windopw (0.25) : size of the rolling window (as a proportion
    of the length of the data)
    
    Output
    variance time-series
    '''

    # compute the size of the rolling window (this must be an integer)
    rw_size=np.floor(roll_window * tseries.shape[0])
    return
    
    













    
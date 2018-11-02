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
    


    
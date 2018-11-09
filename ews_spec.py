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
from lmfit import Model, Parameters
import matplotlib.pyplot as plt

        
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
                  ews = ['smax','cf','aic']):


    '''
    Function to compute the metrics associated with *pspec* that can be
    used as EWS.
    
    Input (default)
    pspec : power spectrum in the form of a Series indexed by frequency
    ews ( ['smax', 'coher_factor', 'aic'] ) : array of strings corresponding 
    to the EWS to be computed. Options include
        'smax' : peak in the power spectrum
        'cf' : coherence factor
        'aic' : Hopf, Fold and Null AIC weights
        'aic_params' : AIC model parameter values
        
                 
    Output
    Series of spectral metrics 
    
    '''
    
    
    # initialise a Series for EWS
    spec_ews = pd.Series([])
    
    # compute smax
    if 'smax' in ews:
        smax = max(pspec)
        # add to DataFrame
        spec_ews['Smax'] = smax
        
        
        
    # compute coherence factor
    if 'cf' in ews:
        
        # frequency at which peak occurs
        w_peak = abs(pspec.idxmax())
        # index location
        
        # power of peak frequency
        power_peak = pspec.max()
        
        # compute the first frequency from -w_peak at which power<power_peak/2
        w_half = next( (w for w in pspec[-w_peak:].index if pspec.loc[w] < power_peak/2 ), 'None')
        
        # if there was no such frequency, or if peak crosses zero frequency,
        # set w_peak = 0 (makes CF=0) 
        if w_half == 'None' or w_half > 0:
            w_peak = 0
            
        else:
            # double the difference between w_half and -w_peak to get the width of the peak
            w_width = 2*(w_half - (-w_peak))
            
        # compute coherence factor (height/relative width)
        coher_factor = power_peak/(w_width/w_peak) if w_peak != 0 else 0

        # add to dataframe
        spec_ews['Coherence factor'] = coher_factor
    

    # compute AIC weights
    if 'aic' in ews:
        
        # put frequency values and power values as a list to use LMFIT
        freq_vals = pspec.index.tolist()
        power_vals = pspec.tolist()
        
        # define models to fit
        def fit_fold(omega,sigma,lam):
            return (sigma**2 / (2*np.pi))*(1/(omega**2+lam**2))
        
        def fit_hopf(omega,sigma,mu,omega0):
            return (sigma**2/(4*np.pi))*(1/((omega+omega0)**2+mu**2)+1/((omega-omega0)**2 +mu**2))
        
        def fit_null(omega,c):
            return c
    
        # assign to Model objects
        fold_model = Model(fit_fold)
        hopf_model = Model(fit_hopf)
        null_model = Model(fit_null)
        
        # set parameter initial values and constraints
        fold_model.set_param_hint('sigma', value=1)
        fold_model.set_param_hint('lam', value=-1, max=0)
        
        hopf_model.set_param_hint('sigma', value=1)
        hopf_model.set_param_hint('mu', value=-1, max=0)
        # condition for S(0) < psi*S(omega0)
        psi = 0.25
        hopf_model.set_param_hint('psi',value=psi,vary=False)
        # introduce new free parameter delta = psi*S(omega0)-S(0) >0
        hopf_model.set_param_hint('delta', value=0.01, min=0, vary=True)
        
        # IMPOSE ANOTHER CONDITION OF W < WMAX SOMEHOW... TO DO!
        
        # write omega0 in terms of delta
        hopf_model.set_param_hint('omega0', expr='sqrt(delta + (mu**2/(4*psi))*(4-3*psi+sqrt(psi**2-16*psi+16)))')
        
        null_model.set_param_hint('c',value=1, vary=True)
                
        # assign initial parameter values and constraints
        fold_params = fold_model.make_params()
        hopf_params = hopf_model.make_params()
        null_params = null_model.make_params()
        
    
        # fit each model to the power spectrum
        fold_result = fold_model.fit(power_vals, fold_params, omega=freq_vals)
        hopf_result = hopf_model.fit(power_vals, hopf_params, omega=freq_vals)
        null_result = null_model.fit(power_vals, null_params, omega=freq_vals)

        # get AIC statistics
        fold_aic = fold_result.aic
        hopf_aic = hopf_result.aic
        null_aic = null_result.aic
    
        # add to dataframe
        spec_ews['AIC fold'] = fold_aic
        spec_ews['AIC hopf'] = hopf_aic
        spec_ews['AIC null'] = null_aic
        
        # print out report (for now)
        print(fold_result.fit_report())
        print(hopf_result.fit_report())
        print(null_result.fit_report())
    
    
        # make a plot of fits
        plt.plot(freq_vals, fold_result.best_fit)
        plt.plot(freq_vals, hopf_result.best_fit)
        plt.plot(freq_vals, null_result.best_fit*np.ones(len(freq_vals)))
    
#        # return parameter values if asked for
#        if 'aic_params' in ews:
#            

    # return DataFrame of metrics
    return spec_ews




    
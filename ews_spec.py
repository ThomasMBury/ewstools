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
from lmfit import Model

        
#--------------------------------
## pspec_welch
#------------------------------

def pspec_welch(yVals,
                dt,
                ham_length=40,
                ham_offset=0.5,
                w_cutoff=1,
                scaling='spectrum'):


    '''
    Function to compute the power spectrum of *series* using Welch's method.
    This involves computing the periodogram with overlapping Hamming windows.
    
    Input (default)
    yVals : array of state values
    dt : time separation between data points
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
    
    # Compute the sampling frequency 
    fs = 1/dt
    # Number of data points
    num_points = len(yVals)
    # If ham_length given as a proportion - compute number of data points in ham_length
    if 0 < ham_length <= 1:
        ham_length = num_points * ham_length
    # If Hamming length given is less than the length of the t-series, make ham_length=length of tseries.
    if ham_length >= num_points:
        ham_length = num_points
    # Compute number of points in offset
    ham_offset_points = int(ham_offset*ham_length)
        
    ## Compute the periodogram using Welch's method (scipy.signal function)
    pspec_raw = signal.welch(yVals,
                               fs,
                               nperseg=ham_length,
                               noverlap=ham_offset_points,
                               return_onesided=False,
                               scaling=scaling)
    
    # Put into a pandas series and index by frequency (scaled by 2*pi)
    pspec_series = pd.Series(pspec_raw[1], index=2*np.pi*pspec_raw[0], name='Power spectrum')
    pspec_series.index.name = 'Frequency'
    
    # sort into ascending frequency
    pspec_series.sort_index(inplace=True)
    
    # append power spectrum with first value (by symmetry)
    pspec_series.at[-min(pspec_series.index)] = pspec_series.iat[0]
    
#    # remove zero-frequency component
#    pspec_series.drop(0, inplace=True)
    
    # impose cutoff frequency
    wmax = w_cutoff*max(pspec_series.index) # cutoff frequency
    pspec_output = pspec_series[-wmax:wmax] # subset of power spectrum
    
    
    return pspec_output




#-----------------------------------------
## Analytical forms for power spectra
#-----------------------------------------
    

def psd_fold(w,sigma,lam):
    return (sigma**2 / (2*np.pi))*(1/(w**2+lam**2))

def psd_hopf(w,sigma,mu,w0):
    return (sigma**2/(4*np.pi))*(1/((w+w0)**2+mu**2)+1/((w-w0)**2 +mu**2))

def psd_null(w,sigma):
    return sigma**2/(2*np.pi) * w**0



#--------------------------------
## Functions to find initalisation parameters for optimisation procedure
#–--------------------------------
    
# Function to compute the initialisation value for mu when fitting Shopf
def mu_init_fun(smax, stot, wdom):
    
    # define chunky term (use \ to continue eqn to new line)
    def alpha(smax, stot, wdom):
        return stot**3 \
        + 9*(np.pi**2)*(wdom**2)*(smax**2)*stot \
        +3*np.sqrt(3)*np.pi*np.sqrt(
                64*(np.pi**4)*(wdom**6)*(smax**6) \
                -13*(np.pi**2)*(wdom**4)*(smax**4)*(stot**2) \
                +2*(wdom**2)*(smax**2)*(stot**4) \
                )
        
    return  -(1/(3*np.pi*smax))*(stot \
             +alpha(smax,stot,wdom)**(1/3) \
             +(stot**2-12*(np.pi**2)*(wdom**2)*(smax**2))/(alpha(smax,stot,wdom)**(1/3)))
 
    
# Function to compute the initialisation value for sigma when fitting Shopf
def sigma_init_hopf_fun(smax,stot,wdom):
    return np.sqrt(
            -2*mu_init_fun(smax,stot,wdom)*stot)

    
# Function to compute the initialisation value for sigma when fitting Sfold
def sigma_init_fold_fun(smax,stot):
    return np.sqrt(
            2*stot**2/(np.pi*smax)
            )



# Function to compute the initialisation value for lambda when fitting Sfold
def lambda_init_fun(smax,stot):
    return -stot/(np.pi*smax)



#------------------------------------
## Functions to fit analytical forms to empirical power spectrum
#–------------------------------------
    
# Fold fit
def fit_fold(pspec,init):
    '''
    Input:
        pspec: power spectrum data as a Series indexed by frequency
        init: Initial parameter guesses [sigma_init, lambda_init]
        
    Output:
        List containing fitted model and AIC score
    '''
    
    
    # Put frequency values and power values as a list to use LMFIT
    freq_vals = pspec.index.tolist()
    power_vals = pspec.tolist()
    
    sigma_init, lambda_init = init
    # Assign model object
    model = Model(psd_fold)
    # Set up constraint S(wMax) < psi_fold*S(0)
    psi_fold = 0.5
    wMax = max(freq_vals)
    # Parameter constraints for sigma
    model.set_param_hint('sigma', value=sigma_init, min=0, max=10*sigma_init)
    # Parameter constraints for lambda
    model.set_param_hint('lam', min=-np.sqrt(psi_fold/(1-psi_fold))*wMax, max=0, value=lambda_init)
    
    # Assign initial parameter values and constraints
    params = model.make_params()        
    # Fit model to the empircal spectrum
    result = model.fit(power_vals, params, w=freq_vals)
    # Compute AIC score
    aic = result.aic
    
    # Export AIC score and model fit
    return [aic, result]



# Function to fit Hopf model to empirical specrum with specified initial parameter guess
def fit_hopf(pspec, init):
    '''
    Input:
        pspec: power spectrum data as a Series indexed by frequency
        init: Initial parameter guesses [sigma_init, mu_init, delta_fit]
        
    Output:
        List containing fitted model and AIC score
    '''
    
    # Put frequency values and power values as a list to use LMFIT
    freq_vals = pspec.index.tolist()
    power_vals = pspec.tolist()
    
    # Assign labels to initialisation values
    sigma_init, mu_init, w0_init = init
    
    
    # If any labels are nan, resort to default values 
    if np.isnan(sigma_init) or np.isnan(mu_init) or np.isnan(w0_init):
        sigma_init, mu_init, w0_init = [1,-0.1,1]
    
    # Constraint parameter
    psi_hopf = 0.2
    
    # Compute initialisation value for the dummy variable delta (direct map with w0)
    # It must be positive to adhere to constraint - thus if negative set to 0.
    delta_init = max(
            w0_init + (mu_init/(2*np.sqrt(psi_hopf)))*np.sqrt(4-3*psi_hopf + np.sqrt(psi_hopf**2-16*psi_hopf+16)),
            0.0001)
    

    # Assign model object 
    model = Model(psd_hopf)
    
    ## Set initialisations parameters in model attributes
    
    # Sigma must be positive, and set a (high) upper bound to avoid runaway computation
    model.set_param_hint('sigma', value=sigma_init, min=0)
    # Psi is a fixed parameter (not used in optimisation)
    model.set_param_hint('psi', value=psi_hopf, vary=False)
    # Mu must be negative 
    model.set_param_hint('mu', value=mu_init, max=0, vary=True)
    # Delta is a dummy parameter, satisfying d = w0 - wThresh (see paper for wThresh). It is allowed to vary, in place of w0.
    model.set_param_hint('delta', value = delta_init, min=0, vary=True)
    # w0 is a fixed parameter dependent on delta (w0 = delta + wThresh)
    model.set_param_hint('w0',expr='delta - (mu/(2*sqrt(psi)))*sqrt(4-3*psi + sqrt(psi**2-16*psi+16))',vary=False)
    
    # Assign initial parameter values and constraints
    params = model.make_params()        
    # Fit model to the empircal spectrum
    result = model.fit(power_vals, params, w=freq_vals)
    # Compute AIC score
    aic = result.aic
    
    # Export AIC score and model fit
    return [aic, result]



# Function to fit Null model to empirical specrum with specified initial parameter guess
def fit_null(pspec, init):
    '''
    Input:
        pspec: power spectrum data as a Series indexed by frequency
        init: Initial parameter guesses [sigma_init]
        
    Output:
        List containing fitted model and AIC score
    '''
    
    # Put frequency values and power values as a list to use LMFIT
    freq_vals = pspec.index.tolist()
    power_vals = pspec.tolist()
    
    sigma_init = init[0]
    
    # Assign model object
    model = Model(psd_null)
    
    # Initial parameter value for Null fit        
    model.set_param_hint('sigma', value=sigma_init, vary=True, min=0, max=10*sigma_init)
    
    # Assign initial parameter values and constraints
    params = model.make_params()        
    # Fit model to the empircal spectrum
    result = model.fit(power_vals, params, w=freq_vals)
    # Compute AIC score
    aic = result.aic
    
    # Export AIC score and model fit
    return [aic, result]




#-----------------------------
## Function to compute n AIC weights from n AIC scores
#–----------------------------

def aic_weights(aic_scores):
    '''
    Input:
        aic_scores: array of AIC scores
    Output:
        Array of the corresponding AIC weights
    '''
    
    # Best AIC score
    aic_best = min(aic_scores)
    
    # Differences in score from best model
    aic_diff = aic_scores - aic_best
    
    # Likelihoods for each model
    llhd = np.exp(-(1/2)*aic_diff)
    
    # Normalise to get AIC weights
    return llhd/sum(llhd)
    
    
    



#--------------------------
## pspec_metrics
#-------------------------



def pspec_metrics(pspec,
                  ews = ['smax','cf','aic'],
                  sweep = False):


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
    sweep (False) : Boolean to determine whether or not to sweep over many 
    initialisation parameters, or just use the single initialisation that 
    is derived from measured metrics (see Methods section).
                 
    Output: 
    A dictionary of spectral metrics obtained from pspec
    
    
    '''
    
    
    # Initialise a dictionary for EWS
    spec_ews = {}
    
    # Compute Smax
    if 'smax' in ews:
        smax = max(pspec)
        # add to DataFrame
        spec_ews['Smax'] = smax
        
        
        
    ## Compute the coherence factor
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
    

    ## Compute AIC weights of fitted analytical forms
    if 'aic' in ews:
        
        # Compute the empirical metrics that allow us to choose sensible initialisation parameters
        # Peak in power spectrum
        smax = pspec.max()
        # Area underneath power spectrum (~ variance)
        stot = pspec.sum()*(pspec.index[1]-pspec.index[0])
        # Dominant frequency (take positive value)
        wdom = abs(pspec.idxmax())
        
#        # Print metrics
#        print('\nSpectrum metrics [smax, stot, wdom]')
#        print([smax, stot, wdom])
        
        ## Create array of initialisation parmaeters        
        
        # Sweep values (as proportion of baseline guess) if sweep = True
        sweep_vals = np.array([0.5,1,1.5]) if sweep else np.array([1])
        
        # Baseline parameter initialisations (computed using empirical spectrum)
        # Sfold
        sigma_init_fold = sigma_init_fold_fun(smax,stot)
        lambda_init = lambda_init_fun(smax,stot)
        # Shopf
        sigma_init_hopf = sigma_init_hopf_fun(smax,stot,wdom)
        mu_init = mu_init_fun(smax,stot,wdom)
        w0_init = wdom
        # Snull
        sigma_init_null = np.sqrt(stot)
        
#        # Print the derived initialisation parameters
#        print('Initial parameter guesses for the Hopf fit [sigma, mu, w0]')
#        print([sigma_init_hopf,mu_init,w0_init])
        
        
        # Arrays of initial values
        init_fold_array = {'sigma': sweep_vals*sigma_init_fold,
                     'lambda': sweep_vals*lambda_init}
        
        init_hopf_array = {'sigma': sweep_vals*sigma_init_hopf,
                     'mu': sweep_vals*mu_init,
                     'w0': sweep_vals*w0_init}

        init_null_array = {'sigma': sweep_vals*sigma_init_null}


        ## Compute AIC values and fits
        
        ## Fold
        
        # Initialise list to store AIC and model fits
        fold_aic_fits = []

        # Sweep over initial parameter guesses and pick best convergence
        for i in range(len(init_fold_array['sigma'])):
            for j in range(len(init_fold_array['lambda'])):
                # Initial parameter guess
                init_fold = [init_fold_array['sigma'][i],init_fold_array['lambda'][j]]
                # Compute fold fit and AIC score
                [aic_temp, model_temp] = fit_fold(pspec, init_fold)
                # Store in list
                fold_aic_fits.append([aic_temp, model_temp])
        # Put list into array
        array_temp = np.array(fold_aic_fits)
        # Pick out the best model
        [aic_fold, model_fold] = array_temp[array_temp[:,0].argmin()]    
                   
        
        ## Hopf
        
        # Initialise list to store AIC and model fits
        hopf_aic_fits = []

        # Sweep over initial parameter guesses and pick best convergence
        for i in range(len(init_hopf_array['sigma'])):
            for j in range(len(init_hopf_array['mu'])):
                for k in range(len(init_hopf_array['w0'])):
                    # Initial parameter guess
                    init_hopf = [init_hopf_array['sigma'][i],init_hopf_array['mu'][j],init_hopf_array['w0'][k]]
                    # Compute fold fit and AIC score
                    [aic_temp, model_temp] = fit_hopf(pspec, init_hopf)
                    # Store in list
                    hopf_aic_fits.append([aic_temp, model_temp])
        # Put list into array
        array_temp = np.array(hopf_aic_fits)
        # Pick out the best model
        [aic_hopf, model_hopf] = array_temp[array_temp[:,0].argmin()]       
        
        
        
        
        ## Null
                
        # Initialise list to store AIC and model fits
        null_aic_fits = []

        # Sweep over initial parameter guesses and pick best convergence
        for i in range(len(init_null_array['sigma'])):
                # Initial parameter guess
                init_null = [init_null_array['sigma'][i]]
                # Compute fold fit and AIC score
                [aic_temp, model_temp] = fit_null(pspec, init_null)
                # Store in list
                null_aic_fits.append([aic_temp, model_temp])
        # Put list into array
        array_temp = np.array(null_aic_fits)
        # Pick out the best model
        [aic_null, model_null] = array_temp[array_temp[:,0].argmin()]   
       
        
        # Compute AIC weights from the AIC scores
        aicw_fold, aicw_hopf, aicw_null = aic_weights([aic_fold, aic_hopf, aic_null])
        
        
#       # Print AIC weights
#        print([aic_fold,aic_hopf,aic_null])
               
        # add to dataframe
        spec_ews['AIC fold'] = aicw_fold
        spec_ews['AIC hopf'] = aicw_hopf
        spec_ews['AIC null'] = aicw_null
        
        
        # export fitted parameter values
        spec_ews['Params fold'] = dict((k, model_fold.values[k]) for k in ('sigma','lam'))  # don't include dummy params 
        spec_ews['Params hopf'] = dict((k, model_hopf.values[k]) for k in ('sigma','mu','w0','delta','psi'))
        spec_ews['Params null'] = model_null.values

#        # Print fitted parameter values
#        print('Fitted parameter values after optimisation [sigma, mu, w0]')
#        print([model_hopf.values[k] for k in ['sigma','mu','w0']])

    # return DataFrame of metrics
    return spec_ews

    
   
    














    
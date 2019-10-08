#################################################################################################################
# ewstools
# Description: Python package for computing, analysing and visualising 
# early warning signals (EWS) in time-series data
# Author: Thomas M Bury
# Web: http://www.math.uwaterloo.ca/~tbury/
# Code repo: https://github.com/ThomasMBury/ewstools
# Documentation: https://ewstools.readthedocs.io/
#
# The MIT License (MIT)
#
# Copyright (c) 2019 Thomas Bury http://www.math.uwaterloo.ca/~tbury/
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#################################################################################################################


#---------------------------------
# Import relevant packages
#--------------------------------

# For numeric computation and DataFrames
import numpy as np
import pandas as pd

# To compute power spectrum using Welch's method
from scipy import signal

import scipy.linalg

# For fitting power spectrum models and computing AIC weights
from lmfit import Model
      


def pspec_welch(yVals,
                dt,
                ham_length=40,
                ham_offset=0.5,
                w_cutoff=1,
                scaling='spectrum'):

    '''
    Computes the power spectrum of a time-series using Welch's method.
    
    The time-series is assumed to be stationary and to have equally spaced
    measurements in time. The power spectrum is computed using Welch's method,
    which computes the power spectrum over a rolling window of subsets of the
    time-series and then takes the average.
    
    Args
    ----
    yVals: array of floats
        Array of time-series values.
    dt: float
        Seperation between data points.
    ham_length: int
        Length of Hamming window (number of data points).
    ham_offset: float
        Hamming offset as a proportion of the Hamming window size.
    w_cutoff: float
        Cutoff frequency used in power spectrum. Given as a proportion of the 
        maximum permissable frequency in the empirical
        power spectrum.
    scaling: {'spectrum', 'density'}
        Whether to compute the power spectrum ('spectrum') or
        the power spectral density ('density'). The power spectral density
        is the power spectrum normalised (such that the area underneath equals one).        
            
    Returns
    -------
    pd.Series: 
        Power values indexed by frequency
        
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
    
    # Sort into ascending frequency
    pspec_series.sort_index(inplace=True)
    
    # Append power spectrum with first value (by symmetry)
    pspec_series.at[-min(pspec_series.index)] = pspec_series.iat[0]
        
    # Impose cutoff frequency
    wmax = w_cutoff*max(pspec_series.index) # cutoff frequency
    pspec_output = pspec_series[-wmax:wmax] # subset of power spectrum
    
    
    return pspec_output






#------------Functional forms of power spectra to fit------------#
    
def psd_fold(w,sigma,lam):
	'''
	Analytical approximation for the power spectrum prior to a Fold bifurcation

	'''
	return (sigma**2 / (2*np.pi))*(1/(w**2+lam**2))
    


def psd_flip(w,sigma,r):
	'''
	Analytical approximation for the power spectrum prior to a Flip bifurcation
	'''
	return (sigma**2 / (2*np.pi))*(1/(1 + r**2 - 2*r*np.cos(w)))



def psd_hopf(w,sigma,mu,w0):
	'''
	Analytical approximation for the power spectrum prior to a Hopf bifurcation

	'''
	return (sigma**2/(4*np.pi))*(1/((w+w0)**2+mu**2)+1/((w-w0)**2 +mu**2))
     
    


def psd_null(w,sigma):
	'''
	Power spectrum of white noise (flat).
	'''
	return sigma**2/(2*np.pi) * w**0
    
    
    



#-------Obtain 'best guess' intitialisation parameters for optimisation------%


def shopf_init(smax, stot, wdom):
    '''
    Compute the 'best guess' initialisation values for sigma, mu and w0,
    when fitting sHopf to the empirical power spectrum.
    
    Args
    ----
    smax: float
        Maximum power in the power spectrum.
    stot: float
        Total power in the power spectrum.
    wdom: float
        Frequency that has the highest power.
        
    Return
    ------
    list of floats: 
        List containing the initialisation parameters [sigma, mu, w0]
        
    '''
    
    # Define chunky term (use \ to continue eqn to new line)
    def alpha(smax, stot, wdom):
        return stot**3 \
        + 9*(np.pi**2)*(wdom**2)*(smax**2)*stot \
        +3*np.sqrt(3)*np.pi*np.sqrt(
                64*(np.pi**4)*(wdom**6)*(smax**6) \
                -13*(np.pi**2)*(wdom**4)*(smax**4)*(stot**2) \
                +2*(wdom**2)*(smax**2)*(stot**4) \
                )
    
    # Initialisation for mu    
    mu = -(1/(3*np.pi*smax))*(stot \
             +alpha(smax,stot,wdom)**(1/3) \
             +(stot**2-12*(np.pi**2)*(wdom**2)*(smax**2))/(alpha(smax,stot,wdom)**(1/3)))
    
    
    # Initialisation for sigma
    sigma = np.sqrt(
            -2*mu*stot)
    
    # Initialisation for w0
    w0 = wdom
    
    # Return list
    return [sigma, mu, w0]



    
    
def sfold_init(smax, stot):
    '''
    Compute the 'best guess' initialisation values for sigma and lamda
    when fitting sfold to the empirical power spectrum.
    
    Args
    --------------
    smax: float
        Maximum power in the power spectrum.
    stot: float
        Total power in the power spectrum.
        
    Return
    -----------------
    list of floats: 
        List containing the initialisation parameters [sigma, lambda]
        
    '''
    
    # Initialisation for sigma
    sigma = np.sqrt(2*stot**2/(np.pi*smax))
    
    # Initialisation for lamda
    lamda = -stot/(np.pi*smax)

    # Return list
    return [sigma, lamda]



def sflip_init(smax, stot):
    '''
    Compute the 'best guess' initialisation values for sigma and r
    when fitting sflip to the empirical power spectrum.
    
    Args
    --------------
    smax: float
        Maximum power in the power spectrum.
    stot: float
        Total power in the power spectrum.
        
    Return
    -----------------
    list of floats: 
        List containing the initialisation parameters [sigma, r]
        
    '''
    
    
    # Initialisation for r
    r =(stot - 2*np.pi*smax)/(stot + 2*np.pi*smax)
    
    # Initialisation for sigma
    sigma = np.sqrt(stot*(1-r**2))
    
    # Return list
    return [sigma, r]



def snull_init(stot):
    '''
    Compute the 'best guess' initialisation values for sigma
    when fitting snull to the empirical power spectrum.
    
    Args
    --------------
    stot: float
        Total power in the power spectrum.
        
    Return
    -----------------
    list of floats: 
        List containing the initialisation parameters [sigma].
        
    '''
    
    # Initialisation for sigma
    sigma = np.sqrt(stot)

    # Return list
    return [sigma]





#---------Run optimisation to compute best fits-----------#
    
# Fold fit
def fit_fold(pspec, init):
    '''
    Fit the Fold power spectrum model to pspec and compute AIC score.
    Uses the package LMFIT for optimisation.
    
    Args
    --------------
    pspec: pd.Series
        Power spectrum data as a Series indexed by frequency.
    init: list of floats
        Initial parameter guesses of the form [sigma_init, lambda_init].
        
    Returns
    ----------------
    list:
        Form [aic, result] where aic is the AIC score for the model fit,
        and result is a handle that contains further information on the fit.

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





# Fold fit
def fit_flip(pspec, init):
    '''
    Fit the Flip power spectrum model to pspec and compute AIC score.
    Uses the package LMFIT for optimisation.
    
    Args
    --------------
    pspec: pd.Series
        Power spectrum data as a Series indexed by frequency.
    init: list of floats
        Initial parameter guesses of the form [sigma_init, r_init].
        
    Returns
    ----------------
    list:
        Form [aic, result] where aic is the AIC score for the model fit,
        and result is a handle that contains further information on the fit.

    '''
    
    
    # Put frequency values and power values as a list to use LMFIT
    freq_vals = pspec.index.tolist()
    power_vals = pspec.tolist()
    
    sigma_init, r_init = init
    # Assign model object
    model = Model(psd_flip)
    # Parameter constraints for sigma
    model.set_param_hint('sigma', value=sigma_init, min=0, max=10*sigma_init)
    # Parameter constraints for r
    model.set_param_hint('r', min=-1, max=0, value=r_init)
    
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
    Fit the Hopf power spectrum model to pspec and compute AIC score.
    Uses the package LMFIT for optimisation.
    
    Args
    --------------
    pspec: pd.Series
        Power spectrum data as a Series indexed by frequency
    init: list of floats
        Initial parameter guesses of the form [sigma_init, mu_init, w0_init]
        
    Returns
    ----------------
    list:
        Form [aic, result] where aic is the AIC score for the model fit,
        and result is a handle that contains further information on the fit.

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
    Fit the Null power spectrum model to pspec and compute AIC score.
    Uses the package LMFIT for optimisation.
    
    Args
    --------------
    pspec: pd.Series
        Power spectrum data as a Series indexed by frequency
    init: list of floats
        Initial parameter guesses of the form [sigma_init]
        
    Returns
    ----------------
    list:
        Form [aic, result] where aic is the AIC score for the model fit,
        and result is a handle that contains further information on the fit.

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





def aic_weights(aic_scores):
    '''
    Computes AIC weights, given AIC scores.
    
    Args
    ----------------
    aic_scores: np.array
        An array of AIC scores
            
    Returns
    -----------------
    np.array
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
    
    
    

#-----------Compute spectral metrics (EWS) from power spectrum------#


def pspec_metrics(pspec,
                  ews = ['smax','cf','aic'],
                  sweep = False):


    '''
    Compute the metrics associated with pspec that can be
    used as EWS.
    
    Args
    -------------------
    pspec: pd.Series
        Power spectrum as a Series indexed by frequency
    ews: list of {'smax', 'cf', 'aic'}
        EWS to be computed. Options include peak in the power spectrum ('smax'),
        coherence factor ('cf'), AIC weights ('aic').
    sweep: bool
        If 'True', sweep over a range of intialisation 
        parameters when optimising to compute AIC scores, at the expense of 
        longer computation. If 'False', intialisation parameter is taken as the
        'best guess'.
    
    Return
    -------------------
    dict:
        A dictionary of spectral EWS obtained from pspec
    
    '''
    
    
    # Initialise a dictionary for EWS
    spec_ews = {}
    
    ## Compute Smax
    if 'smax' in ews:
        smax = max(pspec)
        # add to DataFrame
        spec_ews['Smax'] = smax
        
        
        
    ## Compute the coherence factor
    if 'cf' in ews:
        
        # frequency at which peak occurs
        w_peak = abs(pspec.idxmax())
        
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
        
        ## Create array of initialisation parmaeters        
        
        # Sweep values (as proportion of baseline guess) if sweep = True
        sweep_vals = np.array([0.5,1,1.5]) if sweep else np.array([1])
        
        # Baseline parameter initialisations (computed using empirical spectrum)
        # Sfold
        [sigma_init_fold, lambda_init] = sfold_init(smax,stot)
        # Sflip
        [sigma_init_flip, r_init] = sflip_init(smax,stot)
        # Shopf
        [sigma_init_hopf, mu_init, w0_init] = shopf_init(smax,stot,wdom)
        # Snull
        [sigma_init_null] = snull_init(stot)
                
        
        # Arrays of initial values
        init_fold_array = {'sigma': sweep_vals*sigma_init_fold,
                     'lambda': sweep_vals*lambda_init}
        
        init_flip_array = {'sigma': sweep_vals*sigma_init_flip,
                     'r': sweep_vals*r_init}        
        
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
                   
        
        
         ## Flip
        
        # Initialise list to store AIC and model fits
        flip_aic_fits = []

        # Sweep over initial parameter guesses and pick best convergence
        for i in range(len(init_flip_array['sigma'])):
            for j in range(len(init_flip_array['r'])):
                # Initial parameter guess
                init_flip = [init_flip_array['sigma'][i],init_flip_array['r'][j]]
                # Compute fold fit and AIC score
                [aic_temp, model_temp] = fit_flip(pspec, init_flip)
                # Store in list
                flip_aic_fits.append([aic_temp, model_temp])
        # Put list into array
        array_temp = np.array(flip_aic_fits)
        # Pick out the best model
        [aic_flip, model_flip] = array_temp[array_temp[:,0].argmin()]           
        
        
        
        
        
        
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
        aicw_fold, aicw_flip, aicw_hopf, aicw_null = aic_weights([aic_fold, aic_flip, aic_hopf, aic_null])
        
               
        # Add to Dataframe
        spec_ews['AIC fold'] = aicw_fold
        spec_ews['AIC flip'] = aicw_flip
        spec_ews['AIC hopf'] = aicw_hopf
        spec_ews['AIC null'] = aicw_null
        
        
        # Add fitted parameter values to DataFrame
        spec_ews['Params fold'] = dict((k, model_fold.values[k]) for k in ('sigma','lam'))  # don't include dummy params
        spec_ews['Params flip'] = dict((k, model_flip.values[k]) for k in ('sigma','r'))
        spec_ews['Params hopf'] = dict((k, model_hopf.values[k]) for k in ('sigma','mu','w0','delta','psi'))
        spec_ews['Params null'] = model_null.values


    # Return DataFrame of metrics
    return spec_ews

    





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
    and outputs the eigenvalues, eigenvectors and jacobian.
    Input:
        df_in: DataFrame with two columns indexed by time
    Output:
    	dictionary consisting of
    		- 'Eigenvalues': np.array of eigenvalues
    		- 'Eigenvectors': np.array of eigenvectors
    		- 'Jacobian': pd.DataFrame of Jacobian entries
    '''
    
    # Get the time-separation between data points
    dt = df_in.index[1] -df_in.index[0]
    
    # Compute autocovaraince matrix from columns
    ar_autocov = compute_autocov(df_in)
    
    # Compute the covariance matrix (built in function)
    ar_cov = df_in.cov()
    
    # Estimate of discrete Jacobian (formula in Williamson (2015))
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







   


    
    
    

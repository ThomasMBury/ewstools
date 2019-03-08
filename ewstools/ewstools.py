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

# For detrending time-series
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.ndimage.filters import gaussian_filter as gf

# For fitting power spectrum models and computing AIC weights
from lmfit import Model


# Demo function
def convert(my_name):
    """
    Print a line about converting a notebook.
    Args:
        my_name (str): person's name
    Returns:
        None
    """

    print("I'll convert a notebook for you some day, %s" % (my_name))


#------------------------------
# Main functions
#–-----------------------------

def ews_compute(raw_series,
            roll_window=0.4,
            smooth='Lowess',
            span=0.1,
            band_width=0.2,
            upto='Full',
            ews=['var','ac'], 
            lag_times=[1],
            ham_length=40,
            ham_offset=0.5,
            pspec_roll_offset=20,
            w_cutoff=1,
            sweep=False):
    '''
    Compute temporal and spectral EWS from time-series data.  
    
    Args
    --------
    raw_series: pd.Series
        Time-series data to analyse. Indexed by time.
    roll_window: float (0.4)
        Rolling window size as a proportion of the length of the time-series 
        data.
    smooth: str ('Lowess')
        Detrending options including
            'Gaussian' : Gaussian smoothing
            'Lowess'   : Lowess smoothing
            'None'     : No detrending
    band_width: float (0.2)
        Bandwidth of Gaussian kernel. Taken as a proportion of time-series length if in (0,1), 
            otherwise taken as absolute.
    span: float (0.1)
        Span of time-series data used for Lowess filtering. Taken as a 
        proportion of time-series length if in (0,1), otherwise taken as 
        absolute.
    upto: int or str ('Full')
        Time up to which EWS are computed. Enter 'Full' to use
        the entire time-series. Otherwise enter a time value.
    ews: list of str (['var', 'ac'])
        List of EWS to compute. Options include
             'var'   : Variance
             'ac'    : Autocorrelation
             'sd'    : Standard deviation
             'cv'    : Coefficient of variation
             'skew'  : Skewness
             'kurt'  : Kurtosis
             'smax'  : Peak in the power spectrum
             'cf'    : Coherence factor
             'aic'   : AIC weights
    lag_times: list of int ([1])
        List of lag times at which to compute autocorrelation.
    ham_length: int (40)
        Length of the Hamming window used to compute the power spectrum.
    ham_offset: float (0.5)
        Hamming offset as a proportion of the Hamming window size.
    pspec_roll_offset: int (20)
        Rolling window offset used when computing power spectra. Power spectrum 
        computation is relatively expensive so this is rarely taken as 1 
        (as is the case for the other EWS).
    w_cutoff: float (1)
        Cutoff frequency used in power spectrum. Given as a proportion of the 
        maximum permissable frequency in the empirical power spectrum.
    sweep: bool ('False')
        Choose 'True' to sweep over a range of intialisation 
        parameters when optimising to compute AIC scores, at the expense of 
        longer computation. Otherwise intialisation parameter is taken as the
        'best guess'.
    
    
    Returns
    ------------
    dict: 
        A dictionary with three components.
            'EWS metrics': pd.DataFrame
                A pandas DataFrame indexed by time with columns corresopnding 
            to each EWS.
            'Power spectrum': pd.DataFrame
                A DataFrame of the measured power spectra and the best fits used 
            to give the AIC weights. Indexed by time.
            'Kendall tau': pd.DataFrame
                A DataFrame of the Kendall tau values for each EWS metric.
    
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
    
    
    
    #------------ Compute temporal EWS---------------#
        
    # Compute standard deviation as a Series and add to the DataFrame
    if 'sd' in ews:
        roll_sd = eval_series.rolling(window=rw_size).std()
        df_ews['Standard deviation'] = roll_sd
    
    # Compute variance as a Series and add to the DataFrame
    if 'var' in ews:
        roll_var = eval_series.rolling(window=rw_size).var()
        df_ews['Variance'] = roll_var
    
    # Compute autocorrelation for each lag in lag_times and add to the DataFrame   
    if 'ac' in ews:
        for i in range(len(lag_times)):
            roll_ac = eval_series.rolling(window=rw_size).apply(
        func=lambda x: pd.Series(x).autocorr(lag=lag_times[i]),
        raw=True)
            df_ews['Lag-'+str(lag_times[i])+' AC'] = roll_ac

            
    # Compute Coefficient of Variation (C.V) and add to the DataFrame
    if 'cv' in ews:
        # mean of raw_series
        roll_mean = raw_series.rolling(window=rw_size).mean()
        # standard deviation of residuals
        roll_std = eval_series.rolling(window=rw_size).std()
        # coefficient of variation
        roll_cv = roll_std.divide(roll_mean)
        df_ews['Coefficient of variation'] = roll_cv

    # Compute skewness and add to the DataFrame
    if 'skew' in ews:
        roll_skew = eval_series.rolling(window=rw_size).skew()
        df_ews['Skewness'] = roll_skew

    # Compute Kurtosis and add to DataFrame
    if 'kurt' in ews:
        roll_kurt = eval_series.rolling(window=rw_size).kurt()
        df_ews['Kurtosis'] = roll_kurt




    
    #------------Compute spectral EWS-------------#
    
    ''' In this section we compute newly proposed EWS based on the power spectrum
        of the time-series computed over a rolling window '''
    
   
    # If any of the spectral metrics are listed in the ews vector:
    if 'smax' in ews or 'cf' in ews or 'aic' in ews:

        
        # Number of components in the residual time-series
        num_comps = len(eval_series)
        # Rolling window offset (can make larger to save on computation time)
        roll_offset = int(pspec_roll_offset)
        # Time separation between data points (need for frequency values of power spectrum)
        dt = eval_series.index[1]-eval_series.index[0]
        
        # Initialise a list for the spectral EWS
        list_metrics_append = []
        # Initialise a list for the power spectra
        list_spec_append = []
        
        # Loop through window locations shifted by roll_offset
        for k in np.arange(0, num_comps-(rw_size-1), roll_offset):
            
            # Select subset of series contained in window
            window_series = eval_series.iloc[k:k+rw_size]           
            # Asisgn the time value for the metrics (right end point of window)
            t_point = eval_series.index[k+(rw_size-1)]            
            
            ## Compute the power spectrum using function pspec_welch
            pspec = pspec_welch(window_series, dt, 
                                ham_length=ham_length, 
                                ham_offset=ham_offset,
                                w_cutoff=w_cutoff,
                                scaling='spectrum')
            
            
            ## Compute the spectral EWS using pspec_metrics (dictionary)
            metrics = pspec_metrics(pspec, ews, sweep)
            # Add the time-stamp
            metrics['Time'] = t_point
            # Add metrics (dictionary) to the list
            list_metrics_append.append(metrics)
            
            
            if 'aic' in ews:
                
                ## Obtain power spectrum fits as an array for plotting
                # Create fine-scale frequency values
                wVals = np.linspace(min(pspec.index), max(pspec.index), 100)
                # Fold fit
                pspec_fold = psd_fold(wVals, metrics['Params fold']['sigma'],
                     metrics['Params fold']['lam'])
                # Hopf fit
                pspec_hopf = psd_hopf(wVals, metrics['Params hopf']['sigma'],
                     metrics['Params hopf']['mu'],
                     metrics['Params hopf']['w0'])
                # Null fit
                pspec_null = psd_null(wVals, metrics['Params null']['sigma'])
                
                ## Put spectrum fits into a dataframe
                dic_temp = {'Time': t_point*np.ones(len(wVals)), 
                            'Frequency': wVals,
                            'Fit fold': pspec_fold,
                            'Fit hopf': pspec_hopf, 
                            'Fit null': pspec_null}
                df_pspec_fits = pd.DataFrame(dic_temp)
                # Set the multi-index
                df_pspec_fits.set_index(['Time','Frequency'], inplace=True)
                            
                ## Put empirical power spectrum and fits into the same DataFrames
                # Put empirical power spectrum into a DataFrame and remove indexes         
                df_pspec_empirical = pspec.to_frame().reset_index()
                # Rename column
                df_pspec_empirical.rename(columns={'Power spectrum': 'Empirical'}, inplace=True)
                # Include a column for the time-stamp
                df_pspec_empirical['Time'] = t_point*np.ones(len(pspec))
                # Use a multi-index of ['Time','Frequency']
                df_pspec_empirical.set_index(['Time', 'Frequency'], inplace=True)
                # Concatenate the empirical spectrum and the fits into one DataFrame
                df_pspec_temp = pd.concat([df_pspec_empirical, df_pspec_fits], axis=1)
                # Add spectrum DataFrame to the list  
                list_spec_append.append(df_pspec_temp)
            
            
                 
        # Concatenate the list of power spectra DataFrames to form a single DataFrame
        df_pspec = pd.concat(list_spec_append) if 'aic' in ews else pd.DataFrame()
        
        # Create a DataFrame out of the multiple dictionaries consisting of the spectral metrics
        df_spec_metrics = pd.DataFrame(list_metrics_append)
        df_spec_metrics.set_index('Time', inplace=True)

        
        # Join the spectral EWS DataFrame to the main EWS DataFrame 
        df_ews = df_ews.join(df_spec_metrics)
        
        # Include Smax normalised by Variance
        df_ews['Smax/Var'] = df_ews['Smax']/df_ews['Variance']
        
        
    
    #------------Compute Kendall tau coefficients----------------#
    
    ''' In this section we compute the kendall correlation coefficients for each EWS
        with respect to time. Values close to one indicate high correlation (i.e. EWS
        increasing with time), values close to zero indicate no significant correlation,
        and values close to negative one indicate high negative correlation (i.e. EWS
        decreasing with time).'''
        
                                                                             
    # Put time values as their own series for correlation computation
    time_vals = pd.Series(df_ews.index, index=df_ews.index)

    # List of EWS that can be used for Kendall tau computation
    ktau_metrics = ['Variance','Standard deviation','Kurtosis','Coefficient of variation','Smax','Smax/Var'] + ['Lag-'+str(i)+' AC' for i in lag_times]
    # Find intersection with this list and EWS computed
    ews_list = df_ews.columns.values.tolist()
    ktau_metrics = list( set(ews_list) & set(ktau_metrics) )
    
    # Find Kendall tau for each EWS and store in a DataFrame
    dic_ktau = {x:df_ews[x].corr(time_vals, method='kendall') for x in ktau_metrics} # temporary dictionary
    df_ktau = pd.DataFrame(dic_ktau, index=[0]) # DataFrame (easier for concatenation purposes)
                                                                             
                                                                             
 
    #-------------Organise final output and return--------------#
       
    # Ouptut a dictionary containing EWS DataFrame, power spectra DataFrame, and Kendall tau values
    output_dic = {'EWS metrics': df_ews, 'Kendall tau': df_ktau}
    
    # Add df_pspec to dictionary if it was computed
    if 'smax' in ews or 'cf' in ews or 'aic' in ews:
        output_dic['Power spectrum'] = df_pspec
        
    return output_dic




#------------------------------
# Helper functions
#–-----------------------------


def pspec_welch(yVals,
                dt,
                ham_length=40,
                ham_offset=0.5,
                w_cutoff=1,
                scaling='spectrum'):


    '''
    Compute the power spectrum of yVals using Welch's method.
    This involves computing the periodogram with overlapping Hamming windows.
    
    Args
    ------------------
    yVals: array of float
        Time-series values
    dt: float
        Seperation between time-series points
    ham_length: int (40)
        Length of Hamming window (number of data points).
    ham_offset: float (0.5)
        Hamming offest as a proportion of the Hamming window size.
    w_cutoff: float (1)
        Cutoff frequency used in power spectrum. Given as a proportion of the 
        maximum permissable frequency in the empirical
        power spectrum.
    scaling: str ('spectrum')
        Choose between
            'spectrum' : computes the power spectrum
            'density'  : computes the power spectral density, which is
            normalised (area underneath =1).          
            
    Returns
    --------------------
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
    return (sigma**2 / (2*np.pi))*(1/(w**2+lam**2))

def psd_hopf(w,sigma,mu,w0):
    return (sigma**2/(4*np.pi))*(1/((w+w0)**2+mu**2)+1/((w-w0)**2 +mu**2))

def psd_null(w,sigma):
    return sigma**2/(2*np.pi) * w**0
    
    
    



#-------Obtain 'best guess' intitialisation parameters for optimisation------%


def shopf_init(smax, stot, wdom):
    '''
    Compute the 'best guess' initialisation values for sigma, mu and w0,
    when fitting sHopf to the empirical power spectrum.
    
    Args
    --------------
    smax: float
        Maximum power in the power spectrum.
    stot: float
        Total power in the power spectrum.
    wdom: float
        Frequency that has the highest power.
        
    Return
    -----------------
    list of float: 
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
    list of float: 
        List containing the initialisation parameters [sigma, lambda]
        
    '''
    
    # Initialisation for sigma
    sigma = np.sqrt(2*stot**2/(np.pi*smax))
    
    # Initialisation for lamda
    lamda = -stot/(np.pi*smax)

    # Return list
    return [sigma, lamda]



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
    list of float: 
        List containing the initialisation parameters [sigma]
        
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
    Uses LMFIT
    
    Args
    --------------
    pspec: pd.Series
        Power spectrum data as a Series indexed by frequency
    init: list of float
        Initial parameter guesses of the form [sigma_init, lambda_init]
        
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



# Function to fit Hopf model to empirical specrum with specified initial parameter guess
def fit_hopf(pspec, init):    
    
    '''
    Fit the Hopf power spectrum model to pspec and compute AIC score.
    Uses LMFIT
    
    Args
    --------------
    pspec: pd.Series
        Power spectrum data as a Series indexed by frequency
    init: list of float
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
    Uses LMFIT
    
    Args
    --------------
    pspec: pd.Series
        Power spectrum data as a Series indexed by frequency
    init: list of float
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




#-----------------------------
## Function to compute AIC weights from AIC scores
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
        
        ## Create array of initialisation parmaeters        
        
        # Sweep values (as proportion of baseline guess) if sweep = True
        sweep_vals = np.array([0.5,1,1.5]) if sweep else np.array([1])
        
        # Baseline parameter initialisations (computed using empirical spectrum)
        # Sfold
        [sigma_init_fold, lambda_init] = sfold_init(smax,stot)
        # Shopf
        [sigma_init_hopf, mu_init, w0_init] = shopf_init(smax,stot,wdom)
        # Snull
        [sigma_init_null] = snull_init(stot)
                
        
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

    
   


    
    
    

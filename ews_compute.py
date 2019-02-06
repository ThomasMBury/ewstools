#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 19:11:58 2018

@author: Thomas Bury

Function to compute EWS from time-series data

"""

# Import Python modules
import numpy as np
import pandas as pd
from scipy.ndimage.filters import gaussian_filter as gf


# Import local module
from ews_spec import pspec_welch, pspec_metrics, psd_fold, psd_hopf, psd_null



def ews_compute(raw_series,
            roll_window=0.25,
            smooth=True,
            upto='Full',
            band_width=0.2,
            ews=['var','ac'], 
            lag_times=[1],
            ham_length=40,
            ham_offset=0.5,
            pspec_roll_offset=20, # generally ham_length*ham_offset
            w_cutoff=1,
            sweep=False):
    '''
    Function to compute EWS from time-series data.   
    
    Inputs (default value):
    raw_series : pandas Series indexed by time 
    roll_windopw (0.25) : size of the rolling window (as a proportion
    of the length of the data)
    smooth (True) : if True, series data is detrended with a Gaussian kernel
    upto ('Full') : if 'Full', use entire time-series, ow input time up to which EWS are to be evaluated
    band_width (0.2) : bandwidth of Gaussian kernel (taken as a proportion if in (0,1), otherwise taken as absolute)
    ews (['var,'ac'] : list of strings corresponding to the desired EWS.
         Options include
             'var'   : Variance
             'ac'    : Autocorrelation
             'sd'    : Standard deviation
             'cv'    : Coefficient of variation
             'skew'  : Skewness
             'kurt'  : Kurtosis
             'smax'  : Peak in the power spectrum
             'cf'    : Coherence factor
             'aic'   : AIC weights
             
    lag_times ([1]) : list of integers corresponding to the desired lag times for AC
    ham_length (40) : length of the Hamming window
    ham_offset (0.5) : proportion of Hamimng window to offset by
    pspec_roll_offset (20) : offset of rolling window for pspec computation
    w_cutoff (1) : cutoff frequency (as proportion of size of maximum frequency)
    sweep (False) : whether to sweep over a range of intialisation parameters
                    when optimising to compute AIC scores (longer computation)
    
    Outputs: Dictionary
    'EWS metrics': DataFrame indexed by time with columns csp to each EWS
    'Power spectrum': DataFrame of power spectra and the model fits indexed by time
    'Kendall tau': DataFrame of the kendall tau values for each EWS metric (those that are expected to increase)
    
    '''
    
    # Initialise a DataFrame to store EWS data - indexed by time
    df_ews = pd.DataFrame(raw_series)
    df_ews.columns = ['State variable']
    df_ews.index.rename('Time', inplace=True)
    
    # Select portion of data where EWS are evaluated (e.g only up to bifurcation)
    if upto == 'Full':
        short_series = raw_series
    else: short_series = raw_series.loc[:upto]

    #------------------------------
    ## Data detrending
    #–------------------------------
    
    # Compute the absolute size of the bandwidth if it is given as a proportion
    if 0 < band_width <= 1:
        bw_size = short_series.shape[0]*band_width
    else:
        bw_size = band_width
    
    # Compute smoothed data and residuals if smooth=True.
    if smooth:
        smooth_data = gf(short_series.values, sigma=bw_size, mode='reflect')
        smooth_series = pd.Series(smooth_data, index=short_series.index)
        residuals = short_series.values - smooth_data
        resid_series = pd.Series(residuals,index=short_series.index)
    
        # Add smoothed data and residuals to the EWS DataFrame
        df_ews['Smoothing'] = smooth_series
        df_ews['Residuals'] = resid_series
        
    # Use the residuals for EWS if smooth=True. Otherwise use the raw series
    eval_series = resid_series if smooth else short_series
    
    # Compute the rolling window size (integer value)
    rw_size=int(np.floor(roll_window * raw_series.shape[0]))
    
    
    
    #----------------------------
    ## Compute temporal EWS
    #-----------------------------  
        
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





    
    #--------------------------
    ## Compute spectral EWS
    #--------------------------
    
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
        df_pspec = pd.concat(list_spec_append) if 'aic' in ews else 0
        
        # Create a DataFrame out of the multiple dictionaries consisting of the spectral metrics
        df_spec_metrics = pd.DataFrame(list_metrics_append)
        df_spec_metrics.set_index('Time', inplace=True)

        
        # Join the spectral EWS DataFrame to the main EWS DataFrame 
        df_ews = df_ews.join(df_spec_metrics)
        
        # Include Smax normalised by Variance
        df_ews['Smax/Var'] = df_ews['Smax']/df_ews['Variance']
        
        
    
    #--------------------------
    ## Compute kendall tau values
    #--------------------------
    
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
                                                                             
                                                                             
 
    #----------------------
    ## Final output
    #–----------------------
       
    # Ouptut a dictionary containing EWS DataFrame, power spectra DataFrame, and Kendall tau values
    output_dic = {'EWS metrics': df_ews, 'Power spectrum': df_pspec, 'Kendall tau': df_ktau}
    
    return output_dic












    
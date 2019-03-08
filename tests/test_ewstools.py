"""
Tests for `ewstools` package.
---------------
"""


import pytest
import numpy as np
import pandas as pd


from ewstools import ewstools


def test_ews_compute():
    '''
    Run a time-series through ews_compute and check everything is
    functioning correctly.
    '''
    # Simulate a simple time-series
    tVals = np.arange(0,10,0.1)
    xVals = 5 + np.random.normal(0,1,len(tVals)) 
    series = pd.Series(xVals, index=tVals)
    
    # Run through ews_compute with all possible EWS
    ews = ['var','ac','sd','cv','skew','kurt','smax','cf','aic']
    lag_times = [1,2,3,4,5]
    dict_ews = ewstools.ews_compute(series,
                             ews=ews,
                             lag_times=lag_times,
                             sweep = True
                             )
    
    assert type(dict_ews) == dict
    
    # Obtain components of dict_ews
    df_ews = dict_ews['EWS metrics']
    df_pspec = dict_ews['Power spectrum']
    df_ktau = dict_ews['Kendall tau']
    
    # Check types
    assert type(df_ews) == pd.DataFrame
    assert type(df_pspec) == pd.DataFrame
    assert type(df_ktau) == pd.DataFrame
    
    # Check index
    assert df_ews.index.name == 'Time'
    assert df_pspec.index.names == ['Time','Frequency']    
    
  
    
    
def test_pspec_welch():
    n_points = 100
    dt = 1
    ham_length = 40
    yVals = np.random.normal(0,1,n_points)
    pspec = ewstools.pspec_welch(yVals, dt, ham_length=ham_length)
    
    assert type(pspec) == pd.Series
    assert pspec.shape in [(n_points,),
                           (n_points+1,),
                           (ham_length,),
                           (ham_length+1,)
                           ]
    

def test_psd_forms():
    wVals = np.arange(-3,3,0.1)
    sigma = 0.01
    lamda = -0.1
    mu = -0.1
    w0 = 1
    sFoldVals = ewstools.psd_fold(wVals, sigma, lamda)
    sHopfVals = ewstools.psd_hopf(wVals, sigma, mu, w0)
    sNullVals = ewstools.psd_null(wVals, sigma)
        
    assert type(sFoldVals)==np.ndarray
    assert type(sHopfVals)==np.ndarray
    assert type(sNullVals)==np.ndarray
    


def test_sfold_init():
    '''
    Check that initialisation parameters are computed correctly by plugging
    them into known expressions.
    '''
    stot = 1
    smax = 0.5
    
    [sigma, lamda] = ewstools.sfold_init(smax, stot)
    
    # Values that smax, stot should attain (+/- 1dp)
    smax_assert = sigma**2/(2*np.pi*lamda**2)
    stot_assert = -sigma**2/(2*lamda)
    
    assert smax_assert*(0.99) <= smax <= smax_assert*(1.01)   
    assert stot_assert*(0.99) <= stot <= stot_assert*(1.01)
    
    
def test_shopf_init():
    '''
    Check that initialisation parameters are computed correctly by plugging
    them into known expressions.
    '''
    smax = 0.5
    stot = 1
    wdom = 1
    
    [sigma, mu, w0] = ewstools.shopf_init(smax, stot, wdom)
    
    # Values that smax, stot should attain (+/- 1dp)
    smax_assert = (sigma**2/(4*np.pi*mu**2))*(1+(mu**2/(mu**2+4*w0**2)))
    stot_assert = -sigma**2/(2*mu)
    wdom_assert = w0
    
    assert smax_assert*(0.99) <= smax <= smax_assert*(1.01)   
    assert stot_assert*(0.99) <= stot <= stot_assert*(1.01)
    assert wdom_assert*(0.99) <= wdom <= wdom_assert*(1.01)

    

def test_snull_init():
    '''
    Check that initialisation parameters are computed correctly by plugging
    them into known expressions.
    '''
    stot = 1
    
    [sigma] = ewstools.snull_init(stot)
    
    # Values that smax, stot should attain (+/- 1dp)
    stot_assert = sigma**2

    assert stot_assert*(0.99) <= stot <= stot_assert*(1.01)
    
    

def test_fit_fold():
    '''
    Run a power spectrum through the fitting procedure of the 'fold'
    power spectrum form.
    '''
    
    # Create a power spectrum
    n_points = 100
    dt = 1
    ham_length = 40
    yVals = np.random.normal(0,1,n_points)
    pspec = ewstools.pspec_welch(yVals, dt, ham_length=ham_length)
    
    sigma_init = 0.05
    lambda_init = -0.1
    init = [sigma_init, lambda_init]
    # Run power spectrum in fit_fold
    [aic, model] = ewstools.fit_fold(pspec, init)
    
    assert type(aic) == np.float64
    assert type(model.values) == dict
    
    
def test_fit_hopf():
    '''
    Run a power spectrum through the fitting procedure of the 'hopf'
    power spectrum form.
    '''
    
    # Create a power spectrum
    n_points = 100
    dt = 1
    ham_length = 40
    yVals = np.random.normal(0,1,n_points)
    pspec = ewstools.pspec_welch(yVals, dt, ham_length=ham_length)
    
    sigma_init = 0.05
    mu_init = -0.1
    w0_init = 1
    init = [sigma_init, mu_init, w0_init]
    # Run power spectrum in fit_hopf
    [aic, model] = ewstools.fit_hopf(pspec, init)
    
    assert type(aic) == np.float64
    assert type(model.values) == dict
    

def test_fit_null():
    '''
    Run a power spectrum through the fitting procedure of the 'null'
    power spectrum form.
    '''
    
    # Create a power spectrum
    n_points = 100
    dt = 1
    ham_length = 40
    yVals = np.random.normal(0,1,n_points)
    pspec = ewstools.pspec_welch(yVals, dt, ham_length=ham_length)
    
    sigma_init = 0.05
    init = [sigma_init]
    # Run power spectrum in fit_null
    [aic, model] = ewstools.fit_null(pspec, init)
    
    assert type(aic) == np.float64
    assert type(model.values) == dict
    
    
def test_aic_weights():
    '''
    Run AIC scores through the conversion to AIC weights
    '''
    aic_scores = np.array([-231,-500,-100,5])
    
    aic_weights = ewstools.aic_weights(aic_scores)
    
    assert type(aic_weights) == np.ndarray
    
    
def test_pspec_metrics():
    '''
    Run a power spectrum through pspec_metrics, and check that the spectral
    EWS have the intended format
    '''
    # Create a power spectrum
    n_points = 100
    dt = 1
    ham_length = 40
    yVals = np.random.normal(0,1,n_points)
    pspec = ewstools.pspec_welch(yVals, dt, ham_length=ham_length)
    
    # Run power spectrum in pspec_metrics
    spec_ews = ewstools.pspec_metrics(pspec,
                      ews=['smax','cf','aic'],
                      sweep=True)

    assert type(spec_ews) == dict
    

    




    
    
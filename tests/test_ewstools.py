"""
Tests for `ewstools` package.
---------------
"""


import pytest
import numpy as np
import pandas as pd


# Import ewstools
from ewstools import core
from ewstools import helpers

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
    ews = ['var','ac','sd','cv','skew','kurt','smax','aic','cf','smax/var','smax/mean']
    aic=['Fold','Hopf','Null','Flip']
    lag_times = [1,2,3,4,5]
    dict_ews = core.ews_compute(series,
                             ews=ews,
                             aic=aic,
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
    pspec = helpers.pspec_welch(yVals, dt, ham_length=ham_length)
    
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
    sFoldVals = helpers.psd_fold(wVals, sigma, lamda)
    sHopfVals = helpers.psd_hopf(wVals, sigma, mu, w0)
    sNullVals = helpers.psd_null(wVals, sigma)
        
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
    
    [sigma, lamda] = helpers.sfold_init(smax, stot)
    
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
    
    [sigma, mu, w0] = helpers.shopf_init(smax, stot, wdom)
    
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
    
    [sigma] = helpers.snull_init(stot)
    
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
    pspec = helpers.pspec_welch(yVals, dt, ham_length=ham_length)
    
    sigma_init = 0.05
    lambda_init = -0.1
    init = [sigma_init, lambda_init]
    # Run power spectrum in fit_fold
    [aic, model] = helpers.fit_fold(pspec, init)
    
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
    pspec = helpers.pspec_welch(yVals, dt, ham_length=ham_length)
    
    sigma_init = 0.05
    mu_init = -0.1
    w0_init = 1
    init = [sigma_init, mu_init, w0_init]
    # Run power spectrum in fit_hopf
    [aic, model] = helpers.fit_hopf(pspec, init)
    
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
    pspec = helpers.pspec_welch(yVals, dt, ham_length=ham_length)
    
    sigma_init = 0.05
    init = [sigma_init]
    # Run power spectrum in fit_null
    [aic, model] = helpers.fit_null(pspec, init)
    
    assert type(aic) == np.float64
    assert type(model.values) == dict
    
    
def test_aic_weights():
    '''
    Run AIC scores through the conversion to AIC weights
    '''
    aic_scores = np.array([-231,-500,-100,5])
    
    aic_weights = helpers.aic_weights(aic_scores)
    
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
    pspec = helpers.pspec_welch(yVals, dt, ham_length=ham_length)
    
    # Run power spectrum in pspec_metrics
    spec_ews = helpers.pspec_metrics(pspec,
                      ews=['smax','cf','aic'],
                      sweep=True)

    assert type(spec_ews) == dict
    
   
    
def test_block_bootstrap():
    '''
    Run a time-series through block_bootstrap and check that it produces
	sensible output
    '''
    # Simulate a simple time-series
    tVals = np.arange(0,10,0.1)
    xVals = 5 + np.random.normal(0,1,len(tVals))
    series = pd.Series(xVals, index=tVals) 
    
    # Bootstrap params
    n_samples = 2
    block_size = 10
    bs_types = ['Stationary', 'Circular']
    # Run through block_bootstrap
    for bs_type in bs_types:
        samples = core.block_bootstrap(series,
                                       n_samples,
                                       bs_type=bs_type,
                                       block_size=block_size
                                       )
    assert type(samples) == pd.DataFrame
    assert samples.shape == (n_samples*len(tVals),1) 
	

    

def test_roll_bootstrap():
    '''
    Run a non-stationary time-series through roll_bootstrap and check that it 
    produces sensible output
    '''
    # Simulate a simple time-series
    tVals = np.arange(0,10,0.1)
    xVals = 5*tVals + np.random.normal(0,1,len(tVals))
    series = pd.Series(xVals, index=tVals)

    # Parameters
    n_samples = 2
    
    # Run function
    df_bootstrap = core.roll_bootstrap(series,
                                           n_samples=n_samples)

    assert type(df_bootstrap) == pd.DataFrame
    assert df_bootstrap.index.names==['Time','Sample','Wintime']




def test_mean_ci():
    '''
    Run mean_ci with a dataset, and check that it computes confidence intervlas
    '''
    # Generate data as a pandas Series
    data = pd.Series(np.random.normal(loc=0,scale=1,size=100))
    
    # Compute confidence intevals
    intervals = core.mean_ci(data)
    
    assert type(intervals) == dict
    assert type(intervals['Mean']) == np.float64 or type(intervals['Mean']) ==float


    
    
    
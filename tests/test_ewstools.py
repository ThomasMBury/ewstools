"""
Tests for `ewstools` package.
---------------
"""


import pytest

# For numeric computation and DataFrames
import numpy as np
import pandas as pd


from ewstools import ewstools as ews


def test_convert(capsys):
    """Correct my_name argument prints"""
    ews.convert("Jill")
    captured = capsys.readouterr()
    assert "Jill" in captured.out
    
    
def test_pspec_welch():
    n_points = 100
    dt = 1
    ham_length = 40
    yVals = np.random.normal(0,1,n_points)
    pspec = ews.pspec_welch(yVals, dt)
    
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
    sFoldVals = ews.psd_fold(wVals, sigma, lamda)
    sHopfVals = ews.psd_hopf(wVals, sigma, mu, w0)
    sNullVals = ews.psd_null(wVals, sigma)
        
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
    
    [sigma, lamda] = ews.sfold_init(smax, stot)
    
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
    
    [sigma, mu, w0] = ews.shopf_init(smax, stot, wdom)
    
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
    
    [sigma] = ews.snull_init(stot)
    
    # Values that smax, stot should attain (+/- 1dp)
    stot_assert = sigma**2

    assert stot_assert*(0.99) <= stot <= stot_assert*(1.01)
    
    

  





    
    
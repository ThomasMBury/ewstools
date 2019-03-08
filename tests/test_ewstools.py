"""Tests for `ewstools` package."""
import pytest

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






    
    
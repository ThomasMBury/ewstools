#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 23 21:12:35 2019

@author: tbury


Test for eval_recon function

"""


import pytest
import numpy as np
import pandas as pd

# Import ewstools
from ewstools import core
from ewstools import helpers

# Simulate a simple multi-variate time series
tVals = np.arange(0,10,0.1)
xVals = 5 + np.random.normal(0,1,len(tVals))
yVals = 10 + np.random.normal(0,1,len(tVals))
zVals = 0 + np.random.normal(0,1,len(tVals))

df_test = pd.DataFrame({'x':xVals,'y':yVals,'z':zVals}, index=tVals)

n = len(df_test.columns)



#------------------------------------
# Write test functions

def test_compute_autocov():

    ar_out = helpers.compute_autocov(df_test)
    assert type(ar_out) == np.ndarray
    assert ar_out.shape == (n,n)


def test_eval_recon():
    
    dic_out = helpers.eval_recon(df_test)
    jac = dic_out['Jacobian']
    evals = dic_out['Eigenvalues']
    evecs = dic_out['Eigenvectors']
    
    assert type(dic_out) == dict
    assert type(jac) == pd.DataFrame
    assert type(evals) == np.ndarray
    assert type(evecs) == np.ndarray
    assert jac.shape == (n,n)
    
def test_eval_recon_rolling():
    
    df_out = core.eval_recon_rolling(df_test)
    
    assert type(df_out) == pd.DataFrame
    assert len(df_out) == len(df_test)
    



























#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for `ewstools.models` module
---------------
"""


import pytest
import numpy as np
import pandas as pd

import ewstools


def test_simulate_ricker():
    '''
    Test the simulate_ricker function
    '''
    
    # Test with fixed F
    series = ewstools.models.simulate_ricker(F=0)
    assert type(series) == pd.Series
    assert series.index.name == 'time'


    # Test with variable F
    series = ewstools.models.simulate_ricker(F=[0,2.7])
    assert type(series) == pd.Series
    assert series.index.name == 'time'
    
    # Test with variable r
    series = ewstools.models.simulate_ricker(r=[1,2])
    assert type(series) == pd.Series
    assert series.index.name == 'time'
    
    # Test with large noise to make values <=0
    series = ewstools.models.simulate_ricker(sigma=1)
    assert type(series) == pd.Series
    assert series.index.name == 'time'
    
    
    
def test_simulate_rosen_mac():
    '''
    Test the simulate_rosen_mac function
    '''
    
    # Test with fixed a
    df = ewstools.models.simulate_rosen_mac(a=12)
    assert type(df) == pd.DataFrame
    assert df.index.name == 'time'
    assert 'x' in df.columns
    assert 'y' in df.columns
    
    # Test with variable a
    df = ewstools.models.simulate_rosen_mac(a=[12,16])
    assert type(df) == pd.DataFrame

    # Test with large noise
    df = ewstools.models.simulate_rosen_mac(sigma_x=0.1, sigma_y=0.1)
    assert type(df) == pd.DataFrame




def test_simulate_may():
    '''
    Test the simulate_may function
    '''
    
    # Test with fixed h
    df = ewstools.models.simulate_may(h=0.15)
    assert type(df) == pd.DataFrame
    assert df.index.name == 'time'
    
    
    # Test with variable h
    df = ewstools.models.simulate_may(h=[0.15,0.27])
    assert type(df) == pd.DataFrame
    assert df.index.name == 'time'
    
    # Test with large noise
    df = ewstools.models.simulate_may(sigma=0.1)
    assert type(df) == pd.DataFrame


    






    
    
#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Tests for `ewstools.models` module
---------------
"""


import pytest
import numpy as np
import pandas as pd

from ewstools.models import simulate_ricker




def test_simulate_ricker():
    '''
    Test the simulate_ricker function
    '''
    
    series = simulate_ricker()
    
    assert type(series) == pd.Series
    assert series.index.name == 'time'
    
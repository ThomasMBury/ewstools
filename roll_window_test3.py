#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 23:35:55 2018

@author: tb460
"""

import pandas as pd

from collections import deque
from functools import partial

from ews_spec import pspec_welch, pspec_metrics

def make_class(func, dim_output):

    class your_multi_output_function_class:
        def __init__(self, func, dim_output):
            assert dim_output >= 2
            self.func = func
            self.deques = {i: deque() for i in range(1, dim_output)}

        def f0(self, *args, **kwargs):
            k = self.func(*args, **kwargs)
            for queue in sorted(self.deques):
                self.deques[queue].append(k[queue])
            return k[0]

    def accessor(self, index, *args, **kwargs):
        return self.deques[index].popleft()

    klass = your_multi_output_function_class(func, dim_output)

    for i in range(1, dim_output):
        f = partial(accessor, klass, i)
        setattr(klass, 'f' + str(i), f)

    return klass


def my_fun(window):    
    pspec = pspec_welch(window,1,10)
    df_metrics = pspec_metrics(pspec,ews=['smax','cf','aic'])
    return df_metrics

n=3

rolling_func = make_class(my_fun, n)
# dict to map the function's outputs to new columns. Eg:
agger = {'output_' + str(i): getattr(rolling_func, 'f' + str(i)) for i in range(n)} 
output = pd.Series(range(100)).agg(agger)












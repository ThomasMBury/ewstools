#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 22:20:16 2018

@author: tb460
"""
import pandas as pd

class CountCalls:
    def __init__(self):
        self.counter = 0

    def your_function(self, window):
        retval = (sum(window),window[2])
        self.counter = self.counter + 1
        return retval

my_series=pd.Series(range(30))

TestCounter = CountCalls()

output2=pd.Series.rolling(my_series, window = 5).apply(TestCounter.your_function)

output=my_series.rolling(window=5).apply(TestCounter.your_function)

print(TestCounter.counter)
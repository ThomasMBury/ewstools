#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 23:29:41 2018

@author: tb460
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data={'happy rating':[2,2,3,5,6,7,11,11,23,22,22,22,25]}
df=pd.DataFrame(data)


roll_mean=df.rollring(window=2).mean()

plt.figure();
df.plot();
roll_mean.plot()
plt.legend(loc='best')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  1 23:29:41 2018

@author: tb460
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# create a noisy trajectory
t=np.linspace(0,10,110)
x=np.random.randn(len(t))*0.5

# create a dataframe indexed by time
data={'state variable':x}
df=pd.DataFrame(data,index=t)
df.index.rename('Time',inplace=True)

# parameters
roll_window=20
lag_time=1

# compute rolling stats
roll_var=df.rolling(window=roll_window).var()
roll_ac=pd.rolling_apply(
        df.iloc[:,0],
        window=roll_window,
        func=lambda x: pd.Series(x).autocorr(lag=lag_time))
roll_std=df.rolling(window=roll_window).std()
roll_skew=df.rolling(window=roll_window).skew()


# add rolling stats to dataframe
df.loc[:,'rolling var']=roll_var.iloc[:,0]
df.loc[:,'rolling ac']=roll_ac
df.loc[:,'rolling std']=roll_std.iloc[:,0]
df.loc[:,'rolling skew']=roll_skew.iloc[:,0]


plt.figure();
df.plot();
plt.legend(loc='best')

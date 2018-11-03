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
t=np.linspace(1,10,100)
x=np.random.randn(len(t))*0.5

data={'happy rating':x}
df=pd.DataFrame(data)

roll_window=4

# compute rolling stats
roll_var=df.rolling(window=roll_window).var()
#roll_ac=df.rolling(window=roll_window).apply(lambda x: x.autocorr(lag=1),raw=False)
roll_std=df.rolling(window=roll_window).std()
roll_skew=df.rolling(window=roll_window).skew()


       
       
# add rolling stats to dataframe
df.loc[:,'rolling var']=roll_var.iloc[:,0]
#df.loc[:,'rolling ac']=roll_ac.iloc[:,0]
df.loc[:,'rolling std']=roll_std.iloc[:,0]
df.loc[:,'rolling skew']=roll_skew.iloc[:,0]



plt.figure();
df.plot();
plt.legend(loc='best')

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 17:48:51 2018

@author: tb460

pandas playground - get to grips with DataFrames
"""

import numpy as np
import pandas as pd


my_data=np.array([['','col1','col2'],['row1',1,2],['row2',3,4]])

my_frame=pd.DataFrame(data=my_data[1:,1:],
                      index=my_data[1:,0],
                      columns=my_data[0,1:])

# input to dataframe can take multiple forms
my_2darray = np.array([[1,2,3],[4,5,6]])
my_dict = {1: ['1','3'],2: ['1','2'],3: ['2','4']}
my_df = pd.DataFrame(data=[4,5,6,6],index=range(0,4),columns=['A'])

# input these forms to a dataframe
df1=pd.DataFrame(data=my_2darray, index=range(0,2),columns=['A','B','C'])

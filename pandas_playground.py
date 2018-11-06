#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov  2 17:48:51 2018

@author: tb460

pandas playground - get to grips with DataFrames
"""

import numpy as np
import pandas as pd



# input to dataframe can take multiple forms
my_2darray = np.array([[1,2,3],[4,5,6]])
my_dict = {1: ['1','3'],2: ['1','2'],3: ['2','4']}
my_df = pd.DataFrame(data=[4,5,6,6],index=range(0,4),columns=['A'])

# input these forms to a dataframe
df1=pd.DataFrame(data=my_2darray, index=range(0,2),columns=['A','B','C'])

# use .loc for label based indexing, .iloc for positional indexing
df1.iloc[1,1]
df1.loc[0,'B']




# export pandas dataframe to csv
products = pd.DataFrame({'category': ['Cleaning', 'Cleaning', 'Entertainment', 'Entertainment', 'Tech', 'Tech'],
                        'store': ['Walmart', 'Dia', 'Walmart', 'Fnac', 'Dia','Walmart'],
                        'price':[11.42, 23.50, 19.99, 15.95, 19.99, 111.55],
                        'testscore': [4, 3, 5, 7, 5, 8]})





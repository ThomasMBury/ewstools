#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Aug  7 11:47:31 2019

@author: tbury

Function to reconstruct the Jacobian from time-series data.
See Williamson (2015) for implementation details.

"""

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# Imoprt trajectoreis dataframe
df_traj = pd.read_csv('../../ews_seasonal/models_ricker/ricker_seasonal/data_export/ews_stat_evaltest/traj.csv', index_col=['rb','rnb','Time'])
df_traj.rename(columns={'Non-breeding':'x', 'Breeding':'y'}, inplace=True)

rb_vals = df_traj.index.levels[0]
rnb_vals = df_traj.index.levels[1]

# Select time series to analyse
df_temp = df_traj.loc[(2,-1)]

# Visualise with plot
df_temp[['x','y']].plot()


#------------------------
##  Function to compute lag-1 autocovariance matrix

def compute_autocov(df_in):
    '''
    Computes the autocovariance (lag-1) matrix of n 
    time series provided in df_in.
    
    Using the definition
        phi_ij = < X_i(t) X_j(t+1) >
    for each element of the autocovariance matrix phi.
    
    Input:
        df_in: DataFrame with n columns indexed by time
    Ouptut:
        np.array of autocovariance matrix
    '''
    
    # Obtain column names of df_in
    col_names = df_in.columns
    # Number of variables
    n = len(col_names)
    
    
    # Define function to compute autocovariance of two columns
    def autocov_cols(a,b):
        '''
        Computes autocovariance of two columns (can be the same)
        Input:
            a,b: Series indexed by time
        Output:
            float: autocovariance between the columns
        '''
        
        # Shift the column of b by 1
        b_shift = b.shift(1)
        
        # Put into a dataframe
        df_temp = pd.DataFrame([a,b_shift],axis=1)
        
        # Compute covariance of columns a and b_shift
        cov = df_temp.cov().iloc[0,1]
        
        # Output
        return cov
            
        
    # Compute elements of autocovariance matrix
    list_elements = []
    
    for i in range(n):
        for j in range(n):
            a = df_in[col_names[i]]
            b = df_in[col_names[j]]
            # Compute autocovaraince between cols
            autocov = autocov_cols(a,b)
            # Append to list of elements
            list_elements.append(autocov)
    
    # Create autocovariance matrix from list of elements
    ar_autocov = np.array(list_elements).reshape(n,n)

    # Output
    return ar_autocov


#---------------------------------------
## Function to compute Jacobian reconstruction 
    


def eval_recon(df_in):
    '''
    Constructs estimate of Jacobian matrix from stationary time-series data
    and outputs the eigenvalues
    Input:
        df_in: DataFrame with two columns indexed by time
    Output:
        np.array of of eigenvalues, np.array of evecs
    '''
    
    # Compute covariance matrix
    cov = np.array(df_in.cov())
    
    # Compute autocovariance matrix
    acov = autocov(df_in)
    
    # Estimate of Jacobian (formula in Williamson (2015))
    jac = np.matmul( acov, np.linalg.inv(cov))
    
    # Eigenvalues
    evals, evecs = np.linalg.eig(jac)
    
    return evals, evecs





# Initiate list for eval data
list_evals = []
# Compute eigenvalues for each set of r values
for rb in rb_vals:
    for rnb in rnb_vals:
        df_temp = df_traj.loc[(rb,rnb)]
        
        # If trajecotry is zero set evals to NA
        if df_temp.iloc[-1,0]==0:
            evals = np.array([np.nan,np.nan])
            vec1 = np.array([np.nan,np.nan])
            vec2 = np.array([np.nan,np.nan])
        else:
        # Compute evals and evecs
            evals, evecs = eval_recon(df_temp)
            vec1 = evecs[:,0]
            vec2 = evecs[:,1]
            
        # Choose evecs to lie in the half-plane x>0 (wlog)
        if vec1[0]<0:
            vec1 = -vec1
        if vec2[0]<0:
            vec2 = -vec2
            
        
        # Add data to a dictionary    
        dic_temp = {'rb':rb, 'rnb':rnb, 'eval1':evals[0], 'eval2':evals[1], 
                    'evec1_x':np.real(vec1[0]), 'evec1_y':np.real(vec1[1]),
                    'evec2_x':np.real(vec2[0]), 'evec2_y':np.real(vec2[1])}    
        # Append to list
        list_evals.append(dic_temp)


# Add to dataframe
df_evals = pd.DataFrame(list_evals).set_index(['rb','rnb'])

     


# Columns for real and imaginary parts   
df_evals['eval1_re'] = df_evals['eval1'].apply(lambda x: np.real(x))
df_evals['eval1_im'] = df_evals['eval1'].apply(lambda x: np.imag(x))
df_evals['eval2_re'] = df_evals['eval2'].apply(lambda x: np.real(x))
df_evals['eval2_im'] = df_evals['eval2'].apply(lambda x: np.imag(x))

# Compute absolute values of eigenvector components
df_evals['evec1_x_abs'] = df_evals['evec1_x'].apply(np.abs)
df_evals['evec1_y_abs'] = df_evals['evec1_y'].apply(np.abs)








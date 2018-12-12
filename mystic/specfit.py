#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 11 19:35:09 2018

@author: Thomas Bury
"""

# import required python modules
import numpy as np
from scipy import signal
import pandas as pd
import matplotlib.pyplot as plt

def fit_fold(coeffs,w):
    sigma,lam = coeffs
    return (sigma**2 / (2*np.pi))*(1/(w**2+lam**2))

def fit_hopf(coeffs,w):
    sigma,mu,w0 = coeffs
    return (sigma**2/(4*np.pi))*(1/((w+w0)**2+mu**2)+1/((w-w0)**2 +mu**2))

def fit_null(coeffs,w):
    sigma = coeffs[0]
    return sigma**2/(2*np.pi)* w**0

# import some power spectrum data
pspecs_raw = pd.read_csv("../../generic_models/normal_forms/data_export/fold_ews_1/pspecs.csv",
                   index_col=['Realisation number','Time','Frequency'])
t_vals = pspecs_raw.index.levels[1]
pspec = pspecs_raw.loc[1,t_vals[-10]]['Empirical']


w=pspec.index.values
y=pspec.values

@reduced(lambda x,y: abs(x)+abs(y))
def objective_fold(coeffs, w, y):
    return fit_fold(coeffs,w) - y

@reduced(lambda x,y: abs(x)+abs(y))
def objective_hopf(coeffs, w, y):
    return fit_hopf(coeffs,w) - y

@reduced(lambda x,y: abs(x)+abs(y))
def objective_null(coeffs, w, y):
    return fit_null(coeffs,w) - y


bounds_fold = [(0, 1),(-2, -0.1)]
bounds_hopf = [(0,1), (-2,-0.1), (0,3)]
bounds_null = [(0,1)]

# set up constraints
from mystic.symbolic import generate_constraint, generate_solvers, simplify


cons = """
w0 >= (mu/(2*sqrt(0.1)))*sqrt(4-3*(0.1) + sqrt((0.1)**2-16*(0.1)+16))
"""
var = ['w0','mu']

cf = generate_constraint(generate_solvers(
        simplify(cons, variables=var, all=True)))


args = (w,y)

from mystic.solvers import diffev2

result_fold = diffev2(objective_fold, args=args, x0=bounds_fold, bounds=bounds_fold, npop=40, 
                 ftol=1e-8, gtol=100, disp=False, full_output=True)

result_hopf = diffev2(objective_hopf, args=args, x0=bounds_hopf, bounds=bounds_hopf, constraints=cf, npop=40, 
                 ftol=1e-8, gtol=100, disp=False, full_output=True)

result_null = diffev2(objective_null, args=args, x0=bounds_null, bounds=bounds_null, npop=40, 
                 ftol=1e-8, gtol=100, disp=False, full_output=True)









plt.plot(w,y,
         w,fit_fold(result_fold[0],w),
         w,fit_hopf(result_hopf[0],w),
         w,fit_null(result_null[0],w)
         )





[![PyPI version](https://badge.fury.io/py/ewstools.svg)](https://badge.fury.io/py/ewstools)
[![Build Status](https://travis-ci.com/ThomasMBury/ewstools.svg?branch=master)](https://travis-ci.com/ThomasMBury/ewstools)
[![Coverage Status](https://coveralls.io/repos/github/ThomasMBury/ewstools/badge.svg?branch=master&service=github)](https://coveralls.io/github/ThomasMBury/ewstools?branch=master&service=github)

# ewstools
**Python package for computing, analysing and visualising early warning signals (EWS)
in time-series data. Includes a novel approach to characterise bifurcations using EWS.**

Functionality includes

  - Computing the following EWS
    - Variance metrics (variance, standard deviation, coefficient of variation)
    - Autocorrelation (at specified lag times)
    - Higher moments (skewness, kurtosis)
    - Power spectrum (including maximum frequency, coherence factor and AIC weights csp. to different canonical forms)

  - Block-bootstrapping time-series to obtain confidence bounds on EWS estimates
  
  - Visualisation of EWS with plots of time-series and power spectra.
  
Dependencies include
  - numpy, pandas, seaborn
  - lmfit, arch  
  

## ews_compute.py
File for function `ews_compute`.  
`ews_compute` takes in Series data and outputs user-specified EWS in a DataFrame.


**Input** (default value)
- *raw_series* : pandas Series indexed by time 
- *roll_window* (0.25) : size of the rolling window (as a proportion of the length of the data)
- *smooth* (True) : if True, series data is detrended with a Gaussian kernel
- *band_width* (0.2) : bandwidth of Gaussian kernel
- *ews* ( ['*var*', '*ac*', '*smax*'] ) : list of strings corresponding to the desired EWS. Options include
  - '*var*'   : Variance
  - '*ac*'    : Autocorrelation
  - '*sd*'    : Standard deviation
  - '*cv*'    : Coefficient of variation
  - '*skew*'  : Skewness
  - '*kurt*'  : Kurtosis
  - '*smax*'  : Peak in the power spectrum
  - '*cf*'    : Coherence factor
  - '*aic*'   : AIC weights
- *lag_times* ( [1] ) : list of integers corresponding to the desired lag times for AC
- *ham_length* (40) : length of the Hamming window (used to compute power spectrum)
- *ham_offset* (0.5) : offset of Hamming windows as a proportion of *ham_length*
- *w_cutoff* (1) : cutoff frequency (as a proportion of maximum frequency attainable from data)
    
**Output**
- DataFrame indexed by time with columns corresponding to each EWS



## ews_compute_run.py
An example script that runs `ews_compute` on times-series data of a stochastic simulation of May's 
harvesting model. It also shows how to compute kendall tau values and plot results. This
can be used as a template for EWS of times-series data.


## ews_compute_runMulti.py
An example script that runs `ews_compute` on multiple time-series data and outputs
EWS as a distribution over realisations.
















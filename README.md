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
  
  
## Install:

Install ewstools using pip:
```python
pip install ewstools
```

Installation with conda will be available soon, for those with an Anaconda distribution.


## Documentation




## Demos

For help getting started, take a look at the following demos. They are scripted in iPython notebooks for interactivity.



## Quick walkthrough














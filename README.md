[![PyPI version](https://badge.fury.io/py/ewstools.svg)](https://badge.fury.io/py/ewstools)
[![Downloads](https://pepy.tech/badge/ewstools)](https://pepy.tech/project/ewstools)
[![Documentation Status](https://readthedocs.org/projects/ewstools/badge/?version=latest)](https://ewstools.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/ThomasMBury/ewstools/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ThomasMBury/ewstools/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/ThomasMBury/ewstools/branch/main/graph/badge.svg?token=Q5LGRV6TLF)](https://codecov.io/gh/ThomasMBury/ewstools)

# ewstools
**A Python package for early warning signals (EWS) of bifurcations in time series data.**

## Overview

Many systems across nature and society have the capacity to undergo an abrupt and profound change in their dynamics. From a dynamical systemes perspective, these changes corresopond to bifurcations, which carry some generic features that can be picked up on in time series data ([Scheffer et al. 2009](https://www.nature.com/articles/nature08227)). Two commonly used metrics include variance and lag-1 autocorrelation, though there exist many others (see e.g. [Clements & Ozgul 2018](https://onlinelibrary.wiley.com/doi/full/10.1111/ele.12948)). More recently, deep learning methods have been developed to provide early warning signals, whilst also signalling the type of bifurcation approaching [(Bury et al. 2021)](https://www.pnas.org/doi/10.1073/pnas.2106140118).

The goal of this Python package is to provide a user-friendly toolbox for computing early warning signals in time series data. It complements an existing early warning signals package in R ([Dakos et al. 2012](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041010)). We hope that having an early warning signal toolbox in Python will allow for additional testing, and appeal to those who primarily work in Python. I will try to keep it updated with the latest methods.

Current functionality of *ewstools* includes

  - Detrending of time series using
    - A Gaussian kernel
    - LOWESS (Locally Weighted Scatterplot Smoothing)

  - Computation of CSD-based early warning signals including:
    - Variance and associated metrics (standard deviation, coefficient of variation)
    - Autocorrelation (at specified lag times)
    - Higher-order statistical moments (skewness, kurtosis)
    - Power spectrum and associated metrics (maximum frequency, coherence factor, AIC weights csp. to canonical power spectrum forms)

  - Application of deep learning classifiers for bifurcation prediction as in the study by [Bury et al.](https://www.pnas.org/doi/10.1073/pnas.2106140118).

  - Block-bootstrapping of time-series to obtain confidence bounds on EWS estimates
  
  - Visualisation tools to display EWS.


## Install

You can install *ewstools* with pip using the commands

```bash
pip install --upgrade pip
pip install ewstools
```

[Jupyter notebook](https://jupyter.org/install) is required for the tutorials, and can be installed with the command
```bash
pip install jupyter notebook
```
Package dependencies of *ewstools* are
```bash
'pandas>=1.2.0',
'numpy>=1.20.0',
'plotly>=5.3.0',
'lmfit>=0.9', 
'arch>=4.7',
'statsmodels>=0.12.0',
'scipy>=1.5.0',
```
and should be installed automatically. To use any of the deep learning functionality, you will need to install [TensorFlow](https://www.tensorflow.org/install) v2.0.0 or later.

To install the latest *development* version of *ewstools*, use the command
```bash
pip install git+https://github.com/thomasmbury/ewstools.git#egg=ewstools
```
NB: the development version comes with the risk of undergoing continual changes, and has not undergone the level of scrutiny of official releases.




## Tutorials/Demonstrations

1. [ewstools: An Introduction](./tutorials/tutorial_intro.ipynb)
2. [ewstools: Computing Spectral EWS](./tutorials/tutorial_spectral.ipynb)
3. [ewstools: Deep Learning Classifiers](./tutorials/tutorial_deep_learning.ipynb)



## Quick demo

First we need to import *ewstools* and collect the data we wish to analyse. Here we will run a simulation of the Ricker model, one of the model functions stored in [`ewstools.models`](https://ewstools.readthedocs.io/en/latest/ewstools.html#ewstools-models-submodule).
```python
import ewstools
from ewstools.models import simulate_ricker
series = simulate_ricker(tmax=500, F=[0,2.7])
series.plot();
```
![](tutorials/readme/series.png)

We then make a [`TimeSeries`](https://ewstools.readthedocs.io/en/latest/ewstools.html#ewstools.core.TimeSeries) object, which takes in our data and a transition time (if desired). EWS are not computed beyond the transition time.

```python
ts = ewstools.TimeSeries(data=series, transition=440)
```

We can then detrend, compute EWS and calculate Kendall tau statistics by applying methods to the [`TimeSeries`](https://ewstools.readthedocs.io/en/latest/ewstools.html#ewstools.core.TimeSeries) object:

```python
ts.detrend(method='Lowess', span=0.2)
ts.compute_var(rolling_window=0.5)
ts.compute_auto(lag=1, rolling_window=0.5)
ts.compute_auto(lag=2, rolling_window=0.5)
ts.compute_ktau()
```

Finally, we can view output as an interactive Plotly figure (when run in a Jupyter notebook) using

```python
ts.make_plotly()
```

![](tutorials/readme/ews.png)

More detailed demonstrations can be found in the tutorials, and all methods are listed in the documentation.

## Documentation

Available on [ReadTheDocs](https://ewstools.readthedocs.io/en/latest/).

## Issues

If you have any suggestions or find any bugs, please post them on the [issue tracker](https://github.com/ThomasMBury/ewstools/issues). I also welcome any contributions - please get in touch if you are interested, or submit a pull request if you are familiar with that process.

## Acknowledgements

This work is currently supported by an FRQNT (Fonds de recherche du Québec - Nature et Technologies) postdoctoral research scholarship awarded to Dr. Thomas Bury. In the past, it was supported by NSERC (Natural Sciences and Engineering Research Council) Discovery Grants awarded to Dr. Chris Bauch and Dr. Madhur Anand.

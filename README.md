[![PyPI version](https://badge.fury.io/py/ewstools.svg)](https://badge.fury.io/py/ewstools)
[![Downloads](https://pepy.tech/badge/ewstools)](https://pepy.tech/project/ewstools)
[![Documentation Status](https://readthedocs.org/projects/ewstools/badge/?version=latest)](https://ewstools.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/ThomasMBury/ewstools/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ThomasMBury/ewstools/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/ThomasMBury/ewstools/branch/main/graph/badge.svg?token=Q5LGRV6TLF)](https://codecov.io/gh/ThomasMBury/ewstools)

# ewstools
**A python package for early warning signals (EWS) of bifurcations in time series data.**

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [Installation Guide](#install)
- [Tutorials](#tutorials)
- [Documentation](#documentation)
- [License](./LICENSE)
- [Issues](#issues)
- [Acknowledgements](#acknowledgements)


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


## Repo Contents

- [tutorials](./tutorials): Jupyter notebooks to demonstrate methods - a good place to start
- [docs](./docs): documentation source code running with ReadTheDocs
- [ewstools](./ewstools): source code
- [tests](./tests): unit tests
- [saved_classifiers](./saved_classifiers): pre-trained Tensorflow bifurcation classifiers

## Install

You can install *ewstools* with pip using the commands

```
pip install --upgrade pip
pip install ewstools
```

[Jupyter notebook](https://jupyter.org/install) is required for the tutorials, and can be installed with the command
```
pip install jupyter notebook
```

To install the latest *development* version, use the command
```
pip install git+https://github.com/thomasmbury/ewstools.git#egg=ewstools
```
Note, the development version comes with the risk of undergoing continual changes, and has not undergone the level of scrutiny of official releases.

## Tutorials

A good place to start is with the [tutorials](./tutorials), written as iPython notebooks.

## Documentation

Documentation available on [ReadTheDocs](https://ewstools.readthedocs.io/en/latest/).

## Issues

If you run have any suggestions or spot any bugs with the package, please post on the [issue tracker](https://github.com/ThomasMBury/ewstools/issues)! I also welcome any contributions - please get in touch if you are interested, or submit a pull request if you are familiar with that process.

## Acknowledgements

This work is supported by an FRQNT (Fonds de recherche du Québec - Nature et Technologies) postdoctoral research scholarship awarded to Thomas Bury. 


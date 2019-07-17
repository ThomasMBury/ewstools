[![PyPI version](https://badge.fury.io/py/ewstools.svg)](https://badge.fury.io/py/ewstools)
[![Build Status](https://travis-ci.com/ThomasMBury/ewstools.svg?branch=master)](https://travis-ci.com/ThomasMBury/ewstools)
[![Coverage Status](https://coveralls.io/repos/github/ThomasMBury/ewstools/badge.svg?branch=master&service=github)](https://coveralls.io/github/ThomasMBury/ewstools?branch=master&service=github)
[![DOI](https://zenodo.org/badge/155786429.svg)](https://zenodo.org/badge/latestdoi/155786429)


# ewstools
**Python package for computing, analysing and visualising early warning signals (EWS)
in time series data. Includes a novel approach to characterise bifurcations using EWS.**

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demos](#demo)
- [Documentation]
- [License](./LICENSE)
- [Issues](https://github.com/thomasmbury/ewstools/issues)
- [Contribution]


## Overview

Many natural and artificial systems have the capacity to undergo a sudden change in their dynamics. In the mathematical realm of dynamical systems, these changes corresopond to bifurcations, and theory therein suggests that certain signals, observable in time series data, should precede these bifurcations ([Scheffer et al. 2009](https://www.nature.com/articles/nature08227)). Two commonly used metrics include variance and autocorrelation, though there exist many others (see e.g. [Clements & Ozgul](https://onlinelibrary.wiley.com/doi/full/10.1111/ele.12948)). Our objective with this package is to provide a user-friendly toolbox in Python to compute early warning signals from time series data. It also contains novel tools to extract information on the bifurcation from the power spectrum, results to be published soon.


Functionality of *ewstools* includes

  - Detrending using either
    - Gaussian smoothing
    - A Lowess filter

  - Computation of the following statistical properties over a rolling window:
    - Variance and its derivatives (standard deviation, coefficient of variation)
    - Autocorrelation (at specified lag times)
    - Higher-order moments (skewness, kurtosis)
    - Power spectrum (including maximum frequency, coherence factor and AIC weights csp. to canonical power spectrum forms)

  - Block-bootstrapping time-series to obtain confidence bounds on EWS estimates
  
  - Visualisation of EWS with plots of time-series and power spectra.
  

## Repo Contents


## System Requirements

## Installation Guide

The package *ewstools* requires Python version 3.7 or later to be installed on your system. It may then be installed using pip, by entering the following into your command line.
```python
pip install ewstools
```

## Demos

For interacitve demonstrations on using *ewstools*, please refer to these [iPython notebooks](https://github.com/ThomasMBury/ewstools/tree/master/demos).

## Documentation

Full documentation is available on [ReadTheDocs](https://ewstools.readthedocs.io/en/latest/).

## Contribution

If you are interested in being a contributer, or run into trouble with the package, please post on the [issue tracker](https://github.com/ThomasMBury/ewstools/issues).


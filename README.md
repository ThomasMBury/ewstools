[![PyPI version](https://badge.fury.io/py/ewstools.svg)](https://badge.fury.io/py/ewstools)
[![Build Status](https://travis-ci.com/ThomasMBury/ewstools.svg?branch=master)](https://travis-ci.com/ThomasMBury/ewstools)
[![Coverage Status](https://coveralls.io/repos/github/ThomasMBury/ewstools/badge.svg?branch=master&service=github)](https://coveralls.io/github/ThomasMBury/ewstools?branch=master&service=github)
[![DOI](https://zenodo.org/badge/155786429.svg)](https://zenodo.org/badge/latestdoi/155786429)


# ewstools
**Python package for computing, analysing and visualising early warning signals (EWS)
in time series data. Includes a novel approach to characterise bifurcations using Spectral EWS.**

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demos](#demos)
- [Documentation](#documentation)
- [License](./LICENSE)
- [Issues](https://github.com/thomasmbury/ewstools/issues)
- [Contribution](#contribution)


## Overview

Many natural and artificial systems have the capacity to undergo a sudden change in their dynamics. In the mathematical realm of dynamical systems, these changes corresopond to bifurcations, and theory therein suggests that certain signals, observable in time series data, should precede these bifurcations ([Scheffer et al. 2009](https://www.nature.com/articles/nature08227)). Two commonly used metrics include variance and autocorrelation, though there exist many others (see e.g. [Clements & Ozgul 2018](https://onlinelibrary.wiley.com/doi/full/10.1111/ele.12948)). Our objective with this package is to provide a user-friendly toolbox in Python to compute early warning signals from time series data. This complements another early warning signal toolbox written in R ([Dakos et al. 2012](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041010)), and provides novel tools to extract information on the bifurcation from the power spectrum - results to be published soon.


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

- [demos](./demos): interactive demos in Jupyter notebooks to illustrate use of package
- [docs](./docs): version-controlled package documentation provided in ReadTheDocs
- [ewstools](./ewstools): package code
- [tests](./tests): testing of package functions using pytest


## System Requirements

### Hardware Requirements

*ewstools* can run on a standard computer with enough RAM to support the operations defined by a user. The software has been tested on a computer with the following specs

RAM: 8G
CPU: 2.7 GHz

though the software should run as expected on computers with lower RAM. The runtimes outlined below were generated on the computer with these specs.

### Software Requirements

*ewstools* requires Python 3.7 or higher and has the following package dependencies:
```
pandas==0.24.2
numpy==1.16.2
arch==4.7
lmfit==0.9.12
```
The Python package should be compatible with Windows, Mac, and Linux operating systems. The demonstrations require Jupyter notebook, which can be installed 



## Installation Guide

Friendly instructions for downloading Python 3 on Linux, Mac OS and Windows are available [here](https://realpython.com/installing-python/).

Then, the package *ewstools* may be installed using pip, by entering the following into Terminal (Mac/Linux) or Command Prompt (Windows)
```
pip install ewstools
```
which includes all package dependencies. Installation of the package should about 1 minute on a standard computer. To interact with the demos, Jupyter notebook is required, which can be installed using
```
pip install jupyterlab
```
and takes no longer than a minute to download.


## Demos

For interacitve demonstrations on using *ewstools*, please refer to these [iPython notebooks](https://github.com/ThomasMBury/ewstools/tree/master/demos).

## Documentation

Full documentation is available on [ReadTheDocs](https://ewstools.readthedocs.io/en/latest/).

## Contribution

If you are interested in being a contributer, or run into trouble with the package, please post on the [issue tracker](https://github.com/ThomasMBury/ewstools/issues).


[![PyPI version](https://badge.fury.io/py/ewstools.svg)](https://badge.fury.io/py/ewstools)
[![Build Status](https://travis-ci.com/ThomasMBury/ewstools.svg?branch=master)](https://travis-ci.com/ThomasMBury/ewstools)
[![Coverage Status](https://coveralls.io/repos/github/ThomasMBury/ewstools/badge.svg?branch=master&service=github)](https://coveralls.io/github/ThomasMBury/ewstools?branch=master&service=github)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.3497512.svg)](https://doi.org/10.5281/zenodo.3497512)


# ewstools
**Tools for obtaining early warning signals (EWS) of bifurcations in time series data.**

## Contents

- [Overview](#overview)
- [Repo Contents](#repo-contents)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Tutorials](#tutorials)
- [Documentation](#documentation)
- [License](./LICENSE)
- [Contribution](#contribution)


## Overview

Many systems across nature and society have the capacity to undergo an abrupt and profound change in their dynamics. From a dynamical systemes perspective, these changes corresopond to bifurcations, which carry some generic features that can be picked up on in time series data ([Scheffer et al. 2009](https://www.nature.com/articles/nature08227)). Two commonly used metrics include variance and lag-1 autocorrelation, though there exist many others (see e.g. [Clements & Ozgul 2018](https://onlinelibrary.wiley.com/doi/full/10.1111/ele.12948)). More recently, deep learning methods have been developed to provide early warning signals, whilst also signalling the type of bifurcation approaching [Bury et al.](https://www.pnas.org/doi/10.1073/pnas.2106140118).

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

## System Requirements

### Hardware Requirements

*ewstools* can run on a standard computer with enough RAM to support the operations defined by a user. The software has been tested on a computer with the following specs

RAM: 8G, CPU: 2.7 GHz

though the software should run as expected on computers with lower RAM. The runtimes outlined below were generated on the computer with these specs.

### Software Requirements

Requires Python 3.7 or higher and has package dependencies listed in [requiements_dev.txt](https://github.com/ThomasMBury/ewstools/requirements_dev.txt). The Python package should be compatible with Windows, Mac, and Linux operating systems. The demos require Jupyter notebook.


## Installation Guide

Instructions for downloading Python 3 on Linux, Mac OS and Windows are available [here](https://realpython.com/installing-python/).

*ewstools* may then be installed using pip with the following command:
```
pip install ewstools
```
which should install all package dependencies. To run and interact with the tutorials, Jupyter notebook is required, which can be installed using
```
pip install jupyter notebook
```

## Tutorials

For demonstrations on how to use *ewstools*, please refer to these [iPython notebooks](https://github.com/ThomasMBury/ewstools/tree/master/demos).

## Documentation

Full documentation is available on [ReadTheDocs](https://ewstools.readthedocs.io/en/latest/).

## Contribution

If you run have any suggestions or spot any bugs with the package, please post on the [issue tracker](https://github.com/ThomasMBury/ewstools/issues)! I also welcome any contributions - please get in touch if you are interested, or submit a pull request if you are familiar with that process.


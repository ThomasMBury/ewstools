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
  
  
## Install:

*ewstools* requires Python version 3.7 or later to be installed on your system. Then *ewstools* can be installed using pip, by entering the following into your command line.
```python
pip install ewstools
```

## Demos

iPython notebooks demonstrating how to use the software are available [here](examples/).

## Documentation

Full documentation for *ewstools* is available on [ReadTheDocs](https://ewstools.readthedocs.io/en/latest/).

## Contribution

If you are interested in being a contributer to *ewstools*, or run into trouble with the package, please post on the [issue tracker](https://github.com/ThomasMBury/ewstools/issues).


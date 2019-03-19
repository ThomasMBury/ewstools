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

The package *ewstools* requires Python version 3.7 or later to be installed on your system. It may then be installed using pip, by entering the following into your command line.
```python
pip install ewstools
```

## Demos

For demonstrations/tutorials on using *ewstools*, please refer to these [iPython notebooks](examples/).

## Documentation

Full documentation is available on [ReadTheDocs](https://ewstools.readthedocs.io/en/latest/).

## Contribution

If you are interested in being a contributer, or run into trouble with the package, please post on the [issue tracker](https://github.com/ThomasMBury/ewstools/issues).


[![PyPI version](https://badge.fury.io/py/ewstools.svg)](https://badge.fury.io/py/ewstools)
[![Downloads](https://pepy.tech/badge/ewstools)](https://pepy.tech/project/ewstools)
[![Documentation Status](https://readthedocs.org/projects/ewstools/badge/?version=latest)](https://ewstools.readthedocs.io/en/latest/?badge=latest)
[![tests](https://github.com/ThomasMBury/ewstools/actions/workflows/tests.yml/badge.svg?branch=main)](https://github.com/ThomasMBury/ewstools/actions/workflows/tests.yml)
[![codecov](https://codecov.io/gh/ThomasMBury/ewstools/branch/main/graph/badge.svg?token=Q5LGRV6TLF)](https://codecov.io/gh/ThomasMBury/ewstools)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.05038/status.svg)](https://doi.org/10.21105/joss.05038)

# ewstools
**A Python package for early warning signals (EWS) of bifurcations in time series data.**

## Overview

Many systems in nature and society can undergo critical transitions—sudden, often irreversible shifts in dynamics. Examples include the outbreak of disease, ecosystem collapse, and cardiac arrhythmias. Mathematically, such transitions often correspond to bifurcations (tipping points) in an underlying dynamical system.

[Scheffer et al. (2009)](https://www.nature.com/articles/nature08227) proposed early warning signals (EWS) for bifurcations based on noisy fluctuations in time series data, sparking a surge of related ways to predict bifurcations (see [Dakos et al. (2024)](https://esd.copernicus.org/articles/15/1117/2024/esd-15-1117-2024.html) for a recent review). More recently, deep learing has shown great potential for predicting bifurcations and their type ([Bury et al. 2021](https://www.pnas.org/doi/10.1073/pnas.2106140118)).

`ewstools` is a Python package for computing and visualizing EWS in time series. It complements the R package by ([Dakos et al. 2012](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0041010)) and meets growing demand for Python-based tools ([PYPL, 2022](https://pypl.github.io/PYPL.html)).

Features include:
  - An intuitive, object-oriented framework for computing EWS
  - Detrending methods:
    - Gaussian kernel smoothing
    - LOWESS (Locally Weighted Scatterplot Smoothing)
  - Computation of critical slowing down (CSD) indicators:
    - Variance, standard deviation, coefficient of variation
    - Autocorrelation (at specified lags)
    - Skewness, kurtosis
    - Power spectra metrics
    - Entropy measures
  - Kendall tau statistics to quantify trends
  - Deep learning classifiers for bifurcation prediction ([Bury et al. 2021](https://www.pnas.org/doi/10.1073/pnas.2106140118))
  - Visualization tools
  - Built-in dynamical system models for testing EWS

`ewstools` uses [pandas](https://pandas.pydata.org/) for dataframe handling, [numpy](https://numpy.org/) for fast numerical computing, [plotly](https://plotly.com/graphing-libraries/) for visualisation, [lmfit](https://lmfit.github.io/lmfit-py/) for least-squares minimisation, [arch](https://github.com/bashtage/arch) for bootstrapping methods, [EntropyHub](https://www.entropyhub.xyz/index.html) for entropy computations, [statsmodels](https://www.statsmodels.org/stable/index.html) and [scipy](https://scipy.org/) for detrending methods, and [TensorFlow](https://www.tensorflow.org/install) for deep learning.


## Install

Requires Python 3.8–3.11. Install via:

```bash
pip install --upgrade pip
pip install ewstools
```

For tutorials, install [Jupyter notebook](https://jupyter.org/install):

```bash
pip install jupyter notebook
```

Main dependencies (installed automatically):

```bash
'pandas>=0.23.0',
'numpy>=1.14.0',
'plotly>=2.3.0',
'lmfit>=0.9.0',
'arch>=4.4',
'statsmodels>=0.9.0',
'scipy>=1.0.1',
```

To enable deep learning features, install [TensorFlow](https://www.tensorflow.org/install):

```bash
pip install ewstools[tf]
```
**Note**: TensorFlow for `ewstools` is currently supported on Linux and macOS only.

To install the latest *development* version:

```bash
pip install git+https://github.com/thomasmbury/ewstools.git#egg=ewstools
```
*(Development versions may be unstable.)*


## Tutorials

1. [Introduction to *ewstools*](https://github.com/ThomasMBury/ewstools/tree/main/tutorials/tutorial_intro.ipynb)
2. [Spectral EWS](https://github.com/ThomasMBury/ewstools/tree/main/tutorials/tutorial_spectral.ipynb)
3. [Deep learning classifiers for bifurcation prediction](https://github.com/ThomasMBury/ewstools/tree/main/tutorials/tutorial_deep_learning.ipynb)



## Quick demo

Code in `quick_demo.ipynb`. Import `ewstools` and simulate a time series (e.g., the Ricker model):
```python
import ewstools
from ewstools.models import simulate_ricker
series = simulate_ricker(tmax=500, F=[0,2.7])
series.plot();
```
![](https://github.com/ThomasMBury/ewstools/blob/main/tutorials/images/series.png)

Create a [`TimeSeries`](https://ewstools.readthedocs.io/en/latest/ewstools.html#ewstools.core.TimeSeries) object:

```python
ts = ewstools.TimeSeries(data=series, transition=440)
```

Detrend, compute EWS, and calculate trends:

```python
ts.detrend(method='Lowess', span=0.2)
ts.compute_var(rolling_window=0.5)
ts.compute_auto(lag=1, rolling_window=0.5)
ts.compute_auto(lag=2, rolling_window=0.5)
ts.compute_ktau()
```

Get predictions from deep learning classifiers
```python
for idx, classifier in enumerate(list_classifiers):
    ts.apply_classifier_inc(classifier, inc=10, verbose=0, name=str(idx))
```

Plot results interactively:

```python
ts.make_plotly()
```

![](https://github.com/ThomasMBury/ewstools/blob/main/tutorials/images/ews.png)

For detailed demonstrations, see the tutorials.

## Documentation

Full documentation available on [ReadTheDocs](https://ewstools.readthedocs.io/en/latest/).

## Issues

Found a bug or have a suggestion? Please post it on the [issue tracker](https://github.com/ThomasMBury/ewstools/issues).

Contributions are welcome! Feel free to reach out or submit a pull request.

## Acknowledgements

This work is supported by an FRQNT postdoctoral research scholarship awarded to Dr. Thomas Bury. Previous support was provided by NSERC Discovery Grants awarded to Dr. Chris Bauch and Dr. Madhur Anand.

## Citation info

If you find ewstools useful, please consider starring the repository ⭐ and citing:

Bury, Thomas M. "[ewstools: A Python package for early warning signals of bifurcations in time series data.](https://joss.theoj.org/papers/10.21105/joss.05038.pdf)" *Journal of Open Source Software* 8.82 (2023): 5038.

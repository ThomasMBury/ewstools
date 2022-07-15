# Tutorials
**A collection of iPython notebooks to demonstrate various applications of *ewstools*.**

To run these tutorials, [Jupyter notebook](https://jupyter.org/install) must be installed. Tutorial 1 is a prerequisite to all later tutorials.

### 1. Introduction to *ewstools* (tutorial_intro.ipynb)

- Initialise a TimeSeries object with your data
- Detrend your data using a specific filter and bandwidth
- Compute CSD-based early warning signals (EWS) over a rolling window
- Measure the trend of the EWS with Kendall tau values
- Visualise output

### 2. Spectral EWS (tutorial_spectral.ipynb)

- Compute and visualise changes in the power spectrum over a rolling window
- Compute spectral early warning signals as in [Bury et al. (2020) Royal Soc. Interface](https://royalsocietypublishing.org/doi/full/10.1098/rsif.2020.0482)

### 3. Deep learning classifiers for bifurcation prediction (tutorial_deep_learning.ipynb)

- Import a TensorFlow classifier and obtain its predictions on a section of time series data
- Compute predictions made from an ensemble of classifiers



## Old tutorials

These old tutorials use deprecated functions that will be removed in future versions of *ewstools*.

### ews_fold.ipynb - Deprecated (uses deprecated functions in ewstools)
- Simulates a single stochastic trajectory of the Ricker model going through a Fold bifurcation
- Shows how to use *ewstools* to compute early warning signals
- Visualises the output of *ewstools* graphically
- Run time < 1 min


### ews_bootstrap.ipynb - Deprecated (uses deprecated functions ewstools)
- Simulates single stochastic trajectories of the Ricker model going through a Flip bifurcation
- Uses *ewstools* to compute bootstrapped time-series and the corresponding EWS.
- Visualises the output, comparing the two bifurcations.
- Run time < 3 mins


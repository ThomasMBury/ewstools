---
title: 'ewstools: A Python package for early warning signals of bifurcations in time series data'
tags:
  - Python
  - time series
  - early warning signal
  - tipping point
  - dynamical system
  - bifurcation
authors:
  - name: Thomas M. Bury
    equal-contrib: false
    orcid: 0000-0003-1595-9444
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Department of Physiology, McGill University, Montréal, Canada
   index: 1
 - name: Department of Applied Mathematics, University of Waterloo, Waterloo, Canada
   index: 2
date: 17 August 2022
bibliography: paper.bib
---


# Summary

Many systems in nature and society have the capacity to undergo critical transitions--
sudden and profound changes in their dynamics that are hard to reverse.
Examples include the outbreak of disease, the collapse of an ecosystem, or the onset 
of a cardiac arrhythmia.
From a mathematical perspective, these transitions may be understood as the 
crossing of a bifurcation (tipping point) in an appropriate dynamical system model.
In 2009, Scheffer and colleagues proposed using statistical metrics to signal the
approach of a bifurcation in time series data [@scheffer2009early].
This created massive interest in the subject of EWS from a wide range scientific disciplines.
Now, there exist a multitude of different EWS and associated methods for
anticipating bifurcations [@clements2018indicators].


`ewstools` provides an accessible toolbox for computing, analysing and 
visualising EWS in time seires data. The package provides:

- Time series detrending methods
- A suite of standard statistical metrics that can provide an EWS (e.g. variance, autocorrelation, skew)
- A suite of spectral EWS, which are based on the power spectrum [@bury2020detecting]
- Methods to apply deep learning classifiers for EWS [@bury2021deep]
- Integrated plotting and evaluation functions to quickly check the performance of EWS
- Comprehensive and interactive tutorials


Earlier versions of `ewstools` were used in the following publications:
- @bury2020detecting
- @bury2021deep




It complements a popular EWS package written in R [@dakos2012methods]. 
My hope that having an EWS toolbox in Python will allow for additional testing, 
and appeal to those who primarily work in Python. 




To date, it includes methods to detrend time series



More recently, deep learning 
classifiers have been trained and applied to detect bifurcations, with promising 
results [@bury2021deep]



`ewstools` makes use of several open-source Python packages, including
pandas [@mckinney2010data] for dataframe handling, 
numpy [@harris2020array] for fast numerical computing, 
plotly [@plotly] for visuliastion, 
lmfit [@newville2016lmfit] for least-squares minimisation, 
arch [@sheppard_2015_15681] for bootstrapping methods, 
statsmodels [@seabold2010statsmodels] and scipy [@virtanen2020scipy] for detrending methods, 
and TensorFlow [@abadi2016tensorflow] for deep learning.





# Statement of need



# Usage Example

```python
import ewstools

# Load data and get time series as a pandas Series object
df = pd.read_csv(‘data.csv’)
series = df['x']

# Initialise ewstools TimeSeries object and define transition time
ts = ewstools.TimeSeries(data=series, transition=440)

# Detrend time series
ts.detrend(method='Lowess', span=0.2)

# Compute desired EWS
ts.compute_var(rolling_window=0.5)
ts.compute_auto(lag=1, rolling_window=0.5)
ts.compute_auto(lag=2, rolling_window=0.5)

# Compute performance metrics
ts.compute_ktau()

# Plot results - can be saved as an interactive html file or as a static image
fig = ts.make_plotly()
```

![Output of plotting function in usage example.\label{fig:Figure 1}](figure1.png)


# Documentation

Documentation for `ewstools` is available at 
[https://ewstools.readthedocs.io/en/latest/](https://ewstools.readthedocs.io/en/latest/).
Tutorials in the form of Jupyter notebooks are available at
[https://github.com/ThomasMBury/ewstools/tree/main/tutorials](https://github.com/ThomasMBury/ewstools/tree/main/tutorials).


# Similar tools

The R package 
[earlywarnings](https://cran.r-project.org/web/packages/earlywarnings/index.html) [@dakos2012methods]
computes EWS from time series data, and the R package
[spatialwarnings](https://cran.r-project.org/web/packages/spatialwarnings/index.html) [@genin2018monitoring]
computes EWS from spatial data.


# Acknowledgements

This work is supported by the 
Fonds de Recherche du Québec Nature et technologies (FRQNT)
and Compute Canada. Earlier versions were supported by the 
Natural Sciences and Engineering Research Council of Canada (NSERC).


# References
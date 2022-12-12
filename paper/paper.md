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
This spurred massive interest in the subject of EWS from a wide range scientific disciplines.
Now, there exist a multitude of different EWS and associated methods for
anticipating bifurcations [@clements2018indicators].

More recently, deep learning 
classifiers have been trained and applied to detect bifurcations, with promising 
results [@bury2021deep]



`ewstools` provides an accessible toolbox for computing, analysing and 
visualising EWS in time seires data. The package provides:

- An intuitive, object-oriented framework to compute EWS in a given dataset
- Methods to detrend time series
- A suite of standard temporal EWS such as variance, autocorrelation, skew
- A suite of spectral EWS [@bury2020detecting]
- Methods to apply deep learning classifiers to detect and classify bifurcations [@bury2021deep]
- Integrated plotting and evaluation functions to quickly check performance of EWS
- Interactive tutorials in the form of Jupyter notebooks



`ewstools` makes use of several open-source Python packages, including
pandas [@mckinney2010data] for dataframe handling, 
numpy [@harris2020array] for fast numerical computing, 
plotly [@plotly] for visuliastion, 
lmfit [@newville2016lmfit] for least-squares minimisation, 
arch [@sheppard_2015_15681] for bootstrapping methods, 
statsmodels [@seabold2010statsmodels] and scipy [@virtanen2020scipy] for detrending methods, 
and TensorFlow [@abadi2016tensorflow] for deep learning.




# Statement of need

Critical transitions may occur in many disciplines, including ecology, medicine,
finance, and epidemiology, to name a few. As such, it is important that EWS are made 
widely accessible. To my knowledge, there are two other software packages developed for
computing EWS (
[earlywarnings](https://cran.r-project.org/web/packages/earlywarnings/index.html) by Dakos et al. @dakos2012methods,
and 
[spatialwarnings](https://cran.r-project.org/web/packages/spatialwarnings/index.html) by Génin et al. @genin2018monitoring),
both using the R programming language.
Given the recent surge in popularity of the Python programming language [@stanvcin2019overview],
it makes sense to have a package available for these users.
Moreover, `ewstools` implements recently developed methods [@bury2021deep] that make use of deep learning
and packages therein that are native to Python.



<!-- 
]), both using the R programming language. 




EWS are applicable to a wide range of scientific domains, making it important, it is important that they are made accessible
to researchers spanning many different scientific domains and coding backgrounds.
To my knowledge there exist two other software packages for computing EWS, 
both using the R programming language.
[earlywarnings](https://cran.r-project.org/web/packages/earlywarnings/index.html) [@dakos2012methods] is a
popular package for computing early warning signals in time series data, and 
[spatialwarnings](https://cran.r-project.org/web/packages/spatialwarnings/index.html) [@genin2018monitoring]
was developed to compute EWS in spatial data, with particular application to
ecosystem degradation.
Given the recent surge in popularity of the Python programming language [@stanvcin2019overview],
ewstools provides a convenient tool for these researchers and data scientists who primarily
work in Python.


I believe that an EWS package in Python will complement these existing packages, by allowing
for additional testing and cross validation of EWS, whilst also appealing to those
who primarily work in Python. 
 


The use of python in the area of data science and machine learning has reached
unprecedented levels, largely thanks to its ecosystem of open-source libraries [@stanvcin2019overview].
As such, it is important that a python package exists for EWS.


 -->


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




# Acknowledgements

This work is supported by the 
Fonds de Recherche du Québec Nature et technologies (FRQNT)
and Compute Canada. Earlier versions were supported by the 
Natural Sciences and Engineering Research Council of Canada (NSERC).


# References
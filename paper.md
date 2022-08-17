---
title: 'ewstools: A Python package for early warning signals of bifurcations in time series data.'
tags:
  - Python
  - time series
  - early warning signal
  - tipping point
  - dynamical system
  - bifurcation
  
authors:
  - name: Thomas M. Bury
    orcid: 0000-0003-1595-9444
    affiliation: "1, 2" # (Multiple affiliations must be quoted)
affiliations:
 - name: Department of Physiology, McGill University, Montréal, Canada
   index: 1
 - name: Department of Applied Mathematics, University of Waterloo, Waterloo, Canada
   index: 2
date: 17 August 2022
bibliography: paper.bib


# Summary

Many systems across nature and society have the capacity to undergo an abrupt and 
profound change in their dynamics. From a dynamical systemes perspective, these events 
are often associated with the crossing of a bifurcation. Early warning signals (EWS) 
for bifurcations are therefore in high demand. Two commonly used EWS for bifurcations 
are variance and lag-1 autocorrelation, that are expected to increase prior to many 
bifurcations due to critical slowing down [@scheffer2009early]. There now exist a 
wealth of other EWS based on changes in time series dynamics that are expected to occur 
prior to bifurcations [@clements2018indicators]. More recently, deep learning 
classifiers have been trained and applied to detect bifurcations, with promising 
results [@bury2021deep]



# Statement of need

`ewstools` is a Python package for computing, analysing, and visualising
early warning signals in time series data.



# Acknowledgements

TB acknowledge contributions from Chris Bauch on code for training deep learning 
classifiers. This project is currently supported by the
Fonds de recherche du Québec (FRQ), 
and has received funding in the past from the
Natural Sciences and Engineering Research Council of Canada (NSERC).


# References
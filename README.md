# early_warnings
A module to compute early warning signals (EWS) from time-series data.
Package requirements:
  - standard python libraries: numpy, pandas, scipy, matplotlib
  - LMFIT: download [here](https://lmfit.github.io/lmfit-py/installation.html)


## ews_compute.py
File for function `ews_compute`.  
`ews_compute` takes in Series data and outputs user-specified EWS in a DataFrame.

**Input** (default value)
- *raw_series* : pandas Series indexed by time 
- *roll_window* (0.25) : size of the rolling window (as a proportion of the length of the data)
- *smooth* (True) : if True, series data is detrended with a Gaussian kernel
- *band_width* (0.2) : bandwidth of Gaussian kernel
- *ews* ( ['*var*', '*ac*', '*smax*'] ) : list of strings corresponding to the desired EWS. Options include
  - '*var*'   : Variance
  - '*ac*'    : Autocorrelation
  - '*sd*'    : Standard deviation
  - '*cv*'    : Coefficient of variation
  - '*skew*'  : Skewness
  - '*kurt*'  : Kurtosis
  - '*smax*'  : Peak in the power spectrum
  - '*cf*'    : Coherence factor
  - '*aic*'   : AIC weights
- lag_times ( [1] ) : list of integers corresponding to the desired lag times for AC
    
**Output**
- DataFrame indexed by time with columns corresponding to each EWS



## ews_compute_test.py
An example script that runs `ews_compute`, computes kendall tau values, and plots results.

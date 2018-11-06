# ews_functions
A collection of functions to compute early warning signals (EWS) from time-series data.

## roll_ews_std.py
Function to compute the EWS that have become standard practise, over a rolling window.
The pandas library is used to compute the metrics over a rolling window.

Input (default)
- raw_series : pandas Series indexed by time 
- roll_window (0.25) : size of the rolling window (as a proportion of the length of the data)
- smooth (True) : if True, series data is detrended with a Gaussian kernel
- band_width (0.2) : bandwidth of Gaussian kernel
- ews (['var,'ac'] : list of strings corresponding to the desired EWS.
- Options include
  - 'var'   : Variance
  - 'ac'    : Autocorrelation
  - 'sd'    : Standard deviation
  - 'cv'    : Coefficient of variation
  - 'skew'  : Skewness
  - 'kurt'  : Kurtosis

-hello
-lag_times : list of integers corresponding to the desired lag times for AC
    
Output
- DataFrame indexed by time with columns csp to each EWS
  - nested?

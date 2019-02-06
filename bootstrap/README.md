# Directory for bootstrapping methods

We investigate the use of bootstrapping time-series data to obtain confidence measures of EWS statistics


## roll_bootstrap.py
File for function `roll_bootstrap`.  
`roll_bootstrap` takes in time-series data and:
  (1) computes the residuals from a Loess smoothing of the data
  (2) computes bootstrap samples of the residuals in a rolling window


**Input** (default value)
- *raw_series* : pandas Series indexed by time 
- *span* (0.1) : proportion of the time-series data used for Losess filtering
- *roll_window* (0.25) : size of the rolling window (as a proportion of the length of the data)
- *roll_offset* (1) : number of data points to shift the rollling window upon each iteration
- *upto* ('Full') : if 'Full', uses entire time-series, otherwise uses time-series up to time upto
- *n_samples* (20) : number of bootstrapped samples to compute for each window iteration
- *bs_type* ('Stationary') : type of bootstrapping to perform from ['Stationary', 'Circular']
- *block_size*  (10) : size of resampling blocks used in bootstrapping.
    
**Output**
- DataFrame indexed by realtime, sample number, and window time








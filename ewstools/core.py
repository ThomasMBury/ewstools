#################################################################################################################
# ewstools
# Description: Python package for computing, analysing and visualising
# early warning signals (EWS) in time-series data
# Author: Thomas M Bury
# Web: https://www.thomasbury.net/
# Code repo: https://github.com/ThomasMBury/ewstools
# Documentation: https://ewstools.readthedocs.io/
#
# The MIT License (MIT)
#
# Copyright (c) 2019 Thomas Bury
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#################################################################################################################


# ---------------------------------
# Import relevant packages
# --------------------------------

# For numeric computation and DataFrames
import numpy as np
import pandas as pd

# Module for block-bootstrapping time-series
from arch.bootstrap import StationaryBootstrap, CircularBlockBootstrap, IIDBootstrap

# For detrending time-series
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.ndimage import gaussian_filter as gf

# Import functions from other files in package
import ewstools.helpers as helpers

# Plotly modules for visualisation
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# For deprecating old functions
import deprecation



# ---------------
# Classes
# --------------

class TimeSeries:
    """
    Univariate time series data on which to compute early warning signals.

    Parameters
    ----------
    data : list, numpy.ndarray or pandas.Series
        1D list of data that makes up time series. If given as a pandas.Series,
        then the index, which should represent time values, is carried over.
    transition : float
        Time value at which transition occurs, if any. If defined, early warning signals
        are only computed up to this point in the time series.

    """

    def __init__(self, data, transition=None):

        # Put data into a pandas DataFrame
        if type(data) in [list, np.ndarray]:
            df_state = pd.DataFrame({'time': np.arange(len(data)), 'state': data})
            df_state.set_index('time', inplace=True)
        
        # If given as pandas series, carry index forward
        elif type(data) == pd.Series:
            df_state = pd.DataFrame({'state': data.values})
            df_state.index = data.index
            # Rename index if no name given
            if not df_state.index.name:
                df_state.index.name = 'time'
                
        # If data is not provided as either of these, flag error
        else:
            print('\nERROR: data has been provided as type {}'.format(type(data)))
            print('Make sure to provide data as either a list, np.ndarray or pd.Series\n')
        
        # Set state and transition attributes
        self.state = df_state
        self.transition = float(transition) if transition else transition
        
        # Initialise other attributes
        self.ews = pd.DataFrame(index=df_state.index)
        self.dl_preds = pd.DataFrame()
        self.ktau = dict()
        self.pspec = None
        self.pspec_fits = None
        self.ews_spec = pd.DataFrame()
        

    def detrend(self, method='Gaussian', bandwidth=0.2, span=0.2):
        '''
        Detrend the time series using a chosen method.
        Add column to the dataframe for 'smoothing' and 'residuals'

        Parameters
        ----------
        method : str, optional
            Method of detrending to use. 
            Select from ['Gaussian', 'Lowess']
            The default is 'Gaussian'.
        bandwidth : float, optional
            Bandwidth of Gaussian kernel. Provide as a proportion of data length
            or as a number of data points. As in the R function ksmooth
            (used by the earlywarnings package in R), we define the bandwidth
            such that the kernel has its quartiles at +/- 0.25*bandwidth.
            The default is 0.2.
        span : float, optional
            Span of time-series data used for Lowess filtering. Provide as a 
            proportion of data length or as a number of data points. 
            The default is 0.2.

        Returns
        -------
        None.

        '''

        # Get time series data prior to transition
        if self.transition:
            df_pre = self.state[self.state.index <= self.transition]
        else:
            df_pre = self.state


        if method == 'Gaussian':
            # Get size of bandwidth in terms of num. datapoints if given as a proportion
            if 0 < bandwidth <= 1:
                bw_num = bandwidth * len(df_pre)
            else:
                bw_num = bandwidth

            # Use the gaussian_filter function provided by Scipy
            # Standard deviation of kernel given bandwidth
            # Note that for a Gaussian, quartiles are at +/- 0.675*sigma
            sigma = (0.25 / 0.675) * bw_num
            smooth_values = gf(df_pre['state'].values, sigma=sigma, mode='reflect')
            smooth_series = pd.Series(smooth_values, index=df_pre.index)


        if method == 'Lowess':
            # Convert span to a proportion of the length of the data
            if not 0 < span <= 1:
                span_prop = span / len(df_pre)
            else:
                span_prop = span

            smooth_values = lowess(df_pre['state'].values, 
                                   df_pre.index.values, 
                                   frac=span_prop)[:,1]
            smooth_series = pd.Series(smooth_values, index=df_pre.index)

    
        # Add smoothed data and residuals to the 'state' DataFrame
        self.state["smoothing"] = smooth_series
        self.state["residuals"] = self.state["state"] - self.state["smoothing"]



    def compute_var(self, rolling_window=0.25):
        '''
        Compute variance over a rolling window.
        If residuals have not been computed, computation will be
        performed over state variable.
        
        Put into 'ews' dataframe

        Parameters
        ----------
        rolling_window : float
            Length of rolling window used to compute variance. Can be specified
            as an absolute value or as a proportion of the length of the
            data being analysed. Default is 0.25.
            
        Returns
        -------
        None.

        '''

        # Get time series data prior to transition
        if self.transition:
            df_pre = self.state[self.state.index <= self.transition]
        else:
            df_pre = self.state
    
        # Get absolute size of rollling window if given as a proportion
        if 0 < rolling_window <= 1:
            rw_absolute = int(rolling_window * len(df_pre))
        else:
            rw_absolute = rolling_window

        # If residuals column exists, compute over residuals.
        if 'residuals' in df_pre.columns:
            var_values = df_pre['residuals'].rolling(window=rw_absolute).var()
        # Else, compute over state variable
        else:
            var_values = df_pre['state'].rolling(window=rw_absolute).var()
        
        self.ews['variance'] = var_values
        
        
        
    def compute_std(self, rolling_window=0.25):
        '''
        Compute standard deviation over a rolling window.
        If residuals have not been computed, computation will be
        performed over state variable.
        
        Put into 'ews' dataframe

        Parameters
        ----------
        rolling_window : float
            Length of rolling window used to compute variance. Can be specified
            as an absolute value or as a proportion of the length of the
            data being analysed. Default is 0.25.
            
        Returns
        -------
        None.

        '''

        # Get time series data prior to transition
        if self.transition:
            df_pre = self.state[self.state.index <= self.transition]
        else:
            df_pre = self.state
    
        # Get absolute size of rollling window if given as a proportion
        if 0 < rolling_window <= 1:
            rw_absolute = int(rolling_window * len(df_pre))
        else:
            rw_absolute = rolling_window

        # If residuals column exists, compute over residuals.
        if 'residuals' in df_pre.columns:
            std_values = df_pre['residuals'].rolling(window=rw_absolute).std()
        # Else, compute over state variable
        else:
            std_values = df_pre['state'].rolling(window=rw_absolute).std()
        
        self.ews['std'] = std_values        


    def compute_cv(self, rolling_window=0.25):
        '''
        Compute coefficient of variation over a rolling window.
        This is the standard deviation of the residuals divided by the
        mean of the state variable.
        If residuals have not been computed, computation will be
        performed over state variable.
        
        Put into 'ews' dataframe

        Parameters
        ----------
        rolling_window : float
            Length of rolling window used to compute variance. Can be specified
            as an absolute value or as a proportion of the length of the
            data being analysed. Default is 0.25.
            
        Returns
        -------
        None.

        '''

        # Get time series data prior to transition
        if self.transition:
            df_pre = self.state[self.state.index <= self.transition]
        else:
            df_pre = self.state
    
        # Get absolute size of rollling window if given as a proportion
        if 0 < rolling_window <= 1:
            rw_absolute = int(rolling_window * len(df_pre))
        else:
            rw_absolute = rolling_window

        # Get standard deviation values
        # If residuals column exists, compute over residuals.
        if 'residuals' in df_pre.columns:
            std_values = df_pre['residuals'].rolling(window=rw_absolute).std()
        # Else, compute over state variable
        else:
            std_values = df_pre['state'].rolling(window=rw_absolute).std()
        
        # Get mean values from state variable
        mean_values = df_pre['state'].rolling(window=rw_absolute).mean()
        
        cv_values = std_values/mean_values
        
        self.ews['cv'] = cv_values        






    def compute_auto(self, rolling_window=0.25, lag=1):
        '''
        Compute autocorrelation over a rolling window. Add to dataframe.

        Parameters
        ----------
        rolling_window : float
            Length of rolling window used to compute autocorrelation. Can be specified
            as an absolute value or as a proportion of the length of the
            data being analysed. Default is 0.25.
        lag : int
            Lag of autocorrelation
            
        Returns
        -------
        None.

        '''

        # Get time series data prior to transition
        if self.transition:
            df_pre = self.state[self.state.index <= self.transition]
        else:
            df_pre = self.state
    
        # Get absolute size of rollling window if given as a proportion
        if 0 < rolling_window <= 1:
            rw_absolute = int(rolling_window * len(df_pre))
        else:
            rw_absolute = rolling_window

        # If residuals column exists, compute over residuals.
        if 'residuals' in df_pre.columns:
            ac_values = df_pre['residuals'].rolling(window=rw_absolute).apply(
                func=lambda x: pd.Series(x).autocorr(lag=lag), raw=True
            )
        # Else, compute over state variable
        else:
            ac_values = df_pre['state'].rolling(window=rw_absolute).apply(
                func=lambda x: pd.Series(x).autocorr(lag=lag), raw=True
            )                    
            
        self.ews['ac{}'.format(lag)] = ac_values


    def compute_skew(self, rolling_window=0.25):
        '''
        Compute skew over a rolling window.
        If residuals have not been computed, computation will be
        performed over state variable.
        
        Add to dataframe.

        Parameters
        ----------
        rolling_window : float
            Length of rolling window used to compute variance. Can be specified
            as an absolute value or as a proportion of the length of the
            data being analysed. Default is 0.25.
            
        Returns
        -------
        None.

        '''
        
        # Get time series data prior to transition
        if self.transition:
            df_pre = self.state[self.state.index <= self.transition]
        else:
            df_pre = self.state
    
        # Get absolute size of rollling window if given as a proportion
        if 0 < rolling_window <= 1:
            rw_absolute = int(rolling_window * len(df_pre))
        else:
            rw_absolute = rolling_window

        # If residuals column exists, compute over residuals.
        if 'residuals' in df_pre.columns:
            skew_values = df_pre['residuals'].rolling(window=rw_absolute).skew()
        # Else, compute over state variable
        else:
            skew_values = df_pre['state'].rolling(window=rw_absolute).skew()
        
        self.ews['skew'] = skew_values



    def compute_kurt(self, rolling_window=0.25):
        '''
        Compute kurtosis over a rolling window.
        If residuals have not been computed, computation will be
        performed over state variable.
        
        Add to dataframe.

        Parameters
        ----------
        rolling_window : float
            Length of rolling window used to compute variance. Can be specified
            as an absolute value or as a proportion of the length of the
            data being analysed. Default is 0.25.
            
        Returns
        -------
        None.

        '''

        # Get time series data prior to transition
        if self.transition:
            df_pre = self.state[self.state.index <= self.transition]
        else:
            df_pre = self.state
    
        # Get absolute size of rollling window if given as a proportion
        if 0 < rolling_window <= 1:
            rw_absolute = int(rolling_window * len(df_pre))
        else:
            rw_absolute = rolling_window

        # If residuals column exists, compute over residuals.
        if 'residuals' in df_pre.columns:
            kurt_values = df_pre['residuals'].rolling(window=rw_absolute).kurt()
        # Else, compute over state variable
        else:
            kurt_values = df_pre['state'].rolling(window=rw_absolute).kurt()
        
        self.ews['kurtosis'] = kurt_values




    def compute_ktau(self, tmin='earliest', tmax='latest'):
        '''
        Compute kendall tau values of CSD-based EWS.
        Output is placed in the attribute *ktau*, which is a Python 
        dictionary contianing Kendall tau values for each CSD-based EWS.

        Parameters
        ----------
        tmin : float or 'earliest'
            Start time for kendall tau computation. If 'earliest', then
            time is taken as earliest time point in EWS time series.
        tmax : float or 'latest'
            End time for kendall tau computation. If 'latest', then time is
            taken as latest time point in EWS time series, which could be
            the end of the state time series, or a defined transtion point.

        '''
        
        
        # Get tmin and tmax values if using extrema
        if tmin == 'earliest':
            tmin = self.ews.dropna().index[0]
        
        if tmax == 'latest':
            tmax = self.ews.dropna().index[-1]
        
        # Get cropped data
        df_ews = self.ews[(self.ews.index >= tmin) &\
                          (self.ews.index <= tmax)].copy()
        
        # Include smax in Kendall tau computation if it exists
        if 'smax' in self.ews_spec.columns:
            df_ews = df_ews.join(self.ews_spec['smax'])
            
        # Make a series with the time values
        time_values = pd.Series(data=df_ews.index, index=df_ews.index)
        ktau_out = df_ews.corrwith(time_values, method="kendall", axis=0)
        self.ktau = dict(ktau_out)
        

    def apply_classifier(self, classifier, tmin, tmax, 
                        name='c1', verbose=1):
        '''
        Apply a deep learning classifier to the residual time series from
        tmin to tmax. 
        If time series has not been detrended, apply to the raw data.
        Predictions from the classifier are saved into the attribute dl_preds.

        Parameters
        ----------
        classifier : keras.engine.sequential.Sequential
            Tensorflow classifier
        tmin : float
            Earliest time in time series segment
        tmax : float
            Latest time in time series segment    
        name : str, optional
            Name assigned to the classifier. The default is 'c1'.
        verbose : int, optional
            Verbosity of update messages from TensorFlow. 0 = silent, 1 = progress bar, 2 = single line.
            The default is 1.
        
        Returns
        -------
        None.

        '''
        
        # Length of time series required as input to classifier
        input_len = classifier.layers[0].input_shape[1]
        
        # Get time series segment. Use residuals if detrending performed.
        # Otherwise use state variable.
        if 'residuals' in self.state.columns:
            data = self.state[(self.state.index >= tmin) &\
                                (self.state.index <= tmax)]['residuals'].values      
        else:
            data = self.state[(self.state.index >= tmin) &\
                                (self.state.index <= tmax)]['state'].values
        
        # If time series segment is larger than input dimension of classifier
        if len(data) > input_len:
            print('ERROR: Length of time series segment is too long for the classifier. You can modify the tmin and tmax parameters to select a smaller segment.')
            return
        
        # Normalise by mean of absolute value
        data_norm = data/(abs(data).mean())

        # Prepend with zeros to make appropriate length for classifier
        num_zeros = input_len - len(data_norm)
        input_data = np.concatenate((np.zeros(num_zeros),data_norm)).reshape(1,-1, 1)
        
        # Get DL prediction
        dl_pred = classifier.predict(input_data, verbose=verbose)[0]
        # Put info into dataframe
        dict_dl_pred = {i:val for (i,val) in zip(np.arange(len(dl_pred)), dl_pred)}
        dict_dl_pred['time'] = tmax
        dict_dl_pred['classifier'] = name      
        df_dl_pred = pd.DataFrame(dict_dl_pred, index=[len(self.dl_preds)])

        # Append to dataframe contiaining DL predictions
        self.dl_preds = pd.concat([self.dl_preds, df_dl_pred], ignore_index=True)



    def apply_classifier_inc(self, classifier, inc=10, name='c1', verbose=1):
        '''
        Apply a deep learning classifier to incrementally increasing time
        series lengths. First prediction is made on time series segment from
        data point at index 0 to data point at index inc. Second prediction
        is made on time series segment from 0 to 2*inc. Third prediction is
        made on time series segment from 0 to 3*inc. Etc.

        Parameters
        ----------
        classifier : keras.engine.sequential.Sequential
            TensorFlow classifier.
        inc : int, optional
            Increment to tmax (the end time of each time series segment) after each classification.
            The default is 10.
        name : str, optional
            Name assigned to the classifier. The default is 'c1'.
        verbose : int, optional
            Verbosity of update messages from TensorFlow. 
            0 = silent, 1 = progress bar, 2 = single line.
            The default is 1.

        Returns
        -------
        None.

        '''
        
        tmin = self.state.index[0]
        tend = self.state.index[-1] if not self.transition else self.transition
        
        # Tmax values for each time series segment
        tmax_vals = np.arange(tmin, tend, inc)
        for tmax in tmax_vals:
            self.apply_classifier(classifier, name=name, tmin=tmin, tmax=tmax, 
                                verbose=verbose)
    
    


    def clear_dl_preds(self):
        '''
        Clear the attribute *dl_preds*

        Returns
        -------
        None.
        '''
        
        self.dl_preds = pd.DataFrame()



    def compute_spectrum(self, rolling_window=0.25, ham_length=40,
                         ham_offset=0.5, pspec_roll_offset=20,
                         w_cutoff=1):
        '''
        Compute the power spectrum over a rolling window.
        Stores the power spectra as a DataFrame in TimeSeries.pspec

        Parameters
        ----------
        rolling_window : float
            Length of rolling window used to compute variance. Can be specified
            as an absolute value or as a proportion of the length of the
            data being analysed. Default is 0.25.
        ham_length : int
            Length of the Hamming window used to compute the power spectrum. 
            The default is 40.
        ham_offset : float
            Hamming offset as a proportion of the Hamming window size. 
            The default is 0.5.
        pspec_roll_offset : int
            Rolling window offset used when computing power spectra. Power spectrum
            computation is relatively expensive so this is rarely taken as 1
            (as is the case for the other EWS). The default is 20.
        w_cutoff : floag
            Cutoff frequency used in power spectrum. Given as a proportion of the
            maximum permissable frequency in the empirical power spectrum.

        Returns
        -------
        None.

        '''
        
        # Get time series data prior to transition
        if self.transition:
            df_pre = self.state[self.state.index <= self.transition]
        else:
            df_pre = self.state

        # Get absolute size of rollling window if given as a proportion
        if 0 < rolling_window <= 1:
            rw_absolute = int(rolling_window * len(df_pre))
        else:
            rw_absolute = rolling_window

        # If residuals column exists, compute over residuals.
        if 'residuals' in df_pre.columns:
            series = df_pre['residuals']
        # Else, compute over state variable
        else:
            series = df_pre['state']

        # Number of components in the residual time-series
        num_comps = len(df_pre)

        # Rolling window offset (can make larger to save on computation time)
        roll_offset = int(pspec_roll_offset)
        # Time separation between data points (need for frequency values of power spectrum)
        dt = series.index[1] - series.index[0]

        # Initialise a list to store the power spectra
        list_pspec = []

        # Loop through window locations shifted by roll_offset
        for k in np.arange(0, num_comps - (rw_absolute - 1), roll_offset):

            # Select subset of series contained in window
            window_series = series.iloc[k : k + rw_absolute]
            # Asisgn the time value for the metrics (right end point of window)
            t_point = series.index[k + (rw_absolute - 1)]

            ## Compute the power spectrum using function pspec_welch
            pspec = helpers.pspec_welch(
                window_series,
                dt,
                ham_length=ham_length,
                ham_offset=ham_offset,
                w_cutoff=w_cutoff,
                scaling="spectrum",
            )

            ## Create DataFrame for empirical power spectrum
            df_pspec_empirical = pspec.to_frame().reset_index()
            # Rename column
            df_pspec_empirical.rename(
                columns={"Power spectrum": "empirical"}, inplace=True
            )
            # Include a column for the time-stamp
            df_pspec_empirical['time'] = t_point * np.ones(len(pspec))
            list_pspec.append(df_pspec_empirical)
        
        # Concatenate power spectra dataframes and store in attribute pspec
        self.pspec = pd.concat(list_pspec)



    def compute_smax(self):
        '''
        Compute Smax (the maximum power in the power spectrum).
        This can only be applied after applying compute_spectrum().
        Stores Smax values in TimeSeries.ews_spec

        Returns
        -------
        None.

        '''
        
        if self.pspec is None:
            print('ERROR: The power spectrum must be computed before computing\
                  spectral EWS such as Smax. The power spectrum can be\
                  computed using compute_pspec()')
        
        smax_values = self.pspec[['time','power']].groupby(['time']).max()
        self.ews_spec['smax'] = smax_values




    def compute_spec_type(self, sweep=False):
        '''
        Fit the analytical forms of the Fold, Hopf and Null power spectrum
        to the empirical power spectrum. Get Akaike Information Criterion
        (AIC) weights to determine best fit.
        Store AIC weights in TimeSeries.ews_spec
        Store fitted power spectra in TimeSeries.pspec_fits

        Parameters
        ----------
        sweep : bool
            If 'True', sweep over a range of intialisation
            parameters when optimising to compute AIC scores, at the expense of
            longer computation. If 'False', intialisation parameter is taken as the
            'best guess'. The default is False.

        Returns
        -------
        None.

        '''
        
        
        if self.pspec is None:
            print('ERROR: The power spectrum must be computed before computing\
                  the spectrum type. The power spectrum can be\
                  computed using compute_pspec()')    
    
        list_aic = []
        list_pspec_fits = []
        
        # Loop through time values
        for time in self.pspec['time'].unique():
            pspec = self.pspec[self.pspec['time']==time]
            pspec_series = pspec.set_index('frequency')['power']
            
            ## Compute the AIC values
            metrics = helpers.pspec_metrics(pspec_series, ews=['aic'], sweep=sweep)
            dict_aic = {}
            dict_aic['fold'] = metrics['AIC fold']
            dict_aic['hopf'] = metrics['AIC hopf']
            dict_aic['null'] = metrics['AIC null']
            dict_aic['time'] = time
            list_aic.append(dict_aic)
            
            # Generate data to plot the fitted power spectra

            # Create fine-scale frequency values
            wVals = np.linspace(min(pspec.index), max(pspec.index), 100)
            # Fold fit
            pspec_fold = helpers.psd_fold(
                wVals,
                metrics["Params fold"]["sigma"],
                metrics["Params fold"]["lam"],
            )
            # Hopf fit
            pspec_hopf = helpers.psd_hopf(
                wVals,
                metrics["Params hopf"]["sigma"],
                metrics["Params hopf"]["mu"],
                metrics["Params hopf"]["w0"],
            )
            # Null fit
            pspec_null = helpers.psd_null(wVals, metrics["Params null"]["sigma"])

            ## Put spectrum fits into a dataframe
            dict_fits = {
                'time': time * np.ones(len(wVals)),
                'frequency': wVals,
                "fit fold": pspec_fold,
                "fit hopf": pspec_hopf,
                "fit null": pspec_null,
            }
            df_pspec_fits = pd.DataFrame(dict_fits)
            list_pspec_fits.append(df_pspec_fits)

        # Concatenate
        df_aic = pd.DataFrame(list_aic).set_index('time')
        df_pspec_fits = pd.concat(list_pspec_fits)

        self.ews_spec = self.ews_spec.join(df_aic)
        self.pspec_fits = df_pspec_fits
    



    def make_plotly(self, kendall_tau=True, ens_avg=False):
        '''
        Make an interactive Plotly figure to view all EWS computed

        Parameters
        ----------

        kendall_tau : bool, optional
            Set as true to show Kendall tau values (if they have been computed)
            Default is True.
        ens_avg : bool, optional
            Plot the ensenble average of DL predictions. 
            Default is False.

        Returns
        -------
        Plotly figure

        '''
        
        # Trace colours
        colours = px.colors.qualitative.Plotly
        # Count number of panels for Plotly subplots
        row_count = 0
        # Always a row for state varialbe
        row_count += 1
        # Row for autocorrelation
        ac_labels = [s for s in self.ews.columns if s[:2]=='ac']
        if len(ac_labels)>0:
            row_count+=1
        # Row for each other EWS
        row_count += len(self.ews.columns)-len(ac_labels)
        # Row for Smax if included
        if 'smax' in self.ews_spec.columns:
            row_count+=1
        # Row for AIC weights if computed
        if 'fold' in self.ews_spec.columns:
            row_count+=1
        # Row for DL predictions if computed
        if len(self.dl_preds.columns)>0:
            row_count+=1
        
        num_rows = row_count
        row_count = 1 # reset row counter
            
        # Make Plotly subplots frame
        fig = make_subplots(rows=num_rows, cols=1, 
                            shared_xaxes=True,
                            x_title='Time',
                            vertical_spacing=0.02
                            )
        
        # Plot state variable
        fig.add_trace(
            go.Scatter(x=self.state.index.values,
                       y=self.state['state'].values,
                       name='state',
                       ),
            row=row_count, col=1
            )
        fig.update_yaxes(title='State', row=row_count)

        
        # Plot smoothing if computed
        if 'smoothing' in self.state.columns:
            fig.add_trace(
                go.Scatter(x=self.state.index.values,
                           y=self.state['smoothing'].values,
                           name='smoothing',
                           ),
                row=row_count, col=1
                )
        
        # Plot variance if computed
        if 'variance' in self.ews.columns:
            row_count += 1
            
            # Add kendall tau to name
            if kendall_tau and ('variance' in self.ktau.keys()):
                ktau = self.ktau['variance']
                name = 'variance (ktau={:.2f})'.format(ktau)
            else:
                name = 'variance'
            
            fig.add_trace(
                go.Scatter(x=self.ews.index.values,
                           y=self.ews['variance'].values,
                           name=name,
                           ),
                row=row_count, col=1
                )  
            fig.update_yaxes(title='Variance', row=row_count)
            
        # Plot standard deviation if computed
        if 'std' in self.ews.columns:
            row_count += 1
            
            # Add kendall tau to name
            if kendall_tau and ('std' in self.ktau.keys()):
                ktau = self.ktau['std']
                name = 'std (ktau={:.2f})'.format(ktau)
            else:
                name = 'std'
            
            fig.add_trace(
                go.Scatter(x=self.ews.index.values,
                           y=self.ews['std'].values,
                           name=name,
                           ),
                row=row_count, col=1
                )  
            fig.update_yaxes(title='St. Dev.', row=row_count)
            
        # Plot coefficient of variation if computed
        if 'cv' in self.ews.columns:
            row_count += 1
            
            # Add kendall tau to name
            if kendall_tau and ('cv' in self.ktau.keys()):
                ktau = self.ktau['cv']
                name = 'cv (ktau={:.2f})'.format(ktau)
            else:
                name = 'cv'
            
            fig.add_trace(
                go.Scatter(x=self.ews.index.values,
                           y=self.ews['cv'].values,
                           name=name,
                           ),
                row=row_count, col=1
                )  
            fig.update_yaxes(title='Coeff. of Var.', row=row_count)
            
                                    
            
        # Plot autocorrelation metrics if computed
        if len(ac_labels)!=0:
            row_count+=1
        for ac_label in ac_labels:
            
            # Add kendall tau to name
            if kendall_tau and (ac_label in self.ktau.keys()):
                ktau = self.ktau[ac_label]
                name = '{} (ktau={:.2f})'.format(ac_label, ktau)
            else:
                name = ac_label            
            
            fig.add_trace(
                go.Scatter(x=self.ews.index.values,
                           y=self.ews[ac_label].values,
                           name=name,
                           ),
                row=row_count, col=1
                )  
            fig.update_yaxes(title='Autocorrelation', row=row_count)
            
                     
        # Plot skew if computed
        if 'skew' in self.ews.columns:
            row_count += 1
            
            # Add kendall tau to name
            if kendall_tau and ('skew' in self.ktau.keys()):
                ktau = self.ktau['skew']
                name = 'skew (ktau={:.2f})'.format(ktau)
            else:
                name = 'skew'
                
            fig.add_trace(
                go.Scatter(x=self.ews.index.values,
                           y=self.ews['skew'].values,
                           name=name,
                           ),
                row=row_count, col=1
                )  
            fig.update_yaxes(title='Skew', row=row_count)
                        
        # Plot kurtosis if computed
        if 'kurtosis' in self.ews.columns:
            row_count += 1
            
            # Add kendall tau to name
            if kendall_tau and ('kurtosis' in self.ktau.keys()):
                ktau = self.ktau['kurtosis']
                name = 'kurtosis (ktau={:.2f})'.format(ktau)
            else:
                name = 'kurtosis'   
                
            fig.add_trace(
                go.Scatter(x=self.ews.index.values,
                           y=self.ews['kurtosis'].values,
                           name=name,
                           ),
                row=row_count, col=1
                )  
            fig.update_yaxes(title='Kurtosis', row=row_count)
               
            
        # Plot Smax if computd
        if 'smax' in self.ews_spec.columns:
            row_count += 1
            
            # Add kendall tau to name
            if kendall_tau and ('smax' in self.ktau.keys()):
                ktau = self.ktau['smax']
                name = 'smax (ktau={:.2f})'.format(ktau)
            else:
                name = 'smax'   
                            
            fig.add_trace(
                go.Scatter(x=self.ews_spec.index.values,
                           y=self.ews_spec['smax'].values,
                           name=name,
                           ),
                row=row_count, col=1
                )  
            fig.update_yaxes(title='Smax', row=row_count)
                          
            
        # Plot AIC weights if computd
        if 'fold' in self.ews_spec.columns:
            row_count += 1
            aic_labels = ['fold', 'hopf', 'null']
            for aic_label in aic_labels:
                fig.add_trace(
                    go.Scatter(x=self.ews_spec.index.values,
                               y=self.ews_spec[aic_label].values,
                               name=aic_label,
                               ),
                    row=row_count, col=1
                    )  
                
            fig.update_yaxes(title='AIC weights', row=row_count)
                       
            
                        
        # Plot DL predictions if computed
        if len(self.dl_preds)>0:
            row_count += 1
            class_labels = [s for s in self.dl_preds.columns if s not in ['time', 'classifier']]
            classifiers = self.dl_preds['classifier'].unique()
            
            # If plotting predictions from every classifier
            if not ens_avg:
                # Loop through class labels and classifier names
                for idx, class_label in enumerate(class_labels):
                    for idx2, classifier in enumerate(classifiers):
                        df_plot = self.dl_preds[self.dl_preds['classifier']==classifier]
                        fig.add_trace(
                            go.Scatter(x=df_plot['time'].values,
                                       y=df_plot[class_label].values,
                                       name='DL class {}'.format(class_label),
                                       legendgroup='DL class {}'.format(class_label),
                                       showlegend=True if idx2==0 else False,
                                       line=dict(color=colours[idx])
                                       ),
                            row=row_count, col=1
                            )
                        
            # If plotting ensemble average over classifiers
            else:
                df_plot = self.dl_preds.groupby(['time']).mean().reset_index()
                # Loop through class labels
                for idx, class_label in enumerate(class_labels):
                    fig.add_trace(
                        go.Scatter(x=df_plot['time'].values,
                                   y=df_plot[class_label].values,
                                   name='DL class {}'.format(class_label),
                                   line=dict(color=colours[idx])
                                   ),
                        row=row_count, col=1
                        )    
                    
            fig.update_yaxes(title='DL predictions', row=row_count)
                              



        # Set figure dimensions 
        fig.update_layout(height=200*num_rows,
                          width=800)
        
        # Separation between legend entries
        fig.layout.legend.tracegroupgap = 0

        return fig






# ---------------
# Functions
# ---------------


@deprecation.deprecated(deprecated_in="2.0", removed_in="2.1",
                        current_version="2.0",
                        details="Please use the ewstools.TimeSeries object and its associated methods to compute EWS instead. A tutorial on how to do so is provided in the tutorial_info ipython notebook at https://github.com/ThomasMBury/ewstools/tree/main/tutorials."
                        )

def ews_compute(
    raw_series,
    roll_window=0.4,
    smooth="Lowess",
    span=0.1,
    band_width=0.2,
    upto="Full",
    ews=["var", "ac"],
    lag_times=[1],
    ham_length=40,
    ham_offset=0.5,
    pspec_roll_offset=20,
    w_cutoff=1,
    aic=["Fold", "Hopf", "Null"],
    sweep=False,
    ktau_time="Start",
):

    """
    Compute temporal and spectral EWS from time-series data.

    Args
    ----
    raw_series: pd.Series
        Time-series data to analyse. Indexed by time.
    roll_window: float
        Rolling window size as a proportion of the length of the time-series
        data.
    smooth: {'Gaussian', 'Lowess', 'None'}
        Type of detrending.
    band_width: float
        Bandwidth of Gaussian kernel. Provide as a proportion of data length
        or as absolute value. As in the R function ksmooth
        (used by the earlywarnings package in R), we define the bandwidth
        such that the kernel has its quartiles at +/- 0.25*bandwidth.
    span: float
        Span of time-series data used for Lowess filtering. Taken as a
        proportion of time-series length if in (0,1), otherwise taken as
        absolute.
    upto: int or 'Full'
        Time up to which EWS are computed. Enter 'Full' to use
        the entire time-series. Otherwise enter a time value.
    ews: list of {'var', 'ac', 'sd', 'cv', 'skew', 'kurt', 'smax', 'cf', 'aic','smax/var', 'smax/mean'}.
                List of EWS to compute. Options include variance ('var'),
                autocorrelation ('ac'), standard deviation ('sd'), coefficient
                of variation ('cv'), skewness ('skew'), kurtosis ('kurt'), peak in
                the power spectrum ('smax'), coherence factor ('cf'), AIC weights ('aic'),
                ratio of peak in power spectrum and variance ('smax/var'), ratio of
                peak in power spectrum and mean of time series (smax/mean')
    lag_times: list of int
        List of lag times at which to compute autocorrelation.
    ham_length: int
        Length of the Hamming window used to compute the power spectrum.
    ham_offset: float
        Hamming offset as a proportion of the Hamming window size.
    pspec_roll_offset: int
        Rolling window offset used when computing power spectra. Power spectrum
        computation is relatively expensive so this is rarely taken as 1
        (as is the case for the other EWS).
    w_cutoff: float
        Cutoff frequency used in power spectrum. Given as a proportion of the
        maximum permissable frequency in the empirical power spectrum.
    aic: AIC values to compute
    sweep: bool
        If 'True', sweep over a range of intialisation
        parameters when optimising to compute AIC scores, at the expense of
        longer computation. If 'False', intialisation parameter is taken as the
        'best guess'.
    ktau_time: float or 'Start'
        Time from which to compute Kendall tau values. Defaults to the start of
        the EWS data.

    Returns
    --------
    dict of pd.DataFrames:
        A dictionary with the following entries.
        **'EWS metrics':** A DataFrame indexed by time with columns corresopnding
        to each EWS.
        **'Power spectrum':** A DataFrame of the measured power spectra and the best fits
        used to give the AIC weights. Indexed by time.
        **'Kendall tau':** A DataFrame of the Kendall tau values for each EWS metric.

    """

    # Initialise a DataFrame to store EWS data - indexed by time
    df_ews = pd.DataFrame(raw_series)
    df_ews.columns = ["State variable"]
    df_ews.index.rename('time', inplace=True)

    # Select portion of data where EWS are evaluated (e.g only up to bifurcation)
    if upto == "Full":
        short_series = raw_series
    else:
        short_series = raw_series.loc[:upto]

    # ------Data detrending--------

    # Compute the absolute size of the bandwidth if it is given as a proportion
    if 0 < band_width <= 1:
        bw_size = short_series.shape[0] * band_width
    else:
        bw_size = band_width

    # Compute the Lowess span as a proportion if given as absolute
    if not 0 < span <= 1:
        span = span / short_series.shape[0]
    else:
        span = span

    # Compute smoothed data and residuals
    if smooth == "Gaussian":
        # Use the gaussian_filter function provided by Scipy
        # Standard deviation of kernel given bandwidth
        # Note that for a Gaussian, quartiles are at +/- 0.675*sigma
        sigma = (0.25 / 0.675) * bw_size
        smooth_data = gf(short_series.values, sigma=sigma, mode="reflect")
        smooth_series = pd.Series(smooth_data, index=short_series.index)
        residuals = short_series.values - smooth_data
        resid_series = pd.Series(residuals, index=short_series.index)

        # Add smoothed data and residuals to the EWS DataFrame
        df_ews["Smoothing"] = smooth_series
        df_ews["Residuals"] = resid_series

    if smooth == "Lowess":
        smooth_data = lowess(short_series.values, short_series.index.values, frac=span)[
            :, 1
        ]
        smooth_series = pd.Series(smooth_data, index=short_series.index)
        residuals = short_series.values - smooth_data
        resid_series = pd.Series(residuals, index=short_series.index)

        # Add smoothed data and residuals to the EWS DataFrame
        df_ews["Smoothing"] = smooth_series
        df_ews["Residuals"] = resid_series

    # Use the short_series EWS if smooth='None'. Otherwise use reiduals.
    eval_series = short_series if smooth == "None" else resid_series

    # Compute the rolling window size (integer value)
    rw_size = int(np.floor(roll_window * raw_series.shape[0]))

    # ------------ Compute temporal EWS---------------#

    # Compute standard deviation as a Series and add to the DataFrame
    if "sd" in ews:
        roll_sd = eval_series.rolling(window=rw_size).std()
        df_ews["Standard deviation"] = roll_sd

    # Compute variance as a Series and add to the DataFrame
    if "var" in ews:
        roll_var = eval_series.rolling(window=rw_size).var()
        df_ews["Variance"] = roll_var

    # Compute autocorrelation for each lag in lag_times and add to the DataFrame
    if "ac" in ews:
        for i in range(len(lag_times)):
            roll_ac = eval_series.rolling(window=rw_size).apply(
                func=lambda x: pd.Series(x).autocorr(lag=lag_times[i]), raw=True
            )
            df_ews["Lag-" + str(lag_times[i]) + " AC"] = roll_ac

    # Compute Coefficient of Variation (C.V) and add to the DataFrame
    if "cv" in ews:
        # mean of raw_series
        roll_mean = raw_series.rolling(window=rw_size).mean()
        # standard deviation of residuals
        roll_std = eval_series.rolling(window=rw_size).std()
        # coefficient of variation
        roll_cv = roll_std.divide(roll_mean)
        df_ews["Coefficient of variation"] = roll_cv

    # Compute skewness and add to the DataFrame
    if "skew" in ews:
        roll_skew = eval_series.rolling(window=rw_size).skew()
        df_ews["Skewness"] = roll_skew

    # Compute Kurtosis and add to DataFrame
    if "kurt" in ews:
        roll_kurt = eval_series.rolling(window=rw_size).kurt()
        df_ews["Kurtosis"] = roll_kurt

    # ------------Compute spectral EWS-------------#

    """ In this section we compute newly proposed EWS based on the power spectrum
        of the time-series computed over a rolling window """

    # Empty DataFrame for storage of power spectra
    df_pspec = pd.DataFrame()

    # If any of the spectral metrics are listed in the ews vector, compute power spectra:
    if ("smax" or "cf" or "aic" or "smax/var" or "smax/mean") in ews:

        # Number of components in the residual time-series
        num_comps = len(eval_series)
        # Rolling window offset (can make larger to save on computation time)
        roll_offset = int(pspec_roll_offset)
        # Time separation between data points (need for frequency values of power spectrum)
        dt = eval_series.index[1] - eval_series.index[0]

        # Initialise a list for the spectral EWS
        list_metrics_append = []
        # Initialise a list for the power spectra
        list_spec_append = []

        # Loop through window locations shifted by roll_offset
        for k in np.arange(0, num_comps - (rw_size - 1), roll_offset):

            # Select subset of series contained in window
            window_series = eval_series.iloc[k : k + rw_size]
            # Asisgn the time value for the metrics (right end point of window)
            t_point = eval_series.index[k + (rw_size - 1)]

            ## Compute the power spectrum using function pspec_welch
            pspec = helpers.pspec_welch(
                window_series,
                dt,
                ham_length=ham_length,
                ham_offset=ham_offset,
                w_cutoff=w_cutoff,
                scaling="spectrum",
            )

            ## Create DataFrame for empirical power spectrum
            df_pspec_empirical = pspec.to_frame().reset_index()
            # Rename column
            df_pspec_empirical.rename(
                columns={"Power spectrum": "Empirical"}, inplace=True
            )
            # Include a column for the time-stamp
            df_pspec_empirical['time'] = t_point * np.ones(len(pspec))
            # Use a multi-index of ['time','Frequency']
            df_pspec_empirical.set_index(['time', 'frequency'], inplace=True)

            ## Compute the spectral EWS using pspec_metrics (dictionary)
            metrics = helpers.pspec_metrics(pspec, ews, aic, sweep)
            # Add the time-stamp
            metrics['time'] = t_point
            # Add metrics (dictionary) to the list
            list_metrics_append.append(metrics)

            ## Store the power spectra fits (if aic computed)
            if "aic" in ews:
                # Create fine-scale frequency values
                wVals = np.linspace(min(pspec.index), max(pspec.index), 100)
                # Fold fit
                pspec_fold = helpers.psd_fold(
                    wVals,
                    metrics["Params fold"]["sigma"],
                    metrics["Params fold"]["lam"],
                )
                # Flip fit
                pspec_flip = helpers.psd_flip(
                    wVals, metrics["Params flip"]["sigma"], metrics["Params flip"]["r"]
                )
                # Hopf fit
                pspec_hopf = helpers.psd_hopf(
                    wVals,
                    metrics["Params hopf"]["sigma"],
                    metrics["Params hopf"]["mu"],
                    metrics["Params hopf"]["w0"],
                )
                # Null fit
                pspec_null = helpers.psd_null(wVals, metrics["Params null"]["sigma"])

                ## Put spectrum fits into a dataframe
                dic_temp = {
                    'time': t_point * np.ones(len(wVals)),
                    'frequency': wVals,
                    "Fit fold": pspec_fold,
                    "Fit flip": pspec_flip,
                    "Fit hopf": pspec_hopf,
                    "Fit null": pspec_null,
                }
                df_pspec_fits = pd.DataFrame(dic_temp)
                # Set the multi-index
                df_pspec_fits.set_index(['time', 'frequency'], inplace=True)

            # Concatenate empirical PS with fits
            df_pspec_temp = (
                pd.concat([df_pspec_empirical, df_pspec_fits], axis=1)
                if "aic" in ews
                else df_pspec_empirical
            )
            # Store DataFrame in list
            list_spec_append.append(df_pspec_temp)

        # Concatenate the list of power spectra DataFrames to form a single DataFrame
        df_pspec = pd.concat(list_spec_append)

        # Create a DataFrame out of the multiple dictionaries consisting of the spectral metrics
        df_spec_metrics = pd.DataFrame(list_metrics_append)
        df_spec_metrics.set_index('time', inplace=True)

        # Join the spectral EWS DataFrame to the main EWS DataFrame
        df_ews = df_ews.join(df_spec_metrics)

        # Compute smax/var and add to the DataFrame
        if "smax/var" in ews:
            df_ews["Smax/Var"] = df_ews["Smax"] / df_ews["Variance"]

        # Compute smax normalised by mean and add to the DataFrame
        if "smax/mean" in ews:
            df_ews["Smax/Mean"] = df_ews["Smax"] / roll_mean

    # ------------Compute Kendall tau coefficients----------------#

    """ In this section we compute the kendall correlation coefficients for each EWS
        with respect to time. Values close to one indicate high correlation (i.e. EWS
        increasing with time), values close to zero indicate no significant correlation,
        and values close to negative one indicate high negative correlation (i.e. EWS
        decreasing with time)."""

    # Time at which to start computng Kendall tau values
    if ktau_time == "Start":
        ktau_time = df_ews.index[0]

    else:
        ktau_time = df_ews.index[np.argmin(abs(df_ews.index - ktau_time))]

    # Put time values as their own series for correlation computation
    time_vals = pd.Series(
        df_ews.loc[ktau_time:].index, index=df_ews.loc[ktau_time:].index
    )

    # List of EWS that can be used for Kendall tau computation
    ktau_metrics = [
        "Variance",
        "Standard deviation",
        "Skewness",
        "Kurtosis",
        "Coefficient of variation",
        "Smax",
        "Smax/Var",
        "Smax/Mean",
    ] + ["Lag-" + str(i) + " AC" for i in lag_times]
    # Find intersection with this list and EWS computed
    ews_list = df_ews.columns.values.tolist()
    ktau_metrics = list(set(ews_list) & set(ktau_metrics))

    # Find Kendall tau for each EWS and store in a DataFrame
    dic_ktau = {
        x: df_ews[x].loc[ktau_time:].corr(time_vals, method="kendall")
        for x in ktau_metrics
    }  # temporary dictionary
    df_ktau = pd.DataFrame(
        dic_ktau, index=[0]
    )  # DataFrame (easier for concatenation purposes)

    # -------------Organise final output and return--------------#

    # Ouptut a dictionary containing EWS DataFrame, power spectra DataFrame, and Kendall tau values
    output_dic = {
        "EWS metrics": df_ews,
        "Power spectrum": df_pspec,
        "Kendall tau": df_ktau,
    }

    return output_dic


# -----------------------------
# Eigenvalue reconstruction
# ------------------------------

def eval_recon_rolling(
    df_in,
    roll_window=0.4,
    roll_offset=1,
    smooth="Lowess",
    span=0.1,
    band_width=0.2,
    upto="Full",
):
    """
    Compute reconstructed eigenvalues from residuals of multi-variate
    non-stationary time series

    Args
    ----
    df_in: pd.DataFrame
        Time series data with variables in different columns, indexed by time
    roll_window: float
        Rolling window size as a proportion of the length of the time series
        data.
    roll_offset: int
        Offset of rolling window upon each EWS computation - make larger to save
        on computational time
    smooth: {'Gaussian', 'Lowess', 'None'}
        Type of detrending.
    band_width: float
        Bandwidth of Gaussian kernel. Taken as a proportion of time-series length if in (0,1),
        otherwise taken as absolute.
    span: float
        Span of time-series data used for Lowess filtering. Taken as a
        proportion of time-series length if in (0,1), otherwise taken as
        absolute.
    upto: int or 'Full'
        Time up to which EWS are computed. Enter 'Full' to use
        the entire time-series. Otherwise enter a time value.

    Returns
    --------

        pd.DataFrame:
                DataFrame indexed by time, with columns of the Jacobian,
                eigenvalues and eigenvectors at each point in time.

    """

    # Properties of df_in
    var_names = df_in.columns

    # Select portion of data where EWS are evaluated (e.g only up to bifurcation)
    if upto == "Full":
        df_pre = df_in.copy()
    else:
        df_pre = df_in.loc[:upto]

    # ------Data detrending--------

    # Compute the absolute size of the bandwidth if it is given as a proportion
    if 0 < band_width <= 1:
        bw_size = df_pre.shape[0] * band_width
    else:
        bw_size = band_width

    # Compute the Lowess span as a proportion if given as absolute
    if not 0 < span <= 1:
        span = span / df_pre.shape[0]
    else:
        span = span

    # Compute smoothed data and residuals
    if smooth == "Gaussian":
        # Loop through variables
        for var in var_names:

            smooth_data = gf(df_pre[var].values, sigma=bw_size, mode="reflect")
            smooth_series = pd.Series(smooth_data, index=df_pre.index)
            residuals = df_pre[var].values - smooth_data
            resid_series = pd.Series(residuals, index=df_pre.index)
            # Add smoothed data and residuals to df_pre
            df_pre[var + "_s"] = smooth_series
            df_pre[var + "_r"] = resid_series

    if smooth == "Lowess":
        # Loop through variabless
        for var in var_names:

            smooth_data = lowess(df_pre[var].values, df_pre.index.values, frac=span)[
                :, 1
            ]
            smooth_series = pd.Series(smooth_data, index=df_pre.index)
            residuals = df_pre[var].values - smooth_data
            resid_series = pd.Series(residuals, index=df_pre.index)
            # Add smoothed data and residuals to df_pre
            df_pre[var + "_s"] = smooth_series
            df_pre[var + "_r"] = resid_series

    # Compute the rolling window size (integer value)
    rw_size = int(np.floor(roll_window * df_in.shape[0]))

    # Set up a rolling window

    # Number of components in the residual time-series
    num_comps = len(df_pre)
    # Rolling window offset (can make larger to save on computation time)
    roll_offset = int(roll_offset)

    # Initialise a list of dictionaries containing eval data
    list_evaldata = []

    # Loop through window locations shifted by roll_offset
    for k in np.arange(0, num_comps - (rw_size - 1), roll_offset):

        # Select subset of residuals contained in window
        df_window = df_pre[[var + "_r" for var in var_names]].iloc[k : k + rw_size]
        # Asisgn the time value for the metrics (right end point of window)
        t_point = df_pre.index[k + (rw_size - 1)]

        # Do eigenvalue reconstruction on residuals
        dic_eval_recon = helpers.eval_recon(df_window)
        # Add time component
        dic_eval_recon['time'] = t_point
        # Add them to list
        list_evaldata.append(dic_eval_recon)

    # Create dataframe from list of dicts of eval data
    df_evaldata = pd.DataFrame(list_evaldata)
    df_evaldata.set_index('time', inplace=True)

    # Create output dataframe that merges all useful info
    df_out = pd.concat(
        [
            df_in,
            df_pre[
                [var + "_r" for var in var_names] + [var + "_s" for var in var_names]
            ],
            df_evaldata,
        ],
        axis=1,
    )

    return df_out


# ----------------------------------------------
# Bootstrapping
# –--------------------------------------------


def block_bootstrap(series, n_samples, bs_type="Stationary", block_size=10):

    """
    Computes block-bootstrap samples of series.

    Args
    ----
    series: pd.Series
        Time-series data in the form of a Pandas Series indexed by time
    n_samples: int
        Number of bootstrapped samples to output.
    bs_type: {'Stationary', 'Circular'}
        Type of block-bootstrapping to perform.
    block_size: int
        Size of resampling blocks. Should be big enough to
        capture important frequencies in the series.

    Returns
    -------
    pd.DataFrame:
        DataFrame containing the block-bootstrapped samples of series.
        Indexed by sample number, then time.

    """

    # Set up list for sampled time-series
    list_samples = []

    # Stationary bootstrapping
    if bs_type == "Stationary":
        bs = StationaryBootstrap(block_size, series)

        # Count for sample number
        count = 1
        for data in bs.bootstrap(n_samples):

            df_temp = pd.DataFrame(
                {"sample": count, 'time': series.index.values, "x": data[0][0]}
            )
            list_samples.append(df_temp)
            count += 1

    if bs_type == "Circular":
        bs = CircularBlockBootstrap(block_size, series)

        # Count for sample number
        count = 1
        for data in bs.bootstrap(n_samples):

            df_temp = pd.DataFrame(
                {"sample": count, 'time': series.index.values, "x": data[0][0]}
            )
            list_samples.append(df_temp)
            count += 1

    # Concatenate list of samples
    df_samples = pd.concat(list_samples)
    df_samples.set_index(["sample", 'time'], inplace=True)

    # Output DataFrame of samples
    return df_samples


def roll_bootstrap(
    raw_series,
    span=0.1,
    roll_window=0.25,
    roll_offset=1,
    upto="Full",
    n_samples=20,
    bs_type="Stationary",
    block_size=10,
):

    """
    Smooths raw_series and computes residuals over a rolling window.
    Bootstraps each segment and outputs samples.

    Args
    ----
    raw_series: pd.Series
        Time-series data in the form of a Pandas Seires indexed by time.
    span: float
        Proportion of data used for Loess filtering.
    roll_windopw: float
        Size of the rolling window (as a proportion
        of the length of the data).
    roll_offset: int
        Number of data points to shift the rolling window
        upon each iteration (reduce to increase computation time).
    upto: int/'Full'
        If 'Full', use entire time-series, otherwise input time up
        to which EWS are to be evaluated.
    n_samples: int
        Number of bootstrapped samples to output.
    bs_type: {'Stationary', 'Circular'}
        Type of block-bootstrapping to perform.
    block_size: int
        Size of resampling blocks. Should be big enough to
        capture important frequencies in the series.


    Return
    ------
    pd.DataFrame:
        DataFrame containing the block-bootstrapped samples at each time
        in raw_series. Indexed by time in raw_series, then, sample number,
        then time within the rolling window.

    """

    ## Parameter configuration

    # Compute the rolling window size (integer value)
    rw_size = int(np.floor(roll_window * raw_series.shape[0]))

    # Compute the Lowess span as a proportion if given as absolute
    if not 0 < span <= 1:
        span = span / raw_series.shape[0]
    else:
        span = span

    ## Data detrending

    # Select portion of data up to 'upto'
    if upto == "Full":
        series = raw_series
    else:
        series = raw_series.loc[:upto]

    # Smooth the series and compute the residuals
    smooth_data = lowess(series.values, series.index.values, frac=span)[:, 1]
    residuals = series.values - smooth_data
    resid_series = pd.Series(residuals, index=series.index)

    ## Rolling window over residuals

    # Number of components in the residual time-series
    num_comps = len(resid_series)
    # Make sure window offset is an integer
    roll_offset = int(roll_offset)

    # Initialise a list for the sample residuals at each time point
    list_samples = []

    # Counter
    i = 0

    # Loop through window locations shifted by roll_offset
    for k in np.arange(0, num_comps - (rw_size - 1), roll_offset):

        # Select subset of series contained in window
        window_series = resid_series.iloc[k : k + rw_size]
        # Asisgn the time value for the metrics (right end point of window)
        t_point = resid_series.index[k + (rw_size - 1)]

        # Compute bootstrap samples of residauls within rolling window
        df_samples_temp = block_bootstrap(window_series, n_samples, bs_type, block_size)
        df_samples_temp.reset_index(inplace=True)
        df_samples_temp['wintime'] = df_samples_temp['time']
        
        # Add column with real time (end of window)
        df_samples_temp['time'] = t_point

        # Reorganise index
        
        df_samples_temp.reset_index(inplace=True)
        df_samples_temp.set_index(['time', "sample", 'wintime'], inplace=True)

        # Append the list of samples
        list_samples.append(df_samples_temp)

        #        # Print update
        #        if i % 1 ==0:
        #            print('Bootstrap samples for window at t = %.2f complete' % (t_point))

        i += 1

    ## Organise output DataFrame

    # Concatenate list of samples
    df_samples = pd.concat(list_samples)

    # Output DataFrame
    return df_samples


def mean_ci(data, alpha=0.95):
    """
    Compute confidence intervals (to alpha%) of the mean of data.
    This is performed using bootstrapping.

    Args
    ----
    data: pd.Series
        Data provided as a Pandas Series
    alpha: float
        Confidence percentage.

    Returns
    -------
    dict:
        Dicitonary of mean, lower and upper bound of data
    """

    # Compute the mean of the Series
    mean = data.mean()
    # Obtain the values of the Series as an array
    array = data.values
    # Bootstrap the array (sample with replacement)
    bs = IIDBootstrap(array)
    # Compute confidence intervals of bootstrapped distribution
    ci = bs.conf_int(np.mean, 1000, method="percentile", size=alpha)
    # Lower and upper bounds
    lower = ci[0, 0]
    upper = ci[1, 0]

    # Output dictionary
    dict_out = {"Mean": mean, "Lower": lower, "Upper": upper}
    return dict_out

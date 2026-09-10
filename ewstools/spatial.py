"""
Spatial early warning signals.

Most of `ewstools` treats early warning signals (EWS) as a purely
temporal phenomenon: a rolling window slides along a single time series
(`TimeSeries`) or a handful of co-measured series (`MultiTimeSeries`),
and indicators like variance or lag-1 autocorrelation are tracked over
time. But a large and separate branch of the critical-slowing-down
literature -- starting with Dakos et al. (2010), "Spatial correlation as
leading indicator of catastrophic shifts", and still an active area (see
e.g. MacLaren, Aihara & Masuda, 2025, on generalising spatial EWS from
regular lattices to irregular real-world networks) -- looks instead
across SPACE at a single instant: do neighbouring units of a system
become more correlated with each other as a critical transition
approaches? Moran's I is the standard statistic for this, but it is not
implemented anywhere in `ewstools` today.

This module adds that missing piece, following the existing package's
conventions (`ewstools.core.MultiTimeSeries`): a spatial analogue that
takes one column per spatial unit and computes a spatial indicator at
EVERY time point (cross-sectional at each row), rather than a temporal
indicator over a rolling window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def morans_i(values, weights) -> float:
    """Moran's I spatial autocorrelation statistic.

    I = (N / S0) * [sum_ij w_ij (x_i - xbar)(x_j - xbar)] / [sum_i (x_i - xbar)^2]

    Parameters
    ----------
    values : array-like, shape (n_units,)
        Observed value at each spatial unit.
    weights : array-like, shape (n_units, n_units)
        Spatial weights matrix, w_ij > 0 if units i and j are neighbours,
        0 otherwise (zero diagonal). Not required to be row-standardised.

    Returns
    -------
    float
        Moran's I, or nan if the weights or the values have no variation
        (undefined in that case).
    """
    x = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    n = len(x)
    if w.shape != (n, n):
        raise ValueError(f"weights must be an {n} x {n} matrix to match values")

    deviations = x - x.mean()
    s0 = w.sum()
    if s0 == 0:
        return float("nan")

    denominator = (deviations**2).sum()
    if denominator == 0:
        return float("nan")

    numerator = deviations @ w @ deviations
    return float((n / s0) * (numerator / denominator))


def morans_i_permutation_test(values, weights, n_permutations: int = 500, seed=None) -> dict:
    """Significance of Moran's I by permutation (spatial analogue of a
    surrogate test: reshuffle the values across spatial units while
    keeping the network itself fixed, following the standard approach
    for testing spatial autocorrelation, e.g. Dakos et al., 2010).

    Returns
    -------
    dict with keys: observed_i, p_value, n_permutations, null_mean, null_std.
    `p_value` is one-sided (probability of a permuted I at least as large
    as the observed one) -- appropriate for testing an INCREASE in
    spatial correlation as an early warning signal.
    """
    observed = morans_i(values, weights)
    if np.isnan(observed):
        return {"observed_i": None, "p_value": None, "n_permutations": 0, "null_mean": None, "null_std": None}

    rng = np.random.default_rng(seed)
    values = np.asarray(values, dtype=float)
    null_values = np.empty(n_permutations)
    for k in range(n_permutations):
        null_values[k] = morans_i(rng.permutation(values), weights)

    return {
        "observed_i": observed,
        "p_value": float(np.mean(null_values >= observed)),
        "n_permutations": n_permutations,
        "null_mean": float(np.mean(null_values)),
        "null_std": float(np.std(null_values)),
    }


class SpatialEWS:
    """
    Spatially-resolved data on which to compute spatial early warning
    signals, following the `data`/`state`/`ews` conventions of
    `ewstools.core.MultiTimeSeries`.

    Parameters
    ----------
    data : pandas.DataFrame
        One column per spatial unit (grid cell, node, sensor, region...),
        one row per time point. Index represents time and is carried
        over.
    weights : array-like, shape (n_units, n_units)
        Spatial weights matrix for the units in `data.columns` (same
        order), fixed over time -- the network itself is not assumed to
        change, only the values observed on it.
    transition : float, optional
        Time value at which a transition occurs, if any. If given,
        spatial EWS are only computed up to this point.
    """

    def __init__(self, data, weights, transition=None):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame (one column per spatial unit)")
        weights = np.asarray(weights, dtype=float)
        n_units = data.shape[1]
        if weights.shape != (n_units, n_units):
            raise ValueError(f"weights must be an N x N matrix with N = data.shape[1] = {n_units}")

        self.state = data
        self.weights = weights
        self.transition = float(transition) if transition else transition
        self.var_names = data.columns
        self.ews = pd.DataFrame(index=data.index)
        self.ktau = dict()

    def _pre_transition(self) -> pd.DataFrame:
        if self.transition:
            return self.state[self.state.index <= self.transition]
        return self.state

    def compute_moran(self):
        """Compute Moran's I at every time point. Output stored in
        `self.ews['morans_i']`.
        """
        df_pre = self._pre_transition()
        self.ews["morans_i"] = df_pre.apply(lambda row: morans_i(row.to_numpy(), self.weights), axis=1)

    def compute_moran_significance(self, n_permutations: int = 500, seed=None):
        """Permutation-test p-value for Moran's I at every time point.
        Output stored in `self.ews['morans_i_pvalue']`.
        """
        df_pre = self._pre_transition()
        self.ews["morans_i_pvalue"] = df_pre.apply(
            lambda row: morans_i_permutation_test(row.to_numpy(), self.weights, n_permutations, seed)["p_value"],
            axis=1,
        )

    def compute_ktau(self, tmin="earliest", tmax="latest"):
        """Kendall tau of each spatial EWS against time -- same convention
        as `ewstools.core.TimeSeries.compute_ktau`. Output stored in the
        `self.ktau` dict.
        """
        if tmin == "earliest":
            tmin = self.ews.dropna(how="all").index[0]
        if tmax == "latest":
            tmax = self.ews.dropna(how="all").index[-1]

        df_ews = self.ews[(self.ews.index >= tmin) & (self.ews.index <= tmax)].copy()
        time_values = pd.Series(data=df_ews.index, index=df_ews.index)
        self.ktau = dict(df_ews.corrwith(time_values, method="kendall", axis=0))

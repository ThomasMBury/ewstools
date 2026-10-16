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


"""
Spatial early warning signals.

Computes spatial indicators of critical slowing down (Dakos et al., 2010)
from a field observed over time. Input is a DataFrame with one column per
spatial unit and one row per time point. Indicators are computed
cross-sectionally at each time point, then tracked for a trend in the
same way as the temporal indicators in `ewstools.core`.

References
----------
Dakos, V., van Nes, E. H., Donangelo, R., Fort, H., & Scheffer, M. (2010).
Spatial correlation as leading indicator of catastrophic shifts.
Theoretical Ecology, 3(3), 163–174.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

# Values within a few units in the last place of one another cannot be told
# apart from floating-point rounding noise, so a field this flat is treated
# as constant (Moran's I is undefined for it). A tolerance relative to the
# field's own magnitude, not to its variance: the variance of a constant
# field is not exactly 0 once the mean has been rounded.
_CONSTANT_FIELD_RTOL = 8 * np.finfo(float).eps

# Permutations whose I equals the observed I but comes out one rounding step
# lower are true ties and must be counted as "at least as large".
_TIE_RTOL = 1e-12


def _validate_weights(weights, n_units: int) -> np.ndarray:
    """Return `weights` as a float array, after checking that it is an
    `n_units` x `n_units` matrix of finite values with a zero diagonal."""
    w = np.asarray(weights, dtype=float)
    if w.shape != (n_units, n_units):
        raise ValueError(f"weights must be an {n_units} x {n_units} matrix to match the {n_units} spatial units")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights must be finite (no NaN or inf values)")
    if np.any(np.diag(w) != 0):
        raise ValueError("weights must have a zero diagonal: self-weights inflate Moran's I")
    return w


def _warn_nan(values: pd.Series, n_time_points: int, name: str) -> None:
    n_nan = int(values.isna().sum())
    if n_nan:
        warnings.warn(
            f"{name} is nan for {n_nan} of {n_time_points} time points. Moran's I is "
            "undefined at a time point if any unit is missing at that time, or if the "
            "field has no variation there. Missing units are not dropped: the "
            "whole time point is discarded.",
            RuntimeWarning,
            stacklevel=3,
        )


def lattice_weights(n_rows: int, n_cols: int | None = None, connectivity: str = "rook") -> np.ndarray:
    """Build a spatial weights matrix for a regular rectangular lattice.

    A convenience for the common regular-grid case only -- for irregular
    real-world networks (real adjacency, k-nearest-neighbours, distance
    thresholds), use `libpysal.weights` and pass its dense array here.

    Parameters
    ----------
    n_rows, n_cols : int
        Grid dimensions. `n_cols` defaults to `n_rows` (a square grid).
        Cell `(r, c)` corresponds to flattened index `r * n_cols + c`
        (row-major), matching `numpy`'s default `.flatten()` order.
    connectivity : str
        "rook" (4 orthogonal neighbours) or "queen" (8 neighbours,
        including diagonals).

    Returns
    -------
    numpy.ndarray, shape (n_rows*n_cols, n_rows*n_cols)
    """
    if connectivity not in ("rook", "queen"):
        raise ValueError('connectivity must be "rook" or "queen"')
    n_cols = n_rows if n_cols is None else n_cols
    offsets = [(0, 1), (0, -1), (1, 0), (-1, 0)]
    if connectivity == "queen":
        offsets += [(1, 1), (1, -1), (-1, 1), (-1, -1)]

    n = n_rows * n_cols
    weights = np.zeros((n, n))
    for r in range(n_rows):
        for c in range(n_cols):
            i = r * n_cols + c
            for dr, dc in offsets:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n_rows and 0 <= cc < n_cols:
                    weights[i, rr * n_cols + cc] = 1
    return weights


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
        (undefined in that case). A field whose values agree to within a few
        units in the last place is treated as constant.

    Raises
    ------
    ValueError
        If `weights` is not an ``n x n`` matrix of finite values with a zero
        diagonal.
    """
    x = np.asarray(values, dtype=float)
    w = _validate_weights(weights, len(x))
    return _morans_i(x, w)


def _morans_i(x: np.ndarray, w: np.ndarray) -> float:
    """Moran's I for already-validated float arrays."""
    n = len(x)
    s0 = w.sum()
    if s0 == 0:
        return float("nan")

    # Constant field. An exact `denominator == 0` check misses fields such as
    # np.full(9, 0.23): the computed mean is off by one rounding step, every
    # deviation is the same tiny nonzero number, and I comes out as exactly 1.
    if np.ptp(x) <= _CONSTANT_FIELD_RTOL * np.max(np.abs(x)):
        return float("nan")

    deviations = x - x.mean()
    denominator = (deviations**2).sum()
    numerator = deviations @ w @ deviations
    return float((n / s0) * (numerator / denominator))


def morans_i_permutation_test(values, weights, n_permutations: int = 500, seed=None) -> dict:
    """Significance of Moran's I by permutation (spatial analogue of a
    surrogate test: reshuffle the values across spatial units while
    keeping the network itself fixed, following the standard approach
    for testing spatial autocorrelation, e.g. Dakos et al., 2010).

    Parameters
    ----------
    values, weights
        As for `morans_i`.
    n_permutations : int
        Number of random reshuffles of `values`.
    seed : int, numpy.random.Generator or None
        Seed for the permutations. A `Generator` is used as given, so that
        successive calls draw fresh permutations from the same stream.

    Returns
    -------
    dict with keys: observed_i, p_value, n_permutations, null_mean, null_std.
    `p_value` is one-sided (probability of a permuted I at least as large
    as the observed one) -- appropriate for testing an INCREASE in
    spatial correlation as an early warning signal. The p-value has a
    floor of 1/(n_permutations + 1) (Davison & Hinkley 1997; North,
    Curtis & Sham 2002) so it can never be exactly zero. Permutations whose
    I equals the observed I up to floating-point rounding count as at least
    as large (ties are common for binary or integer fields).
    """
    values = np.asarray(values, dtype=float)
    weights = _validate_weights(weights, len(values))
    observed = _morans_i(values, weights)
    if np.isnan(observed):
        return {"observed_i": None, "p_value": None, "n_permutations": 0, "null_mean": None, "null_std": None}

    rng = np.random.default_rng(seed)
    null_values = np.empty(n_permutations)
    for k in range(n_permutations):
        null_values[k] = _morans_i(rng.permutation(values), weights)

    n_at_least_as_large = np.sum(null_values >= observed - _TIE_RTOL * max(1.0, abs(observed)))
    return {
        "observed_i": observed,
        "p_value": float((1 + n_at_least_as_large) / (1 + n_permutations)),
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
        order). Must be finite with a zero diagonal (checked: self-weights
        would inflate Moran's I). Fixed over time -- the network itself is not assumed to
        change, only the values observed on it. ``weights[i, j]`` refers
        to ``data.columns[i]`` and ``data.columns[j]`` **by position**:
        reordering, merging or sorting the columns after building
        ``weights`` changes the value silently; there is no check. If
        the real network changes over time -- units added or removed,
        links strengthened -- Moran's I trends for that reason alone,
        independently of the system's dynamics (Dakos et al. 2010,
        Fig. 6a). Recompute over a sub-network that is present and
        connected throughout.
    transition : float or timestamp, optional
        Time value (in the units of ``data.index``; a timestamp for a
        datetime index) at which a transition occurs, if any. If given
        (including 0), spatial EWS are only computed up to this point.

    Notes
    -----
    **Trend control.** Moran's I is computed on the raw field at each
    time point. A spatial pattern whose amplitude changes slowly over
    time (for example a gradient that strengthens) produces a trend in
    Moran's I with no change in the system's dynamics. Where such
    patterns are plausible, compute the indicator on residuals from a
    per-unit temporal detrend: pass the field through
    ``MultiTimeSeries.detrend`` and hand the residual columns to
    ``SpatialEWS`` (recipe below). The temporal detrend removes slowly
    varying spatial patterns only. It does not remove a rise in the
    amplitude of spatially coherent fluctuations relative to unit-level
    noise, which also raises Moran's I. Dakos et al. (2010, Fig. 6) show
    two further ways spatial correlation rises with no change in
    proximity to a transition: increased connectivity (6a) and increased
    environmental heterogeneity (6b). Compare the indicator against a
    reference period and against the field's spatial amplitude before
    reading a trend as slowing down.

    **Recipe** (``MultiTimeSeries`` appends columns to the frame it is
    given -- pass a copy)::

        mts = MultiTimeSeries(df.copy(), transition=t_trans)
        mts.detrend(method="Gaussian", bandwidth=0.2)
        resid = mts.state[[f"{c}_residuals" for c in df.columns]].set_axis(
            df.columns, axis=1)
        sews = SpatialEWS(resid, weights=W, transition=t_trans)

    Rows after ``transition`` have NaN residuals and are excluded by
    ``SpatialEWS`` in the same way. Rows containing NaN produce NaN
    Moran's I (see ``compute_moran``).

    **Reference period.** Moran's I depends on the unit set and the
    weights as much as on the field, so a single value is not meaningful
    on its own -- compare a candidate window against a reference period
    (e.g. the same field early in the record, or a period known to be
    far from any transition) rather than reading an absolute level.

    **Single-snapshot inference.** For an analytic significance test on
    ONE time point (rather than a trend across many, which is what this
    class is for), see ``esda.moran.Moran`` (part of `PySAL
    <https://pysal.org/esda/>`_), which additionally supports
    row-standardised weights and conditional/analytical p-values.
    """

    def __init__(self, data, weights, transition=None):
        if not isinstance(data, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame (one column per spatial unit)")
        weights = _validate_weights(weights, data.shape[1])

        self.state = data
        self.weights = weights
        self.transition = transition
        self.var_names = data.columns
        self.ews = pd.DataFrame(index=data.index)
        self.ktau = dict()

    def _pre_transition(self) -> pd.DataFrame:
        if self.transition is not None:
            return self.state[self.state.index <= self.transition]
        return self.state

    def compute_moran(self):
        """Compute Moran's I at every time point. Output stored in
        `self.ews['morans_i']`. Computed on the raw field; for trend
        control see the Notes section above. Rows containing NaN produce
        NaN and are counted in a warning.
        """
        df_pre = self._pre_transition()
        values = df_pre.apply(lambda row: morans_i(row.to_numpy(), self.weights), axis=1)
        self.ews["morans_i"] = values
        _warn_nan(values, len(df_pre), "Moran's I")

    def compute_moran_significance(self, n_permutations: int = 500, seed=None):
        """Permutation-test p-value for Moran's I at every time point.
        Output stored in `self.ews['morans_i_pvalue']`.

        A single random generator, seeded once from `seed`, is shared by all
        time points, so each time point gets its own permutations (the
        results are still reproducible for a given `seed`).
        """
        df_pre = self._pre_transition()
        rng = np.random.default_rng(seed)
        values = df_pre.apply(
            lambda row: morans_i_permutation_test(row.to_numpy(), self.weights, n_permutations, rng)["p_value"],
            axis=1,
        )
        self.ews["morans_i_pvalue"] = values
        _warn_nan(values, len(df_pre), "The Moran's I p-value")

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

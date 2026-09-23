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

import warnings

import numpy as np
import pandas as pd


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
    spatial correlation as an early warning signal. The p-value has a
    floor of 1/(n_permutations + 1) (Davison & Hinkley 1997; North,
    Curtis & Sham 2002) so it can never be exactly zero.
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
        "p_value": float((1 + np.sum(null_values >= observed)) / (1 + n_permutations)),
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
        change, only the values observed on it. ``weights[i, j]`` refers
        to ``data.columns[i]`` and ``data.columns[j]`` **by position**:
        reordering, merging or sorting the columns after building
        ``weights`` changes the value silently; there is no check. If
        the real network changes over time -- units added or removed,
        links strengthened -- Moran's I trends for that reason alone,
        independently of the system's dynamics (Dakos et al. 2010,
        Fig. 6a). Recompute over a sub-network that is present and
        connected throughout.
    transition : float, optional
        Time value at which a transition occurs, if any. If given,
        spatial EWS are only computed up to this point.

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
        `self.ews['morans_i']`. Computed on the raw field; for trend
        control see the Notes section above. Rows containing NaN produce
        NaN and are counted in a warning.
        """
        df_pre = self._pre_transition()
        values = df_pre.apply(lambda row: morans_i(row.to_numpy(), self.weights), axis=1)
        self.ews["morans_i"] = values
        n_nan = int(values.isna().sum())
        if n_nan:
            warnings.warn(
                f"Moran's I is nan for {n_nan} of {len(df_pre)} time points. A time "
                "point yields nan if any unit is missing at that time, or if the "
                "field has no variation there. Missing units are not dropped: the "
                "whole time point is discarded.",
                RuntimeWarning,
                stacklevel=2,
            )

    def compute_moran_significance(self, n_permutations: int = 500, seed=None):
        """Permutation-test p-value for Moran's I at every time point.
        Output stored in `self.ews['morans_i_pvalue']`.
        """
        df_pre = self._pre_transition()
        values = df_pre.apply(
            lambda row: morans_i_permutation_test(row.to_numpy(), self.weights, n_permutations, seed)["p_value"],
            axis=1,
        )
        self.ews["morans_i_pvalue"] = values
        n_nan = int(values.isna().sum())
        if n_nan:
            warnings.warn(
                f"Moran's I is nan for {n_nan} of {len(df_pre)} time points. A time "
                "point yields nan if any unit is missing at that time, or if the "
                "field has no variation there. Missing units are not dropped: the "
                "whole time point is discarded.",
                RuntimeWarning,
                stacklevel=2,
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

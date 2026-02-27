"""Tests for DFA (Detrended Fluctuation Analysis)."""
import numpy as np
import pandas as pd
import ewstools
from ewstools import helpers


def test_dfa_white_noise():
    """White noise has no long-range correlations: alpha ~ 0.5."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(2000)
    alpha = helpers.dfa(x)
    assert not np.isnan(alpha)
    assert 0.35 <= alpha <= 0.65, f"White noise alpha out of range: {alpha:.4f}"


def test_dfa_brownian_motion():
    """Brownian motion (cumulative sum of white noise): alpha ~ 1.5."""
    rng = np.random.default_rng(42)
    x = np.cumsum(rng.standard_normal(2000))
    alpha = helpers.dfa(x)
    assert not np.isnan(alpha)
    assert 1.3 <= alpha <= 1.7, f"Brownian alpha out of range: {alpha:.4f}"


def test_dfa_short_series():
    """DFA returns NaN for series too short to analyse."""
    x = np.random.default_rng(42).standard_normal(5)
    assert np.isnan(helpers.dfa(x, min_scale=4))


def test_dfa_deterministic():
    """DFA is deterministic for identical input."""
    x = np.random.default_rng(99).standard_normal(1000)
    assert helpers.dfa(x) == helpers.dfa(x)


def test_TimeSeries_dfa():
    """DFA integrates correctly through the TimeSeries interface."""
    rng = np.random.default_rng(42)
    tVals = np.arange(0, 10, 0.02)
    data = pd.Series(5 + rng.standard_normal(len(tVals)), index=tVals)
    ts = ewstools.TimeSeries(data, transition=8)
    ts.compute_dfa(rolling_window=0.25)
    assert isinstance(ts.ews, pd.DataFrame)
    assert 'dfa' in ts.ews.columns
    ts.compute_ktau()
    assert isinstance(ts.ktau, dict)
    assert 'dfa' in ts.ktau.keys()


def test_TimeSeries_dfa_no_transition():
    """DFA works on a TimeSeries without a specified transition point."""
    rng = np.random.default_rng(42)
    tVals = np.arange(0, 10, 0.02)
    data = pd.Series(5 + rng.standard_normal(len(tVals)), index=tVals)
    ts = ewstools.TimeSeries(data)
    ts.compute_dfa(rolling_window=0.25)
    assert 'dfa' in ts.ews.columns

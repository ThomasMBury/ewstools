"""A transition at time 0 is a transition like any other.

`TimeSeries` and `MultiTimeSeries` used to test `if self.transition:`, which is False for 0, so a series indexed from a
negative time with ``transition=0`` (a common convention: time zero = tipping point) silently behaved as if no transition
had been given and every indicator was computed over the whole series, past the transition.
"""
import numpy as np
import pandas as pd
import pytest

import ewstools


def _series(n=100, start=-50, seed=0):
    rng = np.random.default_rng(seed)
    return pd.Series(np.cumsum(rng.normal(size=n)) * 0.1, index=np.arange(start, start + n))


def _frame(n=100, start=-50, seed=0):
    rng = np.random.default_rng(seed)
    idx = np.arange(start, start + n)
    return pd.DataFrame({"a": np.cumsum(rng.normal(size=n)) * 0.1, "b": np.cumsum(rng.normal(size=n)) * 0.1}, index=idx)


def test_transition_zero_is_stored_as_a_float():
    ts = ewstools.TimeSeries(_series(), transition=0)
    mts = ewstools.MultiTimeSeries(_frame(), transition=0)
    assert ts.transition == 0.0 and type(ts.transition) == float
    assert mts.transition == 0.0 and type(mts.transition) == float


def test_no_transition_is_still_none():
    assert ewstools.TimeSeries(_series()).transition is None
    assert ewstools.MultiTimeSeries(_frame()).transition is None


@pytest.mark.parametrize("method", ["compute_var", "compute_std", "compute_cv", "compute_skew", "compute_kurt", "compute_auto", "compute_dfa"])
def test_timeseries_indicators_stop_at_a_transition_at_zero(method):
    ts = ewstools.TimeSeries(_series(), transition=0)
    getattr(ts, method)(rolling_window=0.25)
    computed = ts.ews.dropna(how="all")
    assert len(computed) > 0
    assert computed.index.max() <= 0


def test_timeseries_detrend_stops_at_a_transition_at_zero():
    ts = ewstools.TimeSeries(_series(), transition=0)
    ts.detrend()
    assert ts.state["smoothing"].dropna().index.max() <= 0
    assert ts.state["residuals"].isna().loc[1:].all()


def test_timeseries_spectrum_stops_at_a_transition_at_zero():
    ts = ewstools.TimeSeries(_series(n=300, start=-150), transition=0)
    ts.compute_spectrum(rolling_window=0.25, ham_length=20, pspec_roll_offset=10)
    assert ts.pspec["time"].max() <= 0


def test_transition_zero_gives_the_same_result_as_a_shifted_series():
    """The same data with times shifted by +50 and ``transition=50`` must give identical indicators."""
    s = _series()
    a = ewstools.TimeSeries(s, transition=0)
    a.compute_var()
    b = ewstools.TimeSeries(pd.Series(s.values, index=s.index + 50), transition=50)
    b.compute_var()
    np.testing.assert_allclose(a.ews["variance"].dropna().values, b.ews["variance"].dropna().values)


def test_multitimeseries_correlation_and_covariance_stop_at_a_transition_at_zero():
    mts = ewstools.MultiTimeSeries(_frame(), transition=0)
    mts.compute_corr(rolling_window=0.25)
    mts.compute_covar(rolling_window=0.25)
    assert mts.corr.index.get_level_values(0).max() <= 0
    assert mts.covar.index.get_level_values(0).max() <= 0


def test_multitimeseries_detrend_stops_at_a_transition_at_zero():
    mts = ewstools.MultiTimeSeries(_frame(), transition=0)
    mts.detrend()
    assert mts.state["a_smoothing"].dropna().index.max() <= 0


def test_positive_and_missing_transitions_are_unchanged():
    ts = ewstools.TimeSeries(_series(), transition=30)
    ts.compute_var()
    assert ts.ews["variance"].dropna().index.max() <= 30
    full = ewstools.TimeSeries(_series())
    full.compute_var()
    assert full.ews["variance"].dropna().index.max() == 49

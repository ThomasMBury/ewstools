import numpy as np
import pandas as pd
import pytest

from ewstools.spatial import SpatialEWS, morans_i, morans_i_permutation_test


def _grid_rook_weights(n):
    weights = np.zeros((n * n, n * n))
    for r in range(n):
        for c in range(n):
            i = r * n + c
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    weights[i, rr * n + cc] = 1
    return weights


# -- morans_i --------------------------------------------------------------


def test_morans_i_matches_hand_computed_chain_example():
    # Chain 1-2-3-4, values = perfect gradient [1,2,3,4].
    # By hand: xbar=2.5, deviations=[-1.5,-0.5,0.5,1.5], S0=6, numerator=2.5,
    # denominator=5.0, I = (4/6)*(2.5/5.0) = 1/3.
    values = np.array([1.0, 2.0, 3.0, 4.0])
    weights = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
        ],
        dtype=float,
    )
    assert morans_i(values, weights) == pytest.approx(1 / 3, abs=1e-9)


def test_morans_i_checkerboard_gives_strong_negative_autocorrelation():
    n = 4
    grid = np.indices((n, n)).sum(axis=0) % 2
    values = np.where(grid == 0, 1.0, -1.0).flatten()
    assert morans_i(values, _grid_rook_weights(n)) < -0.9


def test_morans_i_smooth_gradient_gives_strong_positive_autocorrelation():
    n = 4
    values = np.indices((n, n))[1].flatten().astype(float)
    assert morans_i(values, _grid_rook_weights(n)) > 0.5


def test_morans_i_constant_values_is_nan():
    values = np.full(9, 3.0)
    assert np.isnan(morans_i(values, _grid_rook_weights(3)))


def test_morans_i_zero_weights_is_nan():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isnan(morans_i(values, np.zeros((4, 4))))


def test_morans_i_shape_mismatch_raises():
    with pytest.raises(ValueError):
        morans_i(np.array([1.0, 2.0, 3.0]), np.zeros((4, 4)))


# -- morans_i_permutation_test ----------------------------------------------


def test_permutation_test_null_mean_matches_theoretical_expectation():
    # Under H0 (permutation), E[I] -> -1/(N-1), a classical property.
    rng = np.random.default_rng(0)
    n = 20
    values = rng.normal(size=n)
    weights = (rng.random((n, n)) < 0.3).astype(float)
    np.fill_diagonal(weights, 0)
    weights = np.maximum(weights, weights.T)

    result = morans_i_permutation_test(values, weights, n_permutations=2000, seed=1)
    assert result["null_mean"] == pytest.approx(-1 / (n - 1), abs=0.05)


def test_permutation_test_p_value_in_bounds_and_matches_observed():
    n = 4
    values = np.indices((n, n))[1].flatten().astype(float)
    result = morans_i_permutation_test(values, _grid_rook_weights(n), n_permutations=500, seed=2)
    assert result["observed_i"] == pytest.approx(morans_i(values, _grid_rook_weights(n)))
    assert 0.0 <= result["p_value"] <= 1.0


# -- SpatialEWS ---------------------------------------------------------------


def _toy_spatial_frame(n_time=20, n_units=9, seed=3):
    rng = np.random.default_rng(seed)
    data = pd.DataFrame(rng.normal(size=(n_time, n_units)), index=np.arange(n_time))
    return data, _grid_rook_weights(3)


def test_spatialews_requires_dataframe():
    _, weights = _toy_spatial_frame()
    with pytest.raises(TypeError):
        SpatialEWS(np.zeros((5, 9)), weights)


def test_spatialews_weights_shape_validation():
    data, _ = _toy_spatial_frame()
    with pytest.raises(ValueError):
        SpatialEWS(data, np.zeros((3, 3)))


def test_spatialews_compute_moran_matches_function_row_by_row():
    data, weights = _toy_spatial_frame()
    spatial = SpatialEWS(data, weights)
    spatial.compute_moran()
    for t, row in data.iterrows():
        assert spatial.ews.loc[t, "morans_i"] == pytest.approx(morans_i(row.to_numpy(), weights))


def test_spatialews_compute_moran_significance_produces_valid_pvalues():
    data, weights = _toy_spatial_frame(n_time=5)
    spatial = SpatialEWS(data, weights)
    spatial.compute_moran_significance(n_permutations=200, seed=4)
    pvals = spatial.ews["morans_i_pvalue"].dropna()
    assert len(pvals) == 5
    assert pvals.between(0, 1).all()


def test_spatialews_compute_ktau_detects_increasing_trend():
    n_time, n_units = 30, 9
    weights = _grid_rook_weights(3)
    rng = np.random.default_rng(5)
    rows = []
    for t in range(n_time):
        # Gradient strength grows over time -> Moran's I should trend up.
        base = np.indices((3, 3))[1].flatten().astype(float)
        rows.append(base * (t / n_time) + rng.normal(scale=0.05, size=n_units))
    data = pd.DataFrame(rows, index=np.arange(n_time))

    spatial = SpatialEWS(data, weights)
    spatial.compute_moran()
    spatial.compute_ktau()
    assert spatial.ktau["morans_i"] > 0.5


def test_spatialews_transition_restricts_computation():
    data, weights = _toy_spatial_frame(n_time=10)
    spatial = SpatialEWS(data, weights, transition=5)
    spatial.compute_moran()
    assert spatial.ews["morans_i"].dropna().index.max() <= 5

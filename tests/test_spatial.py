from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from ewstools.spatial import SpatialEWS, lattice_weights, morans_i, morans_i_permutation_test


def _grid_rook_weights(n):
    return lattice_weights(n, connectivity="rook")


# -- lattice_weights ---------------------------------------------------------


def test_lattice_weights_rook_matches_hand_built_grid():
    n = 3
    hand_built = np.zeros((n * n, n * n))
    for r in range(n):
        for c in range(n):
            i = r * n + c
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
                rr, cc = r + dr, c + dc
                if 0 <= rr < n and 0 <= cc < n:
                    hand_built[i, rr * n + cc] = 1
    assert np.array_equal(lattice_weights(n), hand_built)


def test_lattice_weights_queen_has_more_neighbours_than_rook():
    rook = lattice_weights(3, connectivity="rook")
    queen = lattice_weights(3, connectivity="queen")
    # Centre cell of a 3x3 grid: 4 rook neighbours, 8 queen neighbours.
    assert rook[4].sum() == 4
    assert queen[4].sum() == 8


def test_lattice_weights_rectangular_shape():
    weights = lattice_weights(2, 5)
    assert weights.shape == (10, 10)


def test_lattice_weights_rejects_invalid_connectivity():
    with pytest.raises(ValueError):
        lattice_weights(3, connectivity="bishop")


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


@pytest.mark.parametrize("value", [3.0, 0.1, 0.23, -0.7, 1e6 + 0.1])
@pytest.mark.parametrize("side", [3, 10])
def test_morans_i_constant_values_is_nan(value, side):
    # 3.0 is exactly representable; 0.1, 0.23, ... are not, so the computed mean
    # is off by one rounding step and every deviation is the same tiny nonzero
    # number. An exact `denominator == 0` check returns I = 1.0 for these.
    values = np.full(side * side, value)
    assert np.isnan(morans_i(values, _grid_rook_weights(side)))


@pytest.mark.parametrize("side", [3, 10])
def test_morans_i_constant_values_is_nan_for_all_two_decimal_constants(side):
    weights = _grid_rook_weights(side)
    for c in np.arange(1, 200) / 100:
        assert np.isnan(morans_i(np.full(side * side, round(c, 2)), weights)), c


def test_morans_i_field_flat_to_within_rounding_noise_is_nan():
    # One value differs from the rest by a single unit in the last place
    # (0.1 + 0.2 == 0.30000000000000004 != 0.3): indistinguishable from
    # rounding noise, so not a real spatial pattern.
    values = np.full(9, 0.3)
    values[4] = 0.1 + 0.2
    assert values[4] != 0.3
    assert np.isnan(morans_i(values, _grid_rook_weights(3)))


def test_morans_i_tiny_but_real_variation_is_not_mistaken_for_constant():
    # Guard against a constant-field tolerance that is too wide: a field varying
    # by 1e-12 (relative) is far above float resolution (~1e-16) and must still
    # produce a value.
    rng = np.random.default_rng(0)
    values = 1000.0 * (1 + 1e-12 * rng.normal(size=100))
    assert np.isfinite(morans_i(values, _grid_rook_weights(10)))


def test_morans_i_zero_weights_is_nan():
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert np.isnan(morans_i(values, np.zeros((4, 4))))


def test_morans_i_shape_mismatch_raises():
    with pytest.raises(ValueError):
        morans_i(np.array([1.0, 2.0, 3.0]), np.zeros((4, 4)))


def test_morans_i_rejects_nonzero_diagonal_weights():
    # Self-weights silently inflate I (here from about -0.11 to +0.19).
    values = np.random.default_rng(2).normal(size=9)
    weights = _grid_rook_weights(3)
    with pytest.raises(ValueError, match="zero diagonal"):
        morans_i(values, weights + np.eye(9))


@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_morans_i_rejects_nonfinite_weights(bad):
    values = np.random.default_rng(2).normal(size=9)
    weights = _grid_rook_weights(3)
    weights[0, 1] = bad
    with pytest.raises(ValueError, match="finite"):
        morans_i(values, weights)


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


def _exact_morans_i(values, weights):
    """Moran's I in exact rational arithmetic (integer-valued fields only)."""
    n = len(values)
    s0 = int(weights.sum())
    mean = Fraction(int(sum(values)), n)
    dev = [Fraction(int(v)) - mean for v in values]
    denominator = sum(d * d for d in dev)
    numerator = sum(dev[i] * dev[j] for i, j in zip(*np.nonzero(weights)))
    return Fraction(n, s0) * numerator / denominator


@pytest.mark.parametrize("field_seed", [0, 11, 13])
def test_permutation_test_counts_exact_ties(field_seed):
    # Binary fields produce many permutations whose I is *exactly* equal to the
    # observed I but comes out one rounding step lower in floating point. With an
    # exact `>=` those ties were dropped and the p-value understated (by ~0.01
    # on these fields). Reference: the same permutations, in exact arithmetic.
    weights = _grid_rook_weights(10)
    field = np.random.default_rng(field_seed).integers(0, 2, 100).astype(float)
    n_permutations, seed = 100, 1

    observed = _exact_morans_i(field, weights)
    rng = np.random.default_rng(seed)
    n_ge = sum(_exact_morans_i(rng.permutation(field), weights) >= observed for _ in range(n_permutations))
    expected = float(Fraction(1 + n_ge, 1 + n_permutations))

    result = morans_i_permutation_test(field, weights, n_permutations=n_permutations, seed=seed)
    assert result["p_value"] == expected


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


def test_spatialews_ktau_rises_under_strengthening_gradient_without_trend_control():
    # NEGATIVE CONTROL: this field has no critical slowing down. Its Moran's I
    # rises only because the spatial gradient strengthens over time. See the
    # Trend control section of the SpatialEWS docstring.
    #
    # Grid size and window length also matter for how reliable this bound is:
    # at 3x3/T=30, an equivalent assertion (`> 0.5`) only passes on 526/1000
    # random seeds (mean 0.486, sd 0.096) -- essentially a coin flip. At
    # 6x6/T=60, `> 0.30` holds on 1000/1000 seeds tested here; even at the
    # 0.30 threshold alone a 3x3 grid still fails on ~5% of seeds, confirming
    # grid size (not just the threshold) was the missing parameter.
    grid_side = 6
    n_time, n_units = 60, grid_side * grid_side
    weights = _grid_rook_weights(grid_side)
    rng = np.random.default_rng(5)
    rows = []
    for t in range(n_time):
        # Gradient strength grows over time -> Moran's I should trend up.
        base = np.indices((grid_side, grid_side))[1].flatten().astype(float)
        rows.append(base * (t / n_time) + rng.normal(scale=0.05, size=n_units))
    data = pd.DataFrame(rows, index=np.arange(n_time))

    spatial = SpatialEWS(data, weights)
    spatial.compute_moran()
    spatial.compute_ktau()
    assert spatial.ktau["morans_i"] > 0.30


def test_spatialews_rejects_invalid_weights():
    data, weights = _toy_spatial_frame()
    with pytest.raises(ValueError, match="zero diagonal"):
        SpatialEWS(data, weights + np.eye(9))
    bad = weights.copy()
    bad[0, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        SpatialEWS(data, bad)


def test_spatialews_transition_restricts_computation():
    data, weights = _toy_spatial_frame(n_time=10)
    spatial = SpatialEWS(data, weights, transition=5)
    spatial.compute_moran()
    assert spatial.ews["morans_i"].dropna().index.max() <= 5


def test_spatialews_transition_zero_is_a_transition():
    # `if transition:` treated 0 as "no transition": Moran's I was computed
    # all the way to the end of an index spanning negative and positive values.
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.normal(size=(11, 9)), index=np.arange(-5, 6))
    spatial = SpatialEWS(data, _grid_rook_weights(3), transition=0)
    spatial.compute_moran()
    assert spatial.ews["morans_i"].dropna().index.max() == 0


def test_spatialews_transition_accepts_timestamp():
    rng = np.random.default_rng(0)
    index = pd.date_range("2020-01-01", periods=8)
    data = pd.DataFrame(rng.normal(size=(8, 9)), index=index)
    transition = pd.Timestamp("2020-01-04")
    spatial = SpatialEWS(data, _grid_rook_weights(3), transition=transition)
    spatial.compute_moran()
    assert spatial.ews["morans_i"].dropna().index.max() == transition


def test_compute_moran_significance_uses_different_permutations_at_each_time_point():
    # Identical rows: if every row re-created default_rng(seed), every row would
    # get the same permutations and hence the same p-value. A single generator
    # shared across rows gives independent draws.
    field = np.random.default_rng(0).normal(size=9)
    data = pd.DataFrame(np.tile(field, (8, 1)), index=np.arange(8))
    spatial = SpatialEWS(data, _grid_rook_weights(3))
    spatial.compute_moran_significance(n_permutations=30, seed=0)
    assert spatial.ews["morans_i_pvalue"].nunique() > 1


def test_compute_moran_significance_is_reproducible_for_a_given_seed():
    data, weights = _toy_spatial_frame(n_time=6)
    a = SpatialEWS(data, weights)
    b = SpatialEWS(data, weights)
    a.compute_moran_significance(n_permutations=50, seed=123)
    b.compute_moran_significance(n_permutations=50, seed=123)
    assert a.ews["morans_i_pvalue"].equals(b.ews["morans_i_pvalue"])


def test_compute_moran_warns_on_rows_with_nan():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.normal(size=(6, 9)), index=np.arange(6))
    data.iloc[2, 4] = np.nan
    weights = _grid_rook_weights(3)

    spatial = SpatialEWS(data, weights)
    with pytest.warns(RuntimeWarning, match=r"nan for 1 of 6 time points"):
        spatial.compute_moran()
    assert spatial.ews["morans_i"].isna().sum() == 1

    # The warning for the p-values must talk about p-values, not about Moran's I.
    with pytest.warns(RuntimeWarning, match=r"p-value is nan for 1 of 6 time points"):
        spatial.compute_moran_significance(n_permutations=20, seed=0)
    assert spatial.ews["morans_i_pvalue"].isna().sum() == 1

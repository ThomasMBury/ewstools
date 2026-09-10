import numpy as np
import pytest
from scipy.stats import chi2

from ewstools.pvalues import combine_pvalues_ebm


def test_combine_pvalues_ebm_requires_at_least_two_tests():
    with pytest.raises(ValueError):
        combine_pvalues_ebm(np.zeros((10, 1)), [0.05])


def test_combine_pvalues_ebm_shape_mismatch_raises():
    with pytest.raises(ValueError):
        combine_pvalues_ebm(np.zeros((10, 2)), [0.05, 0.1, 0.2])


def test_combine_pvalues_ebm_independent_data_matches_fisher_method():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(500, 3))  # independent columns -> correlation ~ 0
    p_values = [0.02, 0.03, 0.10]

    result = combine_pvalues_ebm(data, p_values)
    chi2_fisher = -2 * sum(np.log(p_values))
    fisher_p = chi2.sf(chi2_fisher, 2 * len(p_values))

    assert result["c"] == pytest.approx(1.0, abs=0.05)
    assert result["df_ebm"] == pytest.approx(2 * len(p_values), abs=0.5)
    assert result["p_combined"] == pytest.approx(fisher_p, abs=0.02)


def test_combine_pvalues_ebm_positive_correlation_gives_less_extreme_pvalue_than_fisher():
    # Two tests built from strongly correlated underlying data carry
    # redundant evidence; the empirical correction must make the combined
    # p-value LESS significant than naively treating them as independent
    # (Fisher), which is the whole point of the correction.
    rng = np.random.default_rng(1)
    base = rng.normal(size=500)
    data = np.column_stack([base, base + rng.normal(scale=0.05, size=500)])
    p_values = [0.01, 0.01]

    result = combine_pvalues_ebm(data, p_values)
    chi2_fisher = -2 * sum(np.log(p_values))
    fisher_p = chi2.sf(chi2_fisher, 2 * len(p_values))

    assert result["c"] > 1.0
    assert result["p_combined"] > fisher_p


def test_combine_pvalues_ebm_brown_polynomial_at_perfect_correlation():
    # Direct check of Brown's (1975) polynomial itself at rho=1:
    # cov = 1*(3.263 + 1*(0.710 + 1*0.027)) = 4.0 exactly.
    data = np.column_stack([np.arange(100.0), np.arange(100.0) * 2 + 1])  # rho = 1 exactly
    result = combine_pvalues_ebm(data, [0.05, 0.2])
    k = 2
    expected_var = 4 * k + 2 * 4.0
    expected_c = expected_var / (2 * 2 * k)
    assert result["c"] == pytest.approx(expected_c, abs=1e-9)


def test_combine_pvalues_ebm_returns_expected_keys():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(50, 3))
    result = combine_pvalues_ebm(data, [0.1, 0.2, 0.3])
    assert set(result) == {"p_combined", "chi2_fisher", "chi2_ebm", "df_ebm", "c"}
    assert 0.0 <= result["p_combined"] <= 1.0


def test_combine_pvalues_ebm_invariant_to_test_order():
    rng = np.random.default_rng(3)
    data = rng.normal(size=(200, 3))
    p_values = [0.02, 0.15, 0.4]

    original = combine_pvalues_ebm(data, p_values)
    reordered = combine_pvalues_ebm(data[:, [2, 0, 1]], [p_values[2], p_values[0], p_values[1]])

    assert reordered["p_combined"] == pytest.approx(original["p_combined"], abs=1e-9)

"""
Combining correlated p-values.

`ewstools` computes several early warning indicators from the same
underlying data (variance, autocorrelation, skew, a spatial indicator...).
A natural next question -- not currently answered anywhere in the
package -- is: are two or more of these signals unusual AT THE SAME TIME?
Fisher's (1925) classical method for combining independent p-values,

    T = -2 * sum(ln p_i),  T ~ chi2(2k) under H0,

is the obvious tool, but it assumes the p_i are independent. Indicators
computed from the same underlying system are typically correlated, which
biases Fisher's method (too liberal if the correlation is positive, too
conservative if negative).

The Empirical Brown's Method (Poole, Gibbs, Shmulevich, Bernard &
Knijnenburg, 2016, "Combining dependent P-values with an empirical
adaptation of Brown's method", Bioinformatics 32(17), i430-i436) fixes
this: it keeps Fisher's statistic T, but instead of assuming the
textbook chi2(2k) null distribution, it estimates the TRUE variance of T
under the observed correlation structure directly from data (rather than
requiring the covariance to be known analytically, as in Brown's 1975
original method), then fits a rescaled chi-squared distribution to that
corrected mean and variance (method of moments).
"""
from __future__ import annotations

import numpy as np
from scipy.stats import chi2


def combine_pvalues_ebm(data, p_values) -> dict:
    """Combine one-sided, possibly-correlated p-values with the Empirical
    Brown's Method (Poole et al., 2016), building on Brown's (1975)
    polynomial approximation for the covariance of `-2 ln(p)` terms.

    Parameters
    ----------
    data : array-like, shape (n_observations, n_tests)
        The raw data underlying each of the `n_tests` p-values (one
        column per test) -- e.g. the k indicator series each p-value was
        derived from. Used only to estimate the empirical correlation
        between tests; this is what makes the method "empirical" rather
        than requiring the covariance to be assumed or known exactly, as
        in Brown's (1975) original method.
    p_values : array-like, shape (n_tests,)
        The one-sided p-value from each test (already computed
        elsewhere, e.g. by a surrogate or permutation test). A small
        p-value must consistently mean the same thing (e.g. "evidence of
        an approaching transition") across all tests for the combination
        to be meaningful.

    Returns
    -------
    dict with keys:
        p_combined : combined p-value.
        chi2_fisher : Fisher's uncorrected statistic, -2*sum(ln p_i).
        chi2_ebm : chi2_fisher rescaled by the empirical correction factor `c`.
        df_ebm : effective degrees of freedom after correction (<= 2*n_tests
            when p-values are positively correlated).
        c : empirical scale factor (> 1 under positive correlation, which is
            what makes the combined p-value LESS extreme than naive Fisher
            when the underlying tests are positively correlated -- the
            correction Fisher's method is missing).

    References
    ----------
    Brown, M. B. (1975). "A method for combining non-independent,
    one-sided tests of significance." Biometrics, 31(4), 987-992.
    Poole, W., Gibbs, D. L., Shmulevich, I., Bernard, B., & Knijnenburg,
    T. A. (2016). "Combining dependent P-values with an empirical
    adaptation of Brown's method." Bioinformatics, 32(17), i430-i436.
    """
    data = np.asarray(data, dtype=float)
    p_values = np.asarray(p_values, dtype=float)
    k = len(p_values)

    if data.ndim != 2 or data.shape[1] != k:
        raise ValueError("data must have shape (n_observations, n_tests) with n_tests == len(p_values)")
    if k < 2:
        raise ValueError("need at least two p-values to combine")

    clipped = np.clip(p_values, 1e-15, 1.0)
    chi2_fisher = float(-2.0 * np.sum(np.log(clipped)))

    corr = np.corrcoef(data, rowvar=False)

    # Brown's (1975) polynomial approximation of Cov(-2 ln p_i, -2 ln p_j)
    # as a function of the Pearson correlation rho between the underlying
    # (approximately normal) variables i and j.
    cov_sum = 0.0
    for i in range(k):
        for j in range(i + 1, k):
            rho = corr[i, j]
            cov_sum += rho * (3.263 + rho * (0.710 + rho * 0.027))

    var_fisher = 4 * k + 2 * cov_sum
    mean_fisher = 2 * k

    # Method-of-moments fit: T ~ c * chi2(df), matching mean and variance.
    c = var_fisher / (2 * mean_fisher)
    df_ebm = 2 * mean_fisher**2 / var_fisher

    p_combined = float(chi2.sf(chi2_fisher / c, df_ebm))

    return {
        "p_combined": p_combined,
        "chi2_fisher": chi2_fisher,
        "chi2_ebm": chi2_fisher / c,
        "df_ebm": float(df_ebm),
        "c": float(c),
    }

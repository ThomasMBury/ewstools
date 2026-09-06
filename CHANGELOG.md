# Changelog

All notable changes to `ewstools` are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Spatial early warning signals** (`ewstools.spatial`): `morans_i()`,
  `morans_i_permutation_test()`, and a new `SpatialEWS` class (mirroring the
  `data`/`state`/`ews` conventions of `MultiTimeSeries`) for computing Moran's I
  and its significance across space at every time point — the spatial branch of
  the critical-slowing-down literature (Dakos et al., 2010; MacLaren, Aihara &
  Masuda, 2025) had no equivalent in this package. See
  `CONTRIBUTION_spatial_significance.md` for the full write-up.
- **Combining correlated p-values** (`ewstools.pvalues`): `combine_pvalues_ebm()`,
  an implementation of the Empirical Brown's Method (Poole et al., 2016) for
  combining several early-warning indicators' p-values without assuming they are
  independent, which Fisher's method requires but indicators computed from the
  same underlying system rarely satisfy.
- 21 new tests (`tests/test_spatial.py`, `tests/test_pvalues.py`), 0 regressions
  on the existing suite.

## [2.1.3] — 2026-07-28

Maintenance release. **No API changes and no behavioural changes** — existing code and
notebooks run unmodified.

### Fixed

- **Spectral EWS now work on modern SciPy / pandas.** `compute_spectrum()` (and therefore
  `compute_smax()` and `compute_spec_type()`) raised
  `KeyError: 'key of type tuple not found and not a MultiIndex'` on recent stacks.
  SciPy ≥ 1.16 routes `signal.welch` through `ShortTimeFFT`, which slices its input as
  `x[..., i0:i1]`; given a pandas `Series`, pandas ≥ 3.0 reads that tuple as a MultiIndex
  key and raises rather than falling through to positional slicing. The series is now
  converted to an array at the boundary.
  The code fix landed on `main` in March 2026 but had not been published to PyPI, so every
  released install still carried the break. This release ships it.
  Reported in [#474](https://github.com/ThomasMBury/ewstools/issues/474).

- **`compute_entropy(method='kolmogorov')` now works on NumPy 2.** EntropyHub 2.0 — its
  latest release — still uses `np.NaN`, which NumPy removed in 2.0, so `EH.K2En()` and
  `EH.CoSiEn()` raised `AttributeError`. `ewstools`' own source was already clean; the
  alias is restored immediately before the EntropyHub import, and `np.NaN` was only ever
  a spelling of `np.nan`, so nothing changes numerically. This matters *because* of the
  NumPy change below: lifting the `numpy<2` ceiling is what puts users on the stack where
  the break appears, and it was a partial break — `method='sample'` worked while
  `method='kolmogorov'` failed — which stays hidden until someone reaches for that method.
  Verified on NumPy 2.4.6 / pandas 3.0.5 / SciPy 1.17.1: fails without the shim, passes
  with it. Covered by a regression test. The shim can be dropped once EntropyHub ships a
  NumPy 2 compatible release ([EntropyHub#21](https://github.com/MattWillFlood/EntropyHub/issues/21)).

### Changed

- **`numpy < 2.0` ceiling removed.** NumPy 2.x is supported and is now covered by CI. The
  pin blocked installation alongside current scientific-Python environments — which is how
  the above bug reached a user in the first place.

### Added

- **CI matrix extended to Python 3.13 and 3.14**, so modern-stack regressions surface in CI
  rather than in users' notebooks. `fail-fast: false` so one failing version no longer masks
  results for the others.

### Verified

Full test suite plus the spectral tutorial path, in clean containers, with dependencies
resolved from `pyproject.toml` as a user would get them:

| Python | NumPy | pandas | SciPy | tests | spectral calls |
|---|---|---|---|---|---|
| 3.9.25 | 2.0.2 | 2.3.3 | 1.13.1 | 30 passed, 1 skipped | 6/6 |
| 3.12.13 | 2.5.1 | 3.0.3 | 1.18.0 | 30 passed, 1 skipped | 6/6 |
| 3.14.6 | 2.5.1 | 3.0.3 | 1.18.0 | 30 passed, 1 skipped | 6/6 |

The single skip is the TensorFlow deep-learning test, which guards itself with
`pytest.importorskip` where TF has no wheels.

Against released 2.1.2 in the same container, all four spectral entry points fail; on this
branch all four pass.

These container runs predate the `compute_entropy` fix above, which adds a regression test —
hence 30 here and 31 in the suite as shipped. That fix was verified separately, on
NumPy 2.4.6 / pandas 3.0.5 / SciPy 1.17.1.

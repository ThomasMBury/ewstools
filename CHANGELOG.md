# Changelog

All notable changes to `ewstools` are documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [2.1.3] — unreleased

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

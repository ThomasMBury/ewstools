# Contribution: spatial early warning signals + combining correlated p-values

## What was missing

`ewstools` implements the temporal branch of the critical-slowing-down
literature thoroughly — variance, autocorrelation, skew, kurtosis,
spectral indicators, all tracked over a rolling window in time (`TimeSeries`,
`MultiTimeSeries`). It has no equivalent for the **spatial** branch of the
same literature: for systems observed at many locations at once (a
lattice, a sensor network, an administrative territory grid...), spatial
autocorrelation between neighbouring units is a separate, well-established
early warning signal (Dakos et al., 2010, "Spatial correlation as leading
indicator of catastrophic shifts", *Theoretical Ecology*; still an active
area — see MacLaren, Aihara & Masuda, 2025, on generalising these methods
from idealised regular lattices, where almost all validation work has been
done, to irregular real-world networks). Searching `ewstools/core.py` and
`ewstools/helpers.py` for "moran", "spatial", "fisher", "brown", or
"combine_p" turns up nothing — confirmed by direct inspection of both
files before writing any code here, not assumed.

A related, separate gap: `ewstools` never combines its own indicators.
If a user computes both a temporal indicator (say, variance) and a spatial
one (Moran's I) on the same system, there is no tool in the package to ask
"are both unusual at once?" — which needs Fisher's method at minimum, and
a **correlation-aware** version of it if the two indicators (as is typical,
since they are computed from the same underlying system) are not
independent. Fisher's method, applied naively to correlated inputs, is
biased. Confirmed absent by the same search.

## What this contribution adds

- **`ewstools/spatial.py`** — `morans_i()` and `morans_i_permutation_test()`
  as standalone functions, plus a `SpatialEWS` class that follows the
  existing `MultiTimeSeries` conventions (`data` → `state` → `ews`,
  transition-aware, a `compute_ktau()` for trend-testing the resulting
  indicator series exactly like the rest of the package). Computes Moran's
  I — and its permutation-test significance — at every time point from a
  DataFrame with one column per spatial unit and a fixed spatial weights
  matrix.
- **`ewstools/pvalues.py`** — `combine_pvalues_ebm()`, a real implementation
  of the Empirical Brown's Method (Poole, Gibbs, Shmulevich, Bernard &
  Knijnenburg, 2016, *Bioinformatics*), which extends Brown's (1975)
  covariance-corrected version of Fisher's method by estimating the
  correlation between tests empirically from data rather than requiring it
  to be known or assumed.
- 21 new tests (`tests/test_spatial.py`, `tests/test_pvalues.py`): hand-computed
  Moran's I examples, known limiting cases (checkerboard → strong negative
  autocorrelation, smooth gradient → strong positive autocorrelation,
  constant field → undefined), the classical permutation-test property
  `E[I] → -1/(N-1)` under the null, and — for the p-value combiner — that it
  reduces to plain Fisher when the underlying data are independent, and
  produces a *less* extreme combined p-value than naive Fisher when they are
  positively correlated (the whole point of the correction), verified
  against a direct evaluation of Brown's polynomial at a known correlation.
  Full suite run locally: 52 passed, 1 skipped (0 regressions on the 32
  tests that existed before this change).

## Usage angles

- **Research**: any critical-transition study with spatially resolved data
  (ecology, epidemiology, social/territorial data, sensor networks, plasma
  diagnostics) gains a spatial EWS tool in the same package already used for
  the temporal one, instead of having to hand-roll Moran's I separately.
- **Industry / infrastructure monitoring**: sensor networks (power grids,
  structural health monitoring, industrial process control) already produce
  exactly the "one column per unit, one row per time step" shape this class
  expects.
- **Public services**: territorial statistics agencies publishing
  regularly-updated indicators across administrative units can use
  `SpatialEWS` directly on their own panel data without depending on a
  domain-specific tool.

## Status

Not yet submitted as a pull request to `ThomasMBury/ewstools`. See
`/positionnement/ewstools` on the Hélios site for the live, honestly
maintained status once submission happens — this repository is a local
fork/branch pending that decision, and is not yet available even via a
public fork URL.

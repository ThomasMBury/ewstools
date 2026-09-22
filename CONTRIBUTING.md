# Contributing to ewstools

Thanks for considering a contribution. Bug reports and suggestions go on the
[issue tracker](https://github.com/ThomasMBury/ewstools/issues); code comes as a pull request
against `main`.

**Small fixes need none of what follows.** A typo, a broken link, a clearer error message, a bug
with a test that reproduces it — open the pull request. The rest of this page is for new features.

## Scope

**In scope:** code that computes, tests, or visualises an early warning signal (EWS) indicator on
the package's data structures (`TimeSeries`, `MultiTimeSeries`), and simulated models that generate
test data for those indicators. The underlying statistic does not have to be novel or specific to
this field — variance, autocorrelation, entropy and DFA are all general statistics, and they are in
scope because the package computes them over a rolling window *as resilience indicators*.

**Out of scope:** general statistical machinery that is not itself an indicator — multiple-testing
corrections, methods for combining significance across tests, generic fitting or smoothing
utilities. These belong in SciPy, statsmodels, or a geostatistics library such as PySAL (`esda`);
where an implementation already exists there, calling it is preferred to reimplementing it.

If you are unsure which side a contribution falls on, that is what the issue below is for.

## Before writing anything larger than a bug fix

Open an issue naming the indicator, its reference, and where in the package it would live, and wait
for a reply before building. This costs ten minutes and can save a week.

This is a small package maintained alongside a full-time academic job, so a reply can take a couple
of weeks. **If two weeks pass, bump the issue; if another two pass, open a draft pull request
referencing it** and the discussion can continue there. Silence means the maintainer is busy, not
that the answer is no.

## Shape

`helpers.dfa` + `TimeSeries.compute_dfa` + `tests/test_dfa.py` are the pattern to copy — one pure
function, one thin method, one test with a known answer.

- **New indicator maths goes in a pure function over NumPy arrays** in `ewstools/helpers.py`
  (`f(values, ...) -> float` or an array), with no dependence on the package's classes. That
  function is what gets tested and documented.
- **The class method orchestrates** — it crops to the transition, walks the rolling window, calls
  the function, and stores the result in `self.ews`. It holds no mathematics of its own.
- Several older methods in `core.py` compute inline instead. They stay as they are; this rule is
  for new indicators.
- Every new public function or method is a commitment. Prefer extending an existing entry point
  over adding a new one.

## Setting up

```
git clone https://github.com/<you>/ewstools.git && cd ewstools
python -m venv .venv && source .venv/bin/activate
pip install -e . -r requirements_dev.txt
pytest tests/
```

One test skips unless TensorFlow is installed, which is expected — the deep-learning classifier is
optional. The package supports Python 3.9 upwards, so please avoid syntax newer than that.

## Before opening a pull request

- Tests in `tests/` for every new function, including at least one case with a **known answer** — a
  hand-computed example, a limiting case with an analytic value, or a comparison against an
  established implementation. `tests/test_dfa.py` shows the idea.
- A docstring in the NumPy style used in `core.py`, stating the reference for the method.
  (`helpers.py` is not consistent about this; follow `core.py`.)
- A **new module** must be exported from `ewstools/__init__.py` and given a section in
  `docs/source/ewstools.rst`, or it will not appear on Read the Docs. Nothing warns you if you
  forget — the docs build does not notice an unreferenced module. A new *function* in an existing
  module needs neither, since the docs use `automodule` with `:members:`.
- A line in `CHANGELOG.md` under an `## [Unreleased]` heading; add that heading if it is not there,
  which it usually will not be.
- `pytest tests/` passing on a branch rebased on current `main`.

The docs build runs in CI with warnings treated as errors, so a malformed docstring will fail the
pull request.

Merging into `main` does not publish a release; releases are cut separately by tagging a version.

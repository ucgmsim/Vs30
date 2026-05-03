# jaehwi_v1p0 Observation CSV Provenance — Follow-up

**Status:** ✅ Resolved 2026-05-02
**Created:** 2026-05-02
**Surfaced by:** `dev/docs/grid_bounds_alignment_fix_plan.md` work

## Resolution

`vs30/resources/observations/jaehwi_v1p0_independent_observations.csv` was
replaced with the `easting`, `northing`, `vs30`, `uncertainty` columns
extracted directly from a fresh legacy `measured_sites.csv` run, so the
refactored code now reads the same NZTM coordinates Jaehwi's pipeline
produced (including its float32 precision quirk). After the swap, the
residual deviation between refactored and legacy outputs is ~140x
smaller than before:

| | Before swap | After swap |
|---|---:|---:|
| Band 1 (Vs30 mean) max rel diff | 5.78e-3 | 6.1e-5 |
| Band 2 (Vs30 stdv) max rel diff | 1.55e-2 | 1.1e-4 |
| `test_jaehwi_v1p0` rtol | 2e-2 | **2e-4** |

The remaining ~1e-4 floor is from minor numerical/algorithmic
differences in the MVN linear algebra (likely BLAS/LAPACK and library
versions between `oldvs30_venv` and `vs30_venv`), not coordinate drift.
Tightening the tolerance further is not worth chasing at this scale.

The original issue text is preserved below for historical reference.

---

## Issue

Two CSVs that should describe the same 671 jaehwi_v1p0 observations
contain NZTM coordinates that differ at sub-meter scale:

- **Refactored input**: `vs30/resources/observations/jaehwi_v1p0_independent_observations.csv`
- **Legacy output**: produced by jaehwi's fork at
  `/home/arr65/src/jaehwi_fork_vs30/Vs30_2026` as
  `<out>/measured_sites.csv` after running its CLI

Across all 671 stations:

| stat | value |
|------|------:|
| median coord distance | 0.000 m |
| mean coord distance | 0.477 m |
| max coord distance | 1.346 m |

Most observations are stable across the drift. **7 of 671 sit within
~1 m of an IwahashiPike pixel boundary, so the sub-meter drift flips
their assigned terrain ID.** Affected row indices: 87, 154, 166, 186,
193, 249, 270.

## Downstream impact

Each of those 7 flipped terrain IDs alters MVN-conditioned pixels
within MAX_DIST_M (10 km) of the observation. The cascade affects both
output bands:

- Band 1 (Vs30 mean): ~357 / 10511 valid pixels deviate, max 0.58 %
  relative.
- Band 2 (Vs30 stdv): ~20 / 10511 valid pixels deviate, max 1.55 %
  relative.

The stdv band is the binding constraint for the test tolerance, because
conditional variance is more sensitive than conditional mean to which
observations dominate the MVN fit.

The deviation pre-dates and is independent of the grid alignment fix.
It was below `TEST_RTOL = 1e-3` at the old `..100` bounds, so it
silently fell under the floor. The new `..050` bounds redistribute
which observations dominate which output pixels in the MVN fit, and
that change pushes the deviation above 1e-3.

## Mitigation in place

`tests/test_benchmarks.py::test_jaehwi_v1p0` runs at `rtol=2e-2`
(rather than the default 1e-3) with a docstring that names this issue
explicitly and points back to this file. The 2e-2 ceiling matches the
precedent set by `test_foster_2019_approx_points_benchmark` for stdv.

## Proper fix

Regenerate `jaehwi_v1p0_independent_observations.csv` from the raw
upstream observation source using the same coordinate-transform
pipeline the legacy uses (so legacy and refactored CSVs match
bit-for-bit at all 671 rows). After that, restore `test_jaehwi_v1p0`
to use the default `TEST_RTOL`.

Related questions worth answering as part of the same work:

- Do `modified_foster_2019_measured_vs30_independent_observations.csv`,
  `viktor_inferred_vs30_from_cpt.csv`, and
  `foster_2019_approx_measured_vs30_independent_observations.csv` have
  the same drift relative to their corresponding legacy CSVs? If so,
  fix them simultaneously and check whether their benchmark tolerances
  can be tightened too.
- What was the original coord-conversion pipeline jaehwi used? Is it
  documented or recoverable from the upstream observation source?

## Investigative trail

The diagnosis (provenance, not algorithm bug) is documented in
`dev/docs/grid_bounds_semantics_investigation.md` and was confirmed by
in-session analysis on 2026-05-02:

- Running the refactored `category.assign_to_category_terrain` on
  legacy CSV coords reproduces legacy's terrain ID exactly.
- Running it on refactored CSV coords reproduces refactored's
  terrain ID exactly.
- The two answers differ only because the input coords differ.

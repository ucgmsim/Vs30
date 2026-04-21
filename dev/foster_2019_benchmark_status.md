# foster_2019_approx Benchmark Test Status

## Current State (2026-04-21)

The foster_2019_approx benchmark is covered by
`test_foster_2019_approx_points_benchmark` in `tests/test_benchmarks.py`, which
samples the pipeline at 30 deterministic prior-dominated points and
compares against the stored 100 m benchmark raster at the same
coordinates. Both `nproc=1` and `nproc=-1` (all cores) are exercised
via parametrisation. Combined runtime: ~14 s in the default tier.

The earlier full-grid tests (`test_foster_2019_approx_single_process`,
`test_foster_2019_approx_multiprocess`) are removed — they could not be run at
the benchmark's native 100 m resolution (~6 h per test) and failed
immediately at 5 km due to a shape mismatch. See
"History" below for context.

The other three model-version benchmarks were regenerated at 5000 m:

| File | Resolution | Shape |
|------|-----------|-------|
| `modified_foster_2019.tif` | 5000 m | 304 × 212 |
| `jaehwi_v1p0.tif`           | 5000 m | 304 × 212 |
| `viktor_cpt_clustering.tif` | 5000 m | 304 × 212 |

`foster_2019_approx.tif` was not regenerated — it is still the 100 m version
from commit `2dc0dda` (2026-04-14). The points-based test does not need it
to match the other three in resolution because it samples at pixel centres.

## Why a 100 m Run Takes ~6 Hours

The grid is ~185 million pixels. With observations spread across the full
domain, most pixels are "affected" and need MVN conditioning, which is the
dominant cost of the pipeline. Running at that resolution for every
pytest invocation is not viable, so the points-based benchmark was chosen
instead.

## How the Points Benchmark Works

`test_foster_2019_approx_points_benchmark` in `tests/test_benchmarks.py`:

1. Deterministically samples 30 pixel centres from foster_2019_approx.tif whose
   nearest observation is more than `MAX_DIST_M` (10 km) away.
2. Runs `pipeline.points_pipeline` at those NZTM coordinates (both
   `nproc=1` and `nproc=-1` via parametrisation).
3. Reads the benchmark raster at the same coordinates using
   `rasterio.sample`.
4. Asserts tight bounds on the median relative difference (Vs30 < 1e-4,
   Stdv < 1e-3) and looser bounds on the 80th percentile (Vs30 < 1 %,
   Stdv < 2 %).

Prior-dominated pixels are used because the MVN step has zero effect
beyond `MAX_DIST_M`, so those pixels depend only on the categorical
posterior, the hybrid slope modification, and the combination step —
all of which reproduce the paper at float precision in the common case.
The 80th-percentile bounds tolerate a small tail of known outlier
pixels documented in `dev/foster_2019_reproduction_comparison.md`
(categorical/hybrid edge cases, unrelated to MVN).

## History

### The deselected full-grid tests (now removed, 2026-04-21)

Prior to the points benchmark, `test_foster_2019_approx_single_process` and
`test_foster_2019_approx_multiprocess` tried to run the full grid pipeline at
5000 m and compare against the 100 m benchmark. This failed immediately
due to a shape mismatch:

```
AssertionError: Band 1: Valid data masks differ
    actual shape=(328, 225)     # 5000 m output
    expected shape=(16384, 11264)  # 100 m benchmark
```

`assert_arrays_match_raster_benchmark` calls `np.array_equal` on two
different-shape arrays and fails. The test was not "hanging" or "broken
mid-pipeline" — it ran the grid pipeline to completion in ~9 s and then
the comparison step failed. To run at matching 100 m resolution would
take ~6 h per test, which is prohibitive.

Three options were considered before landing on the points benchmark:

1. **Regenerate `foster_2019_approx.tif` at 5000 m from the original R code**
   (Kevin Foster's R code:
   `/home/arr65/src/Kevin_Foster_R_code_vs30_model/Vs30_NZ`). The R
   workflow is not documented in
   `dev/generating_benchmarks_from_legacy_code.md` (the foster_2019_approx
   section is explicitly a stub), and some input data files appear to
   have been lost — see
   `dev/foster_2019_reproduction_comparison.md`.
2. **Leave the test disabled.** Coverage overlap with the other three
   model-version benchmarks is significant (shared categorical CSVs)
   but not total.
3. **Small-subregion 100 m benchmark.** Would need a new benchmark
   raster and a new test.

The points benchmark is effectively option 3 taken to its logical
extreme — 30 individual pixel-sized "subregions" — and reuses the
existing 100 m benchmark raster directly.

### Obs-set investigation (2026-04-21, see below)

An investigation confirmed that the ~1.6 % cohort-B (near-obs) rel diff
is intrinsic to the refactored MVN numerical implementation, not the
observation data. That is why the points benchmark uses cohort A
(prior-dominated) only: cohort B cannot be reproduced at tight
tolerance regardless of obs set.

Remaining supporting coverage:

- `test_grid_points_consistency_fast[foster_2019_approx]` — 3×3 local grids at
  Auckland / Wellington / Christchurch, exercises the full pipeline
  plumbing including Matérn correlation.
- `test_utils.TestMaternCorrelationFunction::*` — Matérn correlation
  function validated against R's `gstat` output.

## Investigation: Is the divergence caused by the observation set? (2026-04-21)

The current production obs file
(`vs30/resources/observations/foster_2019_approx_measured_vs30_independent_observations.csv`)
has 412 rows, but Foster et al. (2019) published 393 rows in supplement 2
(`15_eeri_35_4_suppl_2_es1_online.txt`). It is natural to ask whether the
~1.6 % median cohort-B divergence from `foster_2019_approx.tif` is caused by
this obs-set difference.

Finding: **No.** The obs-set difference is not the cause.

Three obs variants were compared against the benchmark raster using a
**fixed** set of 30 near-obs pixels:

| Variant | Rows | Provenance |
|---------|------|------------|
| A | 412 | Current production CSV |
| B | 382 | Current CSV filtered to one-row-per-supplement-obs (unique nearest match) |
| C | 393 | 382 full-precision current rows + 11 rows reconstructed from the supplement's lossy coords/Vs30 |

The 11 "paper extras" in variant C are the pairs of supplement rows whose
nearest current-CSV row is shared:

- 4 are byte-identical duplicates in the supplement (probably accidental
  data entry) — e.g. the same McGann et al. (2017) row appearing twice.
- 7 are legitimate dual-source measurements of the same station: Kaiser
  et al. (2017) AND Wotherspoon et al. (2013), each reporting its own
  Vs30. The current CSV keeps only the Kaiser row for these.

Results (cohort B, 30 near-obs pixels, fixed across variants):

| Variant | median rel diff | max rel diff | % within 1 % |
|---------|----------------|--------------|--------------|
| A (412) | 1.59 %         | 11.4 %       | 26.7 %       |
| B (382) | 1.63 %         | 13.3 %       | 26.7 %       |
| C (393) | 1.58 %         | 13.4 %       | 26.7 %       |

Per-pixel |Δ| when switching from variant A to C is only ~1 m/s mean /
~11 m/s max — an order of magnitude smaller than the ~1.6 % baseline
divergence from the benchmark itself. Cohort A (prior-dominated) is
identical across all three variants at ~2e-8 median rel diff, as
expected.

**Conclusion:** The ~1.6 % cohort-B drift is intrinsic to the refactored
MVN numerical implementation (legacy R vs Python), not the observation
data. We keep the 412-row CSV as the production file because it has
cleaner provenance (full precision throughout, no reconstructed rows)
and does not include the supplement's 4 accidental duplicates. The
393-row reconstruction lives in
`dev/foster_2019_reproduction_with_paper_obs.py` purely as a diagnostic
and is not used by production code or tests.

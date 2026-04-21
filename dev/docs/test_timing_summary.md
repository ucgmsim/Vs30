# Test Suite Timing Summary

Run date: 2026-04-21
Environment: `vs30_venv` on Linux workstation, Python 3.13

## Two-Tier Layout

| Tier | Command | Wall time | Scope |
|------|---------|-----------|-------|
| Default | `pytest tests/` | ~2 min 59 s | Unit tests, all benchmark tests (including foster_2019_approx points benchmark), 3-city smoke of grid/points consistency |
| Full | `pytest tests/ --runslow` | ~42–45 min | Default + 38-point grid/points consistency for all 4 model versions |

The two previous foster_2019_approx full-grid benchmark tests
(`test_foster_2019_approx_single_process`, `test_foster_2019_approx_multiprocess`) were
replaced by `test_foster_2019_approx_points_benchmark`, which samples the pipeline
at prior-dominated points and compares against the published 100 m
benchmark raster. Both single-process and multiprocess paths are exercised
via parametrisation. See `dev/foster_2019_benchmark_status.md`.

## Default-Tier Timing (2026-04-21)

Command: `pytest tests/ --durations=15`

Result: **44 passed, 4 skipped (slow), 0 deselected, 0 failed. Total wall time 179.21 s (2 min 59 s).**

| Rank | Test | Time (s) |
|------|------|----------|
| 1 | `test_benchmarks.py::test_jaehwi_v1p0_multiprocess` | 29.21 |
| 2 | `test_grid_points_consistency_fast[jaehwi_v1p0]` | 27.54 |
| 3 | `test_grid_points_consistency_fast[foster_2019_approx]` | 23.32 |
| 4 | `test_benchmarks.py::test_viktor_cpt_clustering_single_process` | 23.11 |
| 5 | `test_benchmarks.py::test_viktor_cpt_clustering_multiprocess` | 22.56 |
| 6 | `test_benchmarks.py::test_modified_foster_2019_multiprocess` | 18.78 |
| 7 | `test_benchmarks.py::test_jaehwi_v1p0_single_process` | 12.01 |
| 8 | `test_benchmarks.py::test_foster_2019_approx_points_benchmark[multiprocess]` | 9.78 |
| 9 | `test_benchmarks.py::test_modified_foster_2019_single_process` | 7.76 |
| 10 | `test_benchmarks.py::test_foster_2019_approx_points_benchmark[single_process]` | 3.96 |
| 11 | `test_gapfill.py::test_classify_nodata_excludes_water_and_offshore` | 0.70 |
| 12 | `test_gapfill.py::test_fill_nodata_grid_nearest_neighbor` | 0.35 |

All other tests ran in under 5 ms each (hidden by `--durations` cutoff).

## Slow-Tier Timing (from previous full run, 2026-04-20)

The 4 slow parameterisations of `test_grid_points_consistency_slow` iterate
the full 38-point set for each model version:

| Parameterisation | Time (s) | Time (mm:ss) |
|-------------------|----------|---------------|
| `test_grid_points_consistency_slow[viktor_cpt_clustering]` | 1367.54 | 22:47 |
| `test_grid_points_consistency_slow[modified_foster_2019]` | 466.66 | 07:47 |
| `test_grid_points_consistency_slow[jaehwi_v1p0]` | 314.40 | 05:14 |
| `test_grid_points_consistency_slow[foster_2019_approx]` | 262.18 | 04:22 |

With the two new slow-tier additions (foster_2019_approx, jaehwi_v1p0), the full
`--runslow` run is ~167 s (default tier) + ~2411 s (slow tier) ≈ 43 min.

## Marker Audit (2026-04-21)

Markers were re-assessed based on actual runtimes:

| Test | Before | After | Actual time |
|------|--------|-------|-------------|
| `test_benchmarks.py` (8 tests) | `@pytest.mark.slow` | unmarked | 3.96–29.2 s each |
| `test_benchmarks.py::test_foster_2019_approx_*` | deselected (shape mismatch) | replaced by points benchmark | 3.96 / 9.78 s |
| `test_grid_points_consistency_fast` | unmarked | unmarked, but scope reduced to 3 cities × 2 versions | 23.3–27.5 s each (was 262–314 s) |
| `test_grid_points_consistency_slow` | 2 model versions | 4 model versions (all) | 262–1368 s each |

Rationale:
- Benchmark tests are not slow in the current 5000 m layout — they were
  previously slow when the grid was at 100 m. Default `pytest tests/`
  now runs the regression coverage.
- `test_grid_points_consistency_fast` was named "fast" relative to the
  coastal-distance variants (~5 min vs. ~15 min), not absolutely. It now
  runs a 3-city smoke test instead of the full 38-point sweep.
- `test_grid_points_consistency_slow` absorbs the full-domain versions
  of `foster_2019_approx` and `jaehwi_v1p0`, so the slow tier is now the single
  place that guarantees 38-point × 4-model-version coverage.

## Where the Time Goes

Default tier (2 min 59 s):
- Benchmarks (8 tests, incl. 2 foster_2019_approx points params): 127.2 s (71 %)
- Fast consistency smoke (2 tests): 50.9 s (28 %)
- Unit tests (all other): 1.1 s (1 %)

Slow tier adds (37–40 min):
- Full 38-point consistency × 4 model versions: ~40 min

## Recent Fix

`vs30/parallel.py` previously left NODATA slope samples at observations as
`-9999.0`; `apply_hybrid_geology_modifications` then capped to
`MIN_SLOPE_FOR_LOG=1e-9` and returned the MIN Vs30 for the gid, while the
grid pipeline in `spatial.prepare_observation_data` applies
`LEGACY_OBS_SLOPE_NODATA_SENTINEL=255` (returning MAX Vs30). Fix: mirror
the sentinel replacement inside `parallel.py:156-164`. After the fix all
consistency and benchmark tests pass cleanly at the runtimes above.

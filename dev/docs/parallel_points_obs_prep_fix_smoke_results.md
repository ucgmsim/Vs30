# Points-Pipeline Multiproc Obs-Prep Fix — Smoke Benchmark Results

**Date:** 2026-04-28
**Branch:** `vs30_refactor`
**Predecessors:**
- [Design](parallel_points_obs_prep_fix_design.md)
- [Investigation findings (bug diagnosis)](points_perf_investigation_findings.md)

## Setup

Hardware: Intel Core i7-9700, 8 cores, 32 GiB. Same as the predecessor investigation.

Cells re-run via the points-perf harness (`dev/scripts/investigations/points_features_investigation/`):

- `(N_query=1000, N_obs=35706, nproc=1)` — sequential path; should be unchanged.
- `(N_query=1000, N_obs=35706, nproc=8)` — parallel path; the cell most affected by the bug (151× slower pre-fix).

## Results

| Cell | Pre-fix (s) | Post-fix (s) | Change |
|---|---|---|---|
| nproc=1 | 18.35 | 18.83 s | ≈ unchanged (+2.6% drift) |
| nproc=8 | 2,770 | 127.79 s | 21.7× faster |

## Interpretation

The bug is gone: the nproc=8 cell completed in 127.79 s, down from 2,770 s pre-fix — a 21.7× recovery that confirms the per-chunk obs-prep redundancy was the dominant cost. However, multiproc still loses to the sequential path at this cell size: nproc=8 took 127.79 s vs 18.83 s for nproc=1, a 6.8× slowdown, indicating the inherent multiproc/BLAS-MT tradeoff still dominates for N_query=1000. The nproc=1 baseline drifted by +2.6% (18.35 → 18.83 s), well within the ≈ unchanged threshold and attributable to normal workstation variance.

## Out of scope (deferred per design §2)

- Full re-sweep of the 7 × 3 × {1, 8} × 3 matrix.
- CLI default decision (depends on the full re-sweep, not the smoke).
- `Pool(initializer=...)` optimisation.

# MVN Performance Investigation: Refactored vs Legacy Code

**Date:** 2026-04-20
**Branch:** `known-issue-mvn-phase-leak`
**Model:** `viktor_cpt_clustering` (35,709 observations)
**Grid:** 1000m resolution, full NZ extent

## Summary

The refactored code takes ~276s to produce the full combined grid for
`viktor_cpt_clustering`, compared to ~106s for the legacy code. Both run with
`nproc=1` and multi-threaded BLAS. This document records the measured
breakdown and identifies the sources of the ~2.6x slowdown.

## Pipeline-level timing (refactored, 276s total)

Measured with `time.perf_counter()` instrumentation in `pipeline.py`:

| Stage                         | Geology | Terrain | Total  | % of total |
|-------------------------------|---------|---------|--------|------------|
| `find_affected_pixels`        | 33.9s   | 32.8s   | 66.7s  | 24%        |
| `compute_spatial_adjustments` | 102.0s  | 91.1s   | 193.1s | 70%        |
| `apply_updates`               | 0.0s    | 0.0s    | 0.0s   | 0%         |
| Other (categorical, combine, gap-fill, I/O) | | | ~16s  | 6%         |

## Key differences: refactored vs legacy

The legacy code (`pre-refactor-Vs30-for-comparison/vs30/mvn.py`, function
`_mvn`) does mathematically identical work but with several implementation
differences that explain the performance gap:

1. **`find_affected_pixels` overhead (66.7s, 24%):** Separate pre-filtering
   step via broadcasting. Legacy has no equivalent — handles observation
   selection inline per-pixel.
2. **Movement optimisation** (`_mvn` lines 118-123): Legacy skips pixels
   whose nearest-observation set has not changed since the previous pixel.
   Refactored recomputes per pixel. Deliberately excluded — see below.
3. **Distance computation:** Legacy `np.einsum("ij,ij->i", x, x)` (fused, no
   intermediates) vs refactored `np.sqrt(np.sum((...) ** 2, axis=1))`.
4. **Covariance distance matrix:** Legacy complex-number trick
   `np.abs(x[:, np.newaxis] - x)` vs refactored `scipy.spatial.distance.cdist`.
5. **Object creation per pixel:** Legacy works directly with arrays.
   Refactored creates `PixelData`, `ObservationData`, `SpatialAdjustmentResult`
   per pixel (~147k allocations per phase).

## Estimated attribution of the ~170s gap

| Source | Impact | Confidence |
|--------|--------|------------|
| Movement optimisation (skipped pixels) | ~118s | High (directly measured) |
| All other differences combined | ~52s | High (by subtraction) |

Confirmed by disabling the movement optimisation in the legacy code
(`mvn.py` lines 118-123, commenting out the `continue`) and re-running
the same 1000m benchmark.

## Baseline comparison table

| Run | Scope | Time |
|-----|-------|------|
| Legacy nproc=1, movement opt ON | full pipeline | 106s |
| Legacy nproc=1, movement opt OFF | full pipeline | 224s |
| Refactored nproc=1 (BLAS multi-threaded) | full pipeline | 276s |
| Refactored nproc=6 (BLAS single-threaded) | geology only | ~51 min |
| Refactored nproc=6 (BLAS single-threaded) | terrain only | ~45 min |

The movement optimisation saves **118s** (53%) in the legacy code. Without
it, the legacy code takes 224s — much closer to the refactored 276s. The
remaining ~52s gap comes from the combined effect of items 3-5 above.

## Why we cannot simply port the movement optimisation

The movement optimisation was **deliberately excluded** from the refactored code.
It makes the output depend on processing order: when multiprocessing splits the
grid into row chunks, each chunk starts without cached state (`prev_model_loc`),
so different processor counts cause different pixels to be skipped vs computed.
Even the legacy code cannot reproduce its own output with different `--nproc`
settings (27% of pixels differ, 2.83% mean difference). See
`wiki/why_we_cannot_reproduce_v1p0_grid_exactly.md` for full details.

The refactored code computes every pixel independently, making results fully
deterministic and reproducible regardless of `nproc`.

Any future optimisation to skip redundant distance computations must preserve
this property — i.e. the decision to skip must depend only on the pixel's
location relative to observations, not on which pixel was processed previously.

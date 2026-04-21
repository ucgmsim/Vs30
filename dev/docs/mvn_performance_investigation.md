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

## Per-pixel timing within `compute_spatial_adjustments`

Measured with `VS30_MVN_DIAG=1` instrumentation in `spatial.py`
(500 sampled pixels, every 50th pixel):

| Step                         | Mean (us) | % of per-pixel time |
|------------------------------|-----------|---------------------|
| `select_observations_for_pixel` | 1,085  | 48%                 |
| `build_covariance_matrix`       | 589    | 26%                 |
| `np.linalg.inv`                 | 558    | 25%                 |

Heavy pixels (n_obs >= 400, hitting MAX_POINTS=500 cap): 30 of 500 sampled.
These average 13ms each, with 5ms on matrix inversion alone.

## Key differences: refactored vs legacy

The legacy code (`pre-refactor-Vs30-for-comparison/vs30/mvn.py`, function
`_mvn`) does mathematically identical work but with several implementation
differences that explain the performance gap.

### 1. `find_affected_pixels` overhead (66.7s, 24%)

The refactored code has a separate pre-filtering step that creates
`(n_obs x chunk_size)` boolean arrays via broadcasting to find which pixels
are affected by any observation. The legacy code has no equivalent step -- it
handles observation selection inline per-pixel.

### 2. Movement optimisation (legacy `_mvn` lines 118-123)

The legacy code tracks the previous pixel location and skips re-computing
distances when the pixel hasn't moved far enough for the nearest observation
to change:

```python
movement = _dists(np.atleast_2d(model_loc - prev_model_loc))[0]
if min_dist - movement > max_dist:
    continue
```

For grid processing where consecutive pixels are spatially close, this can
skip many pixels entirely. The refactored code recomputes distances to all
35,709 observations from scratch for every pixel.

### 3. Distance computation method

- **Legacy:** `_dists(obs_locs - model_loc)` uses `np.einsum("ij,ij->i", x, x)`
  -- a single fused operation with no intermediate arrays.
- **Refactored:** `np.sqrt(np.sum((obs_data.locations - pixel.location) ** 2, axis=1))`
  -- creates intermediate difference, squared, and summed arrays.

### 4. Distance matrix for covariance

- **Legacy:** Complex number trick: `_dist_mat(_xy2complex(...))` which is
  `np.abs(x[:, np.newaxis] - x)` -- very compact, single allocation.
- **Refactored:** `scipy.spatial.distance.cdist(all_points, all_points, metric="euclidean")`
  -- more general but has Python function call overhead per pixel.

### 5. Object creation per pixel

- **Legacy:** Works directly with arrays and boolean masks. Zero object
  creation inside the per-pixel loop.
- **Refactored:** Creates per pixel:
  - `PixelData` namedtuple
  - `ObservationData` with 7 array slices (for selected observations)
  - `SpatialAdjustmentResult` namedtuple
  
  That is ~3 objects x ~49k affected pixels = ~147k object allocations per
  phase, plus the array slicing work.

## Estimated attribution of the ~170s gap

| Source | Impact | Confidence |
|--------|--------|------------|
| Movement optimisation (skipped pixels) | ~118s | High (directly measured) |
| All other differences combined | ~52s | High (by subtraction) |

### Confirming the movement optimisation impact

Temporarily disabling the movement optimisation in the legacy code
(`mvn.py` lines 118-123, commenting out the `continue`) and re-running
the same 1000m benchmark:

| Run | Time |
|-----|------|
| Legacy **with** movement optimisation | 106s |
| Legacy **without** movement optimisation | 224s |
| Refactored code | 276s |

The movement optimisation saves **118s** (53%) in the legacy code. Without it,
the legacy code takes 224s — much closer to the refactored 276s. The remaining
~52s gap comes from the combined effect of items 3-5 above (`find_affected_pixels`
overhead, `scipy.cdist` vs `np.einsum`, object creation per pixel, etc.).

## Baseline comparison table

| Run | Scope | Time |
|-----|-------|------|
| Legacy nproc=1, movement opt ON | full pipeline | 106s |
| Legacy nproc=1, movement opt OFF | full pipeline | 224s |
| Refactored nproc=1 (BLAS multi-threaded) | full pipeline | 276s |
| Refactored nproc=6 (BLAS single-threaded) | geology only | ~51 min |
| Refactored nproc=6 (BLAS single-threaded) | terrain only | ~45 min |

## Changes already made

1. **Adaptive nproc fallback** (`pipeline.py`, `constants.py`): When
   observations exceed `MULTIPROCESS_OBSERVATION_THRESHOLD` (1000), the
   pipeline falls back to `nproc=1` so BLAS can parallelise matrix inversions.
   This avoids the ~50 minute runtime with `nproc=6`.

2. **Skip `obs_to_grid_indices` when nproc=1** (`spatial.py`): The
   per-observation grid index lists are only needed for parallel workers.
   Skipping them when `nproc=1` removes dead work, though the measured
   speedup was within noise (~275s vs ~276s), indicating the loop was not the
   expensive part of `find_affected_pixels`.

## Why we cannot simply port the movement optimisation

The movement optimisation was **deliberately excluded** from the refactored code.
It makes the output depend on processing order: when multiprocessing splits the
grid into row chunks, each chunk starts without cached state (`prev_model_loc`),
so different processor counts cause different pixels to be skipped vs computed.
Even the legacy code cannot reproduce its own output with different `--nproc`
settings (27% of pixels differ, 2.83% mean difference). See
`dev/why_we_cannot_reproduce_v1p0_grid_exactly.md` for full details.

The refactored code computes every pixel independently, making results fully
deterministic and reproducible regardless of `nproc`.

Any future optimisation to skip redundant distance computations must preserve
this property — i.e. the decision to skip must depend only on the pixel's
location relative to observations, not on which pixel was processed previously.

## Potential next steps (not yet implemented)

- Replace `scipy.spatial.distance.cdist` with the legacy complex-number
  distance matrix approach or `np.einsum`.
- Replace per-pixel object creation with direct array operations.
- Profile whether `find_affected_pixels` can be eliminated entirely for the
  nproc=1 path by doing observation selection inline (as legacy does).
- Investigate a deterministic skip optimisation: pre-compute which pixels are
  far from all observations (using the bbox mask) and skip them without
  depending on processing order.

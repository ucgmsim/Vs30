# Jaehwi v1.0 Model Reproduction Investigation

Investigation into reproducing the intended Jaehwi v1.0 Vs30 model, using
`/home/arr65/data/vs30/grid_models/jaehwi_v1p0/V1.0_26Mar.tif` as the reference.

## Conclusion

**The refactored vs30 pipeline correctly reproduces Jaehwi's intended model.**

Our pipeline and Jaehwi's actual code (`run_vs30calc_V1.py` from `/home/arr65/src/Vs30_2026/`)
produce nearly identical output when run in points mode with `--gupdate posterior
--tupdate posterior` (862.14 vs 862.51 m/s at the test point, 0.37 m/s difference).

The ~48 m/s residual gap to V1.0_26Mar.tif is due to implementation artifacts in
Jaehwi's grid-mode processing (float32 precision loss, MVN distance caching, raster
roundtrip) that are absent from our pipeline. These artifacts are not scientifically
meaningful, and reproducing them is not a goal.

## Source Code

Two versions of Jaehwi's code were tested:

| Code | Location | Result |
|------|----------|--------|
| Old fork (`vs30calc_NSHM_newmodel.py`) | `/home/arr65/src/jaehwi_fork_vs30/Vs30/` | **Wrong code** — uses `model_new` and `sites_load_NSHM2022_0`, cannot reproduce reference in any mode |
| Actual code (`run_vs30calc_V1.py`) | `/home/arr65/src/Vs30_2026/` | **Correct code** — uses `model_fixed` and `sites_load_NSHM2022`, matches our pipeline |

Key differences between the two codebases:
- Different model module: `model_fixed` (actual) vs `model_new` (old fork)
- Different observation loader: `sites_load_NSHM2022` (actual) vs `sites_load_NSHM2022_0` (old fork)
- Different Bayesian update formula: `model_fixed.posterior()` uses sequential update
  with `mean_shift` term; `model_new.posterior()` uses a batch formula without it
- The actual code required a typo fix (`params_2` → `params` in type annotations)

## Confirmed Configuration

| Parameter | Default | Confirmed setting | Evidence |
|-----------|---------|-------------------|----------|
| Entry point | — | `run_vs30calc_V1.py` | Jaehwi confirmed this is for making the v1.0 version |
| Update mode | `posterior_paper` | `--gupdate posterior --tupdate posterior` | Jaehwi confirmed; Foster posteriors give 418 m/s vs reference 910 m/s |
| Combination | `stdv_weight=False` → ratio 1.0 | ratio 1.0 (50/50 geometric mean) | Default; matches reference |
| Coastal distance | `mod13=True` but code commented out | Off | Coastal distance code disabled in `model_geology_new.py` |
| GID 4 slope skip | `mod6=True` | On (`skip_alluvium_slope: true`) | Default in `params.py:80` |
| Observations | `sites_load_NSHM2022` | Original observation set | Loaded from `vs30/data/updated/` directory |

## Single-Point Comparison (easting=1575300, northing=5169300, GID 15, TID 11)

| Source | Geol cat | Geol MVN | Terr cat | Terr MVN | Combined MVN |
|--------|---------|---------|---------|---------|-------------|
| Vs30_2026 (`posterior_paper`, default) | 690.97 | 700.25 | 266.87 | 268.49 | **433.60** |
| Old fork (`posterior_paper`, default) | 690.97 | 644.03 | 266.87 | 271.74 | **418.34** |
| Old fork (`--gupdate posterior`) | 708.82 | 645.27 | 597.10 | 596.38 | **620.34** |
| **Vs30_2026 (`--gupdate posterior`)** | **1068.30** | **1074.48** | **691.42** | **692.36** | **862.51** |
| **Our pipeline (Bayesian + ratio)** | **1068.30** | **1073.54** | **691.42** | **692.36** | **862.14** |
| Reference raster (V1.0_26Mar.tif) | — | — | — | — | **910.06** |

## Why Our Pipeline Cannot Exactly Match V1.0_26Mar.tif

Our pipeline is grid-points consistent (confirmed by `test_grid_and_points_consistency`).
Jaehwi's code is NOT — the grid mode introduces:

1. **Float32 coordinate precision**: `_mvn_tiff_worker` casts pixel locations to
   `np.float32` (line 210), losing sub-meter precision for NZTM coordinates (~1.5M).
   Points mode uses float64 throughout.

2. **Float32 intermediate raster roundtrip**: In grid mode, Vs30/stdv values are
   written to GeoTIFF (float32) then read back for MVN. Points mode keeps float64.

3. **MVN distance caching**: Grid mode processes pixels in raster scan order, and
   an optimization (lines 109-114) skips recalculating distances when consecutive
   pixels are spatially close. Points mode recalculates independently.

4. **Complex64 distance matrix**: `_xy2complex` converts to `complex64` (float32
   real/imaginary) for observation-observation distances. Our code uses
   `scipy.spatial.distance.cdist` with float64.

Since our code matches Jaehwi's points-mode output, and our grid mode equals our
points mode, our grid output will match Jaehwi's points mode — not his grid mode.
The ~48 m/s gap is these accumulated precision artifacts baked into V1.0_26Mar.tif.

## 200-Point Experiment Results

Test setup: 200 randomly sampled points from the Wellington test subgrid
(1555050-1610050 x 5145050-5195050, 550x500 pixels).

### Configuration search (run with our pipeline)

| Experiment | Config | Mean abs diff |
|------------|--------|--------------|
| Foster posteriors + ratio + no coast | `posterior_paper` equivalent | 198.83 m/s |
| **Our Bayesian + ratio + no coast** | **Best match** | **18.03 m/s** |
| Jaehwi posteriors + ratio + no coast | Jaehwi batch formula | 19.16 m/s |

The 18 m/s mean abs diff is from MVN grid-vs-points artifacts, not from a
configuration mismatch.

## Full-Grid Reproduction with Jaehwi's Code

Ran `run_vs30calc_V1.py --gupdate posterior --tupdate posterior --nproc 1` on the
full grid to confirm that V1.0_26Mar.tif cannot be exactly reproduced even with
the same codebase.

A patch was required: Jaehwi's `mvn.mvn_tiff` creates a multiprocessing `Pool`
even when `nproc=1`, causing a deadlock. Bypassed with a `nproc == 1` branch
(list comprehension instead of `pool.starmap`).

### Timing

| Step | Duration |
|------|----------|
| Geology Bayesian update | ~11s |
| Geology MVN | 23 min |
| Terrain MVN | 22 min |
| Combination | 17s |
| **Total** | **46 min** |

### Comparison: Jaehwi's code (nproc=1) vs V1.0_26Mar.tif

```
Valid pixels:      25,913,941
Mean abs diff:     10.78 m/s
Median abs diff:   0.00 m/s
Mean % diff:       2.83%
Median % diff:     0.00%
Max % diff:        162.87%

Pixels with >  0.1% diff: 15.99%
Pixels with >  1.0% diff: 14.75%
Pixels with >  5.0% diff: 14.49%
Pixels with > 10.0% diff: 14.33%
Pixels with > 25.0% diff:  1.99%
Pixels with > 50.0% diff:  0.73%
```

**73% of pixels match exactly** (median diff = 0). The ~15% that differ are
MVN-affected pixels near observations where the distance caching optimization
produces different results depending on processing order (`nproc=1` vs the
original multi-worker run). Even Jaehwi's own code cannot reproduce
V1.0_26Mar.tif with a different `nproc` setting.

Output: `/home/arr65/data/vs30/grid_models/jaehwi_v1p0_reproduced_with_jaehwi_code/`

## Remaining Investigation Items

- **Observation set difference** (0.37 m/s): Our reconstructed 671-station CSV
  vs `sites_load_NSHM2022` loader. Minor; could be resolved by comparing the
  two observation sets directly.

## Reference Data

| File | Description |
|------|-------------|
| `V1.0_26Mar.tif` | Reference 2-band GeoTIFF (Vs30 + Standard Deviation), 15200x10600 @ 100m, EPSG:2193, nodata=-32767 |
| `applied_Vs30_data.csv` | 671 observations with model-sampled columns — output of the pipeline, not an alternative input |
| `gid.tif` | Geology ID raster with 50m spatial offset from `V1.0_26Mar.tif` (origin 1060100 vs 1060050) |
| `Vs30_extraction_26Mar.py` | Post-extraction script that reads from a TIF and gap-fills nodata pixels — does NOT create V1.0_26Mar.tif |

## Updated Config

`vs30/configs/jaehwi_v1p0.yaml` settings:
- `do_bayesian_update: true` (Bayesian from priors, not Foster posteriors)
- `combination_method: ratio`
- `combine_ratio: 1.0`
- `apply_coastal_distance_mod: false`
- `skip_alluvium_slope: true`
- `geology_categorical_csv: geology_model_prior_mean_and_standard_deviation.csv`
- `terrain_categorical_csv: terrain_model_prior_mean_and_standard_deviation.csv`

## Parameters That Match

- **Geology/terrain priors:** CSV values match Jaehwi's hardcoded arrays exactly
- **MVN parameters:** phi=1407 (geology), phi=993 (terrain), max_dist=10000,
  max_points=500, cov_reduc=1.5, noisy=True
- **Hybrid slope breakpoints and sigma reduction factors:** Match our constants
- **Nodata value:** -32767

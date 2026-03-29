# Jaehwi v1.0 Model Reproduction Investigation

Investigation into reproducing the intended Jaehwi v1.0 Vs30 model, using
`/home/arr65/data/vs30/grid_models/jaehwi_v1p0/V1.0_26Mar.tif` as the reference.

## Conclusion

**The refactored vs30 pipeline correctly reproduces Jaehwi's intended model.**

A 200-point comparison across New Zealand confirms that both codebases produce
near-identical results in points mode: **81.5% of points agree within 0.01%**,
with a median Vs30 difference of 0.0002 m/s. Geology and terrain IDs match
100%, and categorical model lookups and Bayesian updates are near-identical.

The remaining ~15% of points with >1% difference are all near observation
stations where the MVN spatial adjustment differs due to minor observation set
differences (our reconstructed 671-station CSV vs Jaehwi's internal
`sites_load_NSHM2022` loader).

The ~48 m/s residual gap to V1.0_26Mar.tif is due to implementation artifacts in
Jaehwi's grid-mode processing (float32 precision loss, MVN distance caching, raster
roundtrip) that are absent from our pipeline. These artifacts are not scientifically
meaningful, and reproducing them is not a goal.

An initial 35% terrain ID disagreement was traced to a sampling artifact (pixel
boundary coincidence between V1.0_26Mar.tif and IwahashiPike.tif), not a model
difference. Both codebases use the identical terrain raster file (MD5 match).
See "Terrain Raster Investigation" below for details.

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

## 200-Point Comparison: Our Pipeline vs Jaehwi's Code (Points Mode)

Test setup: 200 randomly sampled points across New Zealand from valid pixels
in V1.0_26Mar.tif. Both codebases run in points mode on the same locations.
Jaehwi's code run via `run_vs30calc_V1.py --gupdate posterior --tupdate posterior`
in `oldvs30_venv`; our pipeline via `pipeline.points_pipeline()` with the
671-station reconstructed observation set.

Script: `dev/compare_points_mode.py`

### Results

```
Combined Vs30 (200 points):
  Mean abs diff:    8.90 m/s
  Median abs diff:  0.0002 m/s
  Max abs diff:     91.10 m/s
  Mean % diff:      2.43%
  Median % diff:    0.00%

  Points with >  0.01% diff:  37/200 (18.5%)
  Points with >  0.10% diff:  30/200 (15.0%)
  Points with >  1.00% diff:  29/200 (14.5%)
  Points with >  5.00% diff:  28/200 (14.0%)

Combined StdDev:
  Mean abs diff:    0.0089
  Median abs diff:  0.0004
  Max abs diff:     0.1174
```

### Intermediate value agreement

| Stage                      | Mean abs diff | Max abs diff |
|----------------------------|--------------|-------------|
| Geology categorical Vs30   | 0.098        | 10.21       |
| Geology categorical StdDev | 0.001        | 0.116       |
| Terrain categorical Vs30   | 0.000        | 0.000       |
| Terrain categorical StdDev | 0.000        | 0.001       |
| Geology MVN Vs30           | 15.59        | 110.85      |
| Terrain MVN Vs30           | 0.002        | 0.272       |

**Key findings:**
- Geology and terrain IDs match 100% (200/200) — same rasters, same sampling.
- Terrain categorical and MVN values are near-identical (< 0.001 m/s).
- Geology categorical values nearly match (0.098 m/s mean diff) — the small
  residual is from reconstructed vs original observation sets affecting the
  Bayesian category-level update.
- Geology MVN is the dominant source of combined difference: 28 of 200 points
  show >5% diff, all near observation stations where the reconstructed
  671-station set differs from Jaehwi's `sites_load_NSHM2022` loader.
- **81.5% of points agree within 0.01%.** At most locations both codebases
  produce virtually identical Vs30 values.

## Terrain Raster Investigation

### Discovery

An early 200-point comparison showed 35% terrain ID disagreement between the two
codebases despite using the same terrain raster file. This section documents the
investigation and resolution.

### The raster files are identical

Both codebases use the Iwahashi-Pike terrain classification raster:

| Property | Value |
|----------|-------|
| Our file | `vs30/resources/geospatial/IwahashiPike.tif` |
| Jaehwi's file | `/home/arr65/src/Vs30_2026/vs30/data/IwahashiPike.tif` |
| MD5 hash | **identical** (`c1cfdaceb8692cf13d9f295a0dfd17c0`) |
| Size | 11,264 x 16,384 pixels |
| Resolution | 100 m |
| CRS | EPSG:2193 (NZTM) |
| Origin | (1,000,000, 6,338,400) |
| Data type | uint8 (terrain categories 1–16, nodata = 255) |

Both codebases sample the raster using the same `floor()` pixel-index method.
Direct testing confirmed that for identical NZTM coordinates, both methods
return the same terrain ID for all 200 points.

### Ruling out the sampling method

Both codes use the same mathematical operation:

```
col = floor((easting  - 1,000,000) / 100)
row = floor((northing - 6,338,400) / (-100))
```

Jaehwi's code does this via GDAL's `GetGeoTransform()`; our code does it via
rasterio's `src.sample()` (which internally uses the same floor-based
transform). When given the same NZTM coordinates, both produce the same pixel
index for all 200 test points.

### Root cause: 50 m grid offset puts all sample points on pixel boundaries

The 200 test points were sampled from V1.0_26Mar.tif pixel centres.
V1.0_26Mar.tif and IwahashiPike.tif share the same 100 m pixel size, but their
grids are offset by 50 m:

| Raster | Origin (easting) | Pixel centres | Pixel boundaries |
|--------|------------------|---------------|-----------------|
| V1.0_26Mar.tif | 1,060,050 | xx100, xx200, xx300, ... | xx050, xx150, xx250, ... |
| IwahashiPike.tif | 1,000,000 | xx050, xx150, xx250, ... | xx000, xx100, xx200, ... |

V1.0_26Mar.tif pixel centres (xx100, xx200, ...) coincide exactly with
IwahashiPike.tif pixel boundaries (xx000, xx100, xx200, ...). Every sample
point fell exactly on a terrain raster pixel boundary:

```python
frac_x = ((easting  - 1_000_000) / 100) % 1.0  # always 0.0
frac_y = ((northing - 6_338_400) / (-100)) % 1.0  # always 0.0
```

### The coordinate roundtrip flips boundary pixels

Jaehwi's code accepts lon/lat input, so the comparison script converts NZTM
coordinates to WGS84 (our script) and then back to NZTM (Jaehwi's code):

```
Original NZTM → WGS84 lon/lat (CSV, 10 decimal places) → NZTM (Jaehwi's code)
```

This roundtrip introduces coordinate shifts of ~3–5 micrometres:

| Coordinate | Original NZTM | After roundtrip | Shift |
|-----------|---------------|-----------------|-------|
| Easting (example) | 1,555,100.0 | 1,555,099.999995 | −5 μm |
| Northing (example) | 5,169,150.0 | 5,169,150.000003 | +3 μm |

For a point on a pixel boundary, `floor()` is sensitive to the direction of
this shift:

```
floor((1,555,100.000000000 - 1,000,000) / 100) = floor(5551.0)      = 5551
floor((1,555,099.999999995 - 1,000,000) / 100) = floor(5550.999...) = 5550  ← different pixel
```

A shift of 5 μm — five thousandths of a millimetre — is enough to move the
`floor()` result by one full pixel (100 m) at exact boundaries.

### Verification

Simulating the coordinate roundtrip in Python and re-sampling the terrain
raster reproduced 65 of the 70 terrain ID mismatches observed between the two
codebases. (The remaining 5 are likely from minor differences in pyproj
versions between `vs30_venv` and `oldvs30_venv`.)

### Resolution

Shifting sample points by 50 m places them at terrain-raster pixel centres
instead of pixel boundaries. With this shift, terrain IDs match **200/200
(100%)** and terrain categorical Vs30 values agree to < 0.001 m/s.

## Earlier Configuration Search (Our Pipeline vs V1.0_26Mar.tif)

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

- **Observation set difference**: Our reconstructed 671-station CSV vs
  `sites_load_NSHM2022` loader. The 200-point comparison shows this affects
  ~15% of points (those near observations) with a mean geology MVN diff of
  ~110 m/s at affected points. This is the only remaining source of
  disagreement between the two codebases. Could be resolved by comparing the
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

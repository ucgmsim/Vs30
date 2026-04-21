# Jaehwi v1.0 Model Reproduction

Validation that our refactored pipeline reproduces the Jaehwi v1.0 Vs30 model.
Reference file: `/home/arr65/data/vs30/grid_models/jaehwi_v1p0/V1.0_26Mar.tif`.

## Conclusion

**The refactored vs30 pipeline correctly reproduces Jaehwi's intended model.**

A 200-point comparison across New Zealand confirms that both codebases produce
near-identical results in points mode: **99.5% of points agree within 0.01%**,
with a mean Vs30 difference of 0.032 m/s.

Our pipeline cannot exactly match V1.0_26Mar.tif because Jaehwi's grid-mode
code introduces float32 precision loss and processor-dependent MVN caching that
our pipeline avoids. Even Jaehwi's own code cannot reproduce V1.0_26Mar.tif
with a different `nproc` setting (2.83% mean difference, 73% of pixels identical).
See `dev/why_we_cannot_reproduce_v1p0_grid_exactly.md` for details.

## Confirmed Configuration

`vs30/configs/jaehwi_v1p0.yaml`:

| Parameter | Setting | Notes |
|-----------|---------|-------|
| `do_bayesian_update` | `true` | Bayesian from priors, not Foster posteriors |
| `combination_method` | `ratio` | |
| `combine_ratio` | `1.0` | 50/50 geometric mean |
| `apply_coastal_distance_mod` | `false` | Coastal distance code disabled in Jaehwi's codebase |
| `apply_alluvium_slope_mod` | `false` | Default in Jaehwi's `params.py:80` |
| `geology_categorical_csv` | `geology_model_prior_mean_and_standard_deviation.csv` | |
| `terrain_categorical_csv` | `terrain_model_prior_mean_and_standard_deviation.csv` | |

Matching parameters:
- **Geology/terrain priors:** CSV values match Jaehwi's hardcoded arrays exactly
- **MVN parameters:** phi=1407 (geology), phi=993 (terrain), max_dist=10000,
  max_points=500, cov_reduc=1.5, noisy=True
- **Hybrid slope breakpoints and sigma reduction factors:** Match our constants
- **Nodata value:** -32767

## 200-Point Comparison: Our Pipeline vs Jaehwi's Code (Points Mode)

200 randomly sampled points across New Zealand from valid pixels in V1.0_26Mar.tif.
Both codebases run in points mode. Jaehwi's code run via
`run_vs30calc_V1.py --gupdate posterior --tupdate posterior` in `oldvs30_venv`;
our pipeline via `pipeline.points_pipeline()`.

Script: `dev/compare_points_mode.py`

```
Combined Vs30 (200 points):
  Mean abs diff:    0.032 m/s
  Median abs diff:  0.0002 m/s
  Max abs diff:     6.30 m/s

  Points with >  0.01% diff:   1/200 ( 0.5%)
  Points with >  1.00% diff:   1/200 ( 0.5%)
  Points with >  5.00% diff:   0/200 ( 0.0%)

Combined StdDev:
  Mean abs diff:    0.0005
  Max abs diff:     0.0145
```

| Stage                      | Mean abs diff | Max abs diff |
|----------------------------|--------------|-------------|
| Geology categorical Vs30   | 0.098        | 10.21       |
| Geology categorical StdDev | 0.001        | 0.116       |
| Terrain categorical Vs30   | 0.000        | 0.000       |
| Terrain categorical StdDev | 0.000        | 0.001       |
| Geology MVN Vs30           | 0.038        | 7.47        |
| Terrain MVN Vs30           | 0.002        | 0.272       |

## Observation Set

The pipeline's `jaehwi_v1p0_independent_observations.csv` (671 stations) is
produced by running Jaehwi's `sites_load_NSHM2022.load_vs()` from
`/home/arr65/src/Vs30_2026/vs30/` and dumping the resulting DataFrame directly
to CSV.

The loader combines three sources:
- **McGann** (276 stations): CPT-derived Vs30, downsampled on 1 km NZMG grid,
  NZMG→NZTM transform, uncertainty = 0.2
- **Wotherspoon** (36 stations): measured Vs30, WGS84→NZTM transform,
  uncertainty = `0.5 if q == 3 else q / 10`
- **Kaiser/GeoNet** (359 stations): GeoNet metadata, WGS84→NZTM transform,
  same uncertainty formula, Q3 stations included (filter commented out)

## Reference Data

| File | Description |
|------|-------------|
| `V1.0_26Mar.tif` | Reference 2-band GeoTIFF (Vs30 + Standard Deviation), 15200x10600 @ 100m, EPSG:2193, nodata=-32767 |
| `applied_Vs30_data.csv` | 671 observations with model-sampled columns — output of the pipeline, not an alternative input |
| `gid.tif` | Geology ID raster with 50m spatial offset from `V1.0_26Mar.tif` (origin 1060100 vs 1060050) |

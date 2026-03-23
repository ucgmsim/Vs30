# Jaehwi v1.0 Model Reproduction Investigation

Investigation into the discrepancy between the output of `vs30 grid` for version
`jaehwi_v1p0` and the reference raster
`/home/arr65/data/vs30/grid_models/jaehwi_v1p0/V1.0_26Mar.tif`.

## Reference Data

| File | Description |
|------|-------------|
| `V1.0_26Mar.tif` | Reference 2-band GeoTIFF (Vs30 + Standard Deviation), 15200x10600 @ 100m, EPSG:2193, nodata=-32767 |
| `applied_Vs30_data.csv` | 671 observations with model-sampled columns (gid, geology_vs30, etc.) — output of the pipeline, not an alternative input |
| `gid.tif` | Geology ID raster with 50m spatial offset from `V1.0_26Mar.tif` (origin 1060100 vs 1060050) |
| `Vs30_extraction_26Mar.py` | Post-processing script for extracting/gap-filling values from a TIF at CSV coordinates |

## Source Code

Jaehwi's fork: `/home/arr65/src/jaehwi_fork_vs30/Vs30/`

Entry point: `vs30calc_NSHM_newmodel.py`, which imports:
- `model_new` and `model_geology_new` (model logic)
- `sites_load_NSHM2022_0` (observation loading)
- `mvn` (spatial adjustment)
- `params` (CLI argument parsing and defaults)

## Confirmed Configuration

Determined via systematic experiment matrix (200-point comparison against Wellington
test subgrid). See `dev/jaehwi_v1p0_reproduction/` for scripts and data.

| Parameter | Jaehwi default (`params.py`) | Confirmed setting | Evidence |
|-----------|------------------------------|-------------------|----------|
| Update mode | `posterior_paper` (Foster posteriors) | `posterior` (Bayesian from priors) | Foster posteriors give 198.83 m/s error; Bayesian gives 18.03 m/s |
| Combination | `stdv_weight=False` → ratio 1.0 | ratio 1.0 (50/50 geometric mean) | Matches default and gives better results than stdv weighting |
| Coastal distance | `mod13=True` but code commented out | Off (`apply_coastal_distance_mod: false`) | Coastal distance code disabled in `model_geology_new.py:255-280` |
| GID 4 slope skip | `mod6=True` | On (`skip_alluvium_slope: true`) | Default in `params.py:80` |
| Observations | 671 stations from 3 sources | 671 reconstructed stations | Validated 671/671 spatial match against `applied_Vs30_data.csv` |

**Key finding:** Despite the default being `posterior_paper`, Jaehwi must have used
`-g posterior -t posterior` when running `vs30calc_NSHM_newmodel.py`. The Foster 2019
posteriors (hardcoded in `model_posterior_paper()`) give catastrophically wrong results
for categories with many high-Vs30 observations (e.g., geology cat 15: Foster=690.97,
Bayesian=1068.30, observations mean=1185.75).

## Experiment Results

Test setup: 200 randomly sampled points from the Wellington test subgrid
(1555050-1610050 x 5145050-5195050, 550x500 pixels).

### Configuration search

| Experiment | Config | Mean abs diff | Median | <1 m/s |
|------------|--------|--------------|--------|--------|
| Foster posteriors + ratio + no coast | `posterior_paper` equivalent | 198.83 m/s | — | — |
| **Our Bayesian + ratio + no coast** | **Best match** | **18.03 m/s** | **0.00 m/s** | **76.5%** |
| Jaehwi posteriors + ratio + no coast | Jaehwi batch formula | 19.16 m/s | 0.55 m/s | 61.0% |
| Prior + Bayesian + stdv weighting + coast on | Old `jaehwi_v1p0.yaml` | ~40+ m/s | — | — |

### Per-category error analysis (Bayesian + ratio + no coast)

| Geology ID | Points | Mean error | Max error | Comment |
|------------|--------|-----------|-----------|---------|
| 1-6, 10 | 81 | 0.01-4.55 | 90.98 | Mostly excellent match |
| 7 | 2 | 48.40 | 96.80 | Small sample, MVN-sensitive |
| 8 | 16 | 30.86 | 198.07 | Some MVN outliers |
| 12 | 6 | 26.05 | 156.25 | MVN-sensitive |
| 15 | 95 | 28.33 | 261.83 | Large sample; residual from Bayesian formula differences |

### Remaining discrepancy sources

The ~18 m/s mean abs diff comes from:

1. **Bayesian update formula differences** (~1 m/s contribution): Our sequential update
   includes `mean_shift` residual term and updates stdv; Jaehwi's batch formula does
   neither. Direct comparison: our Bayesian (18.03) vs Jaehwi's exact posteriors (19.16).

2. **MVN spatial adjustment sensitivity** (dominant): Categories with many high-dispersion
   observations (cat 15: 64 obs averaging 1185 m/s) produce large MVN corrections that
   amplify small differences in the categorical model value.

3. **Post-processing gap-fill**: `Vs30_extraction_26Mar.py` applies nearest-neighbor
   infill for nodata pixels within the coastline, which is not replicated in our pipeline.

## Observations Reconstruction

Script: `dev/jaehwi_v1p0_reproduction/reconstruct_observations.py`

Replicates Jaehwi's `sites_load_NSHM2022_0.py` loading logic:
- **McGann** (276): NZMG coords, downsampled on 1km grid, transformed to NZTM
- **Wotherspoon** (36): Filtered out 140 post-v1p0 SCPT/SDMT entries
- **Kaiser/GeoNet** (359): Filtered out 512 post-v1p0 Foster/Perrin-derived entries

Result: 671 observations, 671/671 spatially matched to `applied_Vs30_data.csv` within 1m.

## Updated Config

`vs30/configs/jaehwi_v1p0.yaml` updated to match confirmed parameters:
- `combination_method: ratio` (was `standard_deviation_weighting`)
- `combine_ratio: 1.0` (was empty)
- `apply_coastal_distance_mod: false` (was `true`)
- `skip_alluvium_slope: true` (new parameter)

## Parameters That Match

- **Grid definition:** 1060050-2120050 x 4730050-6250050, 100m spacing
- **Geology/terrain priors:** CSV values match Jaehwi's hardcoded arrays exactly
- **MVN parameters:** phi=1407 (geology), phi=993 (terrain), max_dist=10000,
  max_points=500, cov_reduc=1.5, noisy=True
- **Hybrid slope breakpoints and sigma reduction factors:** Match our constants
- **Nodata value:** -32767

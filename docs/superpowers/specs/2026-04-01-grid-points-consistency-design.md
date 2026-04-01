# Grid-Points Consistency Test Expansion

**Date:** 2026-04-01
**Status:** Approved
**Goal:** Replace the limited 3-point consistency test with broad-coverage property testing across the full NZ domain and all model versions.

## Background

`tests/test_grid_points_consistency.py` verifies that the grid and points pipelines produce the same Vs30 values by running both on a 3km x 3km domain and comparing 3 pixel-center locations. This is limited: it only exercises geology/terrain categories present in that small region, uses a single model configuration, and may miss issues elsewhere in the domain.

The property under test is: **for any on-land location within `constants.FULL_NZ_GRID_CONFIG`, the grid pipeline and points pipeline should produce approximately equal Vs30 and stdv values.**

"Approximately equal" because the two pipelines sample slope and coastal distance differently: the grid pipeline uses grid-resampled rasters while the points pipeline samples directly from source data.

## Design

### Approach: 3x3 Local Grid Per Point

For each test point, generate a 3x3 pixel grid (300m x 300m at 100m resolution) centered on the point, run `grid_pipeline` on it, extract the center pixel value, and compare with `points_pipeline` at the same coordinates.

**Why 3x3 grids minimize discrepancy:** The grid pipeline's MVN step (`spatial.py:445-474`) samples observation slope/coast from grid-resampled rasters for observations *within* the grid, but falls back to direct source sampling for observations *outside* the grid. With a 300m x 300m grid, virtually all observations fall outside, so both pipelines use the same source data for observation conditioning. The only remaining discrepancy is at the query pixel itself:

- **Slope:** The source raster is 270m resolution. Nearest-neighbor reprojection from 270m to 100m at a grid-aligned pixel center returns the same source pixel as `raster.sample_slope_at_points()`.
- **Coastal distance:** GDAL proximity raster vs shapely geometric distance. Small numerical difference.

### Test Points

**Total: ~35-45 points**, stored as a pre-computed CSV fixture at `tests/fixtures/consistency_test_points.csv`.

Columns: `name, longitude, latitude, category` where category is one of `city`, `rare_geology`, `coastal_sensitive`, `observation_sparse`, `geology_boundary`, `random`.

#### Deliberate Points (~10-15)

| Category | Rationale | Examples |
|----------|-----------|---------|
| Major cities | High-impact locations near observations | Auckland, Wellington, Christchurch, Dunedin, Hamilton |
| Rare geology | Categories with few or no observations: GID 1 (peat), GID 5 (lacustrine), GID 9 (outwash), GID 14 (volcanic) | Specific locations on those geology types |
| Coastal-sensitive geology | GID 4 (alluvium) and GID 10 (flood plain) near coast, which receive the coastal distance modification | Coastal Canterbury, Hawke's Bay |
| Observation-sparse | Far from any observation station, testing MVN extrapolation | Remote Fiordland or West Coast |
| Near geology boundary | Transition zone between geology categories | Edge of alluvium/terrace boundary |

#### Random Points (~20-30)

- Generated with `np.random.default_rng(42)` for reproducibility
- Uniform random NZTM coordinates within `FULL_NZ_GRID_CONFIG` bounds
- Filtered to on-land using the coastline shapefile
- Snapped to pixel centers of the 100m NZTM grid
- The generation script (or notebook) is committed alongside the fixture so points can be regenerated if needed

### Model Versions

All four fixed model versions are tested:

| Model | `apply_coastal_distance_mod` | Observations | Tier |
|-------|------------------------------|-------------|------|
| `foster_2019` | false | 416 | fast |
| `jaehwi_v1p0` | false | 671 | fast |
| `modified_foster_2019` | true | 470 | slow |
| `viktor_cpt_clustering` | true | 35,709 | slow |

### Two-Tier Test Structure

The coastal distance computation (`raster.compute_coast_distance_array`) always extends to the full NZ domain (~161M pixels) regardless of target grid size. This makes models with `apply_coastal_distance_mod: true` ~5-10x slower per point than models without it.

#### Fast tier (default CI)

- Models: `foster_2019`, `jaehwi_v1p0`
- Points: all ~40
- Estimated time: **~4-5 minutes**
- No special marker; runs in the default test suite

#### Slow tier (nightly / on-demand)

- Models: `modified_foster_2019`, `viktor_cpt_clustering`
- Points: all ~40
- Estimated time: **~25-30 minutes**
- Marked with `@pytest.mark.slow`
- Excluded from default CI via `pytest -m "not slow"` (or equivalent conftest/pyproject config)

Both tiers use the same comparison logic and point set; only the model version parametrization differs.

### Comparison Logic

For each model version:

1. Load the model config from `constants.MODEL_VERSION_TO_CONFIG[version]`
2. Run `points_pipeline` **once** with all ~40 points batched into a single call (efficient: single shapefile load, single MVN computation)
3. For each point individually:
   a. Create a 3x3 `GridConfig` using `gapfill.create_local_grid_config(easting, northing, FULL_NZ_GRID_CONFIG, half_width=150)` — this snaps to the standard NZ grid and produces a 3-pixel-wide grid centered on the target (nx = 300/100 = 3 pixels per side)
   b. Run `grid_pipeline` with `output_dir=None` (in-memory, no file I/O)
   c. Extract center pixel (row=1, col=1) Vs30 and stdv from the grid result
   d. Compare with the corresponding row from the batched points_pipeline DataFrame
4. Assert approximate equality for all points

### Tolerances

Start with the current test's tolerances:
- Vs30: `rtol=0.03` (3%)
- Stdv: `rtol=0.30` (30%)

The 3x3 grid approach should produce tighter agreement than the current 30x30 grid test. Once the test is running, empirically check whether tolerances can be tightened (target: rtol=0.01 for Vs30, rtol=0.15 for stdv). Tighten only if all points pass comfortably with margin.

### Handling Edge Cases

- **Ocean/nodata points:** Pre-filtered out during point selection. The fixture CSV contains only on-land points.
- **Gap-fill triggering:** Some points may have nodata in the combined model (e.g., GID 0 water at the 3x3 grid scale). These should be identified and excluded during point selection, or the test should handle them by checking that both pipelines produce nodata.
- **Points at domain boundary:** Ensure the 3x3 grid doesn't extend beyond `FULL_NZ_GRID_CONFIG` bounds. Either exclude such points or clamp the grid.

### File Structure

```
tests/
  fixtures/
    consistency_test_points.csv          # Pre-computed test point coordinates
  test_grid_points_consistency.py        # Rewritten test file
dev/
  generate_consistency_test_points.py    # Script to regenerate the fixture CSV
```

The existing `test_grid_points_consistency.py` is replaced entirely. The original 3-point test becomes redundant once the expanded test covers the same domain.

### Dependencies

No new packages. The test uses existing pipeline functions, `gapfill.create_local_grid_config()` for grid alignment, and standard pytest parametrize/markers.

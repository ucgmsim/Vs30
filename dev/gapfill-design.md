# Gap-Fill Design Spec

## Goal

Fill nodata gaps in the VS30 pipeline output using nearest-neighbor
interpolation, reproducing the logic from Jaehwi's
`Vs30_extraction_26Mar.py`. Applied as a post-processing stage in both
grid and points modes.

## Background

The combined VS30 grid has nodata pixels where the source geology or
terrain rasters lack coverage. These fall into three categories:

1. **Water** (GID=0) — ocean, lakes, rivers. Leave as nodata.
2. **Outside coastline** — offshore pixels with no geology data (GID=255
   or GID=0) that fall outside the coastline polygon. Leave as nodata.
3. **On-land gaps** — pixels inside the coastline where the geology raster
   has no coverage (GID=255) or the combined output is otherwise nodata
   for non-water reasons (e.g., terrain raster lacks coverage at that
   pixel). Typically coastal fringes or small islands not covered by the
   source vector data. Fill with nearest valid neighbor.

Only category 3 gets filled. The coastline shapefile
(`nz-coastlines-and-islands-polygons-topo-1500k.shp`, already bundled in
`vs30/resources/geospatial/coast/`) distinguishes categories 2 and 3.

## Architecture

### New module: `vs30/gapfill.py`

Two functions:

**`classify_nodata(combined_vs30, geology_ids, locations)`**

Takes the combined VS30 array (or 1D point values), geology IDs, and an
Nx2 array of NZTM `[easting, northing]` coordinates (consistent with the
convention used by `category.assign_to_category_geology` and other
functions in the codebase). Computes `np.isnan(combined_vs30)` internally
to identify nodata pixels (the in-memory arrays use NaN for nodata; the
`NODATA_VALUE` sentinel is only used for file I/O). Returns a boolean mask
of pixels eligible for filling: `nodata AND geology_id != 0 AND inside
coastline polygon`.

The geology `id_array` is used (not terrain) because the GID=0 water
classification is specific to the geology raster's QMAP source data.
Pixels with valid geology but invalid terrain coverage will have NaN in
the combined output and will pass the `geology_id != 0` filter — this is
correct, as they are on-land gaps that should be filled.

Loads the coastline shapefile via `raster.ensure_shapefile_extracted(
constants.GEOSPATIAL_DIR / constants.COASTLINE_SHAPEFILE_PATH, "coast")`
(matching the existing call in `raster.compute_coastal_distance_at_points`),
unions the geometry, and tests point-in-polygon for nodata pixels that
passed the GID != 0 filter.

If there are no nodata pixels, or none pass the GID != 0 filter, returns
an all-False mask immediately (no-op fast path).

Shared between grid and points modes.

**`fill_nodata_grid(vs30, stdv, geology_ids, profile)`**

Entry point for both grid and points modes. Derives pixel center locations
as an Nx2 array of `[easting, northing]` from the rasterio profile's
affine transform. Passes `vs30` and the locations array to
`classify_nodata` to identify fillable pixels. Builds a `cKDTree` from all
valid (non-NaN) pixel coordinates in the combined output, queries nearest
neighbor for each fillable pixel, and copies both vs30 and stdv values
from the donor pixel. Returns filled copies of both arrays.

The donor values come directly from the combined output grid, which
already has the full pipeline applied (categorical + hybrid + MVN +
combination). No re-computation is needed.

If there are fillable pixels but no valid donor pixels (e.g., grid
entirely over ocean), logs a warning and returns the input arrays
unchanged.

### Grid/points consistency via shared fill algorithm

Both grid mode and points mode use the same `fill_nodata_grid` function
to fill gaps. This guarantees identical fill values at the same location.

In grid mode, the combined output grid is used directly.

In points mode, for each fillable query point, a small local grid is
generated around the point (e.g., 10 km x 10 km), the full pipeline
(categorical + hybrid + MVN + combination) is run on this local grid,
and then `fill_nodata_grid` fills the gaps in this local grid using the
same KDTree nearest-neighbor algorithm as grid mode. The fill value at
the pixel containing the query point is extracted as the result.

For grid/points consistency, the local grid's pixel centers must align
with the grid used in grid mode. This is achieved via a
`gapfill_grid_config` parameter in `points_pipeline` (see Pipeline
Integration below).

### Pipeline integration

**Grid mode (`grid_pipeline`):**

After Stage 5 (model combination), a new Stage 6:

1. If `include_intermediate`: write the pre-fill combined output as
   `combined_vs30_before_gapfill.tif`.
2. Call `gapfill.fill_nodata_grid(combined_vs30, combined_stdv,
   geology_ids, profile)`.
3. Write the gap-filled result as `combined_vs30.tif` (the final output,
   same filename as today).

The geology ID array is computed during `compute_model_grid` but is
currently a local variable not returned to the caller.
`compute_model_grid` must be modified to return `id_array` alongside
`(vs30, stdv, profile)`, changing its return type from
`tuple[np.ndarray, np.ndarray, dict]` to
`tuple[np.ndarray, np.ndarray, np.ndarray, dict]`. Both call sites in
`grid_pipeline` (geology and terrain) must be updated to unpack the new
4-tuple. Only the geology `id_array` is needed for gap-fill; the terrain
`id_array` can be discarded with `_`.

**Points mode (`points_pipeline`):**

A new `gapfill_grid_config` parameter is added to `points_pipeline`:

```python
def points_pipeline(
    ...,
    gapfill_grid_config: GridConfig = constants.DEFAULT_GAPFILL_GRID_CONFIG,
):
```

This defines the grid alignment (origin, spacing) used when generating
local grids for gap-fill. It defaults to `DEFAULT_GAPFILL_GRID_CONFIG`
(defined in `constants.py`), which uses the standard NZ domain at 100m
spacing with the `*050` origin alignment matching `jaehwi_v1p0` and
`modified_foster_2019` configs. Users who need exact alignment with a
specific grid mode run can pass their grid config explicitly.

Gap-fill runs after both the parallel path (`run_parallel_locations`) and
the sequential path produce the combined result DataFrame. Currently the
parallel path returns early from `points_pipeline`. This early return
must be removed so that both paths flow into a shared post-processing
block that includes gap-fill. The shared block:

1. If `include_intermediate`: store pre-fill values as
   `vs30_before_gapfill` and `stdv_before_gapfill` columns.
2. Obtain geology IDs for the query points by calling
   `category.assign_to_category_geology(points)`. This is a redundant
   raster sample (geology IDs were already computed during the geology
   processing stage) but is straightforward and avoids threading IDs
   through both the sequential and parallel code paths.
3. Call `gapfill.classify_nodata` to identify fillable points.
4. For each fillable point, generate a local grid centred on the grid
   pixel from `gapfill_grid_config` that contains the fillable point.
   The local grid's half-width is `constants.GAPFILL_LOCAL_GRID_SIZE_M`
   (default 5000m, giving a 10 km x 10 km grid). The local grid uses the
   same spacing and origin alignment as `gapfill_grid_config`, ensuring
   pixel centers match any full grid run with that config. Run the full
   pipeline stages (categorical + hybrid + MVN + combination) on this
   local grid, then call `gapfill.fill_nodata_grid` to fill gaps.
   Extract the fill value at the pixel containing the query point. If no
   valid donor pixel exists in the local grid, expand the half-width by
   `constants.GAPFILL_LOCAL_GRID_EXPANSION_M` (default 5000m) and retry.
5. Copy the results into the nodata slots. The output coordinates remain
   the user's original query coordinates.

The number of fillable points is expected to be very small (0-5 in
typical use), so running a small local grid per fillable point is cheap.

### Gap-fill is unconditional

Gap-filling is always applied. There is no config flag to disable it.
The `include_intermediate` flag controls whether the pre-fill output is
preserved (as `combined_vs30_before_gapfill.tif` in grid mode, or as
`vs30_before_gapfill`/`stdv_before_gapfill` columns in points mode).

### What does NOT get gap-filled

- Individual geology/terrain model outputs (only the combined output).
- Water pixels (GID=0), including inland lakes and rivers.
- Offshore pixels outside the coastline polygon.

### No maximum fill distance

No distance limit is applied to the nearest-neighbor search. The
reference implementation (`Vs30_extraction_26Mar.py`) has no distance
limit, and the number of on-land gap pixels is small relative to the
total grid. In practice, the nearest valid pixel is typically within a few
hundred metres (adjacent coastal pixel).

## Data flow

### Grid mode

```
combined_vs30 (with nodata gaps, NaN for missing)
        |
        v
fill_nodata_grid(vs30, stdv, geology_ids, profile)
        |
        +-> classify_nodata(combined_vs30, geology_ids, locations)
        |       - np.isnan(combined_vs30) to find nodata
        |       - exclude geology_id == 0 (water)
        |       - check coastline polygon (exclude offshore)
        |       -> fillable_mask
        |
        +-> cKDTree(valid pixel coords) -> nearest valid neighbor
        |
        v
combined_vs30_filled (gaps filled with nearest valid vs30 and stdv)
```

### Points mode

```
combined results (some points NaN)
        |
        v
geology_ids = category.assign_to_category_geology(query_points)
        |
        v
classify_nodata(combined_vs30, geology_ids, locations)
        |
        v
fillable points identified
        |
        v
for each fillable point:
        |
        +-> snap to nearest pixel center in gapfill_grid_config
        +-> define local grid (e.g., 10 km x 10 km) aligned to config
        +-> run pipeline stages on local grid
        +-> fill_nodata_grid on local grid -> extract fill value
        |
        v
copy fill values into original nodata slots (keep original coordinates)
```

## Testing

Two integration tests using the real coastline shapefile (consistent with
the project's testing philosophy of focusing on core scientific
calculations):

1. **Grid mode**: Construct a small grid region that contains known nodata
   pixels of each category (water, offshore, on-land gap), run
   `fill_nodata_grid`, and verify:
   - Water pixels remain nodata.
   - Offshore pixels remain nodata.
   - On-land gap pixels are filled with the nearest valid value.

2. **Points mode**: Query a point at a known on-land gap location, verify
   that the result is non-NaN and matches the expected value from the
   nearest valid shifted-coordinate computation.

The existing `test_compute_at_locations.py` benchmark CSV will need
regeneration if any of the test city locations are affected by gap-filling.

## Edge cases

- **No nodata pixels**: `classify_nodata` returns all-False mask, gap-fill
  is a no-op.
- **Fillable pixels but no valid donors** (e.g., grid entirely over
  ocean): log a warning and return arrays unchanged.
- **All pixels nodata**: same as above — no valid donors, no filling.
- **Coastline shapefile not yet extracted**: `classify_nodata` calls
  `raster.ensure_shapefile_extracted()` before loading, consistent with
  existing usage in `raster.py`.
- **Local grid too small to contain valid donor**: expand the local grid
  half-width by `GAPFILL_LOCAL_GRID_EXPANSION_M` and retry.

## Dependencies

- `shapely` (already a dependency via `geopandas`) for point-in-polygon
- `scipy.spatial.cKDTree` (already a dependency) for nearest-neighbor
- Coastline shapefile already bundled in `vs30/resources/geospatial/coast/`
- `raster.ensure_shapefile_extracted()` for shapefile extraction
- `category.assign_to_category_geology` for GID lookup

## Files touched

- Create: `vs30/gapfill.py`
- Modify: `vs30/pipeline.py` — add Stage 6 in `grid_pipeline` and
  `points_pipeline`; change `compute_model_grid` return type from 3-tuple
  to 4-tuple (add `id_array`); update both call sites in `grid_pipeline`;
  add `gapfill_grid_config` parameter to `points_pipeline`; remove early
  return in parallel points path so both paths flow into shared gap-fill
  post-processing
- Modify: `vs30/constants.py` — add
  `COMBINED_VS30_BEFORE_GAPFILL_FILENAME`,
  `DEFAULT_GAPFILL_GRID_CONFIG`,
  `GAPFILL_LOCAL_GRID_SIZE_M` (default 5000m), and
  `GAPFILL_LOCAL_GRID_EXPANSION_M` (default 5000m)
- Modify: tests as needed for new benchmarks

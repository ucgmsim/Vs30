# Gapfill Memory Optimization

## Problem

`gapfill.fill_nodata_grid` builds a coordinate array and KDTree for every pixel
in the grid. On the full NZ grid (161M pixels at 100m resolution), this requires
10+ GB of RAM — enough to exhaust memory on a typical workstation.

The dominant memory consumers are:

| Structure | Size (161M pixels, float64) |
|-----------|-----------------------------|
| `locations` array (N, 2) | ~2.6 GB |
| KDTree from valid pixels | several GB |
| Shapely points for coastline test | several GB |
| vs30 + stdv float64 copies | ~2.6 GB |

In practice, only a small fraction of pixels are nodata gaps that need filling
(typically small clusters at coastline edges). The current implementation
allocates memory proportional to the total grid size rather than the number of
gaps.

## Design

Reduce memory by restricting coordinate computation, KDTree construction, and
coastline testing to a neighborhood around the nodata gaps. The public interface
of `fill_nodata_grid` and `classify_nodata` is unchanged.

### Algorithm

1. **Identify nodata pixels in index space.** Find row/col positions where
   `vs30` is NaN. This is a cheap boolean operation on the existing array — no
   coordinate computation needed.

2. **Dilate the nodata mask.** Use `scipy.ndimage.binary_dilation` to expand the
   nodata mask by a buffer of `GAPFILL_LOCAL_GRID_SIZE_M / dx` pixels (50 pixels
   at 100m resolution = 5 km). The dilated region defines the "neighborhood" —
   the set of pixels that need coordinates.

3. **Compute float32 coordinates only within the dilated region.** Build a
   `(N_dilated, 2)` locations array in float32 instead of a `(N_total, 2)` array
   in float64. Float32 precision at NZTM magnitudes (~6.25M meters) gives
   worst-case error of ~0.7m — negligible on a 100m grid for nearest-neighbor
   lookup.

4. **Classify nodata pixels.** Extract the corresponding `combined_vs30`,
   `geology_ids`, and `locations` for pixels within the dilated region, then
   pass these subsets to `classify_nodata`. It performs the same water (GID=0)
   and coastline checks as before, but on a much smaller input. This reduces
   the number of shapely points created for the point-in-polygon test.

5. **Build KDTree from valid pixels in the neighborhood.** The tree contains
   only valid (non-NaN) pixels within the dilated region, not the entire grid.

6. **Query and fill.** Find nearest valid donors for fillable pixels. Index back
   into the original float64 vs30/stdv arrays to copy donor values, preserving
   full upstream precision.

7. **Expand-and-retry fallback.** If any fillable pixel has no valid donor within
   the initial buffer, expand the dilation by
   `GAPFILL_LOCAL_GRID_EXPANSION_M / dx` pixels and retry for those pixels only.
   Cap at `GAPFILL_MAX_LOCAL_GRID_HALF_WIDTH_M / dx` pixels. This reuses the
   existing constants from the points pipeline's gap-fill expansion logic
   (`constants.py:240-248`).

### Constants reused

The dilation buffer sizes reuse the existing gap-fill constants in
`constants.py`, converting from meters to pixels using the grid spacing:

| Constant | Value | Pixels (100m grid) | Purpose |
|----------|-------|---------------------|---------|
| `GAPFILL_LOCAL_GRID_SIZE_M` | 5000 | 50 | Initial dilation buffer |
| `GAPFILL_LOCAL_GRID_EXPANSION_M` | 5000 | 50 | Buffer expansion step |
| `GAPFILL_MAX_LOCAL_GRID_HALF_WIDTH_M` | 50000 | 500 | Maximum buffer |

### Memory estimate

For a typical case with ~1000 nodata pixels and a 50-pixel dilation buffer, the
neighborhood contains roughly 100k-200k pixels instead of 161M. Memory for the
coordinate array drops from ~2.6 GB to ~1-2 MB (float32), and the KDTree shrinks
proportionally.

Even in a worst case where gaps are scattered across the grid and the dilated
regions span a large fraction of it, memory usage is bounded by the dilated area
rather than the full grid, and the fallback expansion is capped.

## Scope

### What changes

- **`gapfill.fill_nodata_grid`** — internal restructuring to use dilation-based
  neighborhood instead of full-grid coordinate computation. Public signature
  unchanged.

- **`gapfill.classify_nodata`** — no signature change. It will receive smaller
  input arrays (only nodata candidates within the neighborhood), but this is
  transparent to callers.

### What doesn't change

- Public function signatures for `fill_nodata_grid` and `classify_nodata`
- The three-category classification logic (water / offshore / on-land)
- Nearest-neighbor fill semantics and output values
- `create_local_grid_config` (used by the points pipeline, unrelated)
- `gapfill_benchmark.py` (calls the same `fill_nodata_grid`)
- Pipeline call sites (`pipeline.py:1190`, `pipeline.py:1540`)

## Testing

The existing tests in `test_gapfill.py` validate correctness on small grids and
should continue to pass without modification, since the public interface and
output semantics are unchanged.

No new tests are needed — the optimization is internal to `fill_nodata_grid` and
does not alter observable behavior.

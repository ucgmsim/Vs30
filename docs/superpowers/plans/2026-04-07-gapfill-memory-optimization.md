# Gapfill Memory Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reduce peak memory in `fill_nodata_grid` so it handles the full NZ 161M-pixel grid without exhausting RAM.

**Architecture:** Restrict coordinate computation, KDTree construction, and coastline testing to a small neighborhood around fillable on-land gaps. Classify nodata pixels (water/offshore/on-land) in index space first, then dilate only the fillable mask to define the donor search region. Use float32 for spatial lookup coordinates and index back into the original float64 arrays for donor values.

**Tech Stack:** scipy.ndimage.maximum_filter (separable dilation), scipy.spatial.KDTree, numpy, shapely

**Spec:** `docs/superpowers/specs/2026-04-07-gapfill-memory-optimization-design.md`

---

### Task 1: Verify existing tests pass (baseline)

**Files:**
- Test: `tests/test_gapfill.py`

- [ ] **Step 1: Run the existing gapfill tests**

Run:
```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv && \
pytest tests/test_gapfill.py -v
```

Expected: All 3 tests pass (`test_classify_nodata_excludes_water_and_offshore`, `test_fill_nodata_grid_nearest_neighbor`, `test_create_local_grid_config_expansion`).

---

### Task 2: Add `pixel_coords_float32` helper and `maximum_filter` import

**Files:**
- Modify: `vs30/gapfill.py:1-12` (imports) and add new function after line 12

- [ ] **Step 1: Add the `maximum_filter` import**

In `vs30/gapfill.py`, add `from scipy.ndimage import maximum_filter` to the imports. The import block should read:

```python
import logging

import geopandas as gpd
import numpy as np
import shapely
from scipy.ndimage import maximum_filter
from scipy.spatial import KDTree

from vs30 import config, constants, raster
```

- [ ] **Step 2: Add the `pixel_coords_float32` helper function**

Add this function immediately after the imports (before `classify_nodata`), at line 15:

```python
def pixel_coords_float32(
    rows: np.ndarray,
    cols: np.ndarray,
    transform,
) -> np.ndarray:
    """
    Compute float32 NZTM pixel center coordinates from row/col indices.

    Computes in float64 from the affine transform, then downcasts to float32.
    Float32 precision at NZTM magnitudes (~6.25M meters) gives worst-case
    error of ~0.7m — negligible on a 100m grid for nearest-neighbor lookup.
    """
    eastings = transform.c + transform.a * (cols + 0.5)
    northings = transform.f + transform.e * (rows + 0.5)
    return np.column_stack([
        eastings.astype(np.float32),
        northings.astype(np.float32),
    ])
```

- [ ] **Step 3: Run tests to verify nothing broke**

Run:
```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv && \
pytest tests/test_gapfill.py -v
```

Expected: All 3 tests pass (the helper is not yet called).

- [ ] **Step 4: Commit**

```bash
git add vs30/gapfill.py
git commit -m "refactor(gapfill): add pixel_coords_float32 helper and maximum_filter import

Preparation for memory-optimized fill_nodata_grid. The helper computes
pixel center coordinates in float32, and maximum_filter will be used for
efficient separable dilation of the fillable mask."
```

---

### Task 3: Rewrite `fill_nodata_grid` with classify-before-dilate algorithm

**Files:**
- Modify: `vs30/gapfill.py:85-165` (the `fill_nodata_grid` function body)

The function signature and docstring are unchanged. Replace the function body
(everything after the docstring) with the new implementation.

- [ ] **Step 1: Replace the `fill_nodata_grid` body**

Replace the body of `fill_nodata_grid` (from `nrows, ncols = vs30.shape` through
`return filled_vs30, filled_stdv`) with:

```python
    nrows, ncols = vs30.shape

    # Fast path: no nodata pixels
    nodata_2d = np.isnan(vs30)
    if not np.any(nodata_2d):
        return vs30.copy(), stdv.copy()

    # --- Classification: identify fillable on-land gaps ---
    # Exclude water (GID=0) in index space — no coordinates needed
    candidate_2d = nodata_2d & (geology_ids != 0)
    if not np.any(candidate_2d):
        return vs30.copy(), stdv.copy()

    # Compute float32 coordinates for non-water nodata candidates only
    transform = profile["transform"]
    candidate_rows, candidate_cols = np.where(candidate_2d)
    candidate_locations = pixel_coords_float32(
        candidate_rows, candidate_cols, transform
    )

    # Coastline check via classify_nodata on the small candidate set
    candidate_vs30 = np.full(len(candidate_rows), np.nan)
    candidate_gids = geology_ids[candidate_rows, candidate_cols]
    fillable_of_candidates = classify_nodata(
        candidate_vs30, candidate_gids, candidate_locations
    )

    if not np.any(fillable_of_candidates):
        return vs30.copy(), stdv.copy()

    # Map fillable indices back to 2D grid positions
    fillable_rows = candidate_rows[fillable_of_candidates]
    fillable_cols = candidate_cols[fillable_of_candidates]
    fillable_2d = np.zeros((nrows, ncols), dtype=bool)
    fillable_2d[fillable_rows, fillable_cols] = True
    fillable_locations = candidate_locations[fillable_of_candidates]
    n_fillable = len(fillable_rows)

    # --- Nearest-neighbor fill with expanding buffer ---
    # Reuse the same constants as the points pipeline's gap-fill expansion
    dx = abs(transform.a)
    buffer_pixels = round(constants.GAPFILL_LOCAL_GRID_SIZE_M / dx)
    expansion_pixels = round(constants.GAPFILL_LOCAL_GRID_EXPANSION_M / dx)
    max_buffer_pixels = round(constants.GAPFILL_MAX_LOCAL_GRID_HALF_WIDTH_M / dx)

    filled_vs30 = vs30.copy()
    filled_stdv = stdv.copy()

    while buffer_pixels <= max_buffer_pixels:
        # Dilate only the fillable mask (not the full nodata mask) to define
        # the donor search neighborhood. maximum_filter with a square kernel
        # is separable and runs in O(N) regardless of kernel size.
        struct_size = 2 * buffer_pixels + 1
        neighborhood_2d = maximum_filter(fillable_2d, size=struct_size)

        # Find valid (non-NaN) pixels within the neighborhood
        valid_in_neighborhood = ~nodata_2d & neighborhood_2d
        if not np.any(valid_in_neighborhood):
            logger.info(
                f"  Gap-fill: no valid donors within {buffer_pixels}-pixel "
                f"({buffer_pixels * dx:.0f}m) buffer, expanding"
            )
            buffer_pixels += expansion_pixels
            continue

        # Compute float32 coordinates for valid donor pixels
        valid_rows, valid_cols = np.where(valid_in_neighborhood)
        valid_locations = pixel_coords_float32(
            valid_rows, valid_cols, transform
        )

        # Build KDTree from neighborhood donors and query for fillable pixels
        tree = KDTree(valid_locations)
        _, nn_indices = tree.query(fillable_locations)

        # Copy values from the original float64 arrays to preserve precision
        donor_rows = valid_rows[nn_indices]
        donor_cols = valid_cols[nn_indices]
        filled_vs30[fillable_rows, fillable_cols] = vs30[donor_rows, donor_cols]
        filled_stdv[fillable_rows, fillable_cols] = stdv[donor_rows, donor_cols]

        logger.info(
            f"  Gap-fill: filled {n_fillable} pixel(s) with nearest-neighbor values"
        )
        return filled_vs30, filled_stdv

    # Exhausted all buffer expansions without finding valid donors
    logger.warning(
        f"  Gap-fill: no valid donors found within maximum buffer of "
        f"{max_buffer_pixels} pixels ({max_buffer_pixels * dx:.0f}m). "
        f"{n_fillable} pixel(s) remain unfilled."
    )
    return filled_vs30, filled_stdv
```

- [ ] **Step 2: Run the gapfill tests**

Run:
```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv && \
pytest tests/test_gapfill.py -v
```

Expected: All 3 tests pass. The `test_fill_nodata_grid_nearest_neighbor` test
verifies that the center pixel of a 3x3 grid is filled with 250.0 (the value
from the pixel directly north), which confirms the nearest-neighbor logic works
correctly with the new algorithm.

- [ ] **Step 3: Run the full test suite**

Run:
```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv && \
pytest tests/ -v
```

Expected: All tests pass.

- [ ] **Step 4: Commit**

```bash
git add vs30/gapfill.py
git commit -m "perf(gapfill): classify before dilating to reduce memory usage

Restrict coordinate computation and KDTree construction to a small
neighborhood around fillable on-land gaps instead of the full grid.
This reduces peak memory from 10+ GB to a few MB for typical inputs.

Algorithm: exclude water (GID=0) in index space, run coastline check
on the small candidate set, dilate only the fillable mask to define
the donor search region, then build the KDTree from valid pixels in
that neighborhood. Uses float32 coordinates for spatial lookup and
indexes back into the original float64 arrays for donor values."
```

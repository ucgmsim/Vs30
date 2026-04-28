# Grid Pipeline Cleanup Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement the grid-pipeline cleanups recommended by the perf-features investigation: remove the per-pixel MVN multiproc path, drop the dead `obs_to_grid_indices` field, drop `nproc` from `find_affected_pixels`, and rename the remaining `nproc` to `dbscan_nproc` on grid-only entry points.

**Architecture:** Five atomic commits, each independently reviewable and bisectable. Each commit ends green on `ruff check && ruff format --check` and on the relevant subset of tests. Reference design at `dev/docs/grid_cleanup_design.md`.

**Tech Stack:** Python 3.13, pytest, ruff, mamba `vs30_venv`. Activate with:

```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv
```

(Henceforth abbreviated as `<activate>`.)

---

## File map

Files modified by this plan:

| Path | Change |
|---|---|
| `vs30/spatial.py` | Drop `obs_to_grid_indices` field from `BoundingBoxResult`; simplify `grid_points_in_bbox`, `process_bbox_chunk`, `find_affected_pixels` (remove parallel branch + obs-indices accumulation); drop `nproc` param from `find_affected_pixels`. |
| `vs30/parallel.py` | Delete `run_parallel_spatial_fit` and its private worker `process_pixels_chunk`. Keep `run_parallel_locations` (points mode). |
| `vs30/constants.py` | Delete `MULTIPROCESS_OBSERVATION_THRESHOLD`. |
| `vs30/pipeline.py` | `compute_spatial_adjustment_on_grid`: drop `nproc`, remove threshold guard + multiproc branch. Rename `nproc` → `dbscan_nproc` on `compute_categorical_vs30_updates`, `compute_model_grid`, `grid_pipeline`. |
| `vs30/cli.py` | `grid` and `grid-custom`: rename `--nproc` → `--dbscan-nproc`; pass `dbscan_nproc` to `pipeline.grid_pipeline`. `points` and `points-custom` untouched. |
| `tests/test_benchmarks.py` | Drop the `_multiprocess` grid tests; consolidate `_single_process` ones; update `run_benchmark` to use `dbscan_nproc=1`. |
| `tests/test_grid_points_consistency.py` | Update grid_pipeline call to use `dbscan_nproc=1`. |
| `dev/scripts/investigations/diagnose_jaehwi_mask.py` | Update grid_pipeline call. |
| `dev/scripts/investigations/perf_features_investigation/run_full_pipeline_confirmation.py` | Update grid_pipeline call. |
| `dev/scripts/investigations/perf_features_investigation/bench_utils.py` | Drop `obs_to_grid_indices` keyword from `BoundingBoxResult` construction; remove `bypass_observation_threshold`, `_compute_one`, `run_numerical_equivalence_check`; simplify `time_one_run` (drop nproc parameter and branch). |
| `dev/scripts/investigations/perf_features_investigation/test_bench_utils.py` | Remove tests for the removed harness functions. |

Files NOT modified (in scope guard):

- `vs30/multiprocess.py` — `run_parallel_locations` (points mode) still uses `spawn_context`, `single_threaded_blas`, `resolve_nproc`.
- `pipeline.points_pipeline` — entirely untouched.
- `vs30 points` / `vs30 points-custom` CLI — entirely untouched.
- `tests/test_benchmarks.py::test_foster_2019_approx_points_benchmark` — uses `points_pipeline`, untouched.
- Reference rasters in `tests/benchmarks/` — not regenerated.

---

## Task 1: Drop `obs_to_grid_indices` field + simplify bbox helpers

**Files:**
- Modify: `vs30/spatial.py`
- Modify: `dev/scripts/investigations/perf_features_investigation/bench_utils.py`

The cleanup steps below do not need a new failing test — they are dead-code removals verified by the existing test suite continuing to pass. The investigation findings already documented this field as never-read.

- [ ] **Step 1: Remove the field from the dataclass**

In `vs30/spatial.py`, replace the `BoundingBoxResult` dataclass (lines 237–254) with:

```python
@dataclass
class BoundingBoxResult:
    """
    Result of bounding box search for affected pixels.

    Attributes
    ----------
    mask : ndarray
        Boolean mask of pixels in any observation's bounding box.
    n_affected_pixels : int
        Total number of pixels affected by at least one observation.
    """

    mask: np.ndarray
    n_affected_pixels: int
```

- [ ] **Step 2: Simplify `grid_points_in_bbox` to return only the mask**

In `vs30/spatial.py`, replace the `grid_points_in_bbox` function (lines 463–529) with:

```python
def grid_points_in_bbox(
    grid_locs: np.ndarray,
    obs_eastings_min: np.ndarray,
    obs_eastings_max: np.ndarray,
    obs_northings_min: np.ndarray,
    obs_northings_max: np.ndarray,
) -> np.ndarray:
    """
    Find grid points within bounding boxes of observations using fully vectorized NumPy.

    Uses broadcasting to compute all observation-grid pairs simultaneously.
    Returns a collapsed boolean mask of which grid points fall inside any
    observation's bounding box.

    Parameters
    ----------
    grid_locs : array_like, shape (M, 2)
        Grid point coordinates as (easting, northing) in NZTM.
    obs_eastings_min : ndarray, shape (N, 1)
        Precomputed obs_eastings - max_dist.
    obs_eastings_max : ndarray, shape (N, 1)
        Precomputed obs_eastings + max_dist.
    obs_northings_min : ndarray, shape (N, 1)
        Precomputed obs_northings - max_dist.
    obs_northings_max : ndarray, shape (N, 1)
        Precomputed obs_northings + max_dist.

    Returns
    -------
    chunk_mask : ndarray, shape (M,), dtype=bool
        Boolean array indicating which grid points in this chunk are affected
        by any observation (collapsed with np.any(axis=0)).
    """
    grid_eastings = grid_locs[:, 0]
    grid_northings = grid_locs[:, 1]

    # Broadcasting (n_obs, 1) against (n_grid,) gives an (n_obs, n_grid) mask
    # of which grid points fall in each observation's bounding box.
    in_bbox = (
        (grid_eastings >= obs_eastings_min)
        & (grid_eastings <= obs_eastings_max)
        & (grid_northings >= obs_northings_min)
        & (grid_northings <= obs_northings_max)
    )

    return np.any(in_bbox, axis=0)
```

The `start_grid_idx` parameter and the `build_obs_indices` parameter are both removed — they only existed to support the per-observation index lists.

- [ ] **Step 3: Simplify `process_bbox_chunk`**

In `vs30/spatial.py`, replace the `process_bbox_chunk` function (lines 551–584) with:

```python
def process_bbox_chunk(args: tuple) -> tuple[int, np.ndarray]:
    """
    Worker function for parallel bounding box processing.

    Processes a single chunk of grid points to find which are affected by observations.

    Parameters
    ----------
    args : tuple
        ``(chunk_idx, grid_locs_chunk, obs_bounds)`` where ``obs_bounds`` is
        ``(obs_eastings_min, obs_eastings_max, obs_northings_min, obs_northings_max)``.

    Returns
    -------
    tuple
        ``(chunk_idx, chunk_mask)``.
    """
    chunk_idx, grid_locs_chunk, obs_bounds = args
    obs_eastings_min, obs_eastings_max, obs_northings_min, obs_northings_max = (
        obs_bounds
    )

    chunk_mask = grid_points_in_bbox(
        grid_locs=grid_locs_chunk,
        obs_eastings_min=obs_eastings_min,
        obs_eastings_max=obs_eastings_max,
        obs_northings_min=obs_northings_min,
        obs_northings_max=obs_northings_max,
    )

    return chunk_idx, chunk_mask
```

The `start_idx` and `build_obs_indices` arguments are dropped from the chunk-args tuple.

- [ ] **Step 4: Update `find_affected_pixels` to drop obs-indices accumulation**

In `vs30/spatial.py`, locate the `find_affected_pixels` function (around line 798). The Task 1 changes are localized to:

(a) Remove the `build_obs_indices` derivation and `obs_to_grid_indices` initialization (replace lines ~857–865 with):

```python
    logger.info(f"Processing {n_chunks} chunks of {chunk_size:,} pixels each")

    label = str(model_type).capitalize()
```

(b) Update the `chunk_args` construction (was around lines 871–876) to drop `start_idx` and `build_obs_indices`:

```python
    # Prepare chunk arguments
    chunk_args = []
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(grid_locs))
        grid_locs_chunk = grid_locs[start_idx:end_idx]
        chunk_args.append((chunk_idx, grid_locs_chunk, obs_bounds))
```

(c) Update the result-merging loop (was around lines 909–918) to drop the obs_to_grid_indices accumulation:

```python
    # Merge results from either parallel or sequential processing
    for chunk_idx, chunk_mask in results:
        start_idx = chunk_idx * chunk_size
        valid_points_in_bbox_mask[start_idx : start_idx + len(chunk_mask)] = chunk_mask
```

(d) Update the `BoundingBoxResult` construction at the bottom of the function to drop `obs_to_grid_indices`:

```python
    return BoundingBoxResult(
        mask=grid_points_in_bbox_mask,
        n_affected_pixels=n_affected,
    )
```

The function still has its `nproc` parameter and parallel branch at this point — those are removed in Task 2.

- [ ] **Step 5: Update the harness `make_full_bbox_result`**

In `dev/scripts/investigations/perf_features_investigation/bench_utils.py`, locate `make_full_bbox_result` (around line 120). Replace the construction of the BoundingBoxResult to drop the `obs_to_grid_indices=[]` keyword:

```python
def make_full_bbox_result(
    raster_data: spatial.RasterData, n_obs: int
) -> spatial.BoundingBoxResult:
    """Build a BoundingBoxResult that marks every valid pixel as affected.

    Used to disable the ``find_affected_pixels`` pre-filter for the OFF
    condition. The previous version of this helper allocated ``n_obs``
    copies of ``valid_flat_indices`` to populate the now-removed
    ``obs_to_grid_indices`` field; that wasted work caused an OOM in the
    high-N_obs / large-N_grid sweep cell. With the field removed the
    helper is now a thin wrapper.

    Parameters
    ----------
    raster_data
        Raster whose valid pixels become the affected set.
    n_obs
        Retained for signature stability with callers and tests; unused.

    Returns
    -------
    spatial.BoundingBoxResult
        Mask covers every valid pixel.
    """
    del n_obs  # see docstring; kept in signature for API stability
    mask = np.zeros(raster_data.vs30.size, dtype=bool)
    mask[raster_data.valid_flat_indices] = True
    return spatial.BoundingBoxResult(
        mask=mask,
        n_affected_pixels=int(mask.sum()),
    )
```

- [ ] **Step 6: Run ruff and tests**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
ruff check vs30/spatial.py dev/scripts/investigations/perf_features_investigation/bench_utils.py && \
ruff format --check vs30/spatial.py dev/scripts/investigations/perf_features_investigation/bench_utils.py && \
pytest tests/test_spatial.py dev/scripts/investigations/perf_features_investigation/test_bench_utils.py -v 2>&1 | tail -30
```

Expected: ruff exits 0; both test files pass.

If `tests/test_spatial.py` does not exist, run `pytest tests/ -v 2>&1 | tail -30` instead and confirm green. Any test that imports `obs_to_grid_indices` directly is the only realistic source of failure — verify with grep first if anything fails.

- [ ] **Step 7: Commit**

```bash
cd /home/arr65/src/Vs30 && \
git add vs30/spatial.py dev/scripts/investigations/perf_features_investigation/bench_utils.py && \
git commit -m "$(cat <<'EOF'
refactor(spatial): drop dead obs_to_grid_indices field from BoundingBoxResult

Per the perf-features investigation findings: this field was constructed
in find_affected_pixels (when nproc>1) but never read anywhere
downstream. compute_spatial_adjustments only reads bbox.mask;
run_parallel_spatial_fit takes affected_flat_indices (derived from the
mask), never the per-observation lists. Removing the field also removes
the build_obs_indices toggle on grid_points_in_bbox / process_bbox_chunk
and the merging logic in find_affected_pixels.

The harness make_full_bbox_result is updated in the same commit; the
n_obs argument is kept for API stability.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Drop `nproc` from `find_affected_pixels`; remove its parallel branch

**Files:**
- Modify: `vs30/spatial.py`
- Modify: `vs30/pipeline.py`
- Modify: `dev/scripts/investigations/perf_features_investigation/bench_utils.py`

- [ ] **Step 1: Simplify `find_affected_pixels`**

In `vs30/spatial.py`, replace the `find_affected_pixels` function with the version below. The signature drops `nproc`; the body collapses to the sequential path with chunked processing (the chunking remains for the memory budget).

```python
def find_affected_pixels(
    raster_data: RasterData,
    obs_data: ObservationData,
    max_spatial_boolean_array_memory_gb: float,
    model_type: constants.ModelType,
    max_dist_m: float = constants.MAX_DIST_M,
) -> BoundingBoxResult:
    """
    Find pixels affected by observations using bounding boxes.

    Parameters
    ----------
    raster_data : RasterData
        Raster data object.
    obs_data : ObservationData
        Observation data.
    max_spatial_boolean_array_memory_gb : float
        Memory limit (GB) for boolean arrays in spatial processing.
    model_type : constants.ModelType
        Model type (ModelType.GEOLOGY or ModelType.TERRAIN), used for
        progress bar labelling.
    max_dist_m : float, optional
        Maximum distance for considering observations.

    Returns
    -------
    BoundingBoxResult
        Result containing the affected-pixel mask and pixel count.
    """
    # Get coordinates for valid pixels
    grid_locs = raster_data.get_coordinates()

    n_obs = len(obs_data.locations)

    # Calculate chunk size based on observation count
    chunk_size = calculate_chunk_size(n_obs, max_spatial_boolean_array_memory_gb)
    n_chunks = int(np.ceil(len(grid_locs) / chunk_size))

    # Precompute observation bounds
    obs_eastings = obs_data.locations[:, 0:1]  # (n_obs, 1)
    obs_northings = obs_data.locations[:, 1:2]  # (n_obs, 1)
    obs_eastings_min = obs_eastings - max_dist_m
    obs_eastings_max = obs_eastings + max_dist_m
    obs_northings_min = obs_northings - max_dist_m
    obs_northings_max = obs_northings + max_dist_m

    # Bundle observation bounds for passing to workers
    obs_bounds = (
        obs_eastings_min,
        obs_eastings_max,
        obs_northings_min,
        obs_northings_max,
    )

    valid_points_in_bbox_mask = np.zeros(len(grid_locs), dtype=bool)

    logger.info(f"Processing {n_chunks} chunks of {chunk_size:,} pixels each")

    label = str(model_type).capitalize()

    # Prepare chunk arguments
    chunk_args = []
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(grid_locs))
        grid_locs_chunk = grid_locs[start_idx:end_idx]
        chunk_args.append((chunk_idx, grid_locs_chunk, obs_bounds))

    if n_chunks > 1:
        results = []
        for chunk_idx in tqdm(
            range(n_chunks),
            desc=f"{label}: checking pixels for nearby observations ({n_chunks} chunks)",
            unit="chunk",
        ):
            results.append(process_bbox_chunk(chunk_args[chunk_idx]))
    else:
        logger.info(
            f"{label}: checking {len(grid_locs):,} pixels for nearby observations"
        )
        results = [process_bbox_chunk(chunk_args[0])]

    # Merge results
    for chunk_idx, chunk_mask in results:
        start_idx = chunk_idx * chunk_size
        valid_points_in_bbox_mask[start_idx : start_idx + len(chunk_mask)] = chunk_mask

    # Create full-size mask
    grid_points_in_bbox_mask = np.zeros(raster_data.vs30.size, dtype=bool)
    grid_points_in_bbox_mask[raster_data.valid_flat_indices] = valid_points_in_bbox_mask

    n_affected = int(np.sum(valid_points_in_bbox_mask))
    logger.info(
        f"Bounding box search complete: {n_affected:,} pixels affected "
        f"({n_affected / len(grid_locs) * 100:.1f}% of valid pixels)"
    )

    return BoundingBoxResult(
        mask=grid_points_in_bbox_mask,
        n_affected_pixels=n_affected,
    )
```

The `multiprocess` import in `spatial.py` may now be unused — check at the end of this task with ruff.

- [ ] **Step 2: Remove `import multiprocess` from spatial.py if unused**

After Step 1, run:

```bash
grep -n "multiprocess" /home/arr65/src/Vs30/vs30/spatial.py
```

If the only remaining reference is the top-of-file import (`from vs30 import category, constants, multiprocess, raster, utils`), edit that import line to drop `multiprocess`:

```python
from vs30 import category, constants, raster, utils
```

If there are other references (there shouldn't be — the parallel branch was the only consumer), report the unexpected reference as a concern and stop.

- [ ] **Step 3: Update the call site in `vs30/pipeline.py`**

In `vs30/pipeline.py`, locate the call to `spatial.find_affected_pixels` inside `compute_spatial_adjustment_on_grid` (around lines 538–545). Replace with:

```python
    bbox_result = spatial.find_affected_pixels(
        raster_data,
        obs_data,
        max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
        model_type=model_type,
        max_dist_m=constants.MAX_DIST_M,
    )
```

(Drops the `nproc=nproc_resolved` keyword.)

- [ ] **Step 4: Update the harness call sites in `bench_utils.py`**

In `dev/scripts/investigations/perf_features_investigation/bench_utils.py`, locate the two call sites that pass `nproc` to `spatial.find_affected_pixels`:

**Inside `time_one_run`** (currently around the `if ffap:` block, ~lines 248–258 of the function):

```python
    # ---- Bounding-box phase -----------------------------------------------
    if ffap:
        t0 = time.perf_counter()
        bbox = spatial.find_affected_pixels(
            raster_data,
            obs_data,
            max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
            model_type=constants.ModelType.TERRAIN,
            max_dist_m=max_dist_m,
        )
        t_bbox = time.perf_counter() - t0
    else:
        bbox = make_full_bbox_result(raster_data, n_obs=len(obs_data.locations))
        t_bbox = 0.0
```

**Inside `_compute_one`** (the helper used by `run_numerical_equivalence_check`):

```python
    if ffap:
        bbox = spatial.find_affected_pixels(
            raster_data,
            obs_data,
            max_spatial_boolean_array_memory_gb=1.0,
            model_type=constants.ModelType.TERRAIN,
            max_dist_m=max_dist_m,
        )
    else:
        bbox = make_full_bbox_result(raster_data, n_obs=len(obs_data.locations))
```

(Both calls drop `nproc=nproc`.)

- [ ] **Step 5: Run ruff and tests**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
ruff check vs30/spatial.py vs30/pipeline.py dev/scripts/investigations/perf_features_investigation/bench_utils.py && \
ruff format --check vs30/spatial.py vs30/pipeline.py dev/scripts/investigations/perf_features_investigation/bench_utils.py && \
pytest tests/test_benchmarks.py::test_modified_foster_2019_single_process tests/test_benchmarks.py::test_jaehwi_v1p0_single_process -v 2>&1 | tail -10
```

Expected: ruff clean; the two named benchmark tests pass.

- [ ] **Step 6: Commit**

```bash
cd /home/arr65/src/Vs30 && \
git add vs30/spatial.py vs30/pipeline.py dev/scripts/investigations/perf_features_investigation/bench_utils.py && \
git commit -m "$(cat <<'EOF'
refactor(spatial): drop nproc from find_affected_pixels

Per the perf-features investigation findings: the bbox phase is fast
(sub-second to a few seconds even at large sizes) and the only caller
after the imminent multiproc-spatial-fit removal is
compute_spatial_adjustment_on_grid, which no longer has an nproc to
pass. The chunk-parallel branch is removed; the chunked iteration stays
(it bounds the memory budget for the broadcast array). Sequential
processing across chunks now uses tqdm directly.

Internal callers in vs30/pipeline.py and the perf-investigation harness
updated to drop the nproc keyword.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Delete the parallel spatial-fit path

**Files:**
- Modify: `vs30/parallel.py`
- Modify: `vs30/pipeline.py`
- Modify: `vs30/constants.py`
- Modify: `dev/scripts/investigations/perf_features_investigation/bench_utils.py`
- Modify: `dev/scripts/investigations/perf_features_investigation/test_bench_utils.py`

- [ ] **Step 1: Delete `run_parallel_spatial_fit` and `process_pixels_chunk` from `vs30/parallel.py`**

In `vs30/parallel.py`, delete the `process_pixels_chunk` function (around lines 371–422) and the `run_parallel_spatial_fit` function (around lines 491–605). Everything else in the file stays (it serves the points pipeline).

After deletion, the file should still import `multiprocess` and `parallel`-related primitives that `run_parallel_locations` needs. Verify with:

```bash
grep -n "from vs30 import\|^import" /home/arr65/src/Vs30/vs30/parallel.py
```

If the import line `from vs30 import category, constants, multiprocess, raster, spatial, utils` is now over-broad (e.g., `spatial` is unused after `process_pixels_chunk` is gone), trim it. After this task `parallel.py` should still import `multiprocess`, `constants`, and `utils` at minimum (used by `run_parallel_locations`); `spatial` may also still be needed for the points-flow `compute_spatial_adjustment_at_points` calls in `process_geology_at_points` / `process_terrain_at_points`. **Do not aggressively trim unrelated imports** — only drop ones that are demonstrably unused after the deletions.

- [ ] **Step 2: Simplify `compute_spatial_adjustment_on_grid` in `vs30/pipeline.py`**

In `vs30/pipeline.py`, replace `compute_spatial_adjustment_on_grid` (around lines 411–583). The new version drops `nproc`, the threshold guard, and the parallel branch:

```python
def compute_spatial_adjustment_on_grid(
    vs30_array: np.ndarray,
    stdv_array: np.ndarray,
    profile: dict,
    observations_df: pd.DataFrame,
    model_values_df: pd.DataFrame,
    model_type: constants.ModelType,
    corr_fn: Callable,
    apply_alluvium_slope_mod: bool,
    apply_coastal_distance_mod: bool,
    noisy: bool = True,
    max_spatial_boolean_array_memory_gb: float = 1.0,
    slope_array: np.ndarray | None = None,
    coast_dist_array: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute MVN spatial adjustment on a grid in memory.

    Performs a spatial adjustment of VS30 arrays by:

    1. Constructing RasterData from in-memory arrays.
    2. Loading measurements and mapping them to categories.
    3. Computing spatial fits to update pixels affected by measurements.
    4. Returning the updated arrays.

    Parameters
    ----------
    vs30_array : np.ndarray
        Input VS30 array (2D).
    stdv_array : np.ndarray
        Input standard deviation array (2D).
    profile : dict
        Rasterio profile with transform, crs, nodata.
    observations_df : pd.DataFrame
        DataFrame with measured VS30 values. Must contain columns:
        easting, northing, vs30, uncertainty.
    model_values_df : pd.DataFrame
        DataFrame with updated categorical Vs30 values.
    model_type : ModelType
        Model type: either GEOLOGY or TERRAIN.
    corr_fn : Callable
        Correlation function for spatial adjustment.
    apply_alluvium_slope_mod : bool
        Whether to apply slope-based interpolation for GID 4 (alluvium).
    apply_coastal_distance_mod : bool
        Whether to apply coastal distance modification for GID 4 and GID 10.
    noisy : bool, optional
        Whether to apply noise weighting in spatial adjustment.
    max_spatial_boolean_array_memory_gb : float, optional
        Maximum memory for spatial boolean arrays.
    slope_array : np.ndarray, optional
        Pre-computed slope array (for geology observation data preparation).
    coast_dist_array : np.ndarray, optional
        Pre-computed coast distance array.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (adjusted_vs30, adjusted_stdv) arrays.
    """
    logger.info(f"Starting spatial adjustment for {model_type} model")

    raster_data = spatial.RasterData.from_arrays(
        vs30=vs30_array,
        stdv=stdv_array,
        transform=profile["transform"],
        crs=profile.get("crs", constants.NZTM_CRS),
        nodata=constants.NODATA_VALUE,
    )
    spatial.validate_raster_data(raster_data)
    spatial.validate_observations(observations_df)

    # Model IDs are 1-indexed; convert to 0-indexed array indices.
    mean_col, std_col = raster.select_vs30_columns_by_priority(
        list(model_values_df.columns)
    )
    max_id = model_values_df[constants.STANDARD_ID_COLUMN].max()
    updated_model_table = np.full((max_id, 2), np.nan)
    ids = model_values_df[constants.STANDARD_ID_COLUMN].values.astype(int) - 1
    valid = (ids >= 0) & (ids < max_id)
    updated_model_table[ids[valid], 0] = model_values_df[mean_col].values[valid]
    updated_model_table[ids[valid], 1] = model_values_df[std_col].values[valid]

    logger.info("Preparing observation data for spatial adjustment...")
    obs_data = spatial.prepare_observation_data(
        observations_df,
        raster_data,
        updated_model_table,
        model_type,
        apply_alluvium_slope_mod=apply_alluvium_slope_mod,
        apply_coastal_distance_mod=apply_coastal_distance_mod,
        noisy=noisy,
        slope_array=slope_array,
        coast_dist_array=coast_dist_array,
    )
    n_obs = len(obs_data.locations)
    logger.info(f"Prepared {n_obs} valid observations")

    if n_obs == 0:
        logger.warning(
            "No valid observations found within model bounds. "
            "Returning input arrays unchanged."
        )
        return vs30_array.copy(), stdv_array.copy()

    logger.info("Finding pixels affected by observations...")
    t_bbox_start = time.perf_counter()
    bbox_result = spatial.find_affected_pixels(
        raster_data,
        obs_data,
        max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
        model_type=model_type,
        max_dist_m=constants.MAX_DIST_M,
    )
    t_bbox_elapsed = time.perf_counter() - t_bbox_start
    logger.info(
        f"Found {bbox_result.n_affected_pixels:,} affected pixels "
        f"in {t_bbox_elapsed:.1f}s"
    )

    logger.info("Computing spatial updates...")
    t_spatial_start = time.perf_counter()
    adjusted_vs30, adjusted_stdv = spatial.compute_spatial_adjustments(
        raster_data,
        obs_data,
        bbox_result,
        corr_fn,
        max_dist_m=constants.MAX_DIST_M,
        max_points=constants.MAX_POINTS,
        noisy=noisy,
        cov_reduc=constants.COV_REDUC,
    )
    t_spatial_elapsed = time.perf_counter() - t_spatial_start
    logger.info(f"Spatial adjustments completed in {t_spatial_elapsed:.1f}s")

    return adjusted_vs30, adjusted_stdv
```

- [ ] **Step 3: Update internal call sites**

In `vs30/pipeline.py`, locate the call to `compute_spatial_adjustment_on_grid` inside `compute_model_grid` (around lines 901–916) and remove the `nproc=nproc` keyword:

```python
        current_vs30, current_stdv = compute_spatial_adjustment_on_grid(
            vs30_array=current_vs30,
            stdv_array=current_stdv,
            profile=profile,
            observations_df=observations_df,
            model_values_df=posterior_df,
            model_type=model_type,
            corr_fn=corr_fn,
            apply_alluvium_slope_mod=apply_alluvium_slope_mod,
            apply_coastal_distance_mod=apply_coastal_distance_mod,
            noisy=noisy,
            max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
            slope_array=slope_array,
            coast_dist_array=coast_dist_array,
        )
```

- [ ] **Step 4: Delete `MULTIPROCESS_OBSERVATION_THRESHOLD` from `vs30/constants.py`**

In `vs30/constants.py`, delete the constant + its docstring (around lines 71–76). The block to remove:

```python
# When the number of valid observations exceeds this threshold, the MVN spatial
# adjustment falls back to single-process mode (nproc=1) to allow BLAS to
# parallelise large matrix inversions across all cores. With many observations,
# pixels frequently hit the MAX_POINTS cap, producing large covariance matrices
# where BLAS-level parallelism is far more efficient than Python-level
# multiprocessing with single-threaded BLAS.
MULTIPROCESS_OBSERVATION_THRESHOLD: int = 1000
```

- [ ] **Step 5: Update the harness `bench_utils.py`**

In `dev/scripts/investigations/perf_features_investigation/bench_utils.py`:

(a) Delete the `bypass_observation_threshold` context manager and its `import contextlib` if no other consumer remains. Search:

```bash
grep -n "contextlib" /home/arr65/src/Vs30/dev/scripts/investigations/perf_features_investigation/bench_utils.py
```

If `contextlib` is unused after deleting `bypass_observation_threshold`, drop the `import contextlib` line at the top of the file.

(b) Delete `run_numerical_equivalence_check` and the private `_compute_one` helper (the entire function bodies — both are no longer meaningful with one strategy left).

(c) Replace `time_one_run` with the simplified version below. The `nproc` parameter is removed; the function is always sequential. The comment block about multi-threaded BLAS / single-threaded BLAS is dropped.

```python
def time_one_run(
    raster_data: spatial.RasterData,
    obs_data: spatial.ObservationData,
    ffap: bool,
    rep: int,
    corr_fn=DEFAULT_CORR_FN,
    max_dist_m: int = constants.MAX_DIST_M,
    max_points: int = constants.MAX_POINTS,
    cov_reduc: float = constants.COV_REDUC,
    noisy: bool = True,
    max_spatial_boolean_array_memory_gb: float = 1.0,
) -> dict:
    """Measure one (N_obs, N_grid, ffap, rep) cell.

    Returns a dict suitable for a CSV row.
    """
    # ---- Bounding-box phase -----------------------------------------------
    if ffap:
        t0 = time.perf_counter()
        bbox = spatial.find_affected_pixels(
            raster_data,
            obs_data,
            max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
            model_type=constants.ModelType.TERRAIN,
            max_dist_m=max_dist_m,
        )
        t_bbox = time.perf_counter() - t0
    else:
        bbox = make_full_bbox_result(raster_data, n_obs=len(obs_data.locations))
        t_bbox = 0.0

    # ---- Spatial-adjustment phase -----------------------------------------
    t0 = time.perf_counter()
    spatial.compute_spatial_adjustments(
        raster_data,
        obs_data,
        bbox,
        corr_fn,
        max_dist_m=max_dist_m,
        max_points=max_points,
        noisy=noisy,
        cov_reduc=cov_reduc,
    )
    t_spatial = time.perf_counter() - t0

    return {
        "N_obs": len(obs_data.locations),
        "N_grid_actual": int(raster_data.valid_flat_indices.size),
        "N_affected": int(bbox.n_affected_pixels),
        "ffap": ffap,
        "rep": rep,
        "t_bbox_s": t_bbox,
        "t_spatial_s": t_spatial,
        "t_total_s": t_bbox + t_spatial,
        "peak_rss_mb": _peak_rss_mb(),
        "timestamp_iso": _dt.datetime.now().isoformat(timespec="seconds"),
    }
```

(d) The harness's `parallel` import is no longer needed. Remove `parallel` from the line `from vs30 import config, constants, parallel, pipeline, raster, spatial, utils` if it is unused elsewhere in `bench_utils.py`.

- [ ] **Step 6: Update `test_bench_utils.py`**

In `dev/scripts/investigations/perf_features_investigation/test_bench_utils.py`, delete the following tests (entire function bodies):

- `test_bypass_observation_threshold_restores_original`
- `test_bypass_observation_threshold_restores_on_exception`
- `test_numerical_equivalence_check_passes_on_small_case`

Update `test_time_one_run_returns_expected_keys` to drop the `nproc=1` keyword from the `bench_utils.time_one_run` call and remove `"nproc"` from the `expected_keys` set:

```python
def test_time_one_run_returns_expected_keys() -> None:
    raster_data, _ = bench_utils.make_raster_data(n_target=1000)
    obs_df = bench_utils.subsample_observations(50, seed=42)
    obs_data = bench_utils.prepare_terrain_obs_data(obs_df, raster_data)
    row = bench_utils.time_one_run(
        raster_data=raster_data,
        obs_data=obs_data,
        ffap=True,
        rep=0,
    )
    expected_keys = {
        "N_obs", "N_grid_actual", "N_affected", "ffap", "rep",
        "t_bbox_s", "t_spatial_s", "t_total_s", "peak_rss_mb",
        "timestamp_iso",
    }
    assert expected_keys.issubset(row.keys())
    assert row["t_total_s"] >= row["t_bbox_s"]
    assert row["t_spatial_s"] > 0
    assert row["t_bbox_s"] >= 0
```

The `from vs30 import constants` import in this test file may now be unused — check with grep and remove if so:

```bash
grep -n "constants" /home/arr65/src/Vs30/dev/scripts/investigations/perf_features_investigation/test_bench_utils.py
```

- [ ] **Step 7: Run ruff and tests**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
ruff check vs30/parallel.py vs30/pipeline.py vs30/constants.py \
  dev/scripts/investigations/perf_features_investigation/bench_utils.py \
  dev/scripts/investigations/perf_features_investigation/test_bench_utils.py && \
ruff format --check vs30/parallel.py vs30/pipeline.py vs30/constants.py \
  dev/scripts/investigations/perf_features_investigation/bench_utils.py \
  dev/scripts/investigations/perf_features_investigation/test_bench_utils.py && \
pytest dev/scripts/investigations/perf_features_investigation/test_bench_utils.py -v 2>&1 | tail -15 && \
pytest tests/test_benchmarks.py::test_modified_foster_2019_single_process -v 2>&1 | tail -10
```

Expected: ruff clean; harness tests pass (the deleted tests are gone, the remaining ones pass); benchmark test passes.

- [ ] **Step 8: Commit**

```bash
cd /home/arr65/src/Vs30 && \
git add vs30/parallel.py vs30/pipeline.py vs30/constants.py \
  dev/scripts/investigations/perf_features_investigation/bench_utils.py \
  dev/scripts/investigations/perf_features_investigation/test_bench_utils.py && \
git commit -m "$(cat <<'EOF'
refactor: remove per-pixel MVN multiproc path from grid pipeline

Per the perf-features investigation findings: nproc=8 was 2-110x slower
than nproc=1 in every cell tested (median 61x slower). The path is
removed and the MULTIPROCESS_OBSERVATION_THRESHOLD guard along with it.

- vs30/parallel.py: delete run_parallel_spatial_fit and the private
  process_pixels_chunk worker. run_parallel_locations (points mode)
  is unaffected.
- vs30/pipeline.py: simplify compute_spatial_adjustment_on_grid to a
  single sequential branch; drop nproc parameter; remove the
  MULTIPROCESS_OBSERVATION_THRESHOLD fallback. Internal call sites in
  compute_model_grid updated.
- vs30/constants.py: delete MULTIPROCESS_OBSERVATION_THRESHOLD.
- perf-investigation harness: delete bypass_observation_threshold,
  _compute_one, and run_numerical_equivalence_check (their reason for
  being is gone). Simplify time_one_run; update tests accordingly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Rename `nproc` → `dbscan_nproc` and consolidate grid benchmark tests

**Files:**
- Modify: `vs30/pipeline.py`
- Modify: `vs30/cli.py`
- Modify: `tests/test_benchmarks.py`
- Modify: `tests/test_grid_points_consistency.py`
- Modify: `dev/scripts/investigations/diagnose_jaehwi_mask.py`
- Modify: `dev/scripts/investigations/perf_features_investigation/run_full_pipeline_confirmation.py`

This is one logical change (rename + test consolidation) split across many files. To keep the intermediate state coherent, do all sub-steps in a **single commit**.

- [ ] **Step 1: Rename `nproc` → `dbscan_nproc` in `vs30/pipeline.py`**

Three functions need the rename: `compute_categorical_vs30_updates`, `compute_model_grid`, `grid_pipeline`. For each, rename the parameter, update the docstring, and update internal call sites.

In `compute_categorical_vs30_updates` (around line 168):
- Function signature: `nproc: int = 1` → `dbscan_nproc: int = 1`
- Docstring "nproc : int, optional" → "dbscan_nproc : int, optional", and update the description to "Number of processes for DBSCAN clustering of clustered observations".
- Body: `category.perform_clustering(clustered_observations_df, nproc)` → `category.perform_clustering(clustered_observations_df, dbscan_nproc)`.

In `compute_model_grid` (around line 706):
- Function signature: `nproc: int = 1` → `dbscan_nproc: int = 1`
- Docstring update.
- Body call to `compute_categorical_vs30_updates(..., nproc=nproc)` → `compute_categorical_vs30_updates(..., dbscan_nproc=dbscan_nproc)`.
- The Task 3 step already removed the call to `compute_spatial_adjustment_on_grid(..., nproc=nproc)` so no further update is needed there.

In `grid_pipeline` (around line 939):
- Function signature: `nproc: int = 1` → `dbscan_nproc: int = 1`
- Docstring update.
- Body: two calls to `compute_model_grid(..., nproc=nproc)` (one for geology, one for terrain) → `compute_model_grid(..., dbscan_nproc=dbscan_nproc)`.

**Note:** `points_pipeline` keeps its `nproc` parameter; it is a separate function and out of scope.

- [ ] **Step 2: Rename in `vs30/cli.py`**

Two CLI commands need the rename: `grid` (around line 422) and `grid_custom` (around line 532). For each:

(a) Rename the parameter declaration:

```python
    nproc: typing.Annotated[int, typer.Option()] = -1,
```

becomes

```python
    dbscan_nproc: typing.Annotated[int, typer.Option()] = -1,
```

(b) Update the docstring `nproc : int, optional` → `dbscan_nproc : int, optional`, with a description like "Number of processes for DBSCAN clustering. Use -1 for all cores."

(c) Update the call to `pipeline.grid_pipeline(..., nproc=nproc)` → `pipeline.grid_pipeline(..., dbscan_nproc=dbscan_nproc)`.

**Do not touch** the `points` and `points_custom` commands (around lines 244 and 304); they keep `nproc`.

- [ ] **Step 3: Update `tests/test_benchmarks.py`**

(a) Update `run_benchmark` to take no `nproc` parameter and always use `dbscan_nproc=1`:

```python
def run_benchmark(version: constants.FixedModelVersion, grid: config.GridConfig) -> None:
    """Run the grid pipeline for a fixed model version and compare against the benchmark raster."""
    cfg = load_fixed_model_config(version)

    result = pipeline.grid_pipeline(
        grid_config=grid,
        output_dir=None,
        geology_categorical_csv=cfg["geology_categorical_csv"],
        terrain_categorical_csv=cfg["terrain_categorical_csv"],
        clustered_observations_csv=cfg["clustered_observations_csv"],
        independent_observations_csv=cfg["independent_observations_csv"],
        combination_method=constants.CombinationMethod(cfg["combination_method"]),
        combine_ratio=cfg["combine_ratio"],
        noisy=cfg["noisy"],
        do_bayesian_update=cfg["do_bayesian_update"],
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        apply_coastal_distance_mod=cfg["apply_coastal_distance_mod"],
        fill_gaps=cfg["fill_gaps"],
        geology_corr_fn=cfg["geology_corr_fn"],
        terrain_corr_fn=cfg["terrain_corr_fn"],
        dbscan_nproc=1,
    )

    benchmark = BENCHMARKS_DIR / f"{version}.tif"
    assert_arrays_match_raster_benchmark(
        result["combined_vs30"], result["combined_stdv"], benchmark
    )
```

(b) Delete the three multiprocess test functions:

```python
def test_modified_foster_2019_multiprocess():
    ...

def test_jaehwi_v1p0_multiprocess():
    ...

def test_viktor_cpt_clustering_multiprocess():
    ...
```

(c) Rename the three single-process test functions to drop the `_single_process` suffix and update their bodies:

```python
def test_modified_foster_2019():
    """modified_foster_2019 full-domain pipeline matches benchmark."""
    run_benchmark(constants.FixedModelVersion.MODIFIED_FOSTER_2019, BENCHMARK_NZ_GRID)


def test_jaehwi_v1p0():
    """jaehwi_v1p0 full-domain pipeline matches benchmark."""
    run_benchmark(constants.FixedModelVersion.JAEHWI_V1P0, BENCHMARK_NZ_GRID)


def test_viktor_cpt_clustering():
    """viktor_cpt_clustering full-domain pipeline matches benchmark."""
    run_benchmark(constants.FixedModelVersion.VIKTOR_CPT_CLUSTERING, BENCHMARK_NZ_GRID)
```

(d) The `import os` at the top is no longer used — remove it. Verify with `grep -n "^import os\|os\\." /home/arr65/src/Vs30/tests/test_benchmarks.py`.

(e) **Do not touch** `test_foster_2019_approx_points_benchmark` — it uses `points_pipeline` and its `nproc=[1, -1]` parametrize stays.

- [ ] **Step 4: Update `tests/test_grid_points_consistency.py`**

In `run_grid_pipeline_at_point` (around line 96), update the `pipeline.grid_pipeline` call to use `dbscan_nproc=1`:

```python
    result = pipeline.grid_pipeline(
        grid_config=local_config,
        output_dir=None,
        ...
        dbscan_nproc=1,
        ...
    )
```

(Replace just the `nproc=1` keyword with `dbscan_nproc=1`. All other arguments stay.)

The points pipeline call in the same file (around line 87) keeps `nproc=-1`.

- [ ] **Step 5: Update `dev/scripts/investigations/diagnose_jaehwi_mask.py`**

Update the `pipeline.grid_pipeline` call (around line 32) to use `dbscan_nproc=1`:

```python
    result = pipeline.grid_pipeline(
        grid_config=STANDARD_NZ_GRID,
        output_dir=None,
        ...
        dbscan_nproc=1,
    )
```

- [ ] **Step 6: Update `dev/scripts/investigations/perf_features_investigation/run_full_pipeline_confirmation.py`**

Update the `pipeline.grid_pipeline` call inside `run_cohort` (around line 90) to use `dbscan_nproc=nproc`:

```python
    pipeline.grid_pipeline(
        grid_config=grid,
        output_dir=None,
        ...
        dbscan_nproc=nproc,
    )
```

(Replace just the `nproc=nproc` keyword with `dbscan_nproc=nproc`. The local `nproc` variable in `run_cohort` stays — it's the function parameter that is then passed to grid_pipeline as the renamed kwarg.)

- [ ] **Step 7: Run ruff and tests**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
ruff check vs30/pipeline.py vs30/cli.py tests/test_benchmarks.py tests/test_grid_points_consistency.py \
  dev/scripts/investigations/diagnose_jaehwi_mask.py \
  dev/scripts/investigations/perf_features_investigation/run_full_pipeline_confirmation.py && \
ruff format --check vs30/pipeline.py vs30/cli.py tests/test_benchmarks.py tests/test_grid_points_consistency.py \
  dev/scripts/investigations/diagnose_jaehwi_mask.py \
  dev/scripts/investigations/perf_features_investigation/run_full_pipeline_confirmation.py && \
pytest tests/test_benchmarks.py tests/test_grid_points_consistency.py -v 2>&1 | tail -20
```

Expected: ruff clean; default-tier benchmarks and consistency tests pass.

- [ ] **Step 8: CLI smoke test**

Verify the renamed CLI flag works:

```bash
<activate> && cd /home/arr65/src/Vs30 && \
vs30 grid --help 2>&1 | grep -i "dbscan-nproc"
```

Expected: a line like `--dbscan-nproc INTEGER` appears in the help output.

```bash
<activate> && cd /home/arr65/src/Vs30 && \
mkdir -p /tmp/vs30_smoke && \
vs30 grid --version foster_2019_approx \
  --grid-xmin 1500000 --grid-xmax 1700000 \
  --grid-ymin 5100000 --grid-ymax 5300000 \
  --grid-dx 5000 --grid-dy 5000 \
  --output-dir /tmp/vs30_smoke \
  --dbscan-nproc 1 2>&1 | tail -10
```

Expected: completes without error; `/tmp/vs30_smoke/` contains output rasters. (This run takes 30–60 seconds.)

```bash
<activate> && cd /home/arr65/src/Vs30 && \
! vs30 grid --version foster_2019_approx \
  --grid-xmin 1500000 --grid-xmax 1700000 \
  --grid-ymin 5100000 --grid-ymax 5300000 \
  --grid-dx 5000 --grid-dy 5000 \
  --output-dir /tmp/vs30_smoke \
  --nproc 1 2>&1 | tail -3
```

Expected: error like "No such option: --nproc" — verifies the old flag is gone.

- [ ] **Step 9: Commit**

```bash
cd /home/arr65/src/Vs30 && \
git add vs30/pipeline.py vs30/cli.py tests/test_benchmarks.py tests/test_grid_points_consistency.py \
  dev/scripts/investigations/diagnose_jaehwi_mask.py \
  dev/scripts/investigations/perf_features_investigation/run_full_pipeline_confirmation.py && \
git commit -m "$(cat <<'EOF'
refactor: rename nproc -> dbscan_nproc on grid pipeline; consolidate tests

After the multiproc spatial-fit removal, the only thing the grid
pipeline's nproc parameter controls is DBSCAN clustering inside the
categorical-update step. Rename to make the intent explicit:

- vs30/pipeline.py: rename on compute_categorical_vs30_updates,
  compute_model_grid, grid_pipeline.
- vs30/cli.py: --nproc -> --dbscan-nproc on `vs30 grid` and
  `vs30 grid-custom`. `vs30 points` / `points-custom` keep --nproc
  (their nproc still does double duty pending the points-mode
  investigation).

Tests:
- tests/test_benchmarks.py: drop the three _multiprocess grid tests
  (the multiproc path that justified them no longer exists, and
  sklearn's DBSCAN multiproc invariance is sklearn's job to test).
  Consolidate the three _single_process tests by dropping the suffix.
  run_benchmark no longer takes nproc.
- tests/test_grid_points_consistency.py: rename the kwarg in the grid
  call. Points call unchanged.
- dev scripts (diagnose_jaehwi_mask, run_full_pipeline_confirmation):
  rename the kwarg.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Final validation

**No code changes** — this task is the end-to-end verification gate.

- [ ] **Step 1: Default-tier test suite**

```bash
<activate> && cd /home/arr65/src/Vs30 && pytest tests/ -v 2>&1 | tail -30
```

Expected: all tests pass. (Default tier — unit + benchmark + 3-city consistency, ~3 minutes.)

- [ ] **Step 2: Slow-tier test suite**

```bash
<activate> && cd /home/arr65/src/Vs30 && pytest tests/ --runslow -v 2>&1 | tail -30
```

Expected: all tests pass. (~43 minutes — full 38-point grid/points consistency across all 4 model versions. Run once.)

If any tests fail in either tier, **stop and investigate** — the cleanup is supposed to be functionally inert on the recommended path. Failures are real bugs, not "tolerance to widen" situations.

- [ ] **Step 3: Repo-wide ruff sweep**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
ruff check . && ruff format --check .
```

Expected: clean across the repo.

- [ ] **Step 4: Sanity-check the harness still imports**

The investigation harness has had several files modified. Confirm it still imports cleanly:

```bash
<activate> && cd /home/arr65/src/Vs30 && \
pytest dev/scripts/investigations/perf_features_investigation/test_bench_utils.py -v 2>&1 | tail -15
```

Expected: 8 tests pass (down from 11 — three were removed in Task 3).

- [ ] **Step 5: No additional commit**

This task is validation only. If everything passes, the cleanup is done. If anything fails, fix it in a follow-up commit before declaring success.

---

## Self-review

**Spec coverage** (from `dev/docs/grid_cleanup_design.md`):

- §2 (in-scope `vs30/spatial.py` changes) → Task 1 + Task 2 ✓
- §2 (in-scope `vs30/parallel.py` deletions) → Task 3 ✓
- §2 (`vs30/constants.py` deletion) → Task 3 ✓
- §2 (`vs30/pipeline.py` simplification + rename) → Task 3 + Task 4 ✓
- §2 (`vs30/cli.py` rename) → Task 4 ✓
- §2 (test updates) → Task 4 ✓
- §2 (dev script updates) → Task 4 ✓
- §5.3 (drop the parametrize, run only `dbscan_nproc=1`) → Task 4 ✓
- §6 (8 atomic commits) → 4 implementation tasks (1, 2, 3, 4) + 1 validation task (5). The original §6 had 8 commits split finer; this plan groups commits 4–7 into Task 4 because they form a single logical rename that breaks tests if split. Internal-state coherence prioritised over fine bisection. ✓
- §7 (risks) → addressed in task wording (no parametrize-related risk left after Task 4 Step 3; CLI rename risk addressed in Task 4 Step 8 smoke). ✓

**Placeholder scan:** every code block contains real, runnable code. Every command has expected output. No "TBD" / "TODO" / "fill in".

**Type consistency:** the post-rename signatures (`grid_pipeline(dbscan_nproc=...)`, etc.) used in tests and dev scripts in Task 4 match the production renames in the same task. `BoundingBoxResult(mask=..., n_affected_pixels=...)` (no `obs_to_grid_indices`) is consistently used across Task 1's spatial.py and bench_utils.py changes. The simplified `time_one_run` signature (no `nproc`) in Task 3 matches the test update in the same task.

# vs30 Package Code Review

Review of the `vs30` package (excluding `VsViewer`) against:

- Dead code remaining after recent removals (multiproc/parallel branches, etc.).
- Loops that could hoist work out to outer scope.
- Redundant `x = obj.x` / `x = obj["x"]` extractions outside the affine-transform exemption.
- Imports that should be at the top of the module and reference the parent module (`from vs30 import constants`, then `constants.X`).
- Comments that just restate Python.
- Overly complex code that could be simpler.

Findings are grouped by severity and ordered roughly from highest to lowest impact within each group. Each item gives the location, what is wrong, why it matters, and how to fix it. The intent is that another contributor (human or AI) can take this document and implement each item without re-deriving the analysis.

The codebase has clearly been through several review/cleanup passes already (recent commits: `8cf3617 refactor: round-2 CODE_REVIEW.md cleanup pass`, `0e4402e refactor: apply CODE_REVIEW.md cleanup and performance pass`). Most of the obvious issues have been fixed. What remains is subtler.

---

## Decision matrix (agreed plan)

| #  | Disposition | Notes |
|----|-------------|-------|
| 1  | **Do**      | Option A — extend `grid_pipeline`/`compute_model_grid` to accept pre-loaded DataFrames + `posterior_df`. Folds in Finding 27. |
| 2  | **Do**      | Full refactor; vectorise `np.log(obs_vs30)` once; inline formulas. Combine with Finding 7. |
| 3  | **Do**      | Add `@functools.lru_cache(maxsize=1)` `load_coast_boundary_union`. |
| 4  | **Do**      | Gate `compute_coastal_distance_at_points` behind `apply_coastal_distance_mod`. |
| 5  | **Do**      | Replace `0.5` with `constants.PIXEL_CENTER_OFFSET`. |
| 6  | **Do**      | Option B — strip column names at the CSV-loading boundary. Subsumes Findings 17 and 18. |
| 7  | **Do**      | Iterate only categories with observations; combines with Finding 2. |
| 8  | **Do**      | Retire `BoundingBoxResult`; return `(mask, grid_locs)` directly. |
| 9  | **Do**      | Extract help-text constants; move validation into pipeline functions. |
| 10 | **Do**      | Approach B — `@functools.lru_cache` array+transform; use `data.shape[0]`/`[1]` inline. |
| 11 | Skip        | Observation only — no fix. |
| 12 | **Do**      | Delete "Default from constants" / "from constants.py" docstring noise. |
| 13 | **Do**      | Replace `.values` with `.to_numpy()` throughout. |
| 14 | **Do**      | Delete six numbered inline comments. |
| 15 | **Do**      | Delete `# Snap to nearest pixel center`. |
| 16 | **Do**      | Delete `# Select observations using distance-based filtering`. |
| 17 | **Do**      | Folds into Finding 6. |
| 18 | **Do**      | Folds into Finding 6. |
| 19 | Leave       | Comment is genuinely informative. |
| 20 | Leave       | Comment is genuinely informative. |
| 21 | **Do**      | Add one-line comment on `max_id` table sizing. |
| 22 | Skip        | Leave LUT-building duplicated. |
| 23 | Leave       | Keep `prepare_*_obs_data` as separate functions. |
| 24 | Leave       | Keep `assert ... is not None` as-is. |
| 25 | **Do**      | Extend `grid_pipeline` return-dict docstring. |
| 26 | **Do**      | Defer `safe_log_slope` inside loop. |
| 27 | **Do**      | Folds into Finding 1. |
| 28 | Skip        | Leave `_nan_to_nodata` in `pipeline.py`. |
| 29 | Skip        | Per-test explicitness. |
| 30 | **Do**      | Add `update_with_clustered_data` unit tests. |
| 31 | **Do**      | Add `find_affected_pixels` unit test. |
| 32 | Leave       | Observation only — `from conftest import …` style. |

## Implementation order

Each chunk should be reviewed against the test suite before moving on:

1. Findings **1 + 27** — gap-fill DataFrame plumbing.
2. Findings **2 + 7** — Bayesian update refactor.
3. Finding **3** — coast-boundary union memoisation.
4. Finding **4** — gate outside-grid coast call.
5. Findings **6 + 17 + 18** — strip column names at I/O boundary.
6. Findings **8 + 25** — retire `BoundingBoxResult`; clarify `geology_ids` docstring.
7. Findings **5, 9, 12, 13, 14, 15, 16, 21** — small style/consistency cleanups in one pass.
8. Finding **10** — raster-array memoisation.
9. Finding **26** — defer `safe_log_slope` inside hybrid-mods loop.
10. Findings **30 + 31** — additive unit tests.

---

## High-impact findings

### 1. Points-pipeline gap-fill reloads CSVs and re-runs DBSCAN per fillable point

**Disposition:** Do — Option A (extend `grid_pipeline`/`compute_model_grid` to accept pre-loaded DataFrames + `posterior_df`). Folds in Finding 27.

**Location:** `vs30/pipeline.py:1094-1144` (the `if fill_gaps and model_type == COMBINED:` block in `points_pipeline`), plus `vs30/pipeline.py:735-800` (`fill_one_point_via_local_grid`).

**Problem:** `grid_pipeline_kwargs` is built (lines 1112-1134) to forward **CSV paths** for observations and categorical models, plus the user's `do_bayesian_update` flag. Then for each fillable point, `fill_one_point_via_local_grid` calls `pipeline.grid_pipeline(...)` once per local-grid attempt (up to 10 attempts per point per `GAPFILL_MAX_LOCAL_GRID_HALF_WIDTH_M / GAPFILL_LOCAL_GRID_EXPANSION_M`).

Each call to `grid_pipeline`:

1. Re-reads `clustered_observations_csv` and `independent_observations_csv` (`pipeline.py:610-620`).
2. Re-reads `geology_categorical_csv` and `terrain_categorical_csv` inside `compute_categorical_vs30_updates` (`pipeline.py:206-208`) when `do_bayesian_update=True`.
3. Re-runs DBSCAN clustering of clustered observations (`category.perform_clustering`).
4. Re-runs the sequential Bayesian update loop in `category.update_with_independent_data` (`category.py:261-281`).

For a run with N fillable points and an average of M attempts per point, DBSCAN + Bayesian update happen `O(N*M)` times even though they would yield identical posterior tables every time.

**Why it matters:** The main `points_pipeline` already loaded the CSVs and computed posteriors once at the top of the function (lines 909-983). Re-doing that work for every gap-fill point can dominate the total runtime when there are many on-land NaN gaps to fill.

**Fix:**

1. Extend `pipeline.grid_pipeline` (and `pipeline.compute_model_grid`) to accept already-loaded `clustered_observations_df` / `independent_observations_df` (DataFrames) **and** an optional already-computed `posterior_df` per model type, in addition to the existing CSV-path arguments. When the DataFrames are supplied, skip the CSV reads and the Bayesian update.
2. In `points_pipeline`, when building `grid_pipeline_kwargs`, pass the DataFrames it already has in scope (`clustered_observations_df`, `independent_observations_df`, `geol_model_df`, `terr_model_df`). Remove the CSV paths for those (or keep them as fallbacks).
3. With pre-computed posteriors threaded through, override `do_bayesian_update=False` in the gap-fill kwargs, since the posterior is already computed and reusing it is the whole point of the optimisation.

A self-contained alternative: keep `grid_pipeline` API unchanged, but introduce a tighter helper used only by gap-fill — e.g. `pipeline._gapfill_grid_at_point(local_config, posterior_geol_df, posterior_terr_df, observations_df, ...)` — that runs steps 2-4 of `compute_model_grid` directly with the pre-loaded data.

Note: line 1124 already hard-codes `"dbscan_nproc": 1` inside the kwargs. This is a bandaid that helps a little (single-proc DBSCAN startup is faster than spawning workers for a tiny dataset) but it does **not** address the redundant work above.

---

### 2. Bayesian update inner loop computes `np.log` of the same values twice per observation

**Disposition:** Do — full refactor (vectorise `np.log(obs_vs30)` once; inline formulas). Combine with Finding 7.

**Location:** `vs30/category.py:261-281` (the inner `for vs30_value, uncertainty in zip(...)` loop).

**Problem:** Inside the loop:

```python
new_variance = compute_bayesian_posterior_variance(
    current_std, current_n, uncertainty, current_mean, vs30_value
)
current_mean = compute_bayesian_posterior_mean(
    current_mean, current_n, vs30_value
)
```

`compute_bayesian_posterior_variance` (`category.py:142-177`) computes `np.log(observation_value) - np.log(prior_mean)`. Then `compute_bayesian_posterior_mean` (`category.py:113-139`) computes `np.log(prior_mean)` and `np.log(observation_value)` again, with the **same** values for `prior_mean` and `observation_value` (the variance call uses the not-yet-updated `current_mean`).

So per observation we do four `np.log` calls instead of two.

**Why it matters:** For categories with thousands of observations the redundant log calls add up. Each `np.log` of a Python float is comparable in cost to a few additions; not catastrophic, but it's pure waste.

**Fix:** Pre-compute logs of all observations vectorised, then inline the formulas in the inner loop using the cached logs. Sketch:

```python
log_obs_vs30 = np.log(obs_vs30)  # vectorised, once before the outer loop

for i, row in enumerate(updated_categorical_model_df.itertuples(index=False)):
    category_id = getattr(row, constants.STANDARD_ID_COLUMN)
    cat_mask = obs_ids == category_id
    if not np.any(cat_mask):
        new_means[i] = getattr(row, constants.COL_POSTERIOR_MEAN_INDEPENDENT)
        new_stds[i] = getattr(row, constants.COL_POSTERIOR_STDV_INDEPENDENT)
        new_ns[i] = getattr(row, constants.COL_POSTERIOR_NOBS_INDEPENDENT)
        continue
    cat_log_obs = log_obs_vs30[cat_mask]
    cat_unc = obs_unc[cat_mask]
    current_mean = getattr(row, constants.COL_POSTERIOR_MEAN_INDEPENDENT)
    current_std = getattr(row, constants.COL_POSTERIOR_STDV_INDEPENDENT)
    current_n = getattr(row, constants.COL_POSTERIOR_NOBS_INDEPENDENT)
    log_current_mean = np.log(current_mean)
    for log_obs, unc in zip(cat_log_obs, cat_unc):
        log_residual = log_obs - log_current_mean
        mean_shift = (current_n / (current_n + 1)) * log_residual ** 2
        new_variance = (
            current_n * current_std ** 2 + unc ** 2 + mean_shift
        ) / (current_n + 1)
        log_current_mean = (current_n * log_current_mean + log_obs) / (current_n + 1)
        current_std = np.sqrt(new_variance)
        current_n += 1
    new_means[i] = float(np.exp(log_current_mean))
    new_stds[i] = current_std
    new_ns[i] = current_n
```

Decide whether to keep `compute_bayesian_posterior_mean`/`compute_bayesian_posterior_variance` as testable units. If you keep them, the test suite already covers them (`tests/test_category.py:17-134`); the inner loop should call its own inlined version (or the helpers should grow `_log_space` siblings that take pre-computed logs). Either way, drop the duplicated work.

A lighter-touch alternative if you want to preserve the helpers' API: compute `log_obs_vs30 = np.log(obs_vs30)` once before the outer loop and use it only for the variance call (which currently does its own `np.log(observation_value)`); the mean call still does its own logs. That eliminates one of the four `np.log` calls per observation.

---

### 3. `compute_coastal_distance_at_points` recomputes `coast.boundary.union_all()` every call

**Disposition:** Do — add `@functools.lru_cache(maxsize=1)` `load_coast_boundary_union`.

**Location:** `vs30/raster.py:539-564`, specifically line 560:

```python
coast_boundary = coast_gdf.geometry.boundary.union_all()
```

**Problem:** `union_all()` over the NZ coastline boundary is an expensive shapely operation (the docstring of `load_coast_union` at `raster.py:62-72` already says so for the polygon version). The function is called from `vs30/spatial.py:391`, `vs30/points.py:123`, and `vs30/points.py:247` — i.e. once per pipeline invocation in each of: prepare_observation_data (geology), prepare_geology_obs_data, process_geology_at_points.

When points-mode gap-fill triggers `fill_one_point_via_local_grid` per fillable point, the local `grid_pipeline` runs may also call `compute_coastal_distance_at_points` for any observations that fall outside the local grid (via `spatial.prepare_observation_data`). With small (5–50 km half-width) local grids and country-wide observations, **most** observations are outside the grid, so this fires every time.

**Why it matters:** Every call rebuilds the same boundary geometry from the same cached coast shapefile. The raw points-pipeline already calls it 2× per geology run. Combined with finding 1, gap-fill amplifies it further.

**Fix:** Add a memoised loader that returns the unioned boundary, mirroring the existing `load_coast_union`:

```python
@functools.lru_cache(maxsize=1)
def load_coast_boundary_union():
    """Unioned NZ coastline boundary, cached across calls (used for distance-to-coast)."""
    return load_coast_shapefile().geometry.boundary.union_all()
```

Then in `compute_coastal_distance_at_points`:

```python
def compute_coastal_distance_at_points(points: np.ndarray) -> np.ndarray:
    coast_boundary = load_coast_boundary_union()
    point_geoms = shapely.points(points)
    return np.asarray(shapely.distance(point_geoms, coast_boundary), dtype=np.float64)
```

Drop the now-unused `coast_gdf = load_coast_shapefile()` local in that function.

---

### 4. `apply_coastal_distance_mod=False` still pays for `compute_coastal_distance_at_points` on outside-grid observations

**Disposition:** Do — gate `compute_coastal_distance_at_points` behind `apply_coastal_distance_mod`.

**Location:** `vs30/spatial.py:388-393` inside `prepare_observation_data`.

**Problem:** When `model_type == GEOLOGY` and any observation falls outside the grid, this branch runs unconditionally:

```python
if not np.all(within_grid):
    outside_grid_points = obs_locs[~within_grid]
    slope_obs[~within_grid] = raster.sample_slope_at_points(outside_grid_points)
    coast_obs[~within_grid] = raster.compute_coastal_distance_at_points(
        outside_grid_points
    )
```

`compute_coastal_distance_at_points` is called regardless of `apply_coastal_distance_mod`. If the mod is off, `apply_hybrid_geology_modifications` skips the coastal step (`raster.py:682`) and the `coast_obs` value is never read.

Inside-grid observations get `coast_obs[within_grid] = coast_dist_array[rows, cols]`, where `coast_dist_array` was set by `grid.compute_hybrid_geology_arrays` (line 117-120) to either the real proximity raster or `np.zeros_like(vs30_array)` when the mod is off — i.e. the inside-grid path is already short-circuited correctly.

**Why it matters:** Wasted shapely distance computation when the mod is off. Combined with finding 3, it can be sizeable.

**Fix:** Gate the call:

```python
slope_obs[~within_grid] = raster.sample_slope_at_points(outside_grid_points)
if apply_coastal_distance_mod:
    coast_obs[~within_grid] = raster.compute_coastal_distance_at_points(
        outside_grid_points
    )
# else: coast_obs[~within_grid] stays at the np.empty initial value, which is
# fine because apply_hybrid_geology_modifications won't read it when the
# coastal mod is off.
```

If you'd rather keep `coast_obs` always defined, set the outside-grid slice to `0.0` in the else branch.

The same gating already happens upstream in `vs30/points.py:122-126` and `vs30/points.py:246-250`, so this brings the grid-mode prep in line with the points-mode prep.

---

## Medium-impact findings

### 5. `gapfill.pixel_coords_float32` uses literal `0.5` instead of `constants.PIXEL_CENTER_OFFSET`

**Disposition:** Do — replace `0.5` with `constants.PIXEL_CENTER_OFFSET`.

**Location:** `vs30/gapfill.py:38-39`.

```python
eastings = transform.c + transform.a * (cols + 0.5)
northings = transform.f + transform.e * (rows + 0.5)
```

`constants.PIXEL_CENTER_OFFSET` (`constants.py:393`) exists and is used by `spatial.RasterData.get_coordinates` at `spatial.py:203-204`. The two pieces of code do exactly the same conversion. Inconsistent.

**Fix:** Replace both `0.5` literals with `constants.PIXEL_CENTER_OFFSET`. No semantic change, just consistency with the existing constant.

---

### 6. Column-name whitespace stripping is inconsistent across consumers

**Disposition:** Do — Option B (strip column names at the CSV-loading boundary). Subsumes Findings 17 and 18.

**Location:**

- `vs30/raster.py:329-343` (`create_vs30_arrays_from_ids`): builds `stripped_columns = {c.strip(): c for c in df.columns}`, then resolves stripped names back to originals before indexing the DataFrame.
- `vs30/raster.py:241-293` (`select_vs30_columns_by_priority`): expects already-stripped names; doesn't strip itself.
- `vs30/grid.py:207-208` (`compute_spatial_adjustment_on_grid`): calls `select_vs30_columns_by_priority(list(model_values_df.columns))` — **without** stripping. Then indexes `model_values_df[mean_col]` directly.
- `vs30/category.py:507-509` (`get_vs30_for_ids`): same — calls priority selector on raw columns.

**Problem:** If a user has whitespace in a CSV header (e.g. `" id "` or `"posterior_mean_vs30_km_per_s "`), `create_vs30_arrays_from_ids` (used by the grid pipeline's initial array creation) handles it, but `compute_spatial_adjustment_on_grid` and `get_vs30_for_ids` do not. The pipeline therefore behaves differently depending on which stage it gets to before failing.

**Fix:** Pick one policy and apply it everywhere. Two options:

**Option A (preferred):** Move the stripping into `select_vs30_columns_by_priority`. Make it return both the canonical (stripped) name and the original name needed to index the DataFrame, e.g. return `(mean_col_orig, std_col_orig)`. All three callers stop caring about whitespace.

**Option B:** Strip column names at the I/O boundary instead — i.e. once when the categorical CSV is loaded:

```python
df = pd.read_csv(path, comment="#", skipinitialspace=True)
df.columns = [c.strip() for c in df.columns]
```

Apply this to `pipeline.compute_categorical_vs30_updates` (`pipeline.py:206-208`), the `do_bayesian_update=False` branch (`pipeline.py:362-364`), and the points-pipeline `else` branches (`pipeline.py:960-962`, `981-983`). Then drop the stripping logic from `create_vs30_arrays_from_ids` and assume clean column names everywhere downstream. This is the simpler refactor; pick A only if you want to be defensive against direct in-Python users who construct DataFrames without whitespace cleanup.

---

### 7. `update_with_independent_data` iterates every category even when it has no observations

**Disposition:** Do — iterate only over categories with observations. Combine with Finding 2.

**Location:** `vs30/category.py:261-281`.

**Problem:** The outer loop walks every row of `updated_categorical_model_df` and computes `mask = obs_ids == category_id`. If no observations match the category, `obs_vs30[mask]` is empty and the inner loop is a no-op. The current category's `current_mean`/`current_std`/`current_n` are then written back to `new_means`/`new_stds`/`new_ns`, leaving them equal to the initial posterior values that were already set in lines 242-250 — i.e. the entire iteration was redundant.

`update_with_clustered_data` (`category.py:450-472`) already does this right: it iterates over `unique_ids = valid_sites[STANDARD_ID_COLUMN].unique()` so it only touches categories that actually have data.

**Why it matters:** Mostly a code-clarity issue; for typical geology/terrain category counts (~15 each) the wasted iterations are negligible. But it lets us simplify and align with `update_with_clustered_data`'s style.

**Fix:** Iterate over categories that actually have observations:

```python
n_categories = len(updated_categorical_model_df)
new_means = updated_categorical_model_df[
    constants.COL_POSTERIOR_MEAN_INDEPENDENT
].to_numpy(copy=True)
new_stds = updated_categorical_model_df[
    constants.COL_POSTERIOR_STDV_INDEPENDENT
].to_numpy(copy=True)
new_ns = updated_categorical_model_df[
    constants.COL_POSTERIOR_NOBS_INDEPENDENT
].to_numpy(copy=True)

cat_id_to_row = {
    int(cat_id): i
    for i, cat_id in enumerate(updated_categorical_model_df[constants.STANDARD_ID_COLUMN])
}

for cat_id in np.unique(obs_ids):
    row_idx = cat_id_to_row.get(int(cat_id))
    if row_idx is None:
        continue
    cat_mask = obs_ids == cat_id
    # ...sequential Bayesian update for this category, writing into
    # new_means[row_idx], new_stds[row_idx], new_ns[row_idx]
```

Combine with finding 2 (cache `np.log(obs_vs30)`).

---

### 8. `BoundingBoxResult` dataclass exists for one logging-only field

**Disposition:** Do — retire `BoundingBoxResult`; have `find_affected_pixels` return `(mask, grid_locs)` directly.

**Location:** `vs30/spatial.py:212-227` and `vs30/spatial.py:815`, plus the consumer at `vs30/grid.py:250`.

**Problem:** The dataclass holds two fields:

- `mask: np.ndarray` — the only one used substantively (in `compute_spatial_adjustments`, `spatial.py:863-864`).
- `n_affected_pixels: int` — used only for the log line at `grid.py:250`. It's just `mask.sum()`, computed once and stashed.

The wrapper buys nothing structural; both consumers index a single attribute.

**Fix (low priority, code-cleanliness):** Have `find_affected_pixels` return `mask, grid_locs` directly (drop the wrapper). Move the count computation to the call site that logs it:

```python
bbox_mask, grid_locs = spatial.find_affected_pixels(...)
n_affected = int(bbox_mask.sum())
logger.info(f"Found {n_affected:,} affected pixels in {t_bbox_elapsed:.1f}s")
adjusted_vs30, adjusted_stdv = spatial.compute_spatial_adjustments(
    raster_data, obs_data, bbox_mask, grid_locs, corr_fn, ...
)
```

Update `spatial.compute_spatial_adjustments` to take the mask directly (`bbox_mask`) instead of `bbox_result: BoundingBoxResult`.

Skip if you're keeping the dataclass for future extensibility — but right now there's nothing to extend.

---

### 9. CLI duplication: `model_type != COMBINED and not include_intermediate` validation, plus `Suggested for all of NZ:` help strings

**Disposition:** Do — extract help-text constants; move validation into `pipeline.grid_pipeline`/`points_pipeline`.

**Location:** `vs30/cli.py:168-173` (in `run_points_pipeline`) and `vs30/cli.py:641-646` (in `grid_custom`); separately `vs30/cli.py:391-428` and `vs30/cli.py:530-565` (the duplicated grid-bound help strings).

**Problem 1 — validation duplication:** The check

```python
if model_type != constants.ModelType.COMBINED and not include_intermediate:
    raise typer.BadParameter(...)
```

is implemented twice with identical text. The shared spot is `pipeline.grid_pipeline` and `pipeline.points_pipeline`, which both already accept `model_type` and `include_intermediate`. Move the check into the pipeline functions and delete both CLI copies.

**Problem 2 — duplicated help strings:** Both `grid` and `grid_custom` declare `grid_xmin/xmax/ymin/ymax/dx/dy` with the same `f"Grid {…} coordinate (NZTM, meters). Suggested for all of NZ: {constants.FULL_NZ_GRID_CONFIG.{…}}.”` help text. The two functions otherwise differ only in which scientific parameters they accept. The 6 help strings × 2 functions = 12 duplicated lines.

**Fix:**

1. Define module-level help-text constants in `vs30/cli.py` (or `vs30/constants.py`):
   ```python
   _GRID_XMIN_HELP = (
       f"Grid minimum X coordinate (NZTM, meters). Suggested for all of NZ: "
       f"{constants.FULL_NZ_GRID_CONFIG.grid_xmin}."
   )
   # ... similarly for xmax/ymin/ymax/dx/dy
   ```
   Use them in both `grid` and `grid_custom`.
2. Move the model-type/include-intermediate validation into `pipeline.grid_pipeline` and `pipeline.points_pipeline`, raising `ValueError` (since pipelines shouldn't depend on Typer). The CLI commands can then drop their copies; a `ValueError` from the pipeline will surface to the user with a meaningful message.

---

### 10. `assign_to_category_terrain` re-opens the terrain raster on every call

**Disposition:** Do — Approach B (cache array+transform via `@functools.lru_cache(maxsize=1)`). Use `data.shape[0]`/`[1]` inline rather than introducing local `h, w` aliases.

**Location:** `vs30/category.py:82-110`.

```python
def assign_to_category_terrain(points: np.ndarray) -> np.ndarray:
    with rasterio.open(
        constants.GEOSPATIAL_DIR / constants.TERRAIN_RASTER_FILENAME
    ) as src:
        terrain_ids = np.array(
            [s[0] for s in src.sample(points, indexes=1)], dtype=src.dtypes[0]
        )
        if src.nodata is not None:
            terrain_ids[terrain_ids == src.nodata] = constants.RASTER_ID_NODATA_VALUE
    return terrain_ids
```

Same for `raster.sample_slope_at_points` (`raster.py:515-536`) which opens the slope raster on each call.

**Problem:** During `points_pipeline`, both terrain and slope are sampled at observation locations and at query locations — at least 2 file-open round-trips per file. With gap-fill triggering local grid pipelines per fillable point, the terrain raster (used by `prepare_terrain_obs_data` indirectly via `category.assign_to_category_terrain`) and slope raster get reopened many times.

**Why it matters:** A `rasterio.open` is fast on its own (~ms), but for many fillable points it adds up. Hard to estimate without profiling; flagged because it was easy to spot, not because it's a known hotspot.

**Fix:** Either:

1. Add `@functools.lru_cache(maxsize=1)` wrappers analogous to `load_qmap_shapefile` that return an opened `rasterio.DatasetReader` (note: rasterio readers are not safe to leave open across processes; the cache is for in-process reuse only).
2. **Or** pass the points to be sampled in batches so the file is only opened once per pipeline run. The simpler refactor: have `category.assign_to_category_terrain` and `raster.sample_slope_at_points` accept an optional pre-opened dataset, and have the points pipeline open each file once near the top of the run.

If neither is convenient, this is fine to leave — the cost is small compared to MVN per-pixel work.

---

### 11. `np.log(model_vs30)` happens at the end of `prepare_observation_data` but feeds a `select_observations_for_pixel`/`build_covariance_matrix` chain that only reads it for the *selected* obs

**Disposition:** Skip — observation only; no fix needed.

**Location:** `vs30/spatial.py:419-425` and `vs30/spatial.py:587-591`.

**Problem:** `prepare_observation_data` precomputes `log_model_vs30 = np.log(model_vs30)` for **all** observations (good, this is loop-invariant). `build_covariance_matrix` then uses `obs_data.log_model_vs30[obs_indices]` — only a subset. Nothing wrong here functionally; just noting that the precomputation already saves work and there's no further hoist available.

The same applies to `model_stdv`, `residuals`, `omega`. They're all precomputed and indexed by `obs_indices`. ✓

No fix needed; documenting it so a future contributor doesn't try to "optimize" by lazy-computing logs inside the per-pixel loop.

---

## Low-impact findings (mostly style)

### 12. Docstring lines that say "from constants.py" or "Default from constants" add noise without info

**Disposition:** Do — delete the noted docstring lines.

**Location:**

- `vs30/utils.py:175`: `"Uses K_VALUE and WEIGHT_EPSILON_DIV_BY_ZERO constants from constants.py for standard deviation weighting calculations."` — the `Notes` section of `combine_vs30_models`. The reader can already see `constants.K_VALUE` in the body.
- `vs30/category.py:318`: `"Uses MIN_GROUP and EPS constants from constants.py for DBSCAN parameters."` — same pattern.
- `vs30/grid.py:303-304`: `nodata: float, optional / "No-data value. Default from constants."` — the default value is already visible in the signature: `nodata: float = constants.NODATA_VALUE`.
- `vs30/spatial.py:939, 941, 946`: `"Maximum distance (meters) to consider observations. Default from constants."` etc., for `compute_spatial_adjustment_at_points`. Defaults are already in the signature.

**Fix:** Delete those sentences. Defaults already show in the signature; mentioning where they live in source restates obvious things and goes stale when constants are renamed.

---

### 13. Inconsistent `.values` vs `.to_numpy()` for pandas → numpy conversions

**Disposition:** Do — replace `.values` with `.to_numpy()` throughout.

**Location:** Mixed throughout `vs30/`. Examples:

- `.values`: `vs30/grid.py:212-215`, `vs30/category.py:325-329`, `vs30/spatial.py:336, 356, 359-361`, `vs30/points.py:139-140, 181-182`, `vs30/cli.py:182-183`, `vs30/pipeline.py:104, 1095-1096`.
- `.to_numpy()`: `vs30/raster.py:351-353`, `vs30/category.py:252-254`.

**Fix:** Pick one. Modern pandas (≥ 1.0) prefers `.to_numpy()`; `.values` still works but its return type is less predictable across dtypes (e.g. on extension dtypes `.values` may return an `ExtensionArray` rather than `ndarray`). Doing a one-shot replacement of all `.values` → `.to_numpy()` is safe given the current dtypes (plain numeric float/int columns) and improves consistency.

---

### 14. `# 1. Independent observations posterior` etc. inline comments restate the docstring

**Disposition:** Do — delete the six numbered inline comments.

**Location:** `vs30/raster.py:267-283` inside `select_vs30_columns_by_priority`.

The `priorities` list has inline comments numbering each tuple, mirroring the docstring's enumerated priority list (lines 245-250). The comments add nothing new — the constant names already tell the reader what each tuple represents (`COL_POSTERIOR_MEAN_INDEPENDENT`, etc.), and the docstring above already enumerates the priority order.

**Fix:** Delete the six numbered inline comments. The docstring is canonical and the names are self-documenting.

---

### 15. `# Snap to nearest pixel center` and similar narrative-style comments

**Disposition:** Do — delete the inline comment.

**Location:** `vs30/gapfill.py:234`. The comment "Snap to nearest pixel center" sits above an arithmetic block that uses `gapfill_grid_config.grid_xmin`, `grid_dx`, etc. The block is clear enough on its own once you know what `create_local_grid_config` is doing — and the docstring just above it (`create_local_grid_config` docstring at lines 209-231) already says "Snaps the point to the nearest pixel center in the reference grid config".

**Fix:** Drop the inline comment. (The arithmetic itself uses `gapfill_grid_config.grid_xmin` four times in a single expression; that's awkward but it's a grid-alignment calculation kin to the affine-transform exemption — it's clearer left as is than refactored into a helper.)

---

### 16. `# Select observations using distance-based filtering` restates the function's name

**Disposition:** Do — delete the inline comment.

**Location:** `vs30/spatial.py:632`. The function is `select_observations_for_pixel` and it does distance-based filtering. The comment is one step away from `# select observations`.

**Fix:** Delete it.

---

### 17. `# Strip whitespace from column names without copying the DataFrame`

**Disposition:** Do — folds into Finding 6 (the surrounding code is removed when stripping moves to load time).

**Location:** `vs30/raster.py:329`. The dict-comprehension `{c.strip(): c for c in model_values_df.columns}` is self-explanatory once the reader knows `model_values_df.columns` returns column labels. The "without copying the DataFrame" part adds value (it explains *why* this approach versus `df.rename(columns=...)`) — keep that part. The "Strip whitespace from column names" part doesn't.

**Fix:** Tighten to: `# Build a stripped→original column-name map without copying the DataFrame.`

(Alternatively, finding 6's Option B obviates this comment entirely by doing the strip at load time.)

---

### 18. `# Map stripped column names back to the originals so we can index the DataFrame.`

**Disposition:** Do — folds into Finding 6 (the surrounding code is removed when stripping moves to load time).

**Location:** `vs30/raster.py:340`. Pairs with finding 17. Fine to keep; it explains the back-and-forth that's not obvious from the next three lines.

---

### 19. `# Default to ID_NODATA where the spatial join returned no match.`

**Disposition:** Leave — comment is genuinely informative; keep.

**Location:** `vs30/category.py:40`. Genuinely informative — it tells the reader why we initialise to `RASTER_ID_NODATA_VALUE` instead of zero or `np.nan`. Keep.

---

### 20. `# Cast to float so .at[] assignments below don't downcast.`

**Disposition:** Leave — comment is genuinely informative; keep.

**Location:** `vs30/category.py:429`. Genuinely informative — it explains a subtle pandas behaviour. Keep.

---

### 21. `if max_id` test in `compute_spatial_adjustment_on_grid` has a subtle edge case

**Disposition:** Do — add the one-line clarifying comment.

**Location:** `vs30/grid.py:210-215`.

```python
max_id = model_values_df[constants.STANDARD_ID_COLUMN].max()
updated_model_table = np.full((max_id, 2), np.nan)
ids = model_values_df[constants.STANDARD_ID_COLUMN].values.astype(int) - 1
valid = (ids >= 0) & (ids < max_id)
```

`max_id` is the largest id; the table is sized `max_id` rows; `ids - 1` gives 0-based indices in `[0, max_id - 1]`; `valid` is `[0, max_id)` which is `[0, max_id - 1]` inclusive — so the last valid id (== max_id, after subtracting 1 → max_id - 1) is correctly admitted.

If a row has `id == 0`, `ids == -1`, `valid` is False — the row is skipped, leaving a NaN in the LUT. That's intentional (id=0 is reserved for "water" and shouldn't appear in the categorical model anyway, since `compute_categorical_vs30_updates` drops `COL_MEAN == NODATA_VALUE` rows).

If a row has `id > max_id`, it can't (it's the max).

So the math is sound. The two-line rationale is non-obvious for a future reader, though.

**Fix (optional):** Add a one-line comment explaining the size choice:

```python
# Indices are 1-based ids minus 1, so we need max_id rows (covering 0..max_id-1).
updated_model_table = np.full((max_id, 2), np.nan)
```

Alternatively, this whole block could be replaced with a call into a shared helper that mirrors `raster.create_vs30_arrays_from_ids`'s LUT-building logic (256-row LUT keyed by raw id), at the cost of a slightly larger LUT — see finding 22.

---

### 22. LUT-building logic is duplicated between `raster.create_vs30_arrays_from_ids` and `grid.compute_spatial_adjustment_on_grid`

**Disposition:** Skip — leave LUT-building duplicated; revisit if a third consumer appears.

**Location:** `vs30/raster.py:329-369` and `vs30/grid.py:206-215`.

Both build a numpy table mapping category id → (mean, stdv) using `select_vs30_columns_by_priority`, but with different conventions:

- `create_vs30_arrays_from_ids` builds a 256-row LUT keyed by raw category id (so `lut[5]` is the value for id 5; `lut[255]` is NODATA).
- `compute_spatial_adjustment_on_grid` builds a `max_id`-row table keyed by `id - 1` (so `table[4]` is the value for id 5).

**Why the asymmetry exists:** The first is indexed by `id_array` (a uint8 raster of raw ids); the second is indexed by `model_ids - 1` (after explicit `-1` adjustment in `spatial.prepare_observation_data` at lines 350-352).

**Fix (optional):** Have both consumers share `raster.create_vs30_arrays_from_ids`'s shape (256-row LUT keyed by raw id). Then `prepare_observation_data` can drop the `-1` adjustment at line 350 (`valid_model_ids = model_ids[valid_mask] - 1` becomes `valid_model_ids = model_ids[valid_mask]`), and `compute_spatial_adjustment_on_grid` can either use the existing helper or a small new one in `raster.py` that returns just the (mean_lut, stdv_lut) pair without rasterizing.

This is a style/DRY improvement, not a correctness fix. Skip if you prefer the current local construction.

---

### 23. Two essentially-identical `prepare_*_obs_data` shapes in `points.py`

**Disposition:** Leave — keep `prepare_geology_obs_data` and `prepare_terrain_obs_data` as separate functions.

**Location:** `vs30/points.py:74-144` (`prepare_geology_obs_data`) and `vs30/points.py:147-186` (`prepare_terrain_obs_data`).

The functions differ only in:

- which `category.assign_to_category_*` they call;
- whether they apply hybrid geology mods (geology-only).

There's enough difference (the geology branch has slope/coast sampling and the legacy NODATA sentinel patch) that combining them with a `model_type` argument would introduce a not-very-clean conditional. Not worth merging. Document choice — leave as is.

The same applies to `process_geology_at_points` vs `process_terrain_at_points` (`points.py:189` vs `:287`); their tuple return shapes differ, so unification would be awkward. Leave.

---

### 24. `assert geol_model_df is not None` / `assert profile is not None`

**Disposition:** Leave — keep `assert ... is not None` as-is (load-bearing for `ty`).

**Location:** `vs30/pipeline.py:684, 1008, 1042`.

These are type-narrowing asserts comments-as-code (the inline comment says "invariant: required when run_geology"). They're load-bearing for `ty` (the type checker) but not for runtime correctness — they trip on a programmer error, not a user-facing bad input.

**Fix (low priority):** They're fine as-is. If you want fewer asserts in production code, you can swap to `cast(pd.DataFrame, geol_model_df)` from `typing` at the use sites instead — keeps `ty` happy without the runtime check. Either form is acceptable.

---

### 25. `result["geology_ids"]` in `grid_pipeline` is always populated when geology runs, regardless of `include_intermediate`

**Disposition:** Do — extend the `grid_pipeline` return-dict docstring.

**Location:** `vs30/pipeline.py:646`.

```python
if run_geology:
    ...
    result["geology_ids"] = geol_ids
```

Then at line 778, `fill_one_point_via_local_grid` reads `local_result["geology_ids"]` for the gap-fill classifier. So `geology_ids` must always be present in the result — it's not really intermediate-only. The current code is correct.

**Optional cleanup:** Document this contract explicitly. The function's return-dict docstring (`pipeline.py:582-589`) does mention the keys but it's currently:

```
- "geology_vs30", "geology_stdv" : 2D arrays (when geology is computed)
```

Consider extending the docstring to say: `"geology_ids" : 2D uint8 array (always populated when geology is computed; required by gap-fill)`. Otherwise, no change.

---

### 26. `safe_log_slope` is computed across the whole grid even when not all GIDs are present

**Disposition:** Do — defer `safe_log_slope` inside the loop using the sketched rewrite.

**Location:** `vs30/raster.py:656-662` in `apply_hybrid_geology_modifications`.

```python
safe_log_slope = np.log10(
    np.where(
        (slope_array <= 0) | (slope_array == constants.NODATA_VALUE),
        constants.MIN_SLOPE_FOR_LOG,
        slope_array,
    )
)
```

Computed once before the loop over `HYBRID_GEOLOGY_PARAMS`. If `apply_alluvium_slope_mod=False` and the array contains only GID 4 (no other hybrid GIDs), the array is computed but `safe_log_slope[mask]` is read only for the GID-4 mask, which is then `continue`-skipped — wasted work.

**Why it matters:** Marginal. `np.log10` and `np.where` are vectorised over the whole array. For a national-grid 100m raster (~10^7 pixels) it's still fast. Defer if you want, but the win is small.

**Fix (optional):** Move the `safe_log_slope` computation inside the loop and apply it only to `slope_array[mask]`:

```python
for spec in constants.HYBRID_GEOLOGY_PARAMS:
    mask = id_array == spec.gid
    if not np.any(mask):
        continue
    stdv_array[mask] *= spec.sigma_reduction
    if spec.gid == 4 and not apply_alluvium_slope_mod:
        continue
    spec_slope = slope_array[mask]
    safe_slope = np.where(
        (spec_slope <= 0) | (spec_slope == constants.NODATA_VALUE),
        constants.MIN_SLOPE_FOR_LOG,
        spec_slope,
    )
    interpolated_val = np.interp(
        np.log10(safe_slope), spec.slope_limits, spec.vs30_values_log10
    )
    vs30_array[mask] = 10**interpolated_val
```

Note this also moves the `if not np.any(mask): continue` guard above the `stdv_array[mask] *= spec.sigma_reduction` line. That changes semantics only if the user expects that line to no-op for empty masks — which it already does (in-place broadcast over zero-length view), so this is safe.

This is a refactor of style/efficiency and not strictly necessary.

---

### 27. `compute_categorical_vs30_updates` accepts paths but also an already-loaded categorical CSV would be cheaper for some callers

**Disposition:** Do — folds into Finding 1.

**Location:** `vs30/pipeline.py:137-247`.

The function accepts `categorical_model_csv: Path` and reads the CSV inside (lines 206-208). Both `compute_model_grid` (paths) and `points_pipeline` (paths) pass paths. The CSV is read each time the function runs.

**Why it matters:** Combined with finding 1, every gap-fill iteration re-reads the categorical CSV. The OS file cache makes the byte-level read cheap, but pandas still reparses the CSV.

**Fix:** Accept `categorical_model_df: pd.DataFrame | None = None` as an alternative to `categorical_model_csv: Path | None = None` and document that exactly one must be provided. Or add a thin wrapper `compute_categorical_vs30_updates_from_df` and let the existing function delegate.

---

### 28. `_nan_to_nodata` is used three times but is private and small

**Disposition:** Skip — leave `_nan_to_nodata` in `pipeline.py`.

**Location:** `vs30/pipeline.py:30-32`.

```python
def _nan_to_nodata(arr: np.ndarray) -> np.ndarray:
    """Replace NaNs with ``constants.NODATA_VALUE`` for raster output."""
    return np.where(np.isnan(arr), constants.NODATA_VALUE, arr)
```

Used at `pipeline.py:696, 713`, twice — and only inside `grid_pipeline` for the combined output. It's already small and clearly named; no need to inline. But: it's a candidate to live in `vs30/grid.py` (alongside `write_raster`) so it's reusable from the grid-writing side rather than from the orchestration side. Low priority.

---

## Test-suite findings

### 29. `test_benchmarks.py` and `test_grid_points_consistency.py` duplicate kwarg lists for pipeline calls

**Disposition:** Skip — keep per-test explicitness.

**Location:** `tests/test_benchmarks.py:60-77, 107-123` and `tests/test_grid_points_consistency.py:74-92, 95-123`.

Each of these test helpers builds a long list of kwargs to pass to `pipeline.grid_pipeline` or `pipeline.points_pipeline`, all sourced from a config dict. The same `cfg["geology_categorical_csv"]`, `cfg.get("clustered_observations_csv")`, etc. patterns appear in three places.

**Fix (optional):** A `tests/_pipeline_helpers.py` module (or add to `conftest.py`) with helpers like `kwargs_from_cfg_for_grid(cfg)` and `kwargs_from_cfg_for_points(cfg)` that return the kwarg dict. Each test then does:

```python
result = pipeline.grid_pipeline(**kwargs_from_cfg_for_grid(cfg), grid_config=grid)
```

Skip if you prefer per-test explicitness for documentation reasons.

---

### 30. `tests/test_category.py` does not exercise `update_with_clustered_data`

**Disposition:** Do — add unit tests for `update_with_clustered_data` and `compute_cluster_weighted_mean_and_stddev`.

**Location:** `tests/test_category.py`.

The file tests `compute_bayesian_posterior_*` and `update_with_independent_data`. There's no direct unit test for `category.update_with_clustered_data` or `category.compute_cluster_weighted_mean_and_stddev`. They are exercised indirectly through `tests/test_benchmarks.py::test_viktor_cpt_clustering` and `test_jaehwi_v1p0`, but a small synthetic-cluster test that verifies the geometric-mean-per-cluster formula and the `effective_n` bookkeeping (`category.py:461-463`) would catch refactor regressions much faster than running the slow benchmarks.

**Fix:** Add tests:

1. A 2-cluster + 1-unclustered case with hand-computed expected `effective_n`, mean, stddev.
2. The "all observations belong to one big cluster" edge case (`effective_n == 1`).
3. The `effective_n == 0` early-continue path.

This is additive test coverage; no change to source code.

---

### 31. `tests/test_spatial.py` doesn't exercise `find_affected_pixels` / chunking

**Disposition:** Do — add a unit test for `find_affected_pixels` and `calculate_chunk_size`.

**Location:** `tests/test_spatial.py`.

`spatial.find_affected_pixels` and `spatial.calculate_chunk_size` aren't directly tested. They're exercised through the benchmark tests, but a small unit test that constructs a synthetic 10×10 raster with one observation and verifies the bbox mask + chunk calculation would catch many regressions cheaply.

**Fix:** Add a test that:

1. Builds a `RasterData` from a 10×10 grid with valid pixels everywhere.
2. Constructs a single-observation `ObservationData` at the grid centre.
3. Calls `find_affected_pixels` with a known `max_dist_m` and verifies the affected-pixel count and mask shape.

Same caveat as 30 — additive only.

---

### 32. `from conftest import …` style imports in test modules

**Disposition:** Leave — observation only; `from conftest import …` style is fine.

**Location:** `tests/test_benchmarks.py:30`, `tests/test_grid_points_consistency.py:20`.

Both test modules do:

```python
from conftest import assert_arrays_match_raster_benchmark, load_fixed_model_config
```

This works because pytest puts the test directory on `sys.path`, but it bypasses `conftest.py`'s normal pytest-fixture role. Using helpers directly via import is fine; it's just unconventional. No change required, just flagged for readers used to fixtures-only `conftest.py` files.

---

## Summary

The codebase is in good shape after the recent cleanup passes. The remaining issues fall into three buckets:

1. **Two real performance bugs in the points-mode gap-fill path** (findings 1, 3, 4): redundant CSV/DBSCAN/Bayesian work and uncached coast-boundary union. These can compound badly for large input sets with many on-land NaN gaps.

2. **One small redundancy in the Bayesian update inner loop** (finding 2): duplicate `np.log` calls per observation.

3. **Style-and-consistency cleanups** (findings 5–28): magic-number `0.5`, mixed `.values`/`.to_numpy()`, mixed column-name stripping, restating-the-obvious comments, and a few minor architectural cleanups (move CLI validation into the pipeline, deduplicate help-text strings, retire `BoundingBoxResult` if you don't plan to extend it).

Items 1, 3, and 4 are the high-value fixes; everything else is code-hygiene.

No dead code was found from the recent removals. No improperly-scoped imports were found (all `import` statements are at module top, all parent-module references use `from vs30 import constants` then `constants.X`). The affine-transform-style `x = obj.x` extractions are confined to the affine-transform code in `raster.compute_coast_distance_array` and `spatial.RasterData.get_coordinates`, where the exemption applies.

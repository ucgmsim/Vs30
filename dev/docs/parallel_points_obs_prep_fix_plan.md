# Points-Pipeline Multiproc Obs-Prep Fix — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix the per-chunk obs-prep redundancy bug in `vs30/parallel.py::run_parallel_locations` so multiproc points-mode runs at the speed of its inherent multiproc/BLAS-MT tradeoff rather than at the speed of `1000× redundant raster sampling`.

**Architecture:** Two-step refactor. Step 1 introduces a `PointsObsData` dataclass and two helpers (`prepare_geology_obs_data`, `prepare_terrain_obs_data`) that the per-points functions use internally — pure abstraction, no behaviour change. Step 2 hoists the helper calls up into `run_parallel_locations` and `points_pipeline` so the obs-prep happens once per pipeline call, not once per chunk. Workers receive precomputed `PointsObsData` and lose access to `observations_df` entirely. Step 3 records the smoke benchmark.

**Tech Stack:** Python 3.13, pytest, ruff, mamba `vs30_venv`. Activate with:

```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv
```

(Henceforth abbreviated as `<activate>`.)

**Reference docs:**
- `dev/docs/parallel_points_obs_prep_fix_design.md` — design, scope, risks.
- `dev/docs/points_perf_investigation_findings.md` — diagnoses the bug this work fixes (§4 of that doc).

---

## File map

| Path | Responsibility |
|---|---|
| `vs30/parallel.py` | (modify) Add `PointsObsData` dataclass + two prep helpers; switch `process_*_at_points` and `run_parallel_locations` to take `PointsObsData` instead of `observations_df` |
| `vs30/pipeline.py` | (modify) `points_pipeline` calls the helpers once before sequential and parallel branches; passes `PointsObsData` instead of `observations_df` |
| `dev/docs/parallel_points_obs_prep_fix_smoke_results.md` | (create) One-page record of pre/post smoke timings on the two `(N_query=1000, N_obs=35706)` cells |

No new test files. The validation gate is the existing `tests/test_benchmarks.py` and `tests/test_grid_points_consistency.py`, both of which already exercise `nproc>1` against `points_pipeline`.

---

## Task 1: Introduce `PointsObsData` + helpers (no behaviour change)

Add the dataclass and the two helpers to `vs30/parallel.py`. Make `process_geology_at_points` and `process_terrain_at_points` use the helpers internally — same compute, same per-chunk redundancy, just one level of abstraction added. Net effect on observable behaviour: zero.

This task is bisectable: each commit before this task and the commit at the end of this task should produce identical numerical output for any pipeline call.

**Files:**
- Modify: `vs30/parallel.py`

- [ ] **Step 1: Add the `PointsObsData` dataclass**

Insert at `vs30/parallel.py` immediately after the existing imports (the file currently uses `from dataclasses import dataclass` already, so no new import is needed). The dataclass goes BEFORE the existing `process_geology_at_points` definition (currently line 13).

```python
@dataclass
class PointsObsData:
    """Precomputed observation arrays for use in points-pipeline workers.

    All arrays have length N_obs. ``model_vs30`` and ``model_stdv`` carry
    post-hybrid-mods values for geology, raw categorical values for terrain.
    """

    locations: np.ndarray
    vs30: np.ndarray
    uncertainty: np.ndarray
    model_vs30: np.ndarray
    model_stdv: np.ndarray

    @classmethod
    def empty(cls) -> "PointsObsData":
        return cls(
            locations=np.empty((0, 2)),
            vs30=np.empty(0),
            uncertainty=np.empty(0),
            model_vs30=np.empty(0),
            model_stdv=np.empty(0),
        )
```

- [ ] **Step 2: Add `prepare_geology_obs_data`**

Insert in `vs30/parallel.py` immediately after the `PointsObsData` definition, before `process_geology_at_points`. The body is lifted verbatim from the existing `process_geology_at_points` lines 92–122 (the entire `if len(observations_df) > 0:` obs-prep block; the `compute_spatial_adjustment_at_points` call that follows it is NOT lifted — it stays in the worker).

```python
def prepare_geology_obs_data(
    observations_df: pd.DataFrame,
    geol_model_df: pd.DataFrame,
    apply_alluvium_slope_mod: bool,
    apply_coastal_distance_mod: bool,
) -> PointsObsData:
    """Precompute observation-side geology values for points_pipeline.

    Parameters
    ----------
    observations_df
        Combined observations DataFrame (easting, northing, vs30, uncertainty).
    geol_model_df
        Categorical geology model with Vs30 mean and standard deviation per category.
    apply_alluvium_slope_mod
        Whether to apply the alluvium slope modification.
    apply_coastal_distance_mod
        Whether to apply the coastal distance modification.

    Returns
    -------
    PointsObsData
        Precomputed observation arrays (model_vs30/stdv are post-hybrid-mods).
        Returns ``PointsObsData.empty()`` if ``observations_df`` is empty.
    """
    if len(observations_df) == 0:
        return PointsObsData.empty()

    obs_locs = observations_df[
        [constants.ObservationColumn.EASTING, constants.ObservationColumn.NORTHING]
    ].values
    obs_geol_ids = category.assign_to_category_geology(obs_locs)
    obs_geol_vs30_df = category.get_vs30_for_ids(obs_geol_ids, geol_model_df)

    # Apply hybrid modifications to observation model values so residuals
    # match the grid pipeline; see spatial.prepare_observation_data.
    obs_slope = raster.sample_slope_at_points(obs_locs)
    # Legacy parity: NODATA slope samples at observations are replaced with
    # the 255 sentinel so log10(255) ≈ 2.41 feeds np.interp and returns the
    # MAX Vs30 for the gid; the equivalent grid-pixel handling uses 1e-9
    # and returns the MIN Vs30. See constants.LEGACY_OBS_SLOPE_NODATA_SENTINEL.
    obs_slope = np.where(
        obs_slope < 0, constants.LEGACY_OBS_SLOPE_NODATA_SENTINEL, obs_slope
    )
    obs_coast_dist = (
        raster.compute_coastal_distance_at_points(obs_locs)
        if apply_coastal_distance_mod
        else np.zeros(len(obs_locs))
    )
    obs_model_vs30, obs_model_stdv = raster.apply_hybrid_geology_modifications(
        obs_geol_vs30_df[constants.COL_CATEGORY_VS30_MEAN].values,
        obs_geol_vs30_df[constants.COL_CATEGORY_VS30_STDV].values,
        obs_geol_ids,
        obs_slope,
        obs_coast_dist,
        apply_alluvium_slope_mod=apply_alluvium_slope_mod,
        apply_coastal_distance_mod=apply_coastal_distance_mod,
    )

    return PointsObsData(
        locations=obs_locs,
        vs30=observations_df[constants.ObservationColumn.VS30].values,
        uncertainty=observations_df[constants.ObservationColumn.UNCERTAINTY].values,
        model_vs30=obs_model_vs30,
        model_stdv=obs_model_stdv,
    )
```

- [ ] **Step 3: Add `prepare_terrain_obs_data`**

Insert in `vs30/parallel.py` immediately after `prepare_geology_obs_data`. Lifted verbatim from the existing `process_terrain_at_points` lines 204–208.

```python
def prepare_terrain_obs_data(
    observations_df: pd.DataFrame,
    terr_model_df: pd.DataFrame,
) -> PointsObsData:
    """Precompute observation-side terrain values for points_pipeline.

    Parameters
    ----------
    observations_df
        Combined observations DataFrame (easting, northing, vs30, uncertainty).
    terr_model_df
        Categorical terrain model with Vs30 mean and standard deviation per category.

    Returns
    -------
    PointsObsData
        Precomputed observation arrays.
        Returns ``PointsObsData.empty()`` if ``observations_df`` is empty.
    """
    if len(observations_df) == 0:
        return PointsObsData.empty()

    obs_locs = observations_df[
        [constants.ObservationColumn.EASTING, constants.ObservationColumn.NORTHING]
    ].values
    obs_terr_ids = category.assign_to_category_terrain(obs_locs)
    obs_terr_vs30_df = category.get_vs30_for_ids(obs_terr_ids, terr_model_df)

    return PointsObsData(
        locations=obs_locs,
        vs30=observations_df[constants.ObservationColumn.VS30].values,
        uncertainty=observations_df[constants.ObservationColumn.UNCERTAINTY].values,
        model_vs30=obs_terr_vs30_df[constants.COL_CATEGORY_VS30_MEAN].values,
        model_stdv=obs_terr_vs30_df[constants.COL_CATEGORY_VS30_STDV].values,
    )
```

- [ ] **Step 4: Use `prepare_geology_obs_data` inside `process_geology_at_points`**

Replace the body of `process_geology_at_points` (currently lines 13–151) so that the obs-prep block at the old lines 92–122 is replaced by a call to the new helper. The function signature and observable output are unchanged.

The new body (the **whole** function — replace the old definition completely):

```python
def process_geology_at_points(
    points: np.ndarray,
    model_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    corr_fn: Callable,
    apply_alluvium_slope_mod: bool,
    apply_coastal_distance_mod: bool,
    noisy: bool = False,
    progress_bar: tqdm | None = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Process geology model at points, including hybrid modifications and spatial adjustment.

    Parameters
    ----------
    points : ndarray
        Array of shape (n_points, 2) with (easting, northing) coordinates.
    model_df : DataFrame
        Categorical geology model with Vs30 mean and standard deviation per category.
    observations_df : DataFrame
        Observation data with columns: easting, northing, vs30, uncertainty.
    corr_fn : Callable
        Correlation function for spatial adjustment.
    apply_alluvium_slope_mod : bool
        Whether to apply the alluvium slope modification.
    apply_coastal_distance_mod : bool
        Whether to apply the coastal distance modification.
    noisy : bool
        Whether to apply noise weighting in spatial adjustment.
    progress_bar : tqdm, optional
        External progress bar to update per point during spatial adjustment.

    Returns
    -------
    geol_ids : ndarray
        Geology category IDs at each point.
    geol_vs30 : ndarray
        Initial geology Vs30 values (before hybrid mods).
    geol_stdv : ndarray
        Initial geology standard deviation (before hybrid mods).
    geol_vs30_hybrid : ndarray
        Geology Vs30 after hybrid modifications.
    geol_stdv_hybrid : ndarray
        Geology standard deviation after hybrid modifications.
    geol_mvn_vs30 : ndarray
        Final geology Vs30 after spatial adjustment.
    geol_mvn_stdv : ndarray
        Final geology standard deviation after spatial adjustment.
    """
    geol_ids = category.assign_to_category_geology(points)

    geol_vs30_df = category.get_vs30_for_ids(geol_ids, model_df)
    geol_vs30 = geol_vs30_df[constants.COL_CATEGORY_VS30_MEAN].values
    geol_stdv = geol_vs30_df[constants.COL_CATEGORY_VS30_STDV].values

    slope_at_points = raster.sample_slope_at_points(points)
    coast_dist_at_points = (
        raster.compute_coastal_distance_at_points(points)
        if apply_coastal_distance_mod
        else np.zeros(len(points))
    )

    geol_vs30_hybrid, geol_stdv_hybrid = raster.apply_hybrid_geology_modifications(
        geol_vs30,
        geol_stdv,
        geol_ids,
        slope_at_points,
        coast_dist_at_points,
        apply_alluvium_slope_mod=apply_alluvium_slope_mod,
        apply_coastal_distance_mod=apply_coastal_distance_mod,
    )

    geology_obs_data = prepare_geology_obs_data(
        observations_df,
        model_df,
        apply_alluvium_slope_mod=apply_alluvium_slope_mod,
        apply_coastal_distance_mod=apply_coastal_distance_mod,
    )

    if len(geology_obs_data.locations) > 0:
        geol_mvn_vs30, geol_mvn_stdv = spatial.compute_spatial_adjustment_at_points(
            points=points,
            model_vs30=geol_vs30_hybrid,
            model_stdv=geol_stdv_hybrid,
            obs_locations=geology_obs_data.locations,
            obs_vs30=geology_obs_data.vs30,
            obs_model_vs30=geology_obs_data.model_vs30,
            obs_model_stdv=geology_obs_data.model_stdv,
            obs_uncertainty=geology_obs_data.uncertainty,
            corr_fn=corr_fn,
            noisy=noisy,
            progress_bar=progress_bar,
        )
    else:
        geol_mvn_vs30 = geol_vs30_hybrid
        geol_mvn_stdv = geol_stdv_hybrid

    return (
        geol_ids,
        geol_vs30,
        geol_stdv,
        geol_vs30_hybrid,
        geol_stdv_hybrid,
        geol_mvn_vs30,
        geol_mvn_stdv,
    )
```

- [ ] **Step 5: Use `prepare_terrain_obs_data` inside `process_terrain_at_points`**

Replace the body of `process_terrain_at_points` (currently lines 154–234) so that the obs-prep block at the old lines 204–208 is replaced by a call to the new helper.

The new body (replace the whole function):

```python
def process_terrain_at_points(
    points: np.ndarray,
    model_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    corr_fn: Callable,
    noisy: bool = False,
    progress_bar: tqdm | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process terrain model at points, including spatial adjustment.

    Parameters
    ----------
    points : ndarray
        Array of shape (n_points, 2) with (easting, northing) coordinates.
    model_df : DataFrame
        Categorical terrain model with Vs30 mean and standard deviation per category.
    observations_df : DataFrame
        Observation data with columns: easting, northing, vs30, uncertainty.
    corr_fn : Callable
        Correlation function for spatial adjustment.
    noisy : bool
        Whether to apply noise weighting in spatial adjustment.
    progress_bar : tqdm, optional
        External progress bar to update per point during spatial adjustment.

    Returns
    -------
    terr_ids : ndarray
        Terrain category IDs at each point.
    terr_vs30 : ndarray
        Initial terrain Vs30 values.
    terr_stdv : ndarray
        Initial terrain standard deviation.
    terr_mvn_vs30 : ndarray
        Final terrain Vs30 after spatial adjustment.
    terr_mvn_stdv : ndarray
        Final terrain standard deviation after spatial adjustment.
    """
    terr_ids = category.assign_to_category_terrain(points)

    terr_vs30_df = category.get_vs30_for_ids(terr_ids, model_df)
    terr_vs30 = terr_vs30_df[constants.COL_CATEGORY_VS30_MEAN].values
    terr_stdv = terr_vs30_df[constants.COL_CATEGORY_VS30_STDV].values

    terrain_obs_data = prepare_terrain_obs_data(observations_df, model_df)

    if len(terrain_obs_data.locations) > 0:
        terr_mvn_vs30, terr_mvn_stdv = spatial.compute_spatial_adjustment_at_points(
            points=points,
            model_vs30=terr_vs30,
            model_stdv=terr_stdv,
            obs_locations=terrain_obs_data.locations,
            obs_vs30=terrain_obs_data.vs30,
            obs_model_vs30=terrain_obs_data.model_vs30,
            obs_model_stdv=terrain_obs_data.model_stdv,
            obs_uncertainty=terrain_obs_data.uncertainty,
            corr_fn=corr_fn,
            noisy=noisy,
            progress_bar=progress_bar,
        )
    else:
        terr_mvn_vs30 = terr_vs30
        terr_mvn_stdv = terr_stdv

    return (
        terr_ids,
        terr_vs30,
        terr_stdv,
        terr_mvn_vs30,
        terr_mvn_stdv,
    )
```

- [ ] **Step 6: Verify ruff is clean**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
ruff check vs30/parallel.py && \
ruff format --check vs30/parallel.py
```

If `ruff format --check` fails, run `ruff format vs30/parallel.py` and re-verify.

- [ ] **Step 7: Run the existing test suite**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
pytest tests/ 2>&1 | tail -20
```

Expected: all tests pass (~3 minutes for the default suite). The key tests are `tests/test_benchmarks.py::test_foster_2019_approx_points_benchmark` (parameterised over `nproc ∈ {1, -1}`) and `tests/test_grid_points_consistency.py` (calls `points_pipeline(..., nproc=-1, ...)`). Both must pass; failure means the helper extraction introduced a numerical drift.

- [ ] **Step 8: Commit**

```bash
cd /home/arr65/src/Vs30 && \
git add vs30/parallel.py && \
git commit -m "$(cat <<'EOF'
refactor(parallel): extract PointsObsData + obs-prep helpers

Add a small PointsObsData dataclass and two helpers:
prepare_geology_obs_data and prepare_terrain_obs_data. Use them
internally in process_geology_at_points and process_terrain_at_points
to replace the inline observation-preparation blocks. No behaviour
change yet — both functions still take observations_df and run the
helpers on every call (so the parallel path still pays the per-chunk
cost).

This commit is bisectable: every observable output of points_pipeline
is unchanged. The follow-up commit hoists the helper calls up so they
run once per pipeline call instead of once per chunk, fixing the bug
diagnosed in §4 of points_perf_investigation_findings.md.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Hoist helper calls + fix the bug

This is the commit that fixes the bug. After this task, the obs-prep work happens exactly once per `points_pipeline` call (not once per chunk), and workers no longer receive `observations_df` at all.

**Files:**
- Modify: `vs30/parallel.py`
- Modify: `vs30/pipeline.py`

- [ ] **Step 1: Update `process_geology_at_points` to take `PointsObsData`**

In `vs30/parallel.py`, replace the function definition modified in Task 1, Step 4 with this version. Two changes from the Task 1 version: the parameter is `geology_obs_data: PointsObsData` instead of `observations_df`, and the inline `prepare_geology_obs_data(...)` call is removed.

```python
def process_geology_at_points(
    points: np.ndarray,
    model_df: pd.DataFrame,
    geology_obs_data: PointsObsData,
    corr_fn: Callable,
    apply_alluvium_slope_mod: bool,
    apply_coastal_distance_mod: bool,
    noisy: bool = False,
    progress_bar: tqdm | None = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Process geology model at points, including hybrid modifications and spatial adjustment.

    Parameters
    ----------
    points : ndarray
        Array of shape (n_points, 2) with (easting, northing) coordinates.
    model_df : DataFrame
        Categorical geology model with Vs30 mean and standard deviation per category.
    geology_obs_data : PointsObsData
        Precomputed observation-side geology values from
        ``prepare_geology_obs_data``.
    corr_fn : Callable
        Correlation function for spatial adjustment.
    apply_alluvium_slope_mod : bool
        Whether to apply the alluvium slope modification (to query-point hybrid mods).
    apply_coastal_distance_mod : bool
        Whether to apply the coastal distance modification (to query-point hybrid mods).
    noisy : bool
        Whether to apply noise weighting in spatial adjustment.
    progress_bar : tqdm, optional
        External progress bar to update per point during spatial adjustment.

    Returns
    -------
    geol_ids : ndarray
        Geology category IDs at each point.
    geol_vs30 : ndarray
        Initial geology Vs30 values (before hybrid mods).
    geol_stdv : ndarray
        Initial geology standard deviation (before hybrid mods).
    geol_vs30_hybrid : ndarray
        Geology Vs30 after hybrid modifications.
    geol_stdv_hybrid : ndarray
        Geology standard deviation after hybrid modifications.
    geol_mvn_vs30 : ndarray
        Final geology Vs30 after spatial adjustment.
    geol_mvn_stdv : ndarray
        Final geology standard deviation after spatial adjustment.
    """
    geol_ids = category.assign_to_category_geology(points)

    geol_vs30_df = category.get_vs30_for_ids(geol_ids, model_df)
    geol_vs30 = geol_vs30_df[constants.COL_CATEGORY_VS30_MEAN].values
    geol_stdv = geol_vs30_df[constants.COL_CATEGORY_VS30_STDV].values

    slope_at_points = raster.sample_slope_at_points(points)
    coast_dist_at_points = (
        raster.compute_coastal_distance_at_points(points)
        if apply_coastal_distance_mod
        else np.zeros(len(points))
    )

    geol_vs30_hybrid, geol_stdv_hybrid = raster.apply_hybrid_geology_modifications(
        geol_vs30,
        geol_stdv,
        geol_ids,
        slope_at_points,
        coast_dist_at_points,
        apply_alluvium_slope_mod=apply_alluvium_slope_mod,
        apply_coastal_distance_mod=apply_coastal_distance_mod,
    )

    if len(geology_obs_data.locations) > 0:
        geol_mvn_vs30, geol_mvn_stdv = spatial.compute_spatial_adjustment_at_points(
            points=points,
            model_vs30=geol_vs30_hybrid,
            model_stdv=geol_stdv_hybrid,
            obs_locations=geology_obs_data.locations,
            obs_vs30=geology_obs_data.vs30,
            obs_model_vs30=geology_obs_data.model_vs30,
            obs_model_stdv=geology_obs_data.model_stdv,
            obs_uncertainty=geology_obs_data.uncertainty,
            corr_fn=corr_fn,
            noisy=noisy,
            progress_bar=progress_bar,
        )
    else:
        geol_mvn_vs30 = geol_vs30_hybrid
        geol_mvn_stdv = geol_stdv_hybrid

    return (
        geol_ids,
        geol_vs30,
        geol_stdv,
        geol_vs30_hybrid,
        geol_stdv_hybrid,
        geol_mvn_vs30,
        geol_mvn_stdv,
    )
```

- [ ] **Step 2: Update `process_terrain_at_points` to take `PointsObsData`**

```python
def process_terrain_at_points(
    points: np.ndarray,
    model_df: pd.DataFrame,
    terrain_obs_data: PointsObsData,
    corr_fn: Callable,
    noisy: bool = False,
    progress_bar: tqdm | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process terrain model at points, including spatial adjustment.

    Parameters
    ----------
    points : ndarray
        Array of shape (n_points, 2) with (easting, northing) coordinates.
    model_df : DataFrame
        Categorical terrain model with Vs30 mean and standard deviation per category.
    terrain_obs_data : PointsObsData
        Precomputed observation-side terrain values from
        ``prepare_terrain_obs_data``.
    corr_fn : Callable
        Correlation function for spatial adjustment.
    noisy : bool
        Whether to apply noise weighting in spatial adjustment.
    progress_bar : tqdm, optional
        External progress bar to update per point during spatial adjustment.

    Returns
    -------
    terr_ids : ndarray
        Terrain category IDs at each point.
    terr_vs30 : ndarray
        Initial terrain Vs30 values.
    terr_stdv : ndarray
        Initial terrain standard deviation.
    terr_mvn_vs30 : ndarray
        Final terrain Vs30 after spatial adjustment.
    terr_mvn_stdv : ndarray
        Final terrain standard deviation after spatial adjustment.
    """
    terr_ids = category.assign_to_category_terrain(points)

    terr_vs30_df = category.get_vs30_for_ids(terr_ids, model_df)
    terr_vs30 = terr_vs30_df[constants.COL_CATEGORY_VS30_MEAN].values
    terr_stdv = terr_vs30_df[constants.COL_CATEGORY_VS30_STDV].values

    if len(terrain_obs_data.locations) > 0:
        terr_mvn_vs30, terr_mvn_stdv = spatial.compute_spatial_adjustment_at_points(
            points=points,
            model_vs30=terr_vs30,
            model_stdv=terr_stdv,
            obs_locations=terrain_obs_data.locations,
            obs_vs30=terrain_obs_data.vs30,
            obs_model_vs30=terrain_obs_data.model_vs30,
            obs_model_stdv=terrain_obs_data.model_stdv,
            obs_uncertainty=terrain_obs_data.uncertainty,
            corr_fn=corr_fn,
            noisy=noisy,
            progress_bar=progress_bar,
        )
    else:
        terr_mvn_vs30 = terr_vs30
        terr_mvn_stdv = terr_stdv

    return (
        terr_ids,
        terr_vs30,
        terr_stdv,
        terr_mvn_vs30,
        terr_mvn_stdv,
    )
```

- [ ] **Step 3: Update `process_locations_chunk` to receive precomputed obs data**

In `vs30/parallel.py`, replace `process_locations_chunk` (currently around line 265) with this version. The chunk_args tuple now carries `geology_obs_data` and `terrain_obs_data` instead of `observations_df`; either may be `None` when that branch is not running.

```python
def process_locations_chunk(
    args: tuple,
) -> tuple[int, pd.DataFrame]:  # pragma: no cover
    """
    Worker function: process a chunk of locations through the full pipeline.

    Note: This function is excluded from coverage because it runs in a
    spawned subprocess which cannot be tracked by pytest-cov.

    Parameters
    ----------
    args : tuple
        (points, chunk_id, geology_obs_data, terrain_obs_data, geol_model_df, terr_model_df, config)
        where points is an (N, 2) array of NZTM (easting, northing) coordinates,
        and geology_obs_data / terrain_obs_data are PointsObsData instances
        (or None if that branch is not running).

    Returns
    -------
    tuple
        (chunk_id, result_df) where result_df has all computed columns
    """
    (
        points,
        chunk_id,
        geology_obs_data,
        terrain_obs_data,
        geol_model_df,
        terr_model_df,
        config,
    ) = args

    result = {}

    run_geology = config.model_type in (
        constants.ModelType.GEOLOGY,
        constants.ModelType.COMBINED,
    )
    run_terrain = config.model_type in (
        constants.ModelType.TERRAIN,
        constants.ModelType.COMBINED,
    )

    if run_geology:
        (
            geol_ids,
            geol_vs30,
            geol_stdv,
            geol_vs30_hybrid,
            geol_stdv_hybrid,
            geol_mvn_vs30,
            geol_mvn_stdv,
        ) = process_geology_at_points(
            points,
            geol_model_df,
            geology_obs_data,
            config.geology_corr_fn,
            apply_alluvium_slope_mod=config.apply_alluvium_slope_mod,
            apply_coastal_distance_mod=config.apply_coastal_distance_mod,
            noisy=config.noisy,
        )

        if config.include_intermediate:
            result[constants.COL_GEOLOGY_ID] = geol_ids
            result[constants.COL_GEOLOGY_VS30] = geol_vs30
            result[constants.COL_GEOLOGY_STDV] = geol_stdv
            result[constants.COL_GEOLOGY_VS30_HYBRID] = geol_vs30_hybrid
            result[constants.COL_GEOLOGY_STDV_HYBRID] = geol_stdv_hybrid
            result[constants.COL_GEOLOGY_MVN_VS30] = geol_mvn_vs30
            result[constants.COL_GEOLOGY_MVN_STDV] = geol_mvn_stdv

    if run_terrain:
        (
            terr_ids,
            terr_vs30,
            terr_stdv,
            terr_mvn_vs30,
            terr_mvn_stdv,
        ) = process_terrain_at_points(
            points, terr_model_df, terrain_obs_data, config.terrain_corr_fn, config.noisy
        )

        if config.include_intermediate:
            result[constants.COL_TERRAIN_ID] = terr_ids
            result[constants.COL_TERRAIN_VS30] = terr_vs30
            result[constants.COL_TERRAIN_STDV] = terr_stdv
            result[constants.COL_TERRAIN_MVN_VS30] = terr_mvn_vs30
            result[constants.COL_TERRAIN_MVN_STDV] = terr_mvn_stdv

    if run_geology and run_terrain:
        combined_vs30, combined_stdv = utils.combine_vs30_models(
            geol_mvn_vs30,
            geol_mvn_stdv,
            terr_mvn_vs30,
            terr_mvn_stdv,
            config.combination_method,
            config.combine_ratio,
        )
        result[constants.ObservationColumn.VS30] = combined_vs30
        result[constants.COL_COMBINED_STDV] = combined_stdv
    elif run_geology:
        result[constants.ObservationColumn.VS30] = geol_mvn_vs30
        result[constants.COL_COMBINED_STDV] = geol_mvn_stdv
    elif run_terrain:
        result[constants.ObservationColumn.VS30] = terr_mvn_vs30
        result[constants.COL_COMBINED_STDV] = terr_mvn_stdv

    return chunk_id, pd.DataFrame(result)
```

- [ ] **Step 4: Update `run_parallel_locations` signature + hoist helpers**

Replace `run_parallel_locations` (currently around line 375) with this version. The `observations_df` parameter is replaced by `geology_obs_data: PointsObsData | None` and `terrain_obs_data: PointsObsData | None`. Helpers are NOT called inside this function — the caller (`points_pipeline`) precomputes them.

```python
def run_parallel_locations(
    points: np.ndarray,
    geology_obs_data: PointsObsData | None,
    terrain_obs_data: PointsObsData | None,
    geol_model_df: pd.DataFrame | None,
    terr_model_df: pd.DataFrame | None,
    config: LocationsChunkConfig,
    nproc: int,
) -> pd.DataFrame:
    """
    Process locations in parallel.

    Divides the points array into chunks and processes each chunk
    in a separate process using the full VS30 pipeline. Pass None for
    the model and obs-data fields whose branch is not used by
    ``config.model_type``.

    Parameters
    ----------
    points : ndarray
        Array of shape (N, 2) with NZTM (easting, northing) coordinates.
    geology_obs_data : PointsObsData or None
        Precomputed observation-side geology values from
        ``prepare_geology_obs_data``. Required when running geology;
        otherwise None.
    terrain_obs_data : PointsObsData or None
        Precomputed observation-side terrain values from
        ``prepare_terrain_obs_data``. Required when running terrain;
        otherwise None.
    geol_model_df : DataFrame or None
        Geology categorical model. Required when running geology; otherwise None.
    terr_model_df : DataFrame or None
        Terrain categorical model. Required when running terrain; otherwise None.
    config : LocationsChunkConfig
        Configuration parameters for processing
    nproc : int
        Number of processes to use (must be > 1)

    Returns
    -------
    DataFrame
        Results with vs30, stdv, and intermediate columns (if requested)
    """
    n_chunks = min(len(points), constants.N_PROGRESS_CHUNKS)
    split_indices = np.array_split(range(len(points)), n_chunks)
    chunk_args = [
        (
            points[idx],
            i,
            geology_obs_data,
            terrain_obs_data,
            geol_model_df,
            terr_model_df,
            config,
        )
        for i, idx in enumerate(split_indices)
        if len(idx) > 0
    ]

    with multiprocess.single_threaded_blas():
        with multiprocess.spawn_context.Pool(processes=nproc) as pool:
            results = []
            with tqdm(total=len(points), unit="point") as pbar:
                for chunk_id, result_df in pool.imap(
                    process_locations_chunk, chunk_args
                ):
                    results.append((chunk_id, result_df))
                    pbar.update(len(result_df))

    results.sort(key=lambda x: x[0])
    return pd.concat([r[1] for r in results], ignore_index=True)
```

- [ ] **Step 5: Update `points_pipeline` to call helpers and pass precomputed obs data**

In `vs30/pipeline.py`, modify `points_pipeline` (function starts at line 1219). Two regions need changes:

**Region A**: between the existing observations load (around line 1329, the `logger.info(f"Loaded {len(observations_df)} observations...")` line) and the existing parallel/sequential branch decision (`if nproc_resolved > 1:` at line 1387), insert helper calls that compute the obs data once.

**Region B**: update the parallel branch (around lines 1402–1409) to pass the precomputed obs data instead of `observations_df`. Update the sequential branch (around lines 1439 and 1471) similarly.

Specifically: after line 1380 (immediately after the categorical models are loaded, before `nproc_resolved = multiprocess.resolve_nproc(nproc)`), add:

```python
    # Precompute observation-side data once. Shared by sequential and parallel
    # branches; in the parallel path this is the fix for the per-chunk
    # obs-prep redundancy bug. See dev/docs/parallel_points_obs_prep_fix_design.md.
    geology_obs_data = (
        parallel.prepare_geology_obs_data(
            observations_df,
            geol_model_df,
            apply_alluvium_slope_mod=apply_alluvium_slope_mod,
            apply_coastal_distance_mod=apply_coastal_distance_mod,
        )
        if run_geology
        else None
    )
    terrain_obs_data = (
        parallel.prepare_terrain_obs_data(observations_df, terr_model_df)
        if run_terrain
        else None
    )
```

Then in the parallel branch, replace:

```python
        result_df = parallel.run_parallel_locations(
            points=points,
            observations_df=observations_df,
            geol_model_df=geol_model_df,
            terr_model_df=terr_model_df,
            config=loc_config,
            nproc=nproc_resolved,
        )
```

with:

```python
        result_df = parallel.run_parallel_locations(
            points=points,
            geology_obs_data=geology_obs_data,
            terrain_obs_data=terrain_obs_data,
            geol_model_df=geol_model_df,
            terr_model_df=terr_model_df,
            config=loc_config,
            nproc=nproc_resolved,
        )
```

In the sequential branch, replace the `parallel.process_geology_at_points(...)` call (around lines 1439–1448):

```python
                ) = parallel.process_geology_at_points(
                    points,
                    geol_model_df,
                    observations_df,
                    corr_fn=geology_corr_fn,
                    noisy=noisy,
                    progress_bar=pbar,
                    apply_coastal_distance_mod=apply_coastal_distance_mod,
                    apply_alluvium_slope_mod=apply_alluvium_slope_mod,
                )
```

with:

```python
                ) = parallel.process_geology_at_points(
                    points,
                    geol_model_df,
                    geology_obs_data,
                    corr_fn=geology_corr_fn,
                    noisy=noisy,
                    progress_bar=pbar,
                    apply_coastal_distance_mod=apply_coastal_distance_mod,
                    apply_alluvium_slope_mod=apply_alluvium_slope_mod,
                )
```

And replace the `parallel.process_terrain_at_points(...)` call (around lines 1471–1478):

```python
                ) = parallel.process_terrain_at_points(
                    points,
                    terr_model_df,
                    observations_df,
                    corr_fn=terrain_corr_fn,
                    noisy=noisy,
                    progress_bar=pbar,
                )
```

with:

```python
                ) = parallel.process_terrain_at_points(
                    points,
                    terr_model_df,
                    terrain_obs_data,
                    corr_fn=terrain_corr_fn,
                    noisy=noisy,
                    progress_bar=pbar,
                )
```

The `observations_df` local in `points_pipeline` is still loaded (it's the input to the helpers). After Region A's insertion, the only remaining direct use of `observations_df` is as the helper argument; after Regions B's edits, no other code in `points_pipeline` references it.

- [ ] **Step 6: Verify ruff is clean**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
ruff check vs30/parallel.py vs30/pipeline.py && \
ruff format --check vs30/parallel.py vs30/pipeline.py
```

If `ruff format --check` fails, run `ruff format vs30/parallel.py vs30/pipeline.py` and re-verify.

- [ ] **Step 7: Run the existing test suite**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
pytest tests/ 2>&1 | tail -20
```

Expected: all tests pass. The critical assertions:
- `tests/test_benchmarks.py::test_foster_2019_approx_points_benchmark` passes for both `nproc=1` and `nproc=-1` parametrisations (numerical equivalence between sequential and parallel paths after the refactor).
- `tests/test_grid_points_consistency.py` passes (the points result with `nproc=-1` still matches the grid result, which exercises a totally separate code path — strong evidence no drift was introduced).

If any failure: do NOT proceed to Task 3. Diagnose and fix.

- [ ] **Step 8: Commit**

```bash
cd /home/arr65/src/Vs30 && \
git add vs30/parallel.py vs30/pipeline.py && \
git commit -m "$(cat <<'EOF'
fix(parallel): hoist points-mode obs-prep out of the per-chunk loop

points_pipeline now calls prepare_geology_obs_data and
prepare_terrain_obs_data once before dispatching to either the
sequential or parallel path, then passes the precomputed PointsObsData
through. Workers no longer receive observations_df, so they cannot
recompute the obs-prep — the per-chunk redundancy that made nproc=8
~150x slower than nproc=1 at large N_query is eliminated by
construction.

process_geology_at_points, process_terrain_at_points,
process_locations_chunk, and run_parallel_locations all now take
PointsObsData instead of observations_df. Function signatures change;
no external callers exist outside vs30/pipeline.py.

Diagnosis: dev/docs/points_perf_investigation_findings.md §4
Design: dev/docs/parallel_points_obs_prep_fix_design.md

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Smoke benchmark + record evidence

Run two specific cells from the points-perf harness with the fix in place. Compare to pre-fix numbers. Confirm the bug is gone. Commit a small evidence document.

**Files:**
- Create: `dev/docs/parallel_points_obs_prep_fix_smoke_results.md`

- [ ] **Step 1: Run the smoke benchmark**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
python -c "
import sys
sys.path.insert(0, 'dev/scripts/investigations/points_features_investigation')
import bench_utils
from pathlib import Path

# Materialise obs CSV for N_obs=35706 (the full viktor_cpt set).
out_dir = Path('dev/scripts/investigations/points_features_investigation/obs_csvs')
paths = bench_utils.materialize_obs_csvs(out_dir, [35706])
obs_path = paths[35706]

# Generate 1000 NZ-land query points.
lons, lats = bench_utils.generate_nz_land_points(1000, seed=42)

cfg = bench_utils.load_modified_foster_2019_config()

# Two cells: nproc=1 and nproc=8.
for nproc in [1, 8]:
    print(f'Running N_query=1000, N_obs=35706, nproc={nproc}...')
    row = bench_utils.time_one_run(
        lons=lons, lats=lats, obs_csv_path=obs_path,
        nproc=nproc, rep=0, cfg=cfg,
    )
    print(f'  t_total_s = {row[\"t_total_s\"]:.2f}')
"
```

Expected runtime: nproc=1 in ~18 s; nproc=8 in roughly the same order of magnitude (≤ ~30 s) once the bug is fixed. If nproc=8 takes more than ~5 minutes, the bug is NOT fixed and Task 2 has a regression — return to Task 2 Step 7.

Record both `t_total_s` values for the next step.

- [ ] **Step 2: Write the smoke evidence doc**

Create `dev/docs/parallel_points_obs_prep_fix_smoke_results.md`. Fill in the actual numbers from Step 1.

```markdown
# Points-Pipeline Multiproc Obs-Prep Fix — Smoke Benchmark Results

**Date:** [today]
**Branch:** `vs30_refactor`
**Predecessors:**
- [Design](parallel_points_obs_prep_fix_design.md)
- [Investigation findings (bug diagnosis)](points_perf_investigation_findings.md)

## Setup

Hardware: Intel Core i7-9700, 8 cores, 32 GiB. Same as the predecessor investigation.

Cells re-run via the points-perf harness (`dev/scripts/investigations/points_features_investigation/`):

- `(N_query=1000, N_obs=35706, nproc=1)` — sequential path; should be unchanged.
- `(N_query=1000, N_obs=35706, nproc=8)` — parallel path; the cell most affected by the bug (151× slower pre-fix).

## Results

| Cell | Pre-fix (s) | Post-fix (s) | Change |
|---|---|---|---|
| nproc=1 | 18.35 | [Step 1 number] | [≈ unchanged / small drift expected] |
| nproc=8 | 2,770 | [Step 1 number] | [post / pre ratio]× faster |

## Interpretation

[2-4 sentences:
- Confirm the bug is gone (the nproc=8 cell ran in seconds, not 46 minutes).
- Note whether multiproc actually wins over the sequential path for this cell, OR is in the same ballpark, OR is slower (meaning the inherent multiproc/BLAS-MT tradeoff still loses for points mode at this size). This shapes the deferred CLI-default decision.
- Note any unexpected observations (e.g., variance higher than expected, memory blew up, etc.).
]

## Out of scope (deferred per design §2)

- Full re-sweep of the 7 × 3 × {1, 8} × 3 matrix.
- CLI default decision (depends on the full re-sweep, not the smoke).
- `Pool(initializer=...)` optimisation.
```

- [ ] **Step 3: Commit**

```bash
cd /home/arr65/src/Vs30 && \
git add dev/docs/parallel_points_obs_prep_fix_smoke_results.md && \
git commit -m "$(cat <<'EOF'
docs: smoke benchmark confirms multiproc obs-prep bug is fixed

(N_query=1000, N_obs=35706) ran in [post-fix nproc=8 time] s on
nproc=8, vs 2,770 s pre-fix — a ~[ratio]x recovery. nproc=1 baseline
unchanged at ~18 s. The per-chunk obs-prep redundancy bug diagnosed
in points_perf_investigation_findings.md §4 is resolved.

A full re-sweep of the original 7 x 3 x {1, 8} x 3 matrix is the
natural next piece of work — it answers the inherent multiproc/BLAS-MT
tradeoff question (which the smoke does not), and feeds into the CLI
default decision deferred from this work.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Edit the commit message to use the actual measured numbers.)

---

## Self-review

**Spec coverage** (from `dev/docs/parallel_points_obs_prep_fix_design.md`):

- §1 Purpose → addressed by Tasks 1+2 together (helpers extracted, hoisted; redundancy eliminated).
- §2 In scope:
  - PointsObsData dataclass + helpers → Task 1 Steps 1–3
  - Refactor of process_*_at_points and run_parallel_locations → Task 1 Steps 4–5 (helpers used internally) + Task 2 Steps 1–4 (signature swap)
  - Rewire points_pipeline → Task 2 Step 5
  - Smoke benchmark → Task 3
- §2 Out of scope: confirmed nothing in the plan touches CLI defaults, full re-sweep, Pool initializer, findings amendment, or memory measurement.
- §3.1 New module surface → Task 1 Steps 1–3.
- §3.2 Modified function signatures → Task 1 Steps 4–5 (intermediate state) + Task 2 Steps 1–4 (final state).
- §3.3 Data flow → Task 2 Step 5 (Region A + Region B).
- §3.4 Pickle / serialisation cost → not addressed in code (correctly — it's an observation, not a change). The smoke in Task 3 will surface whether pickle cost is meaningful.
- §4.1 Correctness via existing tests → Task 1 Step 7 + Task 2 Step 7.
- §4.2 Smoke benchmark → Task 3.
- §5 Branch and commit strategy → exactly three commits as designed (Task 1 commit, Task 2 commit, Task 3 commit).
- §6 Risks → mitigations folded into the relevant steps (verbatim lift, existing-tests gate, smoke for the pickle cost concern).
- §7 Out-of-scope follow-ups → not addressed (correctly — they're future work).

**Placeholder scan:** every code block contains complete, runnable code. The smoke results doc skeleton in Task 3 Step 2 has clearly-marked blanks (`[Step 1 number]`, `[≈ unchanged / small drift expected]`, etc.) that the implementer fills from the measured numbers. This is intentional template scaffolding tied to a measurement step, not a TODO.

**Type consistency:**
- `PointsObsData` defined in Task 1 Step 1; same fields used in `prepare_geology_obs_data` (Task 1 Step 2), `prepare_terrain_obs_data` (Task 1 Step 3), the modified `process_*_at_points` (Task 1 Steps 4–5 and Task 2 Steps 1–2), `process_locations_chunk` (Task 2 Step 3), and `run_parallel_locations` (Task 2 Step 4).
- The `geology_obs_data` and `terrain_obs_data` parameter names are used identically across `process_locations_chunk`, `run_parallel_locations`, and `points_pipeline`'s call sites (Task 2 Steps 3, 4, 5).
- `apply_alluvium_slope_mod` and `apply_coastal_distance_mod` continue to flow through the geology helper and the modified `process_geology_at_points` consistently.

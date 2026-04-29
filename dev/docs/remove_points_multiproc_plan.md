# Remove points-pipeline multiprocessing — implementation plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the multiprocessing dispatch path from the points pipeline (validated dead by `dev/docs/points_perf_post_fix_findings.md`), rename `vs30/parallel.py` → `vs30/points.py`, and extract grid-pipeline stage helpers from `vs30/pipeline.py` into a new `vs30/grid.py` for symmetry.

**Architecture:** Pure refactor — no behavior change for the surviving code path. The sequential code path inside `points_pipeline` already covers everything functionally; the parallel branch is being deleted. After the work, `vs30/pipeline.py` contains orchestrators + shared utilities, `vs30/grid.py` contains grid-pipeline stage helpers, `vs30/points.py` contains points-pipeline per-point helpers. One-way dependency: `pipeline → grid` and `pipeline → points`, no back-edges.

**Tech Stack:** Python 3.13, pytest, Typer (CLI), numpy/pandas/rasterio.

**Spec:** [`dev/docs/remove_points_multiproc_design.md`](remove_points_multiproc_design.md). Read it first.

---

## File structure (target end state)

| File | Status | Contents |
|---|---|---|
| `vs30/points.py` | renamed from `parallel.py`, slimmed | `PointsObsData`, `prepare_geology_obs_data`, `prepare_terrain_obs_data`, `process_geology_at_points`, `process_terrain_at_points` |
| `vs30/grid.py` | new | `create_initial_vs30_arrays`, `compute_hybrid_geology_arrays`, `compute_spatial_adjustment_on_grid`, `combine_model_arrays`, `write_raster` |
| `vs30/pipeline.py` | slimmed | shared utilities + `compute_categorical_vs30_updates` + `compute_model_grid` + `grid_pipeline` + `fill_one_point_via_local_grid` + `points_pipeline` |
| `vs30/multiprocess.py` | **deleted** | (was `spawn_context`, `limit_blas_threads`, `resolve_nproc`) |
| `vs30/constants.py` | one constant removed | `N_PROGRESS_CHUNKS` deleted |
| `vs30/cli.py` | updated | `points` / `points_custom` / `run_points_pipeline` now use `dbscan_nproc` instead of `nproc` |
| `vs30/parallel.py` | **deleted** (renamed away) | (was multiprocessing plumbing) |
| `tests/test_benchmarks.py` | updated | parametrize and `nproc` arg dropped |
| `tests/test_grid_points_consistency.py` | updated | `nproc=-1` arg dropped |
| `dev/scripts/investigations/points_features_investigation/run_balanced_blas_supplement.py` | **deleted** | (was sweep runner) |
| `dev/scripts/investigations/points_features_investigation/run_points_sweep.py` | **deleted** | (was sweep runner) |
| `dev/scripts/investigations/points_features_investigation/bench_utils.py` | **deleted** | (was sweep helper) |
| `dev/scripts/investigations/points_features_investigation/test_bench_utils.py` | **deleted** | (was sweep-helper test) |
| `dev/docs/points_perf_post_fix_findings.md` | postscript added | one-paragraph note at top about the follow-up action |
| `tests/conftest.py` (conditional) | possibly trimmed | `pytest_addoption` and `pytest_collection_modifyitems` deleted iff §Task 9 measures slow-tier <15 min |
| `dev/CLAUDE.md` (conditional) | runtime estimate updated | Running Tests section updated to reflect measured times |

Refactor approach: each task produces a green-tests commit. The full default test suite (`pytest tests/`) runs in ~3 min and acts as the safety net throughout.

---

## Task 1: Baseline — confirm starting tests are green

**Files:** None modified.

- [ ] **Step 1.1: Activate the conda environment and run the full test suite.**

```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv && \
cd /home/arr65/src/Vs30 && \
pytest tests/ -v
```

Expected: **all tests pass** (~3 min). If anything is red on `master`/`vs30_refactor` baseline, stop and investigate before proceeding — this plan assumes a green baseline.

- [ ] **Step 1.2: Note current test count and runtime.**

Record the number of tests collected and the runtime; you'll cross-check at the end. (No commit — just observation.)

---

## Task 2: Rename `vs30/parallel.py` → `vs30/points.py`

**Goal:** Pure rename, no behavior change. After this task, `parallel.py` no longer exists; everything that called `parallel.X` now calls `points.X`.

**Files:**
- Rename: `vs30/parallel.py` → `vs30/points.py`
- Modify: `vs30/pipeline.py` (one import line, six call-site references)

- [ ] **Step 2.1: Move the file with `git mv` (preserves history).**

```bash
git mv vs30/parallel.py vs30/points.py
```

- [ ] **Step 2.2: Update the import in `vs30/pipeline.py`.**

Find this block (around line 17–27):

```python
from vs30 import (
    category,
    config,
    constants,
    gapfill,
    multiprocess,
    parallel,
    raster,
    spatial,
    utils,
)
```

Replace `parallel,` with `points,` (keeps alphabetical order: `points` slots between `multiprocess` and `raster`):

```python
from vs30 import (
    category,
    config,
    constants,
    gapfill,
    multiprocess,
    points,
    raster,
    spatial,
    utils,
)
```

- [ ] **Step 2.3: Update the six call-sites in `vs30/pipeline.py`.**

In `points_pipeline`, replace every `parallel.X` reference with `points.X`. The six occurrences are:

| Approximate line | From | To |
|---|---|---|
| ~1385 | `parallel.prepare_geology_obs_data(` | `points.prepare_geology_obs_data(` |
| ~1395 | `parallel.prepare_terrain_obs_data(` | `points.prepare_terrain_obs_data(` |
| ~1408 | `loc_config = parallel.LocationsChunkConfig(` | `loc_config = points.LocationsChunkConfig(` |
| ~1420 | `result_df = parallel.run_parallel_locations(` | `result_df = points.run_parallel_locations(` |
| ~1458 | `) = parallel.process_geology_at_points(` | `) = points.process_geology_at_points(` |
| ~1490 | `) = parallel.process_terrain_at_points(` | `) = points.process_terrain_at_points(` |

Use `sed` for safety (one-liner replacement of `parallel.` → `points.` inside `pipeline.py`):

```bash
grep -n "parallel\." vs30/pipeline.py
```

Expected: exactly 6 hits, all of which are the call-sites above. If you see anything else, stop and investigate — the rename should be 1-to-1.

Then apply the replacement:

```bash
sed -i 's/parallel\./points./g' vs30/pipeline.py
```

Verify by re-grepping (should now show zero hits for `parallel.`):

```bash
grep -n "parallel\." vs30/pipeline.py
```

Expected: no output.

- [ ] **Step 2.4: Run the full test suite.**

```bash
pytest tests/ -v
```

Expected: all tests pass. No behavior changed; only file/symbol names.

- [ ] **Step 2.5: Confirm no stragglers refer to the old name.**

```bash
grep -rn "vs30\.parallel\|from vs30 import.*parallel\|from vs30\.parallel" --include="*.py"
```

Expected: zero hits. (Some hits in `dev/scripts/investigations/points_features_investigation/` are acceptable iff they're in scripts being deleted in Task 7 — but at this point we expect zero matches in `vs30/` or `tests/`.)

- [ ] **Step 2.6: Commit.**

```bash
git add vs30/points.py vs30/pipeline.py
git commit -m "$(cat <<'EOF'
refactor: rename vs30/parallel.py to vs30/points.py

Pure rename — no behavior change. The file no longer does any parallel
work after the upcoming multiprocessing removal; renaming first lets
each subsequent change land cleanly.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3a: Remove `nproc` from the points-pipeline public API

**Goal:** Swap `nproc` → `dbscan_nproc` on `points_pipeline`, `run_points_pipeline`, and the two CLI commands. Delete the parallel-dispatch branch in `points_pipeline`. Drop `multiprocess` import. After this task the parallel branch is gone, but the symbols inside `points.py` (`LocationsChunkConfig`, `process_locations_chunk`, `run_parallel_locations`) are now dead code — they get removed in Task 3b.

**Files:**
- Modify: `vs30/pipeline.py` (`points_pipeline` signature + body + docstring; drop `multiprocess` import)
- Modify: `vs30/cli.py` (`points`, `points_custom`, `run_points_pipeline` signatures + docstrings + call-site forwarding)
- Modify: `tests/test_benchmarks.py` (drop parametrize, drop arg, fix docstring)
- Modify: `tests/test_grid_points_consistency.py` (drop arg)

- [ ] **Step 3a.1: Update `tests/test_grid_points_consistency.py`.**

Around line 87, in `run_points_pipeline_for_version`, find the `points_pipeline` call:

```python
def run_points_pipeline_for_version(cfg: dict, points_df: pd.DataFrame) -> pd.DataFrame:
    """Run points_pipeline once with all test points for a given model config."""
    return pipeline.points_pipeline(
        longitudes=points_df["longitude"].values,
        latitudes=points_df["latitude"].values,
        geology_categorical_csv=cfg["geology_categorical_csv"],
        terrain_categorical_csv=cfg["terrain_categorical_csv"],
        clustered_observations_csv=cfg.get("clustered_observations_csv"),
        independent_observations_csv=cfg.get("independent_observations_csv"),
        combination_method=constants.CombinationMethod(cfg["combination_method"]),
        combine_ratio=cfg.get("combine_ratio"),
        noisy=cfg["noisy"],
        do_bayesian_update=cfg["do_bayesian_update"],
        nproc=-1,
        geology_corr_fn=cfg.get("geology_corr_fn"),
        terrain_corr_fn=cfg.get("terrain_corr_fn"),
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        apply_coastal_distance_mod=cfg["apply_coastal_distance_mod"],
        fill_gaps=cfg.get("fill_gaps", False),
    )
```

Delete the `nproc=-1,` line. Resulting block:

```python
def run_points_pipeline_for_version(cfg: dict, points_df: pd.DataFrame) -> pd.DataFrame:
    """Run points_pipeline once with all test points for a given model config."""
    return pipeline.points_pipeline(
        longitudes=points_df["longitude"].values,
        latitudes=points_df["latitude"].values,
        geology_categorical_csv=cfg["geology_categorical_csv"],
        terrain_categorical_csv=cfg["terrain_categorical_csv"],
        clustered_observations_csv=cfg.get("clustered_observations_csv"),
        independent_observations_csv=cfg.get("independent_observations_csv"),
        combination_method=constants.CombinationMethod(cfg["combination_method"]),
        combine_ratio=cfg.get("combine_ratio"),
        noisy=cfg["noisy"],
        do_bayesian_update=cfg["do_bayesian_update"],
        geology_corr_fn=cfg.get("geology_corr_fn"),
        terrain_corr_fn=cfg.get("terrain_corr_fn"),
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        apply_coastal_distance_mod=cfg["apply_coastal_distance_mod"],
        fill_gaps=cfg.get("fill_gaps", False),
    )
```

(With `nproc=-1` removed, the test now runs the sequential path. The current `points_pipeline` still accepts the `nproc` param so deleting just this argument is safe — `nproc` defaults to 1.)

- [ ] **Step 3a.2: Update `tests/test_benchmarks.py`.**

Find the test (around line 85):

```python
@pytest.mark.parametrize("nproc", [1, -1], ids=["single_process", "multiprocess"])
def test_foster_2019_approx_points_benchmark(nproc):
    """Prior-dominated points from foster_2019_approx.tif match points_pipeline output.

    Tests that the categorical posterior + hybrid slope modification
    reproduce the paper's published Vs30 raster at float precision for
    pixels outside any observation's MVN neighbourhood. Runs in both
    single-process and multiprocess modes to cover both dispatch paths.

    The assertions use median + 80th-percentile thresholds rather than a
    hard rtol because a small minority of prior-dominated pixels have
    larger discrepancies from the paper (up to ~19 % in the worst case)
    due to categorical/hybrid edge-case differences between the legacy
    R pipeline and the refactored Python one. Those outliers are
    independent of MVN conditioning and known to exist. The median must
    stay tight because any drift in the common-case codepath would show
    up there immediately.
    """
```

Three edits:

1. Delete the `@pytest.mark.parametrize` decorator line.
2. Drop the `nproc` parameter from the function signature.
3. Strip the *"Runs in both single-process and multiprocess modes to cover both dispatch paths."* sentence from the docstring.

Resulting header:

```python
def test_foster_2019_approx_points_benchmark():
    """Prior-dominated points from foster_2019_approx.tif match points_pipeline output.

    Tests that the categorical posterior + hybrid slope modification
    reproduce the paper's published Vs30 raster at float precision for
    pixels outside any observation's MVN neighbourhood.

    The assertions use median + 80th-percentile thresholds rather than a
    hard rtol because a small minority of prior-dominated pixels have
    larger discrepancies from the paper (up to ~19 % in the worst case)
    due to categorical/hybrid edge-case differences between the legacy
    R pipeline and the refactored Python one. Those outliers are
    independent of MVN conditioning and known to exist. The median must
    stay tight because any drift in the common-case codepath would show
    up there immediately.
    """
```

Then in the body of that test, find the `pipeline.points_pipeline(...)` call (around line 109-126) and delete the `nproc=nproc,` line. The call becomes:

```python
    result = pipeline.points_pipeline(
        longitudes=np.asarray(lons),
        latitudes=np.asarray(lats),
        geology_categorical_csv=cfg["geology_categorical_csv"],
        terrain_categorical_csv=cfg["terrain_categorical_csv"],
        clustered_observations_csv=cfg.get("clustered_observations_csv"),
        independent_observations_csv=cfg.get("independent_observations_csv"),
        combination_method=constants.CombinationMethod(cfg["combination_method"]),
        combine_ratio=cfg.get("combine_ratio"),
        noisy=cfg["noisy"],
        do_bayesian_update=cfg["do_bayesian_update"],
        geology_corr_fn=cfg.get("geology_corr_fn"),
        terrain_corr_fn=cfg.get("terrain_corr_fn"),
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        apply_coastal_distance_mod=cfg["apply_coastal_distance_mod"],
        fill_gaps=cfg.get("fill_gaps", False),
    )
```

- [ ] **Step 3a.3: Run the test suite to confirm test edits don't break anything.**

```bash
pytest tests/ -v
```

Expected: all tests pass. (The tests now no longer pass `nproc`, so they run the sequential path — but the multiproc path still exists in `points_pipeline`, just unused by tests.)

- [ ] **Step 3a.4: Update `vs30/pipeline.py` — `points_pipeline` signature.**

Find `points_pipeline`'s signature (around line 1219). The `nproc: int = 1,` parameter (around line 1234) gets replaced:

Before:

```python
def points_pipeline(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    apply_alluvium_slope_mod: bool,
    model_type: constants.ModelType = constants.ModelType.COMBINED,
    geology_categorical_csv: Path | None = None,
    terrain_categorical_csv: Path | None = None,
    clustered_observations_csv: Path | None = None,
    independent_observations_csv: Path | None = None,
    combination_method: constants.CombinationMethod = constants.CombinationMethod.STANDARD_DEVIATION_WEIGHTING,
    combine_ratio: float | None = None,
    noisy: bool = True,
    mvn: bool = True,
    do_bayesian_update: bool = False,
    include_intermediate: bool = False,
    nproc: int = 1,
    geology_corr_fn: Callable | None = None,
    terrain_corr_fn: Callable | None = None,
    apply_coastal_distance_mod: bool = True,
    fill_gaps: bool = False,
    gapfill_grid_config: config.GridConfig = constants.FULL_NZ_GRID_CONFIG,
) -> pd.DataFrame:
```

After (swap `nproc: int = 1,` → `dbscan_nproc: int = -1,`):

```python
def points_pipeline(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    apply_alluvium_slope_mod: bool,
    model_type: constants.ModelType = constants.ModelType.COMBINED,
    geology_categorical_csv: Path | None = None,
    terrain_categorical_csv: Path | None = None,
    clustered_observations_csv: Path | None = None,
    independent_observations_csv: Path | None = None,
    combination_method: constants.CombinationMethod = constants.CombinationMethod.STANDARD_DEVIATION_WEIGHTING,
    combine_ratio: float | None = None,
    noisy: bool = True,
    mvn: bool = True,
    do_bayesian_update: bool = False,
    include_intermediate: bool = False,
    dbscan_nproc: int = -1,
    geology_corr_fn: Callable | None = None,
    terrain_corr_fn: Callable | None = None,
    apply_coastal_distance_mod: bool = True,
    fill_gaps: bool = False,
    gapfill_grid_config: config.GridConfig = constants.FULL_NZ_GRID_CONFIG,
) -> pd.DataFrame:
```

- [ ] **Step 3a.5: Update `points_pipeline` docstring.**

Find this block in the docstring (around lines 1286-1289):

```
    nproc : int, optional
        Number of parallel processes. Default 1; set to -1 for all cores.
        See dev/docs/points_perf_post_fix_findings.md — nproc=1 wins in every
        cell tested for this pipeline.
```

Replace with:

```
    dbscan_nproc : int, optional
        Number of processes for DBSCAN clustering of clustered observations
        when do_bayesian_update is True. Default -1 (all cores). Has no
        effect when do_bayesian_update is False.
```

- [ ] **Step 3a.6: Wire `dbscan_nproc` to the two Bayesian-update call sites.**

In `points_pipeline`'s body, find the two `compute_categorical_vs30_updates(...)` calls (around lines 1355-1361 and 1374-1380). Both currently pass `dbscan_nproc=nproc`. Change both to `dbscan_nproc=dbscan_nproc`.

First occurrence (geology):

Before:

```python
        if do_bayesian_update:
            logger.info(
                "Performing Bayesian update of geology categorical model values..."
            )
            geol_model_df = compute_categorical_vs30_updates(
                categorical_model_csv=geology_categorical_csv,
                model_type=constants.ModelType.GEOLOGY,
                clustered_observations_csv=clustered_observations_csv,
                independent_observations_csv=independent_observations_csv,
                dbscan_nproc=nproc,
            )
```

After:

```python
        if do_bayesian_update:
            logger.info(
                "Performing Bayesian update of geology categorical model values..."
            )
            geol_model_df = compute_categorical_vs30_updates(
                categorical_model_csv=geology_categorical_csv,
                model_type=constants.ModelType.GEOLOGY,
                clustered_observations_csv=clustered_observations_csv,
                independent_observations_csv=independent_observations_csv,
                dbscan_nproc=dbscan_nproc,
            )
```

Second occurrence (terrain): same change — `dbscan_nproc=nproc` → `dbscan_nproc=dbscan_nproc`.

- [ ] **Step 3a.7: Delete the parallel-dispatch branch and `multiprocess.resolve_nproc` call.**

Find this block in `points_pipeline` (around lines 1400-1434):

```python
    nproc_resolved = multiprocess.resolve_nproc(nproc)

    # ================================================================
    # Parallel Processing Path
    # ================================================================
    if nproc_resolved > 1:
        logger.info(f"\nProcessing with {nproc_resolved} parallel workers...")

        loc_config = points.LocationsChunkConfig(
            include_intermediate=include_intermediate,
            model_type=model_type,
            combination_method=combination_method,
            combine_ratio=combine_ratio,
            noisy=noisy,
            geology_corr_fn=geology_corr_fn,
            terrain_corr_fn=terrain_corr_fn,
            apply_coastal_distance_mod=apply_coastal_distance_mod,
            apply_alluvium_slope_mod=apply_alluvium_slope_mod,
        )

        result_df = points.run_parallel_locations(
            points=points,
            geology_obs_data=geology_obs_data,
            terrain_obs_data=terrain_obs_data,
            geol_model_df=geol_model_df,
            terr_model_df=terr_model_df,
            config=loc_config,
            nproc=nproc_resolved,
        )

        # Add coordinate columns at the front
        result_df.insert(0, constants.ObservationColumn.EASTING, points[:, 0])
        result_df.insert(1, constants.ObservationColumn.NORTHING, points[:, 1])

        logger.info(f"  Total locations: {len(result_df)}")

    else:
        # ================================================================
        # Sequential Processing Path
        # ================================================================
        result = {}
        result[constants.ObservationColumn.EASTING] = points[:, 0]
```

Delete from `nproc_resolved = multiprocess.resolve_nproc(nproc)` through the closing `else:` line (inclusive of the section comment headers and the entire `if nproc_resolved > 1:` block). The remaining body (formerly inside the `else:`) is dedented one level. The result becomes:

```python
    result = {}
    result[constants.ObservationColumn.EASTING] = points[:, 0]
    result[constants.ObservationColumn.NORTHING] = points[:, 1]

    # --- Stage 1-3: Geology model (categorical lookup, hybrid mods, spatial adjustment) ---
    if run_geology:
        ...
```

(The `result_df = pd.DataFrame(result)` line at the end of the formerly-`else` branch stays unchanged. The downstream `if fill_gaps and model_type == constants.ModelType.COMBINED:` block at the end of `points_pipeline` is unaffected.)

**Tip:** the cleanest way is to identify the start line (`    nproc_resolved = multiprocess.resolve_nproc(nproc)`) and the end line (the closing `else:` of `if nproc_resolved > 1:`), delete everything between and including them, then dedent the remaining `result = {}` block by 4 spaces.

- [ ] **Step 3a.8: Drop the `multiprocess` import from `vs30/pipeline.py`.**

In the import block (around lines 17-27), remove the `multiprocess,` line:

Before:

```python
from vs30 import (
    category,
    config,
    constants,
    gapfill,
    multiprocess,
    points,
    raster,
    spatial,
    utils,
)
```

After:

```python
from vs30 import (
    category,
    config,
    constants,
    gapfill,
    points,
    raster,
    spatial,
    utils,
)
```

- [ ] **Step 3a.9: Update `vs30/cli.py` — `run_points_pipeline` helper.**

Find `run_points_pipeline` signature (around line 121-143). Replace `nproc: int = 1,` (around line 135) with `dbscan_nproc: int = -1,`.

In the docstring (around lines 179-182), replace this block:

```
    nproc : int, optional
        Number of parallel processes. Default 1; set to -1 for all cores.
        See dev/docs/points_perf_post_fix_findings.md — nproc=1 wins in every
        cell tested for this pipeline.
```

With:

```
    dbscan_nproc : int, optional
        Number of processes for DBSCAN clustering of clustered observations
        when do_bayesian_update is True. Default -1 (all cores). Has no
        effect when do_bayesian_update is False.
```

In the body, find the call to `pipeline.points_pipeline(...)` (around line 216). Replace `nproc=nproc,` (line 230) with `dbscan_nproc=dbscan_nproc,`.

- [ ] **Step 3a.10: Update `vs30/cli.py` — `points` command.**

Find the `points` command (around line 246). Replace the Typer option (around line 256):

Before:

```python
    nproc: typing.Annotated[int, typer.Option()] = 1,
```

After:

```python
    dbscan_nproc: typing.Annotated[int, typer.Option()] = -1,
```

In the docstring (around lines 275-278), replace:

```
    nproc : int, optional
        Number of parallel processes. Default 1; set to -1 for all cores.
        See dev/docs/points_perf_post_fix_findings.md — nproc=1 wins in every
        cell tested for this pipeline.
```

With:

```
    dbscan_nproc : int, optional
        Number of processes for DBSCAN clustering. Default -1 (all cores).
```

In the body, find the call to `run_points_pipeline(...)` (around line 282-305). Replace `nproc=nproc,` (line 297) with `dbscan_nproc=dbscan_nproc,`.

- [ ] **Step 3a.11: Update `vs30/cli.py` — `points_custom` command.**

Find the `points_custom` command (around line 308). Replace the Typer option (around line 353):

Before:

```python
    nproc: typing.Annotated[int, typer.Option()] = 1,
```

After:

```python
    dbscan_nproc: typing.Annotated[int, typer.Option()] = -1,
```

In the docstring (around lines 399-402), replace:

```
    nproc : int, optional
        Number of parallel processes. Default 1; set to -1 for all cores.
        See dev/docs/points_perf_post_fix_findings.md — nproc=1 wins in every
        cell tested for this pipeline.
```

With:

```
    dbscan_nproc : int, optional
        Number of processes for DBSCAN clustering. Default -1 (all cores).
```

In the body, find the call to `run_points_pipeline(...)` (around line 404-424). Replace `nproc=nproc,` (line 418) with `dbscan_nproc=dbscan_nproc,`.

- [ ] **Step 3a.12: Run the full test suite.**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 3a.13: Smoke-check the CLI.**

```bash
vs30 points --help 2>&1 | grep -E "nproc|dbscan"
vs30 points-custom --help 2>&1 | grep -E "nproc|dbscan"
```

Expected: each shows `--dbscan-nproc INTEGER` only. Neither shows `--nproc`.

- [ ] **Step 3a.14: Confirm no stale `nproc` references remain in `vs30/cli.py`, `vs30/pipeline.py`, or the two test files.**

```bash
grep -n "[^_a-z]nproc" vs30/cli.py vs30/pipeline.py tests/test_benchmarks.py tests/test_grid_points_consistency.py
```

Expected output (only `dbscan_nproc` matches in `vs30/cli.py`, `vs30/pipeline.py`, and various unrelated DBSCAN-context lines like `dbscan_nproc=dbscan_nproc`). Verify no plain `nproc` (not preceded by `_` or `dbscan_`) remains in these files. The grep negates `[^_a-z]nproc` — any plain `nproc` token that isn't part of `dbscan_nproc` should appear here only as `dbscan_nproc=X` patterns.

If unsure, use a stricter check:

```bash
grep -n -E "(^|[^a-zA-Z_])nproc([^a-zA-Z_]|$)" vs30/cli.py vs30/pipeline.py tests/test_benchmarks.py tests/test_grid_points_consistency.py
```

Expected: zero hits in `cli.py`, `pipeline.py`, `test_benchmarks.py`, `test_grid_points_consistency.py`. (Hits in `vs30/points.py` are still expected — they go away in Task 3b.)

- [ ] **Step 3a.15: Commit.**

```bash
git add vs30/pipeline.py vs30/cli.py tests/test_benchmarks.py tests/test_grid_points_consistency.py
git commit -m "$(cat <<'EOF'
refactor(points): remove nproc; add dbscan_nproc; delete parallel branch

Per dev/docs/points_perf_post_fix_findings.md §5.1, nproc=1 wins every
(N_query, N_obs) cell tested. Drop the multiprocessing dispatch in
points_pipeline along with the nproc parameter on points_pipeline,
run_points_pipeline, and the points / points-custom CLI commands.

Add a dbscan_nproc parameter (default -1) for DBSCAN parallelism when
do_bayesian_update is True, mirroring grid_pipeline's pattern.

Tests updated to drop the now-irrelevant single-process / multiprocess
parametrization. The dead helper symbols (LocationsChunkConfig,
process_locations_chunk, run_parallel_locations) inside vs30/points.py
become unreachable after this commit and are removed in the next.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3b: Strip dead multiprocessing code from `vs30/points.py`

**Goal:** Remove the symbols inside `points.py` that are now unreachable: `LocationsChunkConfig`, `process_locations_chunk`, `run_parallel_locations`. Drop the `multiprocess` and `utils` imports (both unused after the deletion). Update the module docstring.

**Files:**
- Modify: `vs30/points.py`

- [ ] **Step 3b.1: Confirm the symbols are truly unreachable from production code.**

```bash
grep -rn "LocationsChunkConfig\|process_locations_chunk\|run_parallel_locations" --include="*.py"
```

Expected: hits only inside `vs30/points.py` itself (their definitions) and possibly in `dev/scripts/investigations/points_features_investigation/` (those scripts are deleted in Task 7). Zero hits in `vs30/pipeline.py`, `vs30/cli.py`, `tests/`.

- [ ] **Step 3b.2: Edit `vs30/points.py`.**

Three edits:

1. **Module docstring** — replace line 1:

   Before:
   ```python
   """Multiprocessing support for parallel spatial adjustment."""
   ```

   After:
   ```python
   """Per-point pipeline helpers: observation-side precomputation and per-query-point geology/terrain processing."""
   ```

2. **Imports** — find this block (around lines 3-10):

   ```python
   from collections.abc import Callable
   from dataclasses import dataclass

   import numpy as np
   import pandas as pd
   from tqdm import tqdm

   from vs30 import category, constants, multiprocess, raster, spatial, utils
   ```

   Drop `multiprocess` and `utils` (both unused after the deletions below):

   ```python
   from collections.abc import Callable
   from dataclasses import dataclass

   import numpy as np
   import pandas as pd
   from tqdm import tqdm

   from vs30 import category, constants, raster, spatial
   ```

3. **Delete the trailing block** — every line from `@dataclass` (around line 320, the start of `LocationsChunkConfig`) to the end of the file (closing `return pd.concat(...)` of `run_parallel_locations`, around line 538). The file now ends with the closing of `process_terrain_at_points` (around line 317).

Use `wc -l` to verify the file shrank significantly:

```bash
wc -l vs30/points.py
```

Expected: ~317 lines (down from ~539).

- [ ] **Step 3b.3: Verify Python can still import the module.**

```bash
python -c "from vs30 import points; print(points.PointsObsData, points.prepare_geology_obs_data, points.prepare_terrain_obs_data, points.process_geology_at_points, points.process_terrain_at_points)"
```

Expected: prints the five symbols' repr without raising.

- [ ] **Step 3b.4: Confirm the deleted symbols are gone.**

```bash
python -c "from vs30 import points; print(hasattr(points, 'LocationsChunkConfig'), hasattr(points, 'process_locations_chunk'), hasattr(points, 'run_parallel_locations'))"
```

Expected: `False False False`.

- [ ] **Step 3b.5: Run the full test suite.**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 3b.6: Commit.**

```bash
git add vs30/points.py
git commit -m "$(cat <<'EOF'
refactor(points): drop unreachable multiproc helpers

After removing the parallel dispatch path, LocationsChunkConfig,
process_locations_chunk, and run_parallel_locations are dead code.
Drop them along with the now-unused multiprocess and utils imports.
Update the module docstring to reflect what's left.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Delete `vs30/multiprocess.py` and `N_PROGRESS_CHUNKS`

**Goal:** Remove the now-orphaned `vs30/multiprocess.py` module and the `N_PROGRESS_CHUNKS` constant.

**Files:**
- Delete: `vs30/multiprocess.py`
- Modify: `vs30/constants.py` (remove one constant line)

- [ ] **Step 4.1: Confirm `vs30/multiprocess.py` has zero callers in production code or tests.**

```bash
grep -rn "from vs30 import.*multiprocess\|from vs30\.multiprocess\|vs30\.multiprocess" vs30/ tests/ --include="*.py"
```

Expected: zero hits. (Hits in `dev/scripts/investigations/points_features_investigation/run_balanced_blas_supplement.py` are acceptable — that script gets deleted in Task 7.)

- [ ] **Step 4.2: Delete the file.**

```bash
git rm vs30/multiprocess.py
```

- [ ] **Step 4.3: Confirm `N_PROGRESS_CHUNKS` has zero callers.**

```bash
grep -rn "N_PROGRESS_CHUNKS" --include="*.py"
```

Expected: only one hit, the definition line in `vs30/constants.py:279`.

- [ ] **Step 4.4: Remove the constant from `vs30/constants.py`.**

Open `vs30/constants.py`, find around line 279:

```python
N_PROGRESS_CHUNKS: int = 1000
```

Delete that line. If there's a comment immediately above explaining the constant, delete it too. (Check by reading 5 lines of context around line 279 first.)

- [ ] **Step 4.5: Verify Python can still import constants.**

```bash
python -c "from vs30 import constants; print('OK')"
```

Expected: prints `OK`.

- [ ] **Step 4.6: Run the full test suite.**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 4.7: Run the broader verification grep.**

```bash
grep -rn "run_parallel_locations\|process_locations_chunk\|LocationsChunkConfig\|N_PROGRESS_CHUNKS\|resolve_nproc\|spawn_context\|limit_blas_threads" vs30/ tests/ --include="*.py"
```

Expected: zero hits inside `vs30/` and `tests/`. (Hits in `dev/scripts/investigations/points_features_investigation/` are still expected at this stage; they get deleted in Task 7.)

- [ ] **Step 4.8: Commit.**

```bash
git add vs30/multiprocess.py vs30/constants.py
git commit -m "$(cat <<'EOF'
refactor: delete vs30/multiprocess.py and N_PROGRESS_CHUNKS

Both became orphaned after the points-pipeline multiprocessing removal:
spawn_context, limit_blas_threads, and resolve_nproc had no remaining
production callers, and N_PROGRESS_CHUNKS was used only inside the
deleted run_parallel_locations.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Extract grid-stage helpers into `vs30/grid.py`

**Goal:** Move five symbols from `vs30/pipeline.py` to a new `vs30/grid.py`. Update call-sites in `pipeline.py` (inside `compute_model_grid` and `grid_pipeline`).

**Symbols moved:** `create_initial_vs30_arrays`, `compute_hybrid_geology_arrays`, `compute_spatial_adjustment_on_grid`, `combine_model_arrays`, `write_raster`. (`compute_model_grid` stays in `pipeline.py` per the design refinement — it orchestrates Stage 1 → Stages 2-5 and would otherwise create a circular import.)

**Files:**
- Create: `vs30/grid.py`
- Modify: `vs30/pipeline.py` (delete moved bodies, update imports, update call-sites with `grid.` prefix)

- [ ] **Step 5.1: Create `vs30/grid.py` with the new module header.**

Write this file (full content shown — copy verbatim, then paste the moved function bodies under it):

```python
"""Grid-pipeline stage helpers (initial arrays, hybrid mods, spatial adjustment, combination, raster writing)."""

import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from vs30 import config, constants, raster, spatial, utils

logger = logging.getLogger(__name__)
```

- [ ] **Step 5.2: Move `create_initial_vs30_arrays` from `pipeline.py` to `grid.py`.**

In `vs30/pipeline.py`, locate the function (around lines 286-335). Cut the entire definition (including the `# ====` banner immediately above it labelled *Stage 2: Create initial VS30 arrays from categorical model*).

In `vs30/grid.py`, paste the function definition (without the banner — `grid.py` is purely stage helpers, so the Stage-N banners are noise there).

- [ ] **Step 5.3: Move `compute_hybrid_geology_arrays`.**

Same procedure: cut from `pipeline.py` (lines 343-407, plus its Stage 3 banner), paste body into `grid.py`.

- [ ] **Step 5.4: Move `compute_spatial_adjustment_on_grid`.**

Cut from `pipeline.py` (lines 414-549, plus its Stage 4 banner), paste body into `grid.py`.

- [ ] **Step 5.5: Move `combine_model_arrays`.**

Cut from `pipeline.py` (lines 557-609, plus its Stage 5 banner), paste body into `grid.py`.

- [ ] **Step 5.6: Move `write_raster`.**

Cut from `pipeline.py` (lines 617-664, plus its `# ====` banner labelled *Raster file writing helper*), paste body into `grid.py`.

- [ ] **Step 5.7: Verify `vs30/grid.py` imports cleanly.**

```bash
python -c "from vs30 import grid; print(grid.create_initial_vs30_arrays, grid.compute_hybrid_geology_arrays, grid.compute_spatial_adjustment_on_grid, grid.combine_model_arrays, grid.write_raster)"
```

Expected: prints the five function reprs without raising.

- [ ] **Step 5.8: Update `vs30/pipeline.py` imports.**

In the import block, drop `raster,` and `spatial,` (no longer used in `pipeline.py` after the move) and add `grid,`. Result:

Before:

```python
from vs30 import (
    category,
    config,
    constants,
    gapfill,
    points,
    raster,
    spatial,
    utils,
)
```

After:

```python
from vs30 import (
    category,
    config,
    constants,
    gapfill,
    grid,
    points,
    utils,
)
```

(`utils` stays — `compute_categorical_vs30_updates` and `load_and_assign_observations` use `utils.validate_csv_columns`.)

- [ ] **Step 5.9: Update `compute_model_grid` call-sites in `pipeline.py`.**

Inside `compute_model_grid`, six function calls now need a `grid.` prefix. Locate each and update:

| Approximate line | From | To |
|---|---|---|
| ~778 | `vs30_array, stdv_array, id_array, profile = create_initial_vs30_arrays(` | `vs30_array, stdv_array, id_array, profile = grid.create_initial_vs30_arrays(` |
| ~788 | `write_raster(` | `grid.write_raster(` |
| ~802 | `write_raster(` | `grid.write_raster(` |
| ~818 | `compute_hybrid_geology_arrays(` | `grid.compute_hybrid_geology_arrays(` |
| ~830 | `write_raster(` | `grid.write_raster(` |
| ~836 | `write_raster(` | `grid.write_raster(` |
| ~843 | `write_raster(` | `grid.write_raster(` |
| ~868 | `current_vs30, current_stdv = compute_spatial_adjustment_on_grid(` | `current_vs30, current_stdv = grid.compute_spatial_adjustment_on_grid(` |
| ~889 | `write_raster(` | `grid.write_raster(` |

(Exact line numbers will drift because of the deletions; use the surrounding context to locate each call.)

- [ ] **Step 5.10: Update `grid_pipeline` call-sites in `pipeline.py`.**

Inside `grid_pipeline` (around lines 905-1143 originally; lower after the cuts), update:

| Approximate function | From | To |
|---|---|---|
| `combine_model_arrays(` (~1073) | `combine_model_arrays(` | `grid.combine_model_arrays(` |
| `write_raster(` (~1091) | `write_raster(` | `grid.write_raster(` |
| `write_raster(` (~1119) | `write_raster(` | `grid.write_raster(` |

- [ ] **Step 5.11: Verify no stragglers in `pipeline.py`.**

```bash
grep -nE "(^|[^.a-zA-Z_])(create_initial_vs30_arrays|compute_hybrid_geology_arrays|compute_spatial_adjustment_on_grid|combine_model_arrays|write_raster)\(" vs30/pipeline.py
```

Expected: every hit is preceded by `grid.` (i.e. `grid.create_initial_vs30_arrays(`, etc.). If any bare call remains (without the `grid.` prefix), update it.

- [ ] **Step 5.12: Verify Python can import everything cleanly.**

```bash
python -c "from vs30 import pipeline; print(pipeline.grid_pipeline, pipeline.points_pipeline, pipeline.compute_model_grid, pipeline.fill_one_point_via_local_grid)"
```

Expected: prints the four function reprs without raising.

- [ ] **Step 5.13: Run the full test suite.**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 5.14: Run the verification greps.**

```bash
grep -rn "from vs30 import.*grid\|from vs30\.grid" vs30/ tests/ --include="*.py"
grep -rn "from vs30 import.*points\|from vs30\.points" vs30/ tests/ --include="*.py"
```

Expected: each shows hits only in `vs30/pipeline.py` (the import statement).

- [ ] **Step 5.15: Commit.**

```bash
git add vs30/grid.py vs30/pipeline.py
git commit -m "$(cat <<'EOF'
refactor: extract grid-stage helpers from pipeline.py to vs30/grid.py

Move create_initial_vs30_arrays, compute_hybrid_geology_arrays,
compute_spatial_adjustment_on_grid, combine_model_arrays, and
write_raster to a new vs30/grid.py module. compute_model_grid stays in
pipeline.py because it calls Stage 1 (compute_categorical_vs30_updates)
and Stages 2-5 (now in grid.py); putting it in grid.py would create a
circular import.

Result: pipeline.py contains orchestrators + shared utilities; grid.py
contains grid-pipeline stage helpers; points.py (renamed earlier from
parallel.py) contains points-pipeline per-point helpers. One-way
dependency pipeline -> grid and pipeline -> points.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: Delete dev investigation runner scripts

**Goal:** Remove the four runner/helper scripts in `dev/scripts/investigations/points_features_investigation/` that would silently break after the parameter removal. Keep the analysis scripts, result CSVs, figures, sweep logs, and README — the historical record.

**Files (deleted):**
- `dev/scripts/investigations/points_features_investigation/run_balanced_blas_supplement.py`
- `dev/scripts/investigations/points_features_investigation/run_points_sweep.py`
- `dev/scripts/investigations/points_features_investigation/bench_utils.py`
- `dev/scripts/investigations/points_features_investigation/test_bench_utils.py`

- [ ] **Step 6.1: Delete the four scripts.**

```bash
git rm dev/scripts/investigations/points_features_investigation/run_balanced_blas_supplement.py \
       dev/scripts/investigations/points_features_investigation/run_points_sweep.py \
       dev/scripts/investigations/points_features_investigation/bench_utils.py \
       dev/scripts/investigations/points_features_investigation/test_bench_utils.py
```

- [ ] **Step 6.2: Verify the analysis scripts are still intact and the result data is preserved.**

```bash
ls dev/scripts/investigations/points_features_investigation/
```

Expected files (note: order may vary):

```
README.md
__pycache__
analyze_points_post_fix_results.py
analyze_points_results.py
figures
obs_csvs
results_balanced_blas_supplement.csv
results_points.csv
results_points_medians.csv
results_points_partial_with_buggy_nproc8.csv
results_points_post_fix.csv
results_points_post_fix_combined_medians.csv
results_points_post_fix_combined_wide.csv
results_points_post_fix_medians.csv
results_points_post_fix_speedup.csv
results_points_speedup.csv
sweep.log
sweep_balanced_blas_supplement.log
sweep_post_fix.log
```

- [ ] **Step 6.3: Run the verification grep again — should now be fully clean.**

```bash
grep -rn "run_parallel_locations\|process_locations_chunk\|LocationsChunkConfig\|N_PROGRESS_CHUNKS\|resolve_nproc\|spawn_context\|limit_blas_threads" --include="*.py"
```

Expected: zero hits anywhere.

```bash
grep -rn "from vs30 import.*multiprocess\|from vs30\.multiprocess\|from vs30 import.*parallel\|from vs30\.parallel" --include="*.py"
```

Expected: zero hits anywhere.

- [ ] **Step 6.4: Run the full test suite (sanity check — these scripts aren't on the test path, but verify nothing else broke).**

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 6.5: Commit.**

```bash
git commit -m "$(cat <<'EOF'
chore(dev): delete points-pipeline sweep runner scripts

The runners (run_points_sweep.py, run_balanced_blas_supplement.py,
bench_utils.py, test_bench_utils.py) all called points_pipeline with
nproc= and would silently break after the parameter removal. The
analysis scripts (analyze_points_results.py,
analyze_points_post_fix_results.py), result CSVs, figures, sweep logs,
and README are preserved — the findings doc stays the authoritative
historical record and the analyses remain runnable against the saved
CSVs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Add findings-doc postscript

**Goal:** Document at the top of `dev/docs/points_perf_post_fix_findings.md` that the §5.1 recommendation was superseded by full removal.

**Files:**
- Modify: `dev/docs/points_perf_post_fix_findings.md`

- [ ] **Step 7.1: Find the multiproc-removal commit hash.**

```bash
git log --oneline -10 | grep -iE "remove nproc|multiproc"
```

Identify the SHA of the commit from Task 3a (subject begins `refactor(points): remove nproc; add dbscan_nproc; delete parallel branch`). Note the SHA for the postscript text.

- [ ] **Step 7.2: Edit `dev/docs/points_perf_post_fix_findings.md`.**

Find this line near the top of the file (line 5):

```
**Status:** Complete — sweep done, balanced-BLAS supplement done, analysis written, CLI default change recommended.
```

Just below the **Predecessors** block (around line 11, after the closing predecessor bullet point and before the `## 1. Summary` heading), insert this postscript section:

```markdown
## Postscript (2026-04-29)

§5.1 recommended changing the CLI default from `-1` to `1`. That landed in
commit `3d0b190`. After further consideration, the multiprocessing path was
removed entirely (see commit `<HASH-FROM-STEP-7.1>`). The `nproc` parameter on
`points_pipeline` / `vs30 points` / `vs30 points-custom` is gone;
`dbscan_nproc` was added to the points commands to preserve user control over
DBSCAN parallelism in the bayesian-update path (mirroring `grid_pipeline`).
The §3.x data and §5.2/§5.3 conclusions are unchanged. The runner scripts in
`dev/scripts/investigations/points_features_investigation/`
(`run_points_sweep.py`, `run_balanced_blas_supplement.py`, `bench_utils.py`,
`test_bench_utils.py`) were also removed — see git history if a re-run is
needed.

```

Replace `<HASH-FROM-STEP-7.1>` with the actual SHA.

- [ ] **Step 7.3: Commit.**

```bash
git add dev/docs/points_perf_post_fix_findings.md
git commit -m "$(cat <<'EOF'
docs(findings): postscript — points-pipeline multiproc removed

§5.1 recommended dropping the CLI default to 1; the multiprocessing
path has now been removed entirely. Note the supersession at the top of
the findings doc so a future reader doesn't act on the old §5.1
recommendation.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: Slow-tier reassessment

**Goal:** Time the slow tier and decide whether to remove the `@pytest.mark.slow` infrastructure (per design §2.10, threshold ~15 min).

**Files (conditional):**
- Modify: `tests/test_grid_points_consistency.py` (decorator + docstring)
- Modify: `tests/conftest.py` (delete `pytest_addoption`, `pytest_collection_modifyitems`)
- Modify: `dev/CLAUDE.md` (Running Tests section)

- [ ] **Step 8.1: Time the slow tier.**

```bash
time pytest tests/test_grid_points_consistency.py::test_grid_points_consistency_slow --runslow -v
```

Wait for completion. Note the wall-clock from `time`'s `real` line.

- [ ] **Step 8.2: Decide which branch to take.**

Apply the design's decision rule:

- If **measured time < 15 min**: take **Branch A** (remove infrastructure). Continue with steps 8.3–8.7.
- If **measured time ≥ 15 min**: take **Branch B** (keep marker, update docs). Skip to step 8.8.

### Branch A — measured time under 15 min: remove the slow infrastructure

- [ ] **Step 8.3a: Drop the `@pytest.mark.slow` decorator from `tests/test_grid_points_consistency.py`.**

Find this block at the bottom (around lines 213-218):

```python
@pytest.mark.slow
@pytest.mark.parametrize("version", ALL_VERSIONS, ids=lambda v: v.value)
def test_grid_points_consistency_slow(version):
    """Full grid/points consistency: all 38 points for all 4 model versions."""
    check_consistency_for_version(version)
```

Delete the `@pytest.mark.slow` line. (Optional: rename `test_grid_points_consistency_slow` to `test_grid_points_consistency_full`, since the "slow" qualifier no longer applies. If you rename, update the docstring on the same line accordingly.)

If you keep the name `test_grid_points_consistency_slow`, leave the function body otherwise untouched. The test now runs in the default tier.

- [ ] **Step 8.4a: Update the test module docstring.**

In `tests/test_grid_points_consistency.py`, find the module docstring (lines 1-15):

```python
"""
Test that grid and points pipelines produce consistent Vs30 values
across the full NZ domain for all fixed model versions.

For each test point, a tiny 3x3 grid (300m x 300m at 100m resolution) is
generated and run through grid_pipeline.  The center pixel is compared
against the batched points_pipeline result at the same coordinates.

The test is split into two tiers:
- Fast tier: smoke test over 3 representative cities for the two
  model versions without coastal distance (foster_2019_approx, jaehwi_v1p0).
  Runs in ~45 s and is enabled by default.
- Slow tier (--runslow): full 38-point coverage for all 4 model versions.
  Runs in ~35-45 minutes.
"""
```

Replace with (incorporating the measured runtime — substitute `<MEASURED>` with the actual minutes-and-seconds from step 8.1):

```python
"""
Test that grid and points pipelines produce consistent Vs30 values
across the full NZ domain for all fixed model versions.

For each test point, a tiny 3x3 grid (300m x 300m at 100m resolution) is
generated and run through grid_pipeline.  The center pixel is compared
against the batched points_pipeline result at the same coordinates.

The test is split into two test functions:
- Fast: smoke test over 3 representative cities for the two model
  versions without coastal distance (foster_2019_approx, jaehwi_v1p0).
  Runs in ~45 s.
- Full: 38-point coverage for all 4 model versions.
  Runs in ~<MEASURED>.
"""
```

- [ ] **Step 8.5a: Delete `pytest_addoption` and `pytest_collection_modifyitems` from `tests/conftest.py`.**

Find the block (lines 17-31):

```python
def pytest_addoption(parser):
    """Register the --runslow command-line option."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests marked @pytest.mark.slow unless --runslow is given."""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="use --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)
```

Delete this entire block. Also confirm whether `pytest` is still imported elsewhere in `conftest.py` — yes, it is (for `pytest.exit`, `pytest.fixture`, `pytest.approx`), so leave the `import pytest` line alone.

- [ ] **Step 8.6a: Update `dev/CLAUDE.md` Running Tests section.**

Find this block in `dev/CLAUDE.md`:

```markdown
## Running Tests

The suite has two tiers:

```bash
# Default — unit tests, benchmark tests, 3-city grid/points smoke. ~3 min.
pytest tests/

# Default + full 38-point grid/points consistency across all 4 model versions. ~43 min.
pytest tests/ --runslow
```

Tests decorated with `@pytest.mark.slow` are skipped unless `--runslow` is
passed (see `conftest.py::pytest_collection_modifyitems`).
```

Replace with (substitute `<NEW-TOTAL>` with the runtime of `pytest tests/` after step 8.5a's changes — re-time it in step 8.7a):

```markdown
## Running Tests

```bash
pytest tests/
```

Runs in ~<NEW-TOTAL>. Includes unit tests, benchmark tests, the 3-city
grid/points smoke check, and the full 38-point grid/points consistency
across all 4 model versions.
```

- [ ] **Step 8.7a: Re-time the now-merged tier and substitute.**

```bash
time pytest tests/ -v
```

Note the `real` wall-clock. Substitute it into `dev/CLAUDE.md` (replacing `<NEW-TOTAL>`) and into the test module docstring (replacing `<MEASURED>` with the slow-tier-only number from step 8.1, if you prefer to break out the two parts separately, or update both to the merged number — your call; the docstring's intent is to give a reader a sense of cost).

Then run the suite once more (now without `--runslow` since the flag is gone):

```bash
pytest tests/ -v
```

Expected: all tests pass.

- [ ] **Step 8.8a: Commit Branch A.**

```bash
git add tests/test_grid_points_consistency.py tests/conftest.py dev/CLAUDE.md
git commit -m "$(cat <<'EOF'
test: remove slow-tier infrastructure

The full 38-point grid/points consistency suite now runs in well under
15 min after recent perf work. Drop @pytest.mark.slow and the
--runslow plumbing in conftest.py — one merged tier is simpler.
Update CLAUDE.md and the test module docstring with the new runtime.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Skip to Task 9.

### Branch B — measured time 15 min or more: keep the marker

- [ ] **Step 8.8b: Update `dev/CLAUDE.md` Running Tests section with the measured time.**

Find the block:

```markdown
# Default + full 38-point grid/points consistency across all 4 model versions. ~43 min.
pytest tests/ --runslow
```

Replace the `~43 min` with the actual measured wall-clock from step 8.1.

- [ ] **Step 8.9b: Commit Branch B.**

```bash
git add dev/CLAUDE.md
git commit -m "$(cat <<'EOF'
docs(CLAUDE): update slow-tier runtime estimate

Measured slow-tier runtime is now ~<MEASURED> (was ~43 min before
recent perf fixes). The slow infrastructure stays; the threshold for
removing it (~15 min) was not met.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

Continue to Task 9.

---

## Task 9: Final verification

**Goal:** Single end-to-end pass of the design's verification plan (§4) to confirm everything is consistent.

**Files:** None modified (verification only).

- [ ] **Step 9.1: Full test suite.**

If Branch A was taken in Task 8:

```bash
pytest tests/ -v
```

If Branch B was taken in Task 8:

```bash
pytest tests/ -v --runslow
```

Expected: all tests pass.

- [ ] **Step 9.2: Verification greps — must all return zero hits.**

```bash
grep -rn "run_parallel_locations\|process_locations_chunk\|LocationsChunkConfig\|N_PROGRESS_CHUNKS\|resolve_nproc\|spawn_context\|limit_blas_threads" --include="*.py"
```

Expected: zero hits.

```bash
grep -rn "from vs30 import.*multiprocess\|from vs30\.multiprocess\|from vs30 import.*parallel\|from vs30\.parallel" --include="*.py"
```

Expected: zero hits.

- [ ] **Step 9.3: New-import grep — only in `pipeline.py`.**

```bash
grep -rn "from vs30 import.*points\|from vs30\.points" vs30/ tests/ --include="*.py"
grep -rn "from vs30 import.*grid\|from vs30\.grid" vs30/ tests/ --include="*.py"
```

Expected: each shows hits only inside `vs30/pipeline.py`.

- [ ] **Step 9.4: CLI smoke check.**

```bash
vs30 points --help 2>&1 | grep -E "nproc|dbscan"
vs30 points-custom --help 2>&1 | grep -E "nproc|dbscan"
```

Expected: each shows `--dbscan-nproc INTEGER` only. Neither shows `--nproc`.

- [ ] **Step 9.5: Eyeball the final file structure.**

```bash
ls vs30/
wc -l vs30/pipeline.py vs30/grid.py vs30/points.py
```

Expected files in `vs30/`: `category.py`, `cli.py`, `config.py`, `constants.py`, `gapfill.py`, `grid.py` (new), `pipeline.py`, `points.py` (renamed from `parallel.py`), `raster.py`, `spatial.py`, `utils.py`. No `parallel.py`. No `multiprocess.py`.

Expected line counts approximately:
- `pipeline.py`: ~1,140 lines
- `grid.py`: ~355 lines
- `points.py`: ~317 lines

- [ ] **Step 9.6: Confirm git history is clean.**

```bash
git log --oneline | head -15
```

Expected: a clean sequence of focused commits, one per task. Each commit subject names what changed in scope.

---

## Self-review notes (for the executing engineer)

After Task 9 passes:

1. **Spec coverage:** every section/requirement of `dev/docs/remove_points_multiproc_design.md` should map to a specific task in this plan.
2. **Functional behavior:** the surviving sequential-processing path inside `points_pipeline` is unchanged. Tests cover it via `test_foster_2019_approx_points_benchmark` (3 cities × 38 points smoke + benchmarks) and the grid/points consistency tests.
3. **Reviewability:** each commit is small enough to review independently. If a task's diff feels too big, split it (e.g., Task 3a could split into "tests + cli.py" and "pipeline.py" if reviewers prefer).
4. **Rollback:** if any task lands and reveals an issue, `git revert <SHA>` is safe — earlier tasks don't depend on later ones. Tasks 2, 3a, 3b, 4 must land in order; Task 5 can land before or after Task 6/7/8 functionally; Task 7 needs Task 3a's commit hash; Task 8 must come after the multiproc removal (otherwise the slow tier still runs the buggy parallel path).

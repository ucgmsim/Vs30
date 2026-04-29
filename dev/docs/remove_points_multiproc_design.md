# Remove multiprocessing from the points pipeline — design

**Date:** 2026-04-29
**Branch:** `vs30_refactor`
**Status:** Design — pending implementation plan.
**Predecessors:**
- [Points-mode post-fix performance findings](points_perf_post_fix_findings.md) — established that `nproc=1` wins every (N_query × N_obs) cell tested, by 1.05× to >180×.
- [Points-pipeline obs-prep fix — design](parallel_points_obs_prep_fix_design.md) — predecessor fix that closed the per-chunk obs-prep redundancy bug.

## 1. Goal

Remove the multiprocessing dispatch path from the points pipeline. The post-fix
sweep (predecessor findings doc §5.1) showed `nproc=1` wins every cell tested
across all 21 (N_query × N_obs) cells and all 7 nproc/BLAS-thread
configurations. The recommendation in §5.1 was to change the CLI default from
`-1` to `1` (which landed in commit `3d0b190`); this design goes further and
removes the multiprocessing path entirely. The sequential path already covers
everything functionally — keeping the parallel branch would just be dead weight
that misleads future readers and bloats the test matrix.

The work also takes the opportunity to:
- Rename `vs30/parallel.py` → `vs30/points.py` (the file no longer does any
  parallel work).
- Extract grid-pipeline stage helpers from `vs30/pipeline.py` into a new
  `vs30/grid.py`, mirroring the points/grid module split. `pipeline.py` keeps
  the top-level orchestrators and shared utilities.
- Reassess whether `tests/test_grid_points_consistency_slow` still earns its
  `@pytest.mark.slow` marker after recent perf work.

## 2. Scope of changes

### 2.1 New module: `vs30/points.py`

Renamed from `parallel.py`. Multiprocessing plumbing removed. Surviving symbols
move verbatim:

- `PointsObsData` dataclass (with `empty()` classmethod)
- `prepare_geology_obs_data`
- `prepare_terrain_obs_data`
- `process_geology_at_points`
- `process_terrain_at_points`

Removed (relative to current `parallel.py`):

- `LocationsChunkConfig` dataclass
- `process_locations_chunk` worker function
- `run_parallel_locations` dispatcher

Module docstring updates from
`"Multiprocessing support for parallel spatial adjustment."` to something
accurate, e.g. *"Per-point pipeline helpers: observation-side precomputation
and per-query-point geology/terrain processing."* The `multiprocess` import
goes away.

### 2.2 New module: `vs30/grid.py`

Extracted from `pipeline.py`. Symbols move verbatim:

- `create_initial_vs30_arrays` (current `pipeline.py:286-335`) — Stage 2
- `compute_hybrid_geology_arrays` (current `pipeline.py:343-407`) — Stage 3
- `compute_spatial_adjustment_on_grid` (current `pipeline.py:414-549`) — Stage 4
- `combine_model_arrays` (current `pipeline.py:557-609`) — Stage 5
- `write_raster` (current `pipeline.py:617-664`)

Plus their imports re-declared in `grid.py`. New module docstring describes
*"Grid-pipeline stage helpers (initial arrays, hybrid mods, spatial adjustment,
combination, raster writing)."*

`compute_model_grid` (the per-model orchestrator at `pipeline.py:672-897`)
intentionally **stays in `pipeline.py`** — it calls Stage 1
(`compute_categorical_vs30_updates`, lives in `pipeline.py`) and Stages 2-5
(now in `grid.py`). Putting it in `grid.py` would introduce a circular import
(`grid.py` → `pipeline.py` for Stage 1; `pipeline.py` → `grid.py` for Stages
2-5). Keeping orchestrators in `pipeline.py` and stage helpers in `grid.py` /
`points.py` gives a clean one-way dependency: `pipeline → grid` and
`pipeline → points`, with no back-edges.

### 2.3 `vs30/pipeline.py` — slimmed to orchestrators + shared utilities

Stays:

- Shared utilities: `read_observations_csv`, `read_categorical_csv`,
  `default_correlation_functions`, `load_and_assign_observations`,
  `collect_observation_csvs` (used by both pipelines).
- `compute_categorical_vs30_updates` (Stage 1 — used by both `points_pipeline`
  and `compute_model_grid`).
- `compute_model_grid` (per-model grid orchestrator — stays here per §2.2's
  rationale; now calls Stages 2-5 via `grid.X`).
- `grid_pipeline` (top-level grid orchestrator). Now imports
  `from vs30 import grid` and calls `grid.combine_model_arrays`,
  `grid.write_raster` directly; reaches Stages 2-5 indirectly via
  `compute_model_grid`.
- `fill_one_point_via_local_grid` (calls `grid_pipeline` from same module —
  unchanged).
- `points_pipeline` (top-level points orchestrator). Now imports
  `from vs30 import points` instead of `parallel`.

Changes inside `points_pipeline`:

- Drop `nproc: int = 1` parameter.
- Add `dbscan_nproc: int = -1` parameter (mirrors `grid_pipeline`).
- Wire `dbscan_nproc` to the two `compute_categorical_vs30_updates(...)` calls
  in the bayesian-update branches (currently they pass `nproc`, which conflated
  two unrelated concerns).
- Delete the entire `if nproc_resolved > 1:` branch (current
  `pipeline.py:1400-1434`). The remaining `else` (sequential) branch becomes
  the single path.
- Drop the `multiprocess` import; drop the `multiprocess.resolve_nproc(nproc)`
  call.
- Update the docstring's `nproc` section to describe `dbscan_nproc` instead;
  drop the post-fix-findings reference (the param it documented is gone).

### 2.4 `vs30/multiprocess.py` — delete

After the changes above, none of its three exports (`spawn_context`,
`limit_blas_threads`, `resolve_nproc`) have any remaining production callers.
The only external caller (`run_balanced_blas_supplement.py` in dev/) is being
deleted in §2.7.

### 2.5 `vs30/constants.py`

- Remove `N_PROGRESS_CHUNKS` (only used by the deleted `run_parallel_locations`).

### 2.6 `vs30/cli.py`

- `points` command (`cli.py:246-305`): replace `nproc: int = 1` Typer option
  with `dbscan_nproc: int = -1` (matching `grid` command's flag name and
  default).
- `points_custom` command (`cli.py:308-424`): same swap.
- `run_points_pipeline` helper (`cli.py:121-243`): same swap. All call-site
  forwarding updated.
- Drop the post-fix-findings note from each `nproc` docstring; replace with a
  standard `dbscan_nproc` description matching the grid commands'.

### 2.7 Tests

`tests/test_benchmarks.py`:

- Drop `@pytest.mark.parametrize("nproc", [1, -1], ...)` from
  `test_foster_2019_approx_points_benchmark` (line 85).
- Drop the `nproc` parameter on the test function and the `nproc=nproc` arg in
  the `points_pipeline(...)` call.
- Strip the *"Runs in both single-process and multiprocess modes to cover both
  dispatch paths."* sentence from the docstring.

`tests/test_grid_points_consistency.py`:

- Drop `nproc=-1` argument from the `points_pipeline(...)` call in
  `run_points_pipeline_for_version` (line 87). No other test changes.

### 2.8 Dev investigation cleanup

Delete (would silently break after the parameter removal — they call
`points_pipeline` with `nproc=`):

- `dev/scripts/investigations/points_features_investigation/run_balanced_blas_supplement.py`
- `dev/scripts/investigations/points_features_investigation/run_points_sweep.py`
- `dev/scripts/investigations/points_features_investigation/bench_utils.py`
- `dev/scripts/investigations/points_features_investigation/test_bench_utils.py`

Keep (still functional + reproducibility value — pure CSV consumers, all
result data, README):

- `analyze_points_results.py`, `analyze_points_post_fix_results.py`
- `results_*.csv`, `figures/`, `sweep*.log`, `obs_csvs/`, `README.md`

### 2.9 Findings doc postscript

Add a postscript paragraph at the top of `dev/docs/points_perf_post_fix_findings.md`
(just under the **Status** line):

> **Postscript (2026-04-29):** §5.1 recommended changing the CLI default from
> `-1` to `1`. That landed in commit `3d0b190`. After further consideration,
> the multiprocessing path was removed entirely (see commit `<hash-tbd>`).
> The `nproc` parameter on `points_pipeline` / `vs30 points` /
> `vs30 points-custom` is gone; `dbscan_nproc` was added to the points
> commands to preserve user control over DBSCAN parallelism in the
> bayesian-update path (mirroring `grid_pipeline`). The §3.x data and
> §5.2/§5.3 conclusions are unchanged. The runner scripts in
> `dev/scripts/investigations/points_features_investigation/`
> (`run_points_sweep.py`, `run_balanced_blas_supplement.py`,
> `bench_utils.py`, `test_bench_utils.py`) were also removed — see git
> history if a re-run is needed.

This keeps the doc honest about what was actually done rather than letting
§5.1's recommendation silently outlive itself.

### 2.10 Slow-tier reassessment

After everything above lands, time the slow tier with:

```bash
pytest tests/test_grid_points_consistency.py::test_grid_points_consistency_slow \
       --runslow -v
```

The current `dev/CLAUDE.md` claim of ~43 min predates several perf-relevant
fixes (`fd24129` test consolidation, `827a71b` obs-prep fix, `9e5a35d` GID 4
alluvium fix). Decision criterion:

- **If under ~15 min total: remove the slow infrastructure entirely.** The
  simplification benefit (one fewer test tier, one fewer flag for devs to know)
  outweighs the longer pre-push run. Specifically:
  - Drop `@pytest.mark.slow` from `test_grid_points_consistency_slow`
    (`tests/test_grid_points_consistency.py:214`).
  - Delete `pytest_addoption` and `pytest_collection_modifyitems` in
    `tests/conftest.py:17-31`.
  - Strip the *Slow tier* description from the test module docstring
    (`tests/test_grid_points_consistency.py:12-14`); merge the two tiers'
    description into one paragraph reflecting the new unified tier.
  - Update `dev/CLAUDE.md` *Running Tests* section: drop the `--runslow` line,
    replace the default-tier runtime with the new measured value.
- **If over 15 min:** keep the marker; just include `--runslow` in the
  verification plan (§4) and update `dev/CLAUDE.md`'s runtime estimate to the
  measured value.

The 15-min threshold is generous on purpose — the goal is to take the
simplification unless it would meaningfully hurt the dev loop.

## 3. Out of scope

- Not changing `grid_pipeline` / `points_pipeline` signatures beyond the
  `nproc` → `dbscan_nproc` swap on the points side.
- Not touching `vs30/spatial.py`, `vs30/category.py`, `vs30/raster.py`,
  `vs30/gapfill.py`, `vs30/utils.py`, `vs30/config.py`,
  or `vs30/constants.py` (other than removing `N_PROGRESS_CHUNKS`).
- Not changing the CLI's grid commands.
- Not renaming `pipeline.py` (it's still earning its name as the
  orchestration layer + shared utilities, even after the `grid.py` extraction).
- Not changing `category.perform_clustering`'s nproc semantics.
- Not extracting the shared utilities from `pipeline.py` into a separate
  utilities module — they live happily there.

## 4. Verification plan

Run after implementation:

1. **Full test suite (both tiers):**

   ```bash
   pytest tests/ -v --runslow
   ```

   If §2.10 removes the slow infrastructure, the `--runslow` flag goes away
   and the command collapses to `pytest tests/ -v`.

2. **Greps must return zero hits** anywhere in `vs30/` or `tests/`:

   ```bash
   grep -rn "run_parallel_locations\|process_locations_chunk\|LocationsChunkConfig\|N_PROGRESS_CHUNKS\|resolve_nproc\|spawn_context\|limit_blas_threads" --include="*.py"
   grep -rn "from vs30 import.*multiprocess\|from vs30.multiprocess\|from vs30 import.*parallel\|from vs30.parallel" --include="*.py"
   ```

3. **New imports must resolve correctly** — only in `pipeline.py`:

   ```bash
   grep -rn "from vs30 import.*points\|from vs30.points" vs30/ tests/ --include="*.py"
   grep -rn "from vs30 import.*grid\|from vs30.grid" vs30/ tests/ --include="*.py"
   ```

4. **Smoke check via the CLI**:

   ```bash
   vs30 points --help    # must show --dbscan-nproc, must not show --nproc
   vs30 points-custom --help    # same
   ```

## 5. Risks and mitigations

- **Mistake during the rename / extraction.** Mitigation: the rename is
  mechanical — same symbols, same bodies, just a different file. Imports get
  updated in lock-step. The full test suite catches functional regressions.
- **Stale references in dev scripts or comments.** Mitigation: the verification
  greps in §4 catch these by name.
- **Findings doc becoming inaccurate.** Mitigation: the postscript explicitly
  records the deviation from §5.1 and the rationale.

## 6. Implementation order

Suggested sequence (to keep diffs reviewable and tests green at each step):

1. Rename `parallel.py` → `points.py`; update imports in `pipeline.py`. Run
   tests — should be green.
2. Inside `points.py`, delete `LocationsChunkConfig`, `process_locations_chunk`,
   `run_parallel_locations`. Update module docstring. Drop `multiprocess`
   import.
3. In `pipeline.py`'s `points_pipeline`, swap `nproc` → `dbscan_nproc`, delete
   the parallel branch, drop `multiprocess` import. Update docstring.
4. In `cli.py`, swap `nproc` → `dbscan_nproc` on `points`, `points_custom`,
   and `run_points_pipeline`. Update docstrings.
5. Update tests (`test_benchmarks.py`, `test_grid_points_consistency.py`).
6. Run the full test suite — should be green.
7. Delete `vs30/multiprocess.py`. Remove `N_PROGRESS_CHUNKS` from `constants.py`.
8. Run the verification greps in §4. Run tests again.
9. Extract grid-stage helpers (`create_initial_vs30_arrays`,
   `compute_hybrid_geology_arrays`, `compute_spatial_adjustment_on_grid`,
   `combine_model_arrays`, `write_raster`) from `pipeline.py` into new
   `vs30/grid.py`. Update `compute_model_grid` (which stays in `pipeline.py`)
   and `grid_pipeline` to call `grid.X` for the moved symbols.
10. Run tests. Run verification greps.
11. Delete the dev runner scripts (§2.8).
12. Add the findings-doc postscript (§2.9). Use the actual commit hash of
    the commit that removed the multiprocessing path (whatever the
    implementation plan groups into a single commit for the removal).
13. Time the slow tier. Apply §2.10's decision branch.
14. Final verification run.

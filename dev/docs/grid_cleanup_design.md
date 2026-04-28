# Grid Pipeline Cleanup — Design

**Date:** 2026-04-28
**Branch:** `vs30_refactor`
**Status:** Design (pre-implementation)
**Predecessor:** [`perf_features_investigation_findings.md`](perf_features_investigation_findings.md)

## 1. Purpose

Implement the recommendations from the performance-features investigation
for **grid mode only**:

1. **Remove the per-pixel MVN multiprocessing path.** It is detrimental in
   every regime tested (2×–110× slower than `nproc=1`).
2. **Drop the dead `BoundingBoxResult.obs_to_grid_indices` field.** It is
   constructed but never read; downstream callers always recompute
   per-pixel observation membership from scratch via
   `select_observations_for_pixel`.
3. **Drop the `nproc` parameter from `find_affected_pixels`.** Its
   internal parallel chunk-processing branch was bundled with the larger
   multiproc story; the bbox phase is fast enough that any savings would
   be marginal at best, and the only caller after this cleanup
   (`compute_spatial_adjustment_on_grid`) no longer has an `nproc` to
   pass anyway.
4. **Rename the remaining `nproc` on grid-only entry points** to
   `dbscan_nproc`, since after the cleanup it is exclusively used for
   DBSCAN clustering inside the categorical-update step.

`find_affected_pixels` itself stays (the investigation showed it is
beneficial in every regime tested at `nproc=1`).

## 2. Scope

### In scope

- `vs30/spatial.py` — drop `obs_to_grid_indices` field and its
  construction; drop `nproc` from `find_affected_pixels`; remove
  `build_obs_indices` parameter from helper functions.
- `vs30/parallel.py` — delete `run_parallel_spatial_fit` and its private
  worker `process_pixels_chunk`.
- `vs30/constants.py` — delete `MULTIPROCESS_OBSERVATION_THRESHOLD`.
- `vs30/pipeline.py` — drop `nproc` from
  `compute_spatial_adjustment_on_grid`; remove its multiproc branch and
  threshold guard; rename `nproc` → `dbscan_nproc` on
  `compute_categorical_vs30_updates`, `compute_model_grid`, and
  `grid_pipeline`.
- `vs30/cli.py` — rename `--nproc` → `--dbscan-nproc` on the `grid` and
  `grid-custom` Typer commands; update the kwarg passed to
  `pipeline.grid_pipeline`.
- `tests/test_benchmarks.py` — drop the `nproc=[1, -1]` parametrize on
  grid benchmark tests; rename the parameter to `dbscan_nproc`.
- `tests/test_grid_points_consistency.py` — update grid_pipeline call to
  use the renamed kwarg.
- `dev/scripts/investigations/diagnose_jaehwi_mask.py` and
  `dev/scripts/investigations/perf_features_investigation/run_full_pipeline_confirmation.py`
   — update grid_pipeline calls to use the renamed kwarg.

### Explicitly NOT in scope

- `pipeline.points_pipeline` — entirely untouched. The points pipeline
  still uses `nproc` for both clustering AND parallel locations; that
  asymmetry will be resolved (or not) in a separate investigation.
- `vs30/parallel.py::run_parallel_locations` and helpers — kept; points
  mode still uses them.
- `vs30/multiprocess.py` (`spawn_context`, `single_threaded_blas`,
  `resolve_nproc`) — kept as-is; `run_parallel_locations` still depends
  on these.
- The `nproc` parameter on `points_pipeline`, `vs30 points`,
  `vs30 points-custom` — entirely untouched.
- Reference rasters (`tests/benchmarks/*.tif`) — not regenerated; the
  cleanup is expected to be bit-identical to the existing nproc=1 path.

## 3. Behavioural guarantee

**No expected output change** for the recommended-and-now-only path
(`nproc=1`, `ffap=ON`). The cleanup:

- Removes the multiproc branch in `compute_spatial_adjustment_on_grid`
  but the existing nproc=1 path is already exercised by the test suite.
- Removes the `obs_to_grid_indices` field, which no caller reads.
- Removes the `find_affected_pixels` parallel branch, but the sequential
  branch produces an identical `mask`.
- Renames `nproc` → `dbscan_nproc` — purely a parameter-name change,
  no semantics change.

The bench reference rasters in `tests/benchmarks/` were generated at
`nproc=1` (per the existing test code) so they continue to match.

## 4. API changes

### Public CLI

```diff
- vs30 grid --version <X> ... --nproc <N>
+ vs30 grid --version <X> ... --dbscan-nproc <N>

- vs30 grid-custom --geology-csv <X> ... --nproc <N>
+ vs30 grid-custom --geology-csv <X> ... --dbscan-nproc <N>
```

`vs30 points` and `vs30 points-custom` are **unchanged**: they keep
`--nproc` because that parameter still does double-duty (clustering +
parallel locations) until the points-mode investigation concludes.

### Public Python

```diff
 pipeline.grid_pipeline(
     grid_config=...,
     ...,
-    nproc=1,
+    dbscan_nproc=1,
 )

 pipeline.compute_model_grid(
     model_type=...,
     ...,
-    nproc=1,
+    dbscan_nproc=1,
 )

 pipeline.compute_categorical_vs30_updates(
     categorical_model_csv=...,
     ...,
-    nproc=1,
+    dbscan_nproc=1,
 )

 pipeline.compute_spatial_adjustment_on_grid(
     vs30_array=...,
     ...,
-    nproc=1,
 )

 spatial.find_affected_pixels(
     raster_data=...,
     ...,
-    nproc=1,
 )
```

No deprecation alias. Project is research-stage with no external users
mentioned; the rename is a clean break.

### Internal helpers

```diff
 spatial.grid_points_in_bbox(
     grid_locs=...,
     obs_eastings_min=...,
     ...,
-    build_obs_indices=True,
-) -> tuple[ndarray, list[ndarray]]:
+) -> ndarray:  # chunk_mask only
```

```diff
 spatial.process_bbox_chunk(args: tuple) -> tuple:
-    chunk_idx, grid_locs_chunk, start_idx, obs_bounds, build_obs_indices = args
+    chunk_idx, grid_locs_chunk, start_idx, obs_bounds = args
-    return chunk_idx, chunk_mask, obs_to_grid_indices
+    return chunk_idx, chunk_mask
```

### Dataclass

```diff
 @dataclass
 class BoundingBoxResult:
     mask: np.ndarray
-    obs_to_grid_indices: list[np.ndarray]
     n_affected_pixels: int
```

## 5. Test plan

### 5.1 Per-commit

After each atomic commit:

```bash
ruff check <touched files>
ruff format --check <touched files>
pytest <relevant test file>
```

### 5.2 Final

After the last commit:

```bash
# Default tier — unit + benchmark + 3-city consistency, ~3 min
pytest tests/

# Slow tier — full 38-point grid/points consistency, ~43 min
pytest tests/ --runslow

# CLI smoke — small region with the new flag name
vs30 grid --version foster_2019_approx \
  --grid-xmin 1500000 --grid-xmax 1700000 \
  --grid-ymin 5100000 --grid-ymax 5300000 \
  --grid-dx 5000 --grid-dy 5000 \
  --output-dir /tmp/vs30_smoke \
  --dbscan-nproc 1
```

### 5.3 Test parametrize change

The grid benchmark tests in `tests/test_benchmarks.py` currently
parametrize over `nproc=[1, -1]`. With multiproc spatial-fit gone, the
only thing the `nproc=-1` variant exercises is sklearn's DBSCAN under
multiple worker processes — an invariant sklearn already guarantees.
**Drop the parametrize**, run only `dbscan_nproc=1`. Halves the test
runtime; no meaningful coverage loss.

The points test (`test_foster_2019_approx_points_benchmark`) keeps its
`nproc=[1, -1]` parametrize unchanged — points mode still has both
spatial and parallel-locations multiprocessing in scope.

## 6. Implementation order

A sequence of small atomic commits, each independently reviewable and
bisectable. Each commit ends green on `ruff check && ruff format --check`
and on the relevant subset of tests.

| # | Commit | Files |
|---|---|---|
| 1 | Drop `obs_to_grid_indices` field + simplify bbox helpers. | `vs30/spatial.py`, `dev/scripts/investigations/perf_features_investigation/bench_utils.py` |
| 2 | Drop `nproc` from `find_affected_pixels`; remove its parallel branch. | `vs30/spatial.py` |
| 3 | Delete `run_parallel_spatial_fit` and `process_pixels_chunk`; drop `nproc` from `compute_spatial_adjustment_on_grid`; remove threshold guard and constant. | `vs30/parallel.py`, `vs30/pipeline.py`, `vs30/constants.py` |
| 4 | Rename `nproc` → `dbscan_nproc` on the three remaining pipeline functions (`compute_categorical_vs30_updates`, `compute_model_grid`, `grid_pipeline`). | `vs30/pipeline.py` |
| 5 | CLI rename `--nproc` → `--dbscan-nproc` on grid commands. | `vs30/cli.py` |
| 6 | Update grid tests: drop parametrize, rename arg. | `tests/test_benchmarks.py`, `tests/test_grid_points_consistency.py` |
| 7 | Update dev scripts. | `dev/scripts/investigations/diagnose_jaehwi_mask.py`, `dev/scripts/investigations/perf_features_investigation/run_full_pipeline_confirmation.py` |
| 8 | Run `pytest tests/ --runslow` and the CLI smoke; if green, commit nothing extra and the cleanup is done. | (validation only) |

## 7. Risks

| Risk | Mitigation |
|---|---|
| A reference raster was generated under nproc>1 and the cleanup changes output. | The current test code passes `nproc=1` to `run_benchmark` for all three full-domain grid benchmarks (lines 153, 163, 173 of `test_benchmarks.py`). Reference rasters are nproc=1 outputs; cleanup is the same path. No expected difference. |
| External script imports `MULTIPROCESS_OBSERVATION_THRESHOLD`. | `grep -rn 'MULTIPROCESS_OBSERVATION_THRESHOLD' .` shows the constant is referenced only in `vs30/constants.py` (definition), `vs30/pipeline.py` (read site), and the perf investigation harness (where it's monkey-patched but already understood as internal). No external dependency. |
| The CLI rename breaks user wrapper scripts. | Project is research-stage. The rename is announced in commit messages; no deprecation alias. |
| `run_parallel_locations` silently breaks because of a shared symbol that gets removed. | Audit (already done): `run_parallel_locations` uses `multiprocess.spawn_context`, `multiprocess.single_threaded_blas`, `multiprocess.resolve_nproc`, and the worker `process_locations_chunk` — all preserved. |
| `find_affected_pixels` callers in dev scripts use `nproc>1`. | Already verified — none of the dev scripts touched in this cleanup pass `nproc` to `find_affected_pixels` directly; they all go through `pipeline.grid_pipeline`, which loses the parameter as part of the cleanup. |

## 8. Out-of-scope follow-ups

These items are noted here so they are not lost, but are **not** part of
this cleanup:

- **Points-mode investigation.** Re-run the perf investigation
  methodology against `points_pipeline` and `parallel.run_parallel_locations`,
  decide whether to remove the points-mode multiproc and / or add
  `find_affected_pixels` to the points path.
- **`MIN_DIST_ENFORCED` clamp.** §7.4 of the findings doc noted the
  clamp causes a tiny systematic stdv difference between ffap=ON and
  ffap=OFF for far-from-obs pixels. With `ffap=ON` becoming the only
  path, this is moot for grid mode going forward; left as a note in
  case it surfaces for any future `find_affected_pixels=False` debugging.

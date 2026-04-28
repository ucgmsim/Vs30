# Points-Pipeline Multiproc Obs-Prep Fix — Design

**Date:** 2026-04-28
**Branch:** `vs30_refactor`
**Status:** Design (pre-implementation)
**Predecessors:**
- [Points-perf investigation findings](points_perf_investigation_findings.md) — diagnoses the bug this work fixes (§4 of that doc).
- [Points-perf investigation design](points_perf_investigation_design.md) — the prior investigation, whose §2 placed production-code changes out of scope and deferred the fix to a separate piece of work (this one).

## 1. Purpose

`vs30/parallel.py::run_parallel_locations` currently splits the query points
into up to `N_PROGRESS_CHUNKS = 1000` chunks and sends each chunk to a worker
along with the full `observations_df`. Each worker, for every chunk it
receives, re-runs the full observation-preparation pipeline on all `N_obs`
observation rows: categorical lookup, slope raster sampling, coastal-distance
sampling, and (for geology) hybrid Vs30 modifications.

For an `N_query = 1000`, `N_obs = 35,706`, `nproc = 8` cell, that means the
35,706-row observation prep is done 1,000 times instead of once — about 125
times per worker. The result: that cell ran in ~2,770 s vs ~18 s on `nproc=1`,
a 151× slowdown that scales linearly with `N_query`.

This work eliminates the redundancy by extracting observation prep into two
helpers that run exactly once per pipeline call. Workers receive precomputed
`PointsObsData` and lose the ability to recompute it.

## 2. Scope

### In scope

- New dataclass `PointsObsData` and helpers `prepare_geology_obs_data`,
  `prepare_terrain_obs_data` in `vs30/parallel.py`.
- Refactor of `process_geology_at_points`, `process_terrain_at_points`,
  `process_locations_chunk`, and `run_parallel_locations` in
  `vs30/parallel.py` to take `PointsObsData` instead of `observations_df`.
- Rewire `points_pipeline` (in `vs30/pipeline.py`) so both the sequential and
  the parallel branches call the helpers once before dispatching.
- Smoke benchmark of two harness cells before and after the fix to confirm
  the slowdown is gone.

### Out of scope

- **CLI default change.** With the bug fixed, the right default for
  `vs30 points --nproc` depends on whether multiproc actually wins after the
  refactor. The findings doc's §5.1 recommendation (`-1` → `1`) was
  conditional on the bug existing. Defer to a follow-up that re-measures.
- **Full re-sweep with the fix.** Re-running the original `7 × 3 × {1,8} × 3`
  matrix to answer "does multiproc inherently win for points mode" is
  separate work — its own piece, with its own findings doc.
- **`Pool(initializer=...)` optimisation.** Worth doing only if the smoke
  shows pickle overhead is a bottleneck. Not pre-optimising.
- **Findings-doc amendment.** The existing
  `points_perf_investigation_findings.md` captured the state at
  investigation-close time and stays as-is. The smoke numbers from this work
  go into the implementation plan and commit messages.

## 3. Design

### 3.1 New module surface (`vs30/parallel.py`)

One small dataclass and two helpers. Both branches share the same data
shape; the helpers differ only in how they build it.

```python
@dataclass
class PointsObsData:
    """Precomputed observation arrays for use in points-pipeline workers.

    Shape contract: every array has length N_obs.

    For geology, ``model_vs30`` and ``model_stdv`` carry post-hybrid-mods
    values (slope/coast applied). For terrain, they carry the raw categorical
    values (no hybrid mods). Either way, they are exactly what
    ``compute_spatial_adjustment_at_points`` expects to receive.
    """
    locations: np.ndarray       # (N, 2) easting, northing
    vs30: np.ndarray            # (N,)   measured vs30
    uncertainty: np.ndarray     # (N,)   per-obs uncertainty
    model_vs30: np.ndarray      # (N,)   model vs30 at obs locations
    model_stdv: np.ndarray      # (N,)   model stdv at obs locations

    @classmethod
    def empty(cls) -> "PointsObsData": ...   # mirrors spatial.ObservationData.empty()


def prepare_geology_obs_data(
    observations_df: pd.DataFrame,
    geol_model_df: pd.DataFrame,
    apply_alluvium_slope_mod: bool,
    apply_coastal_distance_mod: bool,
) -> PointsObsData:
    """Lifted verbatim from process_geology_at_points lines 92-122."""


def prepare_terrain_obs_data(
    observations_df: pd.DataFrame,
    terr_model_df: pd.DataFrame,
) -> PointsObsData:
    """Lifted verbatim from process_terrain_at_points lines 204-208."""
```

Both helpers return `PointsObsData.empty()` for an empty input DataFrame
(the pre-fix code's `if len(observations_df) > 0` branch).

### 3.2 Modified function signatures

| Function | Before | After |
|---|---|---|
| `process_geology_at_points` | takes `observations_df`, does obs-prep internally | takes `geology_obs_data: PointsObsData`, body is just query-point work + spatial adjustment |
| `process_terrain_at_points` | takes `observations_df` | takes `terrain_obs_data: PointsObsData` |
| `process_locations_chunk` (chunk_args tuple) | `(points, chunk_id, observations_df, geol_model_df, terr_model_df, config)` | `(points, chunk_id, geology_obs_data, terrain_obs_data, geol_model_df, terr_model_df, config)` |
| `run_parallel_locations` | takes `observations_df` | takes `geology_obs_data: PointsObsData \| None` and `terrain_obs_data: PointsObsData \| None` (`None` when that branch is not running) |
| `points_pipeline` (sequential and parallel branches) | passes `observations_df` to either `process_*_at_points` or `run_parallel_locations` | calls `prepare_*_obs_data` once, passes the precomputed `PointsObsData` to the appropriate downstream function |

The geology helper preserves the existing `apply_alluvium_slope_mod` and
`apply_coastal_distance_mod` flags. These still parameterise the
*observation-side* hybrid mods. The same flags continue to parameterise the
*query-point-side* hybrid mods inside `process_geology_at_points` (which
remain inside the worker, as they should — they operate on the chunk's
points only).

### 3.3 Data flow

```
points_pipeline (in pipeline.py)
  ├─ Load observations_df (one CSV read)
  ├─ Load categorical models (geology, terrain, optionally Bayesian-updated)
  ├─ geology_obs_data = prepare_geology_obs_data(obs_df, geol_model_df, ...)   # if run_geology
  ├─ terrain_obs_data = prepare_terrain_obs_data(obs_df, terr_model_df)        # if run_terrain
  └─ if nproc > 1:
       run_parallel_locations(points, geology_obs_data, terrain_obs_data, ...)
         ├─ split points into n_chunks
         ├─ chunk_args = [(points[idx], i, geology_obs_data, terrain_obs_data, ...) for ...]
         └─ pool.imap(process_locations_chunk, chunk_args)
              └─ each worker calls process_*_at_points(chunk_points, ..., obs_data, ...)
     else (sequential):
       directly call process_*_at_points(all_points, ..., obs_data, ...)
```

The obs-prep work happens exactly **once** per `points_pipeline` call,
regardless of `nproc`. Workers cannot recompute it because they do not
receive `observations_df`.

### 3.4 Pickle / serialisation cost

After the fix, each chunk's `chunk_args` tuple still gets serialised once per
chunk via `pool.imap`. The new shape carries `PointsObsData` instead of
`observations_df`; the byte sizes are similar (~1.4 MB at `N_obs = 35,706`).
With 1,000 chunks that's roughly 10 s of pickle overhead — vs the bug's
~2,750 s of redundant raster sampling, a >250× improvement.

A `Pool(initializer=...)` pattern would eliminate even that 10 s by sending
the obs data once per worker instead of once per chunk. Out of scope for
this work; revisit if smoke results suggest it matters.

## 4. Validation

### 4.1 Correctness — relies on existing tests

No new pytest files. Two existing tests already exercise the parallel
points path with `nproc > 1` and would catch any worker-vs-sequential drift
introduced by the refactor:

- `tests/test_benchmarks.py::test_foster_2019_approx_points_benchmark` —
  parameterised over `nproc ∈ {1, -1}`. Compares `points_pipeline` output to
  the prior-dominated `foster_2019_approx.tif` raster.
- `tests/test_grid_points_consistency.py` — runs `points_pipeline` with
  `nproc=-1` and asserts the points result matches the **grid** result.
  Grid mode does not go through `parallel.py`'s points code at all, so any
  drift in the refactor manifests as a numerical mismatch.

Together they cover sequential-vs-parallel parity and parallel-vs-grid
parity, which is sufficient correctness coverage for this refactor.

### 4.2 Performance — smoke benchmark

Re-run two harness cells with the fix in place. The harness already exists at
`dev/scripts/investigations/points_features_investigation/`; we invoke
`bench_utils.time_one_run` directly:

| Cell | Pre-fix (s) | Post-fix expectation |
|---|---|---|
| `N_query=1000, N_obs=35706, nproc=1` | 18.4 | ~18 s (unchanged — sequential path was already correct) |
| `N_query=1000, N_obs=35706, nproc=8` | 2,770 | ≤ ~30 s (multiproc/BLAS-MT tradeoff dominates instead of the bug) |

The exact post-fix `nproc=8` number is the answer to "does multiproc
actually win after the bug is gone?" — relevant for the deferred CLI-default
decision but not load-bearing for this piece. The success criterion is
simply "the slowdown is gone" — anything within a small constant factor of
the `nproc=1` baseline qualifies.

Outputs go into a small CSV under the harness directory, gitignored. The
numbers get cited in the implementation plan and the final commit message.

## 5. Branch and commit strategy

- Continue on `vs30_refactor` (matches the project's recent practice of
  incremental commits on the long-running working branch).
- Expected commit shape:
  - One commit for the helper extraction + dataclass (no behavioural change
    yet — `process_*_at_points` and the parallel path keep the old
    `observations_df` parameter and call the helpers internally, so the
    refactor is bisectable).
  - One commit that switches the parallel path and `points_pipeline` over
    to `PointsObsData`, removing the now-dead `observations_df` plumbing.
    This is the commit that fixes the bug.
  - One commit recording the smoke benchmark evidence (CSV is gitignored;
    commit body cites the numbers).

The bisectable two-step refactor matters because the second commit is the
one whose performance impact we want to demonstrate clearly.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Subtle numerical drift introduced by separating obs-prep from query-point work (e.g., array dtype mismatches, missing legacy-parity sentinel handling). | Helpers are lifted **verbatim** from the existing function bodies, including the `LEGACY_OBS_SLOPE_NODATA_SENTINEL` sentinel substitution (`parallel.py:106-108`). Existing benchmark + consistency tests are the gate. |
| Empty-observations edge case (no observations at all). | Helpers return `PointsObsData.empty()`; `process_*_at_points` retains its `if len(geology_obs_data.locations) > 0` guard, just sourced from the dataclass instead of the DataFrame. |
| Pickle serialisation overhead at large `N_obs` exceeds expectations and dominates the per-point work. | Quantified in the smoke (§4.2). If meaningful, the `Pool(initializer=...)` optimisation is the documented next step. |
| External callers exist somewhere we missed. | `grep` confirmed no external callers; only `pipeline.py` and the test files reference these functions. |

## 7. Out-of-scope follow-ups

These are the natural next pieces of work after this fix lands:

1. **Full re-sweep** of the original `nproc ∈ {1, 8}` matrix to answer the
   inherent multiproc/BLAS-MT tradeoff question — its own brainstorm →
   design → plan cycle.
2. **CLI default decision** based on (1)'s measurements. Likely a one-line
   change in `vs30/cli.py:254` and `:349`.
3. **Pool initializer optimisation** if the smoke shows pickle cost is
   meaningful. Same scope as a small targeted refactor.

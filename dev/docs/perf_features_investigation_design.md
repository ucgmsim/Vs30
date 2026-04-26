# Performance Features Investigation — Design

**Date:** 2026-04-26
**Branch:** `vs30_refactor`
**Status:** Design (pre-implementation)

## 1. Purpose

The Vs30 package contains two performance-oriented features whose effect has
never been explicitly measured:

1. **Multiprocessing of the per-pixel MVN loop** (`vs30/parallel.py`,
   gated by `nproc` and the
   `MULTIPROCESS_OBSERVATION_THRESHOLD = 1000` fallback in `pipeline.py`).
2. **`find_affected_pixels`** (`vs30/spatial.py`) — a bounding-box pre-filter
   that identifies which valid pixels lie within `MAX_DIST_M` of any
   observation, so the spatial-adjustment loop touches only those pixels.

Both add code complexity. Each one's continued existence should be justified
by measured benefit. This investigation determines, for each feature,
whether it is **beneficial / neutral / detrimental** across the
realistic parameter space, and produces a clear recommendation:

- **Keep** the feature if there is any meaningful regime where it is the
  fastest option.
- **Remove** the feature (and simplify the code) if it is neutral or
  detrimental everywhere.

## 2. Scope

In scope:

- **Grid mode only.** Both features currently exist in
  `compute_spatial_adjustment_on_grid`. Points mode (`points_pipeline`) is
  out of scope for this investigation; the conclusions for grid mode will
  inform whether a follow-up investigation in points mode is worthwhile.
- The two features in isolation and in combination (since both can be on or
  off independently).

Out of scope:

- Refactoring or rewriting either feature. The investigation produces a
  recommendation; any refactor based on it is a separate piece of work.
- Points-mode performance (deferred).
- Bayesian update / DBSCAN clustering / hybrid geology / gap-fill
  performance — not affected by either feature.

## 3. Background — current implementation

### 3.1 Multiprocessing

`compute_spatial_adjustment_on_grid` (in `vs30/pipeline.py`) resolves
`nproc` and either:

- Takes the **sequential** path
  (`spatial.compute_spatial_adjustments`) which loops over affected
  pixels in a single process, letting BLAS parallelise each
  `np.linalg.inv` across all cores, OR
- Takes the **parallel** path
  (`parallel.run_parallel_spatial_fit`) which splits the affected pixels
  into chunks and processes them in `nproc` worker processes, each pinned
  to single-threaded BLAS via `multiprocess.single_threaded_blas`.

A guard at lines ~525–534 of `pipeline.py` forces `nproc = 1` whenever
`n_obs > MULTIPROCESS_OBSERVATION_THRESHOLD = 1000`. The justification in
the constant's docstring is that with many observations, pixels frequently
hit `MAX_POINTS = 500`, producing 500×500 covariance matrices where BLAS
parallelism dominates Python-level multiprocessing.

The investigation will both **verify the threshold's premise** and **search
for any (N_obs, N_grid) regime where multiprocessing wins**.

### 3.2 `find_affected_pixels`

For each observation, `find_affected_pixels` builds an axis-aligned
bounding box of side `2 * MAX_DIST_M` (= 20 km) centred on the observation
and marks every grid pixel inside any box as "affected". The result is a
boolean mask + per-observation index lists.

`compute_spatial_adjustments` then iterates only over affected pixels.
Inside the loop, `select_observations_for_pixel` still computes the
exact Euclidean distance from the pixel to **every** observation
(via `np.einsum`), uses `np.partition` to find the cutoff distance, and
returns the within-range observations.

The premise is that for sparse-observation grids the bbox pre-filter
eliminates most pixels cheaply, so the more expensive per-pixel exact
distance + cutoff is amortised over a much smaller pixel set. The
investigation will quantify whether this premise holds across realistic
configurations.

## 4. Methodology

Two phases:

### 4.1 Phase 1 — Isolated MVN parameter sweep

Build a synthetic `RasterData` of a target valid-pixel count, subsample
observations from the largest available pool
(`viktor_inferred_vs30_from_cpt.csv`, 35,709 obs), and time **only** the
two functions under investigation:

- `find_affected_pixels` (or its skipped equivalent), measured separately
  as `t_bbox`.
- `compute_spatial_adjustments` or `run_parallel_spatial_fit`, measured
  as `t_spatial`.

Total effective time per cell is `t_bbox + t_spatial`. This isolation lets
us iterate cheaply (~seconds per cell) and span a much wider parameter
space than full-pipeline runs allow.

### 4.2 Phase 2 — Full-pipeline confirmation

Run `vs30 grid` end-to-end on **2–4 representative configurations** chosen
from Phase 1 conclusions (typically the corners of the regime: sparse-obs
+ coarse-grid, sparse-obs + fine-grid, dense-obs + coarse-grid, dense-obs
+ fine-grid). Confirm that the Phase 1 ordering of strategies is preserved
end-to-end. If a Phase 2 run inverts the Phase 1 ordering, that is a
finding to investigate before publishing recommendations.

## 5. Test matrix (Phase 1)

| Parameter | Values | Notes |
|---|---|---|
| `N_obs` | 50, 100, 250, 500, 1000, 2500, 5000, 10000, 35709 | Subsampled from `viktor_inferred_vs30_from_cpt.csv` with seeded RNG. The full set is 35,709. |
| `N_grid` (target valid pixels) | 1k, 10k, 100k, 1M | Realised by clipping the standard NZ grid to a bounding subdomain at varying resolutions. Actual valid-pixel count logged as `N_grid_actual`. |
| `nproc` | 1 (multi-threaded BLAS), 8 (single-threaded BLAS) | Endpoints of the strategy space on this 8-core machine. |
| `ffap` (find_affected_pixels) | ON, OFF | OFF = pass a fully-true `BoundingBoxResult` so every valid pixel is iterated. |
| Repetitions | 3 (median reported) | Smooth out OS jitter. |

**Total cells:** 9 × 4 × 2 × 2 = 144 unique configurations × 3 reps = **432 runs**.

Cells where the expected wall time exceeds **30 minutes** (e.g.,
N_obs=35709, N_grid=1M, nproc=8, ffap=OFF) will be skipped or run with
fewer reps, with the omission noted in the results CSV. Wall budget for
Phase 1: ~3–6 hours.

### 5.1 Phase 2 confirmation set

| Cohort | N_obs | Resolution | Approx N_valid_pixels |
|---|---|---|---|
| sparse / coarse | 470 (modified_foster_2019) | 5000 m | ~50k |
| sparse / fine | 470 | 500 m | ~5M |
| dense / coarse | 35,709 (viktor_cpt) | 5000 m | ~50k |
| dense / fine | 35,709 | 500 m | ~5M |

Each cohort run with whichever strategy Phase 1 recommends, and once with
the opposite strategy as a contrast. Wall budget for Phase 2: ~2–4 hours.

## 6. Harness structure

New directory: `dev/scripts/investigations/perf_features_investigation/`.

```
perf_features_investigation/
├── bench_utils.py                    # shared helpers
├── run_isolated_sweep.py             # Phase 1 driver
├── run_full_pipeline_confirmation.py # Phase 2 driver
├── analyze_results.py                # tables + figures from CSV
└── README.md                         # how to reproduce
```

### 6.1 `bench_utils.py`

- `subsample_observations(n: int, seed: int) -> pd.DataFrame` — deterministic subsample of viktor_cpt observations.
- `make_raster_data(n_target: int) -> tuple[RasterData, dict]` — clip the standard NZ grid to a region containing approximately `n_target` valid pixels; build a real `RasterData` (with valid arrays so `validate_raster_data` passes). Reuses `pipeline.create_initial_vs30_arrays` with `model_type=TERRAIN` so the pixels are realistic and slope/coast arrays are not needed.
- `make_full_bbox_result(raster_data) -> BoundingBoxResult` — returns a `BoundingBoxResult` with `mask = raster_data.valid_mask.flatten()`, used to disable the bbox pre-filter.
- `time_one_run(...) -> dict` — runs a single (N_obs, N_grid, nproc, ffap) cell, returns timing dict suitable for CSV row.
- Context manager `bypass_observation_threshold()` — temporarily sets `MULTIPROCESS_OBSERVATION_THRESHOLD` so the guard doesn't pre-empt the multiproc condition we want to measure. (Implemented as monkey-patching the constant on the `pipeline` module — the harness alone uses this; production code path is untouched.)

### 6.2 `run_isolated_sweep.py`

```
for N_obs in N_OBS_VALUES:
    obs = subsample_observations(N_obs, seed=42)
    for N_grid in N_GRID_VALUES:
        raster_data, profile = make_raster_data(N_grid)
        # Build ObservationData once per (N_obs, N_grid) — depends on raster_data.
        # model_type=TERRAIN is used because it does not require slope/coast
        # arrays; both features being measured (bbox + multiproc) are downstream
        # of prepare_observation_data and are agnostic to model_type, so the
        # choice does not affect relative timings.
        obs_data = spatial.prepare_observation_data(obs, raster_data, ..., model_type=TERRAIN)
        run numerical-equivalence guardrail (small case, once per script run)
        for nproc in [1, 8]:
            for ffap in [True, False]:
                for rep in range(N_REPS):
                    row = time_one_run(...)
                    rows.append(row)
                    write CSV incrementally so partial results survive a crash
```

Output: `results_isolated.csv` with columns:

```
N_obs, N_grid_target, N_grid_actual, N_affected, nproc, ffap, rep,
t_bbox_s, t_spatial_s, t_total_s, peak_rss_mb, timestamp_iso
```

### 6.3 `run_full_pipeline_confirmation.py`

For each cohort in §5.1, builds the `GridConfig`, calls
`pipeline.grid_pipeline` directly, captures wall time and the
`compute_spatial_adjustment_on_grid` sub-timings already logged by the
production code. Output: `results_full_pipeline.csv`.

### 6.4 `analyze_results.py`

Loads both CSVs, computes:

- **Per-configuration medians** across reps (drop reps with > 2σ outlier).
- **Speedup tables**: for each (N_obs, N_grid), report
  `speedup_multiproc = t_total_nproc1 / t_total_nproc8` and
  `speedup_ffap = t_total_ffap_off / t_total_ffap_on`. Speedup > 1 means the
  feature wins.
- **Heatmaps** over the (N_obs, N_grid) plane, one per question, written
  to `figures/`:
  - `multiproc_speedup_ffap_on.png`, `multiproc_speedup_ffap_off.png`
  - `ffap_speedup_nproc1.png`, `ffap_speedup_nproc8.png`
- **Crossover identification**: where each speedup crosses 1.0, with
  ±10% confidence bands.

## 7. Numerical equivalence guardrail

Before the sweep, the harness runs **one** small (N_obs=200, N_grid≈5k)
case across all four (nproc × ffap) combinations and asserts the
resulting `(updated_vs30, updated_stdv)` arrays match within
`atol=1e-9, rtol=1e-7`. Any mismatch halts the sweep — it indicates a
logic divergence between the variants and the timings would compare
non-equivalent computations. (Numerical determinism across these
variants is already a property the production code is supposed to have;
the guardrail confirms it before we lean on it.)

## 8. Outputs and findings doc structure

The deliverable `.md` file is `dev/docs/perf_features_investigation_findings.md`.
Sections:

1. **Summary.** One-line recommendation per feature
   (keep / remove / keep-with-revised-threshold).
2. **Methodology.** Brief — points to this design doc for full detail.
3. **Phase 1 results.**
   - `find_affected_pixels`: heatmaps + table of crossover points.
   - Multiprocessing: heatmaps + table of crossover points.
   - Combined (best strategy per regime).
4. **Phase 2 results.** End-to-end timings for the 4 cohorts; comparison
   to the Phase 1 prediction.
5. **Conclusions and recommendations.**
   - Per-feature: beneficial / neutral / detrimental, with quantified
     evidence.
   - If keeping: any threshold / heuristic that should be added or revised.
   - If removing: a brief sketch of the simplification (no implementation —
     that's a separate task).
6. **Reproducibility.** How to re-run from `perf_features_investigation/`.

## 9. Risks and mitigations

| Risk | Mitigation |
|---|---|
| OS noise inflates variance, masking real differences. | 3 reps + median; close other workloads; user has confirmed they will not use the workstation during runs. |
| Memory pressure on the largest cells (N_obs=35,709, N_grid=1M, ffap=OFF) crashes the run. | Skip cells whose pre-estimated peak RSS exceeds 24 GB (75% of 32 GB system RAM); log skipped reasons. |
| `MULTIPROCESS_OBSERVATION_THRESHOLD` guard hides the very regime we want to measure. | `bypass_observation_threshold()` context manager around each measurement. |
| Subsampled viktor_cpt observations are still spatially clustered and don't represent the small-N obs regime well. | Note this in the findings doc; if needed, also run a small confirmation with the foster_2019 / modified_foster_2019 / jaehwi_v1p0 observation files. |
| Results vary with hardware (BLAS implementation, core count). | Findings doc records hardware exactly. Conclusions phrased as "on this hardware" and "expected to generalise to BLAS-supporting CPUs"; no claim of universality. |
| Phase 2 inverts a Phase 1 conclusion. | Treat as a finding, not a failure: investigate the divergence (likely I/O, file write, or Bayesian update overhead) before recommending. |

## 10. Branch and commit strategy

- Work on the current branch `vs30_refactor`.
- Harness scripts committed under `dev/scripts/investigations/perf_features_investigation/`.
- This design doc and the eventual findings doc committed under `dev/docs/`.
- `results_*.csv` and `figures/` gitignored — they are reproducible
  outputs, not source. (Add a `.gitignore` entry inside the new directory.)
- Commits split logically: harness in one, design doc in one, findings + analysis in one. Production code is **not** modified by this investigation.

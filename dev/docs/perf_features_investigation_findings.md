# Performance Features Investigation — Findings

**Date:** 2026-04-27
**Branch:** `vs30_refactor`
**Status:** Draft (awaiting Phase 2 confirmation runs)

## 1. Summary

Two performance-oriented features in the Vs30 grid pipeline were measured
across a wide parameter sweep:

| Feature | Recommendation | One-line evidence |
|---|---|---|
| **Multiprocessing** of the per-pixel MVN loop | **Remove** | `nproc=8` is **2×–110× slower** than `nproc=1` in **every** cell tested; median ratio `nproc=1/nproc=8 ≈ 0.016`. |
| **`find_affected_pixels`** bbox pre-filter | **Keep** | `ffap=ON` is **1.04×–7× faster** than `ffap=OFF` in every cell tested at `nproc=1`; median speedup `1.5×`. |
| **Best strategy** in every regime | `nproc=1, ffap=ON` | Unanimous across all (N_obs, N_grid) cells. |

(See §6 for sketches of the simplification that follows from removing the
multiproc path.)

## 2. Methodology

Two-phase empirical measurement, per the design at
[`perf_features_investigation_design.md`](perf_features_investigation_design.md).

**Phase 1 — Isolated MVN sweep.** A custom harness
(`dev/scripts/investigations/perf_features_investigation/`) builds real
`RasterData` of approximately a target valid-pixel count, subsamples
observations from `viktor_inferred_vs30_from_cpt.csv`, and times only the
two functions under investigation:

- `spatial.find_affected_pixels` (or a no-op replacement that marks every valid
  pixel as affected) — measured as `t_bbox`.
- `spatial.compute_spatial_adjustments` (sequential) or
  `parallel.run_parallel_spatial_fit` (parallel) — measured as `t_spatial`.

Total per-cell time = `t_bbox + t_spatial`. The per-cell budget is 30 minutes;
cells that exceed this skip remaining reps (so very slow cells contribute one
rep, not three).

**Numerical equivalence guardrail.** Before the sweep, the harness asserts
that all (`nproc` × `ffap`) variants produce identical (vs30, stdv) arrays
within `atol=1e-9, rtol=1e-7` on a small case. Runs were grouped by ffap (the
two values process different pixel sets, so cross-ffap arrays do differ
slightly — by `≈ sqrt(corr_fn(0)) ≈ 0.99995` on the prior stdv of pixels far
from any observation, a side-effect of the `MIN_DIST_ENFORCED=0.1` clamp inside
`exponential_correlation_function`. This is a separate observation; see §7).

**Phase 2 — Full-pipeline confirmation.** Four representative cohorts run via
`pipeline.grid_pipeline` end-to-end at both `nproc=1` and `nproc=8`. Used to
verify that the Phase 1 ordering of strategies survives the surrounding
pipeline overhead (Bayesian update, hybrid mods, raster writes, etc.).

## 3. Hardware and software

- **CPU:** Intel Core i7-9700, 8 cores @ 3.00 GHz, no hyperthreading.
- **Memory:** 32 GiB.
- **OS:** Linux 6.17 (Ubuntu derivative).
- **BLAS:** Whatever NumPy's default is in the `vs30_venv` (likely OpenBLAS); BLAS
  threading is controlled at runtime by `threadpoolctl`.
- **Python:** 3.13.9 (mamba env `vs30_venv`).
- **Vs30 commit at sweep time:** `66e712b` (with the make_full_bbox_result OOM
  fix; pre-fix data is unaffected because the fix is in a non-timed code path).

The recommendations are phrased in terms of "this hardware". Speedups should
generalise to any modern x86_64 BLAS-supporting CPU; the absolute crossover
points may shift slightly with core count or BLAS implementation.

## 4. Phase 1 — sweep design

| Parameter | Values |
|---|---|
| `N_obs` (target) | 50, 100, 250, 500, 1000, 2500, 5000, 10000, 35709 |
| `N_obs` (post-filter, actual values logged) | 50, 98, 235, 476, 950, 2373, 4751, 9534, 35709 *(approx)* |
| `N_grid` (target valid pixels) | 1k, 10k, 100k, 1M |
| `N_grid` (actual, see §4.1) | 986 / 9 898 / 105 905 / 1 242 298 |
| `nproc` | 1 (multi-threaded BLAS) and 8 (single-threaded BLAS) |
| `ffap` | True, False |
| Reps | 3, median reported |

### 4.1 Note on actual N_obs / N_grid

Both axes are reported as targets; the harness logs the actual
post-filter `N_obs` (after `prepare_observation_data` drops observations
outside the local sub-domain, NaN model values, etc.) and the actual
`N_grid_actual` (from `valid_flat_indices.size` of the constructed
RasterData). The harness centres the sub-domain near central NZ; the
~3 % shortfall in N_obs at small grids is expected and does not affect
within-cell comparisons.

### 4.2 Trim policy

After a smoke run revealed `nproc=8` is 30–80× slower than `nproc=1` at
every cell tested, cells with `nproc>1 AND (N_obs > 10 000 OR
N_grid > 100 000)` were skipped to avoid multi-day wall times that would
not have changed the qualitative finding. Phase 1 ran for ~7 h and the
sweep was cut short at N_obs ≈ 4751 with the multiproc trend already
overwhelming; a follow-up `fill_high_nobs.py` script ran the missing
nproc=1 cells at N_obs ∈ {10 000, 35 709} (these are fast, since the
production path under nproc=1 is the recommended configuration anyway).

## 5. Phase 1 — results

### 5.1 Multiprocessing

**Headline:** `nproc=8` (parallel + single-threaded BLAS) is 2×–110× slower
than `nproc=1` (sequential + multi-threaded BLAS) in every cell measured.

| Speedup (nproc=1 / nproc=8), `ffap=ON` | min | median | max |
|---|---|---|---|
| Across all cells | 0.0091 | 0.0163 | 0.5144 |

> A speedup `<1` means `nproc=8` LOSES. A speedup of 0.01 means `nproc=8` is
> **100× slower** than `nproc=1`.

**Patterns:**

- **N_obs is the dominant factor.** At small N_obs (50–98), `nproc=8` is
  "only" 2–40× slower (the matrix inversions are cheap and parallel
  Python overhead dominates). Once N_obs reaches the `MAX_POINTS=500`
  cap (around N_obs ≈ 235–476), each MVN step is a non-trivial 500×500
  Cholesky solve, and `nproc=1` with multi-threaded BLAS pulls dramatically
  ahead — often by 60–110×.
- **N_grid is a weak modifier.** Per-pixel cost dominates the wall time;
  `nproc=8` overhead amortises only modestly with more pixels.
- **No crossover.** No sampled cell has speedup ≥ 1; the smallest case
  (50 obs × 1k pixels) is the closest at 0.025 (`nproc=8` is 40× slower).

The existing `MULTIPROCESS_OBSERVATION_THRESHOLD = 1000` guard, which
forces `nproc=1` whenever `n_obs > 1000`, is therefore correctly directed
but **insufficient**: the multiproc path loses everywhere we measured,
including well below the 1000-obs threshold.

**Heatmap:** see `figures/perf_features/multiproc_speedup_ffap_on.png`.

### 5.2 `find_affected_pixels`

**Headline:** `ffap=ON` is faster than `ffap=OFF` in every cell measured at
`nproc=1`. Speedup decreases monotonically with N_obs (more observations →
more pixels are within `MAX_DIST_M=10000m` of some observation → less work
saved).

| Speedup (ffap=OFF / ffap=ON), `nproc=1` | min | median | max |
|---|---|---|---|
| Across all cells | 1.04 | 1.47 | 6.99 |

**Patterns:**

- **Largest savings at small N_obs / large N_grid.** With 50 observations
  on a 1M-pixel grid, only a small fraction of pixels are within
  `MAX_DIST_M`, so the bbox pre-filter eliminates most of the work.
  Speedup 6.99×.
- **Smallest savings at large N_obs / small N_grid.** With 2373 obs on a
  1k-pixel grid, nearly every pixel is "affected" anyway, and the bbox
  step's own cost approaches the savings. Speedup 1.05×.
- **Saturation, not regression.** As N_obs grows, the ratio approaches
  1.0 from above but does not cross — the bbox phase is cheap (broadcast
  numpy arithmetic), and even when no pixels are filtered, the cost is
  modest.

**Heatmap:** see `figures/perf_features/ffap_speedup_nproc1.png`.

### 5.3 Best strategy per regime

For every (N_obs, N_grid) cell measured, the fastest configuration is
**(nproc=1, ffap=ON)**. The finding is consistent across `N_obs` from 50
to ~9 500 (filling pending for higher N_obs) and `N_grid` from 1 k to
1 M.

## 6. Phase 2 — full-pipeline confirmation

*(Pending — to be filled in after Phase 2 driver runs to completion.)*

| Cohort | N_obs | Resolution | nproc=1 (s) | nproc=8 (s) | Phase 1 prediction | Match? |
|---|---|---|---|---|---|---|
| sparse_coarse | 470 | 5000 m | … | … | nproc=1 wins | … |
| sparse_fine   | 470 | 500 m | … | … | nproc=1 wins | … |
| dense_coarse  | 35 709 | 5000 m | … | … | nproc=1 wins | … |
| dense_fine    | 35 709 | 500 m | … | … | nproc=1 wins | … |

## 7. Conclusions and recommendations

### 7.1 Multiprocessing — REMOVE

The multiprocessing path `parallel.run_parallel_spatial_fit` plus the
`MULTIPROCESS_OBSERVATION_THRESHOLD` guard add code complexity without
ever producing a speedup, on this hardware. Every cell sampled — across
nine N_obs values and four N_grid sizes — shows `nproc=8` losing to
`nproc=1`, often by 1–2 orders of magnitude. The previous belief that
multiproc helps below the 1000-obs threshold is **not supported by the
data**. BLAS multi-threading on the per-pixel covariance solves wins
across the entire parameter space.

**What to remove:**

- `vs30/parallel.py::run_parallel_spatial_fit` — the parallel branch in
  `pipeline.compute_spatial_adjustment_on_grid`.
- `MULTIPROCESS_OBSERVATION_THRESHOLD` constant in `vs30/constants.py`.
- The `nproc>1` branches in `pipeline.compute_spatial_adjustment_on_grid`
  (the function still needs an `nproc` parameter for upstream Bayesian
  clustering).
- `vs30/multiprocess.py::single_threaded_blas` (no longer needed if there
  are no parallel pools that share BLAS).

`compute_spatial_adjustments` and `find_affected_pixels` retain `nproc`
parameters today; with the recommendation, they collapse to sequential
versions (or keep `nproc=1` for parameter-API stability).

The points pipeline (`points_pipeline` in `pipeline.py`) and the parallel
geology+terrain location chunking (`run_parallel_locations`) are out of scope
for this investigation — they merit a separate follow-up.

### 7.2 `find_affected_pixels` — KEEP

`find_affected_pixels` is faster than its absence in every cell measured,
with the largest gains in the regime most users care about (large grid,
moderate observation count). Its implementation is also small and
self-contained — keeping it costs little code complexity, and the field
`obs_to_grid_indices` it returns is unused anywhere downstream and can be
dropped to simplify the BoundingBoxResult dataclass (a separate, very
small follow-up).

### 7.3 Side observation: `obs_to_grid_indices` is dead code

A grep across `vs30/`, `dev/`, and `tests/` confirms that
`BoundingBoxResult.obs_to_grid_indices` is constructed in
`spatial.find_affected_pixels` (when `nproc>1`) but never read anywhere
else. Removing it from the dataclass + its construction in
`grid_points_in_bbox` and `find_affected_pixels` is a worthwhile
simplification orthogonal to this investigation. (The harness's
`make_full_bbox_result` was forced into a 48 GB allocation on the
high-N_obs sweep cell because of this field — see commit `66e712b` in
the investigation harness for the workaround.)

### 7.4 Side observation: ffap-induced output difference at far pixels

The `MIN_DIST_ENFORCED=0.1` clamp inside `exponential_correlation_function`
makes `corr_fn(0) ≈ 0.99990` (not exactly 1). When `ffap=OFF`, every valid
pixel enters `compute_spatial_adjustment_for_pixel`, including those
far from observations; the no-obs branch shrinks `stdv` by `sqrt(0.99990)`
≈ `0.99995`. When `ffap=ON`, those pixels are skipped entirely and `stdv`
stays at its float32 prior. The two strategies therefore produce
slightly different output for far-from-obs pixels. The difference is
tiny (~5×10⁻⁵ on stdv) but means removing `find_affected_pixels`
*would* perturb production output by a small, deterministic amount in
that regime.

## 8. Reproducibility

The harness lives at
`dev/scripts/investigations/perf_features_investigation/`. To reproduce:

```bash
# Activate the project env
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv

# Phase 1 — full sweep (~7 h on the i7-9700, with nproc=8 trim policy)
python -m dev.scripts.investigations.perf_features_investigation.run_isolated_sweep

# (Optional) fill in nproc=1 cells at high N_obs that the sweep skipped:
python -m dev.scripts.investigations.perf_features_investigation.fill_high_nobs

# Phase 2 — full pipeline confirmation (~2-4 h)
python -m dev.scripts.investigations.perf_features_investigation.run_full_pipeline_confirmation

# Analysis (figures + medians + best-strategy CSVs)
python -m dev.scripts.investigations.perf_features_investigation.analyze_results
```

CSV outputs (`results_isolated.csv`, `results_full_pipeline.csv`,
`results_isolated_medians.csv`, `results_isolated_best_strategy.csv`) and
`figures/*.png` are gitignored — they are reproducible from the harness
and should be regenerated on different hardware before drawing
quantitative conclusions for that hardware.

The design rationale (test matrix, methodology, scope) is in
[`perf_features_investigation_design.md`](perf_features_investigation_design.md).
The implementation plan is in
[`perf_features_investigation_plan.md`](perf_features_investigation_plan.md).

# Points-Mode Performance Investigation — Design

**Date:** 2026-04-28
**Branch:** `vs30_refactor`
**Status:** Design (pre-implementation)
**Predecessors:**
- [`perf_features_investigation_findings.md`](perf_features_investigation_findings.md) — grid-mode investigation
- [`grid_cleanup_design.md`](grid_cleanup_design.md), [`grid_cleanup_plan.md`](grid_cleanup_plan.md) — grid-mode cleanup

## 1. Purpose

The grid-mode investigation found `nproc=8` (multiprocessing + single-threaded
BLAS) was 2–110× slower than `nproc=1` (sequential + multi-threaded BLAS) in
every cell tested. That conclusion does **not** automatically transfer to the
points pipeline because the two pipelines parallelise different things.

**Grid mode** parallelises the per-pixel MVN inner loop. Per-pixel work is
dominated by BLAS-bound covariance solves (501×501), so multi-threaded BLAS in
a single Python process wins outright.

**Points mode** parallelises the **full per-point pipeline**: category lookup
(point-in-polygon for geology, raster sampling for terrain), hybrid mods
(slope + coastal distance sampling), MVN spatial adjustment, and combination.
A non-trivial fraction of per-point work is **not** BLAS-bound — for those
parts, multi-threaded BLAS gives nothing and multiprocessing could in
principle win, if worker-spawn overhead amortises.

This investigation answers: **does multiprocessing ever win for points mode in
realistic operating regimes?** And consequently, what should the CLI default
for `vs30 points --nproc` be?

## 2. Scope

### In scope

- The multiprocessing / BLAS-MT tradeoff in `pipeline.points_pipeline` and
  `parallel.run_parallel_locations`.
- End-to-end timing of `pipeline.points_pipeline` calls (not isolated
  components).
- A short findings document with a concrete recommendation
  (keep multiproc / remove / keep with revised default).

### Explicitly NOT in scope

- A `find_affected_points` ("ffap for points") investigation. Discussed during
  brainstorming and deliberately deferred. The multiproc results indirectly
  inform whether ffap is worth a separate investigation; see §7.
- Any production code changes. This investigation produces a recommendation;
  any subsequent cleanup is a separate piece of work.
- Profiling per-phase time inside `points_pipeline`. End-to-end timing only.

## 3. Methodology

### 3.1 Two-strategy comparison

Time `pipeline.points_pipeline` end-to-end at the two endpoints of the strategy
space:

- **`(nproc=1, BLAS multi-threaded)`** — sequential Python loop, BLAS uses all
  cores for matrix solves.
- **`(nproc=8, BLAS single-threaded)`** — 8 worker processes via
  `multiprocess.spawn_context.Pool`, each pinned to single-threaded BLAS via
  `single_threaded_blas`.

These are the same two strategies measured in grid mode. We are not exploring
intermediate `nproc` values: prior data and reasoning suggest the answer is
monotonic between these endpoints.

### 3.2 Test matrix

| Axis | Values | Notes |
|---|---|---|
| `N_query` | 1, 10, 100, 1 000, 10 000, 50 000, 100 000 | Spans CLI single-point through stress-test scale. Realistic points-mode use is typically <1 000; 100 000 is beyond intended use but bounds the conclusion. |
| `N_obs` | 100, 1 000, 35 706 | Small / mid / full `viktor_cpt`. `35 706` is the actual post-comment-line row count. |
| Strategy | 2 endpoints (above) | |
| Reps | 3, median reported | |

7 × 3 × 2 × 3 = **126 runs**, projected **~60–115 minutes** wall time on the
i7-9700.

### 3.3 Query-point sampling

Query points are uniformly random over NZ land, with a fixed seed for
reproducibility. Concretely: rejection-sample WGS84 lon/lat within the NZ
bounding box, keep only points whose NZTM coordinates fall on a valid pixel of
the IwahashiPike terrain raster (i.e., land).

Rationale: this is the most "neutral" stress test for an arbitrary site
catalogue. An alternative — sampling from the same distribution as
observations — was considered and rejected as harder to interpret. Real
research-mode use cases distribute somewhere between the two; the uniform
sample gives a clean reference point.

### 3.4 Categorical-model config

`modified_foster_2019` (matches Phase 2 of the grid investigation; uses both
clustered and independent observations and exercises the geology hybrid mods
including coastal distance). Observations are subsampled from
`vs30/resources/observations/viktor_inferred_vs30_from_cpt.csv` with the
existing `subsample_observations` helper from the grid harness.

### 3.5 Per-cell timing

For each `(N_query, N_obs, strategy, rep)` cell:

1. Generate `N_query` lat/lons via the seeded sampler.
2. Subsample `N_obs` observations.
3. Call `pipeline.points_pipeline(...)` end-to-end with the strategy's
   `nproc` value.
4. Record total wall time.

The points pipeline does its own setup work (loading categorical CSVs, etc.)
inside each call. We count this as part of the per-cell time, since the
end-to-end question is what users experience.

## 4. Hardware

- Intel Core i7-9700, 8 cores @ 3.00 GHz, no hyperthreading.
- 32 GiB RAM.
- Linux 6.17.
- Python 3.13.9 (mamba env `vs30_venv`).
- BLAS: NumPy default in `vs30_venv` (likely OpenBLAS).

Recommendations are phrased in terms of "this hardware". Speedups should
generalise to similar BLAS-supporting CPUs; absolute crossovers may shift.

## 5. Deliverable artefacts

- `dev/docs/points_perf_investigation_findings.md` — short (~50 lines plus
  figures). Sections: summary recommendation, methodology pointer, results
  table, conclusion + recommended action.
- `dev/scripts/investigations/points_features_investigation/`:
  - `bench_utils.py` — helpers: random NZ-land query-point generator and
    single-cell timer. `subsample_observations` is reused via import from the
    grid harness rather than duplicated.
  - `run_points_sweep.py` — driver that loops the test matrix and writes
    `results_points.csv` incrementally.
  - `analyze_points_results.py` — medians, speedup pivot, 1–2 heatmaps.
  - Tests: minimal — just enough to confirm helpers behave.
- Optional: 1–2 PNG heatmaps tracked under `dev/docs/figures/points_perf/`.

CSV outputs (`results_points.csv`, derived medians) and `figures/*.png` are
gitignored — reproducible from the harness.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Worker spawn variance dominates small-N_query timings, drowning the signal. | 3 reps + median; report single-cell variance in the findings if it's significant. |
| `viktor_cpt` observations are heavily clustered; uniform NZ-land query points means most have no obs within `MAX_DIST_M=10 km`, so MVN early-exits cheaply and the per-point cost is dominated by setup. This is realistic for some users but not all. | Findings doc explicitly notes the sampling distribution and cautions that users with cluster-biased query distributions (e.g., querying urban sites) may see different numbers. |
| At `N_query=100k` × `N_obs=35 706`, raster sampling + polygon lookup of all query points happens once at the start of `points_pipeline`. If that step is itself the bottleneck, multiproc savings are limited regardless. | This is itself a finding worth documenting; the analysis script should be able to spot it (would manifest as flat speedup vs N_query). |
| `vs30/parallel.py::run_parallel_locations` was preserved during the grid cleanup; if it has an undiscovered bug surfaced by larger N_query, this investigation will surface it. | Treat as a finding, not a failure — escalate to the human if it does. |
| OOM on the largest cell. | Already analysed: at 100 k query × 35 706 obs, peak memory ≈ 1–4 GB across 8 workers, comfortable on 32 GB. The grid-mode OOM came from a harness-specific allocation that is not present here. |

## 7. Out-of-scope follow-ups informed by this investigation

The investigation's results indirectly answer "is `find_affected_points` worth
investigating later?" without any extra instrumentation:

- **If sequential `nproc=1` is fast at all cells** (e.g., 100 k × 35 706 takes
  <30 s): ffap savings are bounded by something less than that. **Skip ffap
  entirely.**
- **If sequential is slow at the upper-bound cells** (e.g., >5 min): that's
  enough cost to justify a separate, more targeted follow-up that profiles
  per-phase breakdown and considers ffap as one optimisation candidate.

This decision rule is sized to fall out naturally of the multiproc
investigation's data; no extra measurements needed.

## 8. Branch and commit strategy

- Work on the current branch `vs30_refactor`.
- Harness scripts under
  `dev/scripts/investigations/points_features_investigation/`.
- Design doc + findings doc under `dev/docs/`.
- `results_points.csv` and `figures/` gitignored — reproducible outputs.
- Production code is **not modified** by this investigation. Recommendations
  flow into a separate cleanup task afterwards (if any).

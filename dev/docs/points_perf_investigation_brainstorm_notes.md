# Points-Mode Performance Investigation — Brainstorm Checkpoint

**Status:** In-progress brainstorm. **Not yet a final spec.** Continuation point
for when we resume after debugging the
`test_grid_points_consistency_slow[jaehwi_v1p0]` failure that surfaced during
the grid cleanup's slow-tier validation.

## Where we are in the brainstorming workflow

| Step | Status |
|---|---|
| 1. Explore project context | ✅ Done (this session has full context on points pipeline) |
| 2. Visual companion | ✅ Skipped (text-only investigation) |
| 3. Clarifying questions | ✅ Done — see scope decisions below |
| 4. Propose 2–3 approaches | ✅ Done — converged on a small targeted investigation |
| 5. Present design | ⚙️ **In progress.** Last open question to user: "OK to lock the scope as multiproc / BLAS-MT tradeoff only, end-to-end timing?" |
| 6. Write design doc | Not started. Save to `dev/docs/points_perf_investigation_design.md`. |
| 7. Spec self-review | Not started |
| 8. User reviews written spec | Not started |
| 9. Transition to writing-plans | Not started |

When resuming: confirm the locked scope below with the user, then write the
formal spec doc.

## Locked scope decisions

The user approved each of these explicitly during the brainstorm.

- **Scope: multiprocessing / BLAS tradeoff only.** No `find_affected_points`
  ("ffap for points") investigation in this round. Rationale: the bottleneck
  framing is genuinely different from grid mode and the ffap savings would be
  bounded by the per-point distance-check cost which is already small.
- **Methodology: two strategy endpoints only.**
  - `(nproc=1, BLAS multi-threaded)`
  - `(nproc=8, BLAS single-threaded)`
- **Test matrix:**

  | Axis | Values |
  |---|---|
  | `N_query` | 1, 10, 100, 1 000, 10 000, 50 000, 100 000 |
  | `N_obs` | 100, 1 000, 35 706 |
  | Strategy | 2 endpoints (above) |
  | Reps | 3 (median reported) |

  Total: 7 × 3 × 2 × 3 = **126 runs**, projected **~60–115 minutes** wall time.

- **Query-point sampling:** uniform random over NZ land (fixed seed). Rationale:
  most "neutral" stress test for an arbitrary site catalogue. The cluster-biased
  alternative was considered and rejected as harder to interpret.
- **Categorical model config:** `modified_foster_2019` (matches Phase 2 of the
  grid investigation; uses both clustered and independent observations and
  exercises the geology hybrid mods).
- **Observations source:** `vs30/resources/observations/viktor_inferred_vs30_from_cpt.csv`.
  Subsample with the existing `subsample_observations` helper from the grid harness.

## ffap follow-up framing (agreed)

The multiproc investigation **indirectly** informs whether `find_affected_points`
is worth a separate investigation. The decision rule that emerged from the
brainstorm:

- **If sequential nproc=1 is fast at the upper-bound stress cell (e.g.,
  100k × 35 706 takes <30 s):** ffap savings are bounded by something less than
  that, so skip ffap entirely.
- **If sequential is slow at the largest cells (e.g., >5 min):** that's enough
  cost to justify a follow-up investigation that profiles the per-phase
  breakdown and considers ffap as one candidate optimisation.

So we do **not** need to instrument the multiproc investigation for ffap-specific
measurements; the magnitude of total per-cell timing answers the meta-question
naturally.

## Deliverable artefacts (planned)

- `dev/docs/points_perf_investigation_design.md` — short (~80–100 lines, much
  smaller than the 274-line grid design doc).
- `dev/scripts/investigations/points_features_investigation/`
  - `bench_utils.py` — shared helpers (random NZ-land query-point generation;
    deterministic seed; reuse `subsample_observations` from the grid harness if
    practical, otherwise duplicate).
  - `run_points_sweep.py` — driver that loops the matrix, calls
    `pipeline.points_pipeline` end-to-end, writes `results_points.csv`.
  - `analyze_points_results.py` — small analysis: medians, speedup pivots,
    1–2 heatmaps.
- `dev/docs/points_perf_investigation_findings.md` — short (~50 lines plus
  figures), structured as: summary recommendation, methodology, results table,
  conclusions.

## Cross-references / dependencies

- Grid investigation findings: `dev/docs/perf_features_investigation_findings.md`
- Grid cleanup design: `dev/docs/grid_cleanup_design.md`
- Grid cleanup plan: `dev/docs/grid_cleanup_plan.md`
- Existing harness for grid mode: `dev/scripts/investigations/perf_features_investigation/`

## Resume instructions

1. **First**: deal with the test failure
   `tests/test_grid_points_consistency.py::test_grid_points_consistency_slow[jaehwi_v1p0]`
   surfaced by `pytest --runslow` in the grid cleanup's Task 5 final validation.
   That blocks closing out the grid cleanup. The default tier passed; only the
   slow tier failed; the failing test compares grid vs points pipeline output
   for jaehwi_v1p0 across 38 NZ-wide points.
2. **Then**: re-engage with the user on the points-mode brainstorm, starting
   from "lock the scope summarised here" → write the formal spec doc → user
   review → invoke writing-plans.

This file can be deleted once the formal spec doc is written and committed.

# Points-Mode Post-Fix Performance Investigation — Design

**Date:** 2026-04-28
**Branch:** `vs30_refactor`
**Status:** Design (pre-implementation)
**Predecessors:**
- [Points-perf investigation findings (bug diagnosis)](points_perf_investigation_findings.md)
- [Points-pipeline obs-prep fix — design](parallel_points_obs_prep_fix_design.md)
- [Points-pipeline obs-prep fix — smoke results](parallel_points_obs_prep_fix_smoke_results.md)

## 1. Purpose

The prior investigation diagnosed a per-chunk obs-prep redundancy bug in
`vs30/parallel.py::run_parallel_locations` that made `nproc=8` runs 151×
slower than `nproc=1` at the worst observed cell. That bug has now been
fixed (commit `0fa92b3`), and a smoke benchmark on a single cell confirmed
the fix works (2,770 s → 128 s, ~21.7× recovery).

The smoke also surfaced that `nproc=8` is still ~6.8× *slower* than
`nproc=1` at `(N_query=1000, N_obs=35706)` — but that's now the inherent
multiproc/BLAS-MT tradeoff, not the bug. **The original investigation's
question — "does multiproc ever win for points mode in realistic regimes?"
— remains unanswered with clean data.** This investigation runs the full
sweep with the bug fixed to answer it, and acts on the answer by setting
the `vs30 points --nproc` CLI default appropriately.

## 2. Scope

### In scope

- Full sweep of `7 × 3 × {1, 2, 4, 8} × 3 = 252` cells. Same `N_query` and
  `N_obs` axes as the prior investigation, with two new intermediate
  `nproc` values (2 and 4) added to surface any sweet spot.
- New analysis script that computes per-cell medians, speedup pivot
  (relative to `nproc=1`), and a "best-nproc-per-cell" categorical heatmap.
- New findings document with the multiproc-tradeoff conclusions and a
  CLI-default recommendation.
- One-line CLI default change in `vs30/cli.py` based on the findings.

### Out of scope (deferrals)

- **`Pool(initializer=...)` optimisation.** Eliminates per-chunk pickle of
  `PointsObsData` (~10 s overhead at full N_obs). Worth pursuing only if
  the sweep shows pickle cost is a meaningful fraction of best-case
  multiproc time. The findings will explicitly call this out as a follow-up
  candidate, with a yes/no recommendation based on data.
- **Larger `N_query` / `N_obs` values.** The 7×3 grid covers realistic use.
- **Other model versions.** `modified_foster_2019` only, matching the prior
  investigation. Other versions could differ at large N_query but exploring
  that is its own investigation.
- **Memory measurements.** Same `RUSAGE_SELF` caveat as before — under
  `nproc>1` only the parent's RSS is captured. The findings will note this
  as a limitation rather than report memory.

## 3. Design

### 3.1 Sweep matrix and harness changes

| Axis | Values | Notes |
|---|---|---|
| `N_query` | 1, 10, 100, 1k, 10k, 50k, 100k | Same as prior. |
| `N_obs` | 100, 1k, 35706 | Same as prior. |
| `nproc` | **1, 2, 4, 8** | Two new intermediate values. |
| Reps | 3 | Median reported. |

**Total**: 252 cells. Estimated wall time: 2–4 hours on the i7-9700.

Driver changes (one file: `dev/scripts/investigations/points_features_investigation/run_points_sweep.py`):

- Restore `NPROC_VALUES = [1, 2, 4, 8]` (was `[1]` after the prior phase
  trim). Replace the inline comment block — the trim story is no longer
  relevant; the new context is "post-fix re-sweep across the multiproc
  endpoint range".
- Change `OUT_CSV = HERE / "results_points.csv"` to
  `OUT_CSV = HERE / "results_points_post_fix.csv"`. Isolates the new data
  from the prior phase's `results_points.csv` (pre-fix nproc=1-only) and
  `results_points_partial_with_buggy_nproc8.csv` (bug evidence). Both
  pre-existing files are preserved untouched as historical record.

The driver's other behaviour is unchanged: pre-generates the largest-N_query
NZ-land pool once, materialises one obs CSV per N_obs, writes results
incrementally to the CSV.

### 3.2 Analysis

New file `dev/scripts/investigations/points_features_investigation/analyze_points_post_fix_results.py`. Reads only the new post-fix CSV (does not combine with prior-phase data — those phases were measured against
different code states and conflating them would obscure the comparison).

**Outputs** (gitignored, regenerable):

- `results_points_post_fix_medians.csv` — 84 rows (21 cells × 4 nproc values).
- `results_points_post_fix_speedup.csv` — 21 rows; columns `t1`, `t2`, `t4`,
  `t8`, `speedup_2_vs_1`, `speedup_4_vs_1`, `speedup_8_vs_1`.
- `figures/speedup_nproc8_vs_1_post_fix.png` — endpoint speedup heatmap;
  direct visual analogue of the prior phase's `multiproc_speedup_points_buggy.png`.
- `figures/best_nproc_per_cell.png` — categorical heatmap labelling each
  (N_query, N_obs) cell with the `nproc ∈ {1, 2, 4, 8}` that minimises
  wall time, with the speedup over `nproc=1` shown alongside. Surfaces any
  sweet-spot at intermediate `nproc`.

Two heatmaps are sufficient. Per-nproc absolute-time heatmaps could be
added later but would dilute attention; the medians CSV is the raw-data
authority for citation in the findings.

The existing `analyze_points_results.py` is untouched. It still produces
the prior-phase artefacts on demand (e.g., regenerating
`multiproc_speedup_points_buggy.png` from `results_points_partial_with_buggy_nproc8.csv`).

### 3.3 Findings document

`dev/docs/points_perf_post_fix_findings.md`. Self-contained, cites all
predecessor docs (prior findings, bug-fix design, smoke results). Target
length ~120–180 lines.

Sections:

1. **Summary** — three-question table:
   - "Does multiproc ever win for points mode now?" — data-driven yes/no/
     "above threshold T".
   - "What should `vs30 points --nproc` default be?" — data-driven; concrete.
   - "Is there a sweet-spot at intermediate nproc?" — data-driven from the
     best-nproc map.
2. **Methodology** — brief, refers to predecessor docs for the bug, fix,
   and smoke verification.
3. **Results** — three subsections:
   - 3.1: Absolute wall time per nproc, table of medians from
     `results_points_post_fix_medians.csv`.
   - 3.2: Speedup vs nproc=1, with the speedup heatmap embedded.
   - 3.3: Best nproc per cell, with the categorical heatmap embedded.
4. **Comparison to pre-fix** — short before/after table at the headline
   cells (`N_query=1000, N_obs=35706` is the canonical comparison point
   from the smoke).
5. **Conclusions and recommendations** —
   - 5.1: CLI default decision (one of: keep `1`, change to `2`/`4`/`-1`),
     with concrete file:line references.
   - 5.2: Pool initializer optimisation follow-up (yes/no, based on the
     ratio of best-multiproc time to pure-pickle estimate).
6. **Hardware and software** — same hardware as predecessors; record the
   vs30 commit at sweep time.
7. **Reproducibility** — runnable commands for the sweep + analysis.

### 3.4 CLI default decision rule

The CLI default change is mechanical given the analysis output:

| Best-nproc map outcome | Recommended default |
|---|---|
| `nproc=1` wins everywhere or in most realistic cells (`N_query ≤ 10k`) | Keep `1`. |
| Some intermediate `nproc` (`2` or `4`) wins broadly | Change to that value. |
| `nproc=-1` (=8 on this hardware) wins broadly across realistic cells | Change to `-1`. |
| Mixed / no clear winner | Keep `1` (safest); document override guidance. |

If the recommendation is non-trivial, the change is a one-line edit at
`vs30/cli.py:254` (the `points` command) and `vs30/cli.py:349` (the
`points_custom` command). The commit message cites the findings doc as the
motivating source.

## 4. Validation

The sweep itself produces validation evidence — there is no separate test
harness. Existing pytest suite (`tests/test_benchmarks.py`,
`tests/test_grid_points_consistency.py`) continues to gate correctness
across both the parallel and sequential code paths. They were re-run after
the bug fix and remain the correctness gate; no further test changes are
needed for this work since the production-side change is at most a one-line
default-value edit.

A re-run of the pytest suite is performed once before the CLI default
commit lands, to confirm the suite still passes after the (potential)
default change.

## 5. Branch and commit strategy

- Continue on `vs30_refactor`.
- Expected commits, in order:
  1. **harness**: restore `NPROC_VALUES`, change `OUT_CSV` path, update
     inline comment in `run_points_sweep.py`.
  2. **analysis**: add `analyze_points_post_fix_results.py`.
  3. *(execution: no commit — sweep CSV and figures are gitignored.)*
  4. **findings**: add `dev/docs/points_perf_post_fix_findings.md` with
     copied heatmaps under `dev/docs/figures/points_perf_post_fix/`.
  5. **CLI default**: one-line (or two-line) change in `vs30/cli.py`
     conditional on the findings recommending it. Skipped if the
     recommendation is "keep `1`".

5 expected commits (4 if the CLI default doesn't change). The sweep runs
unattended in the background between commits 2 and 4.

## 6. Risks and mitigations

| Risk | Mitigation |
|---|---|
| Variance higher than expected at intermediate nproc values, making the best-nproc map noisy. | 3 reps + median (matches prior). If a cell's reps disagree on which nproc wins, the analysis flags it (no concrete plan needed — findings doc notes the ambiguity). |
| `nproc=2` or `nproc=4` triggers an unexpected code path (unlikely — `multiprocess.Pool(processes=K)` and `single_threaded_blas()` are nproc-agnostic), causing a failure mid-sweep. | The driver's existing per-cell `try/except` writes a `None`-valued failed row and continues. Failed cells are flagged at the end of the sweep and re-run (or noted in findings). |
| Sweep takes longer than 4 hours on this hardware. | Acceptable — investigation time, not user-facing. The driver writes incrementally so partial results are usable. |
| `Pool(initializer=...)` would dominate the best-multiproc time, but we don't include the optimisation here. | Findings explicitly recommends a follow-up if data warrants. Not blocking for the CLI default decision: the decision is between current options, not hypothetical optimisations. |
| CLI default change inadvertently breaks a downstream user's script that relied on the old default. | The CLI default is the only externally observable change in this whole work, so it's the only blast-radius concern. The change is documented in the findings; the code change is one-line and trivially revertable. |

## 7. Out-of-scope follow-ups (informed by this work)

- **`Pool(initializer=...)` for `PointsObsData`** if the findings show
  pickle cost is meaningfully high. Its own brainstorm → design → plan
  cycle.
- **Re-tuning `N_PROGRESS_CHUNKS`.** Currently `1000`. With the bug fixed,
  fewer chunks reduce pickle volume; smaller chunks improve progress-bar
  granularity. The right value is data-driven and could be revisited if
  the post-fix data shows chunk-overhead dominating.

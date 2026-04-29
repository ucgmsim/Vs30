# Points-Mode Performance Investigation Harness

Code to measure the multiprocessing / BLAS-MT tradeoff in
`pipeline.points_pipeline` end-to-end. Used for two investigations on this
branch:

1. **Original investigation + bug fix**: discovered a per-chunk obs-prep
   redundancy bug in `vs30/parallel.py::run_parallel_locations` and fixed it.
2. **Post-fix investigation**: measured the inherent multiproc/BLAS-MT
   tradeoff with the bug fixed, plus a balanced-BLAS supplement at
   intermediate `nproc` values.

See, in chronological order:
- `dev/docs/points_perf_investigation_design.md` — original design.
- `dev/docs/points_perf_investigation_findings.md` — bug-discovery findings.
- `dev/docs/parallel_points_obs_prep_fix_design.md` — bug-fix design.
- `dev/docs/parallel_points_obs_prep_fix_smoke_results.md` — bug-fix smoke verification.
- `dev/docs/points_perf_post_fix_design.md` — post-fix-investigation design.
- `dev/docs/points_perf_post_fix_findings.md` — **current findings + recommendations**.

## Files

| File | Role |
|---|---|
| `bench_utils.py` | Shared helpers (NZ-land query-point generation, obs-CSV materialisation, single-cell timer) |
| `test_bench_utils.py` | Unit tests for the helpers |
| `run_points_sweep.py` | Sweep driver — full `(N_query × N_obs × nproc × rep)` matrix |
| `run_balanced_blas_supplement.py` | Supplement driver — `nproc ∈ {2, 4, 6}` only, with the production balanced-BLAS fix in place |
| `analyze_points_results.py` | Analyser for the original investigation's CSVs (kept for reproducibility of pre-fix artefacts) |
| `analyze_points_post_fix_results.py` | **Current analyser** — reads both the post-fix sweep and the balanced-BLAS supplement and emits combined heatmaps + CSVs |

## Run order (current investigation)

```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv

# Main post-fix sweep (~3-4 hr; 252 cells)
python -m dev.scripts.investigations.points_features_investigation.run_points_sweep

# Balanced-BLAS supplement (~2 hr; 63 cells, nproc=2,4,6)
python -m dev.scripts.investigations.points_features_investigation.run_balanced_blas_supplement

# Combined analysis
python -m dev.scripts.investigations.points_features_investigation.analyze_points_post_fix_results
```

Outputs (gitignored, regenerable):
- `results_points.csv` — pre-fix nproc=1-only sweep (preserved as historical record)
- `results_points_partial_with_buggy_nproc8.csv` — pre-fix partial with the obs-prep bug evidence (preserved)
- `results_points_post_fix.csv` — post-fix 4-nproc sweep
- `results_balanced_blas_supplement.csv` — balanced-BLAS supplement
- `results_points_post_fix_medians.csv`, `results_points_post_fix_speedup.csv` — derived from the post-fix sweep
- `results_points_post_fix_combined_medians.csv`, `results_points_post_fix_combined_wide.csv` — combined post-fix + supplement
- `obs_csvs/` — materialised subsampled observation CSVs
- `figures/*.png` — heatmaps

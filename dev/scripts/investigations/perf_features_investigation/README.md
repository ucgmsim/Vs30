# Performance Features Investigation Harness

Code to measure the effect of multiprocessing and `find_affected_pixels`
on `vs30` grid-mode performance. The investigation is complete — its
findings drove the multiproc removal in the grid pipeline, and the
recommendations now live in the codebase. The harness is preserved here
for reproducibility on different hardware.

## Run order

```bash
# Activate env (assumes mamba/vs30_venv already configured)
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv

# Phase 1: isolated MVN sweep
python -m dev.scripts.investigations.perf_features_investigation.run_isolated_sweep

# Phase 2: full-pipeline confirmation
python -m dev.scripts.investigations.perf_features_investigation.run_full_pipeline_confirmation

# Analyse
python -m dev.scripts.investigations.perf_features_investigation.analyze_results
```

Outputs:
- `results_isolated.csv`, `results_full_pipeline.csv` — raw timings (gitignored)
- `figures/*.png` — speedup heatmaps (gitignored)

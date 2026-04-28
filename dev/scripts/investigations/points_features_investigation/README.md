# Points-Mode Performance Investigation Harness

Code to measure the multiprocessing / BLAS-MT tradeoff in
`pipeline.points_pipeline` end-to-end.

See:
- `dev/docs/points_perf_investigation_design.md` — methodology and rationale
- `dev/docs/points_perf_investigation_findings.md` — results and recommendations

## Run order

```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv

# Run the sweep (~60-115 minutes)
python -m dev.scripts.investigations.points_features_investigation.run_points_sweep

# Analyse
python -m dev.scripts.investigations.points_features_investigation.analyze_points_results
```

Outputs (gitignored, regenerable):
- `results_points.csv` — raw per-cell timings
- `results_points_medians.csv` — per-cell medians
- `obs_csvs/` — materialised subsampled observation CSVs
- `figures/*.png` — heatmaps

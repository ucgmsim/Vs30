"""Phase 1 driver — isolated MVN parameter sweep.

Loops the (N_obs x N_grid x nproc x ffap x rep) matrix and writes one CSV
row per cell. CSV is written incrementally so partial results survive an
interrupt or crash.

Run with::

    python -m dev.scripts.investigations.perf_features_investigation.run_isolated_sweep
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import bench_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("perf_sweep")

OUT_CSV = Path(__file__).parent / "results_isolated.csv"

CSV_FIELDS = [
    "N_obs",
    "N_grid_target",
    "N_grid_actual",
    "N_affected",
    "nproc",
    "ffap",
    "rep",
    "t_bbox_s",
    "t_spatial_s",
    "t_total_s",
    "peak_rss_mb",
    "timestamp_iso",
]

# Full sweep — overridden when --smoke is passed.
N_OBS_VALUES = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 35709]
N_GRID_VALUES = [1_000, 10_000, 100_000, 1_000_000]
NPROC_VALUES = [1, 8]
FFAP_VALUES = [True, False]
N_REPS = 3

# Skip cells whose previous-rep total time exceeds this — the sweep budget
# is finite and very long cells contribute little additional information.
PER_CELL_TIME_BUDGET_S = 1800.0


def _append_row(row: dict) -> None:
    new_file = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _run_cell(raster_data, obs_data, n_obs, n_grid_target, nproc, ffap, n_reps) -> None:
    for rep in range(n_reps):
        logger.info(
            f"  cell N_obs={n_obs:>6} N_grid_target={n_grid_target:>9,} "
            f"nproc={nproc} ffap={int(ffap)} rep={rep}"
        )
        try:
            row = bench_utils.time_one_run(
                raster_data=raster_data,
                obs_data=obs_data,
                nproc=nproc,
                ffap=ffap,
                rep=rep,
            )
        except Exception:
            logger.exception("    cell failed — recording empty row and moving on")
            row = {k: None for k in CSV_FIELDS}
            row.update(
                {
                    "N_obs": n_obs,
                    "N_grid_target": n_grid_target,
                    "nproc": nproc,
                    "ffap": ffap,
                    "rep": rep,
                }
            )
            _append_row(row)
            return
        row["N_grid_target"] = n_grid_target
        _append_row(row)
        if row["t_total_s"] > PER_CELL_TIME_BUDGET_S and rep < n_reps - 1:
            logger.warning(
                f"    cell exceeded budget ({row['t_total_s']:.1f}s) — skipping remaining reps"
            )
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny matrix (3 N_obs x 2 N_grid x 2 nproc x 2 ffap x 1 rep)",
    )
    args = parser.parse_args()

    if args.smoke:
        n_obs_values = [50, 250, 1000]
        n_grid_values = [1_000, 10_000]
        n_reps = 1
    else:
        n_obs_values = N_OBS_VALUES
        n_grid_values = N_GRID_VALUES
        n_reps = N_REPS

    logger.info("Numerical equivalence guardrail starting...")
    bench_utils.run_numerical_equivalence_check(
        n_obs=200, n_target=1_000, nproc_options=(1, 8)
    )
    logger.info("Numerical equivalence guardrail passed.")

    for n_obs in n_obs_values:
        obs_df = bench_utils.subsample_observations(n_obs, seed=42)
        for n_grid in n_grid_values:
            logger.info(
                f"Building raster data: N_obs={n_obs}, N_grid_target={n_grid:,}"
            )
            raster_data, _ = bench_utils.make_raster_data(n_grid)
            obs_data = bench_utils.prepare_terrain_obs_data(obs_df, raster_data)
            for nproc in NPROC_VALUES:
                for ffap in FFAP_VALUES:
                    _run_cell(raster_data, obs_data, n_obs, n_grid, nproc, ffap, n_reps)

    logger.info(f"Sweep complete - results at {OUT_CSV}")


if __name__ == "__main__":
    main()

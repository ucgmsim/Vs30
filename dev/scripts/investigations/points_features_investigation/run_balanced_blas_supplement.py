"""Balanced-BLAS supplement sweep — nproc=2 and nproc=4 only.

Re-runs the same (N_query x N_obs) matrix as run_points_sweep.py, but only
for the intermediate nproc values (2 and 4), with the production code now
allocating BLAS threads proportionally to nproc (commit 827a71b). The
single-threaded-BLAS data for those cells lives in results_points_post_fix.csv;
this script's output, results_balanced_blas_supplement.csv, lets the
analysis quantify the recovery from balanced-BLAS allocation.

nproc=1 is unaffected (sequential path bypasses the balanced-BLAS code) and
nproc=8 = cpu_count saturates the CPU either way, so neither needs re-running.

Run with::

    python -m dev.scripts.investigations.points_features_investigation.run_balanced_blas_supplement
"""

import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import bench_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("balanced_blas_supplement")

HERE = Path(__file__).parent
OUT_CSV = HERE / "results_balanced_blas_supplement.csv"
OBS_DIR = HERE / "obs_csvs"

CSV_FIELDS = [
    "N_query",
    "N_obs",
    "nproc",
    "rep",
    "t_total_s",
    "peak_rss_mb",
    "timestamp_iso",
]

# Same N_query and N_obs axes as run_points_sweep.py; only intermediate nproc.
N_QUERY_VALUES = [1, 10, 100, 1_000, 10_000, 50_000, 100_000]
N_OBS_VALUES = [100, 1_000, 35_706]
NPROC_VALUES = [2, 4]
N_REPS = 3


def _append_row(row: dict) -> None:
    new_file = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    logger.info("Loading modified_foster_2019 config...")
    cfg = bench_utils.load_modified_foster_2019_config()

    logger.info(f"Materialising obs CSVs for N_obs in {N_OBS_VALUES}")
    obs_paths = bench_utils.materialize_obs_csvs(OBS_DIR, N_OBS_VALUES)

    largest_n_query = max(N_QUERY_VALUES)
    logger.info(f"Pre-generating {largest_n_query:,} NZ-land query points...")
    lons_pool, lats_pool = bench_utils.generate_nz_land_points(largest_n_query, seed=42)
    logger.info(f"Pool ready ({len(lons_pool):,} points).")

    for n_query in N_QUERY_VALUES:
        lons = lons_pool[:n_query]
        lats = lats_pool[:n_query]
        for n_obs in N_OBS_VALUES:
            obs_csv_path = obs_paths[n_obs]
            for nproc in NPROC_VALUES:
                for rep in range(N_REPS):
                    logger.info(
                        f"  cell N_query={n_query:>6} N_obs={n_obs:>6} "
                        f"nproc={nproc} rep={rep}"
                    )
                    try:
                        row = bench_utils.time_one_run(
                            lons=lons,
                            lats=lats,
                            obs_csv_path=obs_csv_path,
                            nproc=nproc,
                            rep=rep,
                            cfg=cfg,
                        )
                    except Exception:
                        logger.exception(
                            "    cell failed — recording empty row and moving on"
                        )
                        row = {k: None for k in CSV_FIELDS}
                        row.update(
                            {
                                "N_query": n_query,
                                "N_obs": n_obs,
                                "nproc": nproc,
                                "rep": rep,
                            }
                        )
                    _append_row(row)

    logger.info(f"Sweep complete - results at {OUT_CSV}")


if __name__ == "__main__":
    main()

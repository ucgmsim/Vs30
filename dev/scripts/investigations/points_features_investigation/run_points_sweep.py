"""Points-mode performance sweep driver.

Loops the (N_query x N_obs x nproc x rep) matrix and writes one CSV row per
cell. CSV is written incrementally so partial results survive an interrupt.

Run with::

    python -m dev.scripts.investigations.points_features_investigation.run_points_sweep
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import bench_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("points_sweep")

HERE = Path(__file__).parent
OUT_CSV = HERE / "results_points.csv"
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

# Full sweep — overridden by --smoke.
N_QUERY_VALUES = [1, 10, 100, 1_000, 10_000, 50_000, 100_000]
N_OBS_VALUES = [100, 1_000, 35_706]
# Trimmed from [1, 8] mid-investigation: a per-chunk obs-prep redundancy bug
# in vs30/parallel.py::run_parallel_locations made nproc=8 cells take >150x
# longer than nproc=1 at large N_query. The bug, not the inherent
# multiproc/BLAS-MT tradeoff, dominates those measurements. We complete the
# nproc=1 sweep to answer the §7 ffap question; a separate piece of work will
# fix the bug and re-measure nproc=8 cleanly. Pre-trim partial data with the
# buggy nproc=8 cells is preserved in
# results_points_partial_with_buggy_nproc8.csv (gitignored, regenerable).
NPROC_VALUES = [1]
N_REPS = 3


def _append_row(row: dict) -> None:
    new_file = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny matrix (3 N_query x 1 N_obs x 2 nproc x 1 rep) for verification.",
    )
    args = parser.parse_args()

    if args.smoke:
        n_query_values = [1, 10, 100]
        n_obs_values = [100]
        n_reps = 1
    else:
        n_query_values = N_QUERY_VALUES
        n_obs_values = N_OBS_VALUES
        n_reps = N_REPS

    logger.info("Loading modified_foster_2019 config...")
    cfg = bench_utils.load_modified_foster_2019_config()

    logger.info(f"Materialising obs CSVs for N_obs in {n_obs_values}")
    obs_paths = bench_utils.materialize_obs_csvs(OBS_DIR, n_obs_values)

    largest_n_query = max(n_query_values)
    logger.info(f"Pre-generating {largest_n_query:,} NZ-land query points...")
    lons_pool, lats_pool = bench_utils.generate_nz_land_points(largest_n_query, seed=42)
    logger.info(f"Pool ready ({len(lons_pool):,} points).")

    for n_query in n_query_values:
        # Sub-sample the pre-generated pool deterministically (first n_query points).
        lons = lons_pool[:n_query]
        lats = lats_pool[:n_query]
        for n_obs in n_obs_values:
            obs_csv_path = obs_paths[n_obs]
            for nproc in NPROC_VALUES:
                for rep in range(n_reps):
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

"""Fill in missing high-N_obs nproc=1 cells after Phase 1 was stopped early.

The Phase 1 sweep was stopped after ~285 cells (mostly through N_obs=5000)
to avoid 9-10 more hours of waiting on slow nproc=8 cells. The conclusions
on multiproc do not need more nproc=8 data — the trend is unambiguous.
This script runs only the **fast** nproc=1 cells at the unrun high N_obs
values, completing the ffap-vs-N_obs picture (especially at N_grid=1M
where ffap savings are largest).

Run::

    python -m dev.scripts.investigations.perf_features_investigation.fill_high_nobs
"""

import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import bench_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("fill_high_nobs")

OUT_CSV = Path(__file__).parent / "results_isolated.csv"

CSV_FIELDS = [
    "N_obs",
    "N_grid_target",
    "N_grid_actual",
    "N_affected",
    "ffap",
    "rep",
    "t_bbox_s",
    "t_spatial_s",
    "t_total_s",
    "peak_rss_mb",
    "timestamp_iso",
]

# Only the cells the Phase 1 sweep didn't reach. nproc=1 only — nproc=8 at
# these sizes was already known to be far slower from cells we did run.
# Note: the 5000 and 10000 tiers were completed by earlier fill-in
# attempts; only the 35706 tier remains. The viktor_cpt CSV has 35706
# rows after the comment-line filter (the design-doc figure of 35709 was
# a wc-l count that included comment lines).
N_OBS_VALUES = [35706]
N_GRID_VALUES = [1_000, 10_000, 100_000, 1_000_000]
N_REPS = 3


def _append(row: dict) -> None:
    new_file = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    for n_obs in N_OBS_VALUES:
        obs_df = bench_utils.subsample_observations(n_obs, seed=42)
        for n_grid in N_GRID_VALUES:
            logger.info(f"Building raster for N_obs={n_obs} N_grid_target={n_grid:,}")
            raster_data, _ = bench_utils.make_raster_data(n_grid)
            obs_data = bench_utils.prepare_terrain_obs_data(obs_df, raster_data)
            for ffap in (True, False):
                for rep in range(N_REPS):
                    logger.info(
                        f"  cell N_obs={n_obs:>6} N_grid_target={n_grid:>9,} "
                        f"ffap={int(ffap)} rep={rep}"
                    )
                    row = bench_utils.time_one_run(
                        raster_data=raster_data,
                        obs_data=obs_data,
                        ffap=ffap,
                        rep=rep,
                    )
                    row["N_grid_target"] = n_grid
                    _append(row)
    logger.info("Fill-in complete.")


if __name__ == "__main__":
    main()

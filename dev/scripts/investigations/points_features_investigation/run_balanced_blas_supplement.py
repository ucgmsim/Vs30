"""Balanced-BLAS supplement sweep — nproc in {2, 4, 6}.

Re-runs the same (N_query x N_obs) matrix as run_points_sweep.py for nproc=2,
4, and 6 with BLAS threads set so workers collectively saturate the 8-core CPU:

- nproc=2: production formula gives 4 BLAS threads each (2 x 4 = 8 cores).
- nproc=4: production formula gives 2 BLAS threads each (4 x 2 = 8 cores).
- nproc=6: production formula would give 1 BLAS thread each (only 6 cores
    active). Investigation override: force 2 BLAS threads (12 threads on 8
    cores; mild oversubscription but full CPU utilisation). Implemented by
    monkey-patching ``vs30.multiprocess.limit_blas_threads`` for the cell.

nproc=1 is unaffected (sequential path bypasses the balanced-BLAS code) and
nproc=8 = cpu_count already saturates the CPU; both have valid post-fix data
in results_points_post_fix.csv already.

Single rep per cell. Per-cell variance in the original 3-rep runs was
sub-1%, so single-rep is statistically reliable for this comparison.

Run with::

    python -m dev.scripts.investigations.points_features_investigation.run_balanced_blas_supplement
"""

import contextlib
import csv
import logging
import sys
from pathlib import Path

import threadpoolctl

sys.path.insert(0, str(Path(__file__).parent))

import bench_utils
from vs30 import multiprocess as _mp

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

N_QUERY_VALUES = [1, 10, 100, 1_000, 10_000, 50_000, 100_000]
N_OBS_VALUES = [100, 1_000, 35_706]
NPROC_VALUES = [2, 4, 6]
N_REPS = 1

# Override production's max(1, cpu_count // nproc) BLAS allocation for nproc
# values where floor-division leaves cores idle on this 8-core machine.
# Override 2 threads on nproc=6 = 12 threads on 8 cores (mild oversubscription
# for full CPU utilisation, vs production's 6 cores active).
_BLAS_OVERRIDES = {6: 2}

_original_limit_blas_threads = _mp.limit_blas_threads


@contextlib.contextmanager
def _blas_override_for_nproc(nproc: int):
    """Monkey-patch multiprocess.limit_blas_threads when nproc has an override.

    Production code in run_parallel_locations calls
    ``multiprocess.limit_blas_threads(threads=max(1, cpu_count // nproc))``.
    For nproc values listed in ``_BLAS_OVERRIDES``, we replace that helper
    with one that ignores the kwargs and uses our override threads value
    instead. The replacement is restored after the cell finishes.
    """
    if nproc not in _BLAS_OVERRIDES:
        yield
        return

    threads = _BLAS_OVERRIDES[nproc]

    @contextlib.contextmanager
    def _override(**_kwargs):
        with threadpoolctl.threadpool_limits(limits=threads, user_api="blas"):
            yield

    _mp.limit_blas_threads = _override
    try:
        yield
    finally:
        _mp.limit_blas_threads = _original_limit_blas_threads


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
                        with _blas_override_for_nproc(nproc):
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

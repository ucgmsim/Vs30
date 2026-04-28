"""Phase 2 driver — full-pipeline confirmation runs.

Runs pipeline.grid_pipeline end-to-end for four (obs density, resolution)
cohorts and records wall-time. Used to confirm that the Phase 1 ordering
of strategies survives the surrounding pipeline overhead.

Run::

    python -m dev.scripts.investigations.perf_features_investigation.run_full_pipeline_confirmation
"""

import argparse
import csv
import datetime as dt
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import bench_utils  # noqa: F401  -- ensures repo root is on sys.path

from vs30 import config, constants, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("perf_full_pipeline")

HERE = Path(__file__).parent
OUT_CSV = HERE / "results_full_pipeline.csv"

CSV_FIELDS = [
    "cohort_label",
    "N_obs_label",
    "resolution_m",
    "nproc",
    "t_total_s",
    "timestamp_iso",
]

# (cohort_label, model_version_for_resources, resolution_m)
# The model version supplies all the resources (categorical CSVs, observation
# CSVs, correlation functions); we just override the resolution.
COHORTS = [
    ("sparse_coarse", constants.FixedModelVersion.MODIFIED_FOSTER_2019, 5000),
    ("sparse_fine", constants.FixedModelVersion.MODIFIED_FOSTER_2019, 500),
    ("dense_coarse", constants.FixedModelVersion.VIKTOR_CPT_CLUSTERING, 5000),
    ("dense_fine", constants.FixedModelVersion.VIKTOR_CPT_CLUSTERING, 500),
]


def _load_cli_config(version: constants.FixedModelVersion) -> dict:
    """Reuse cli.load_model_config to get fully-resolved config + corr fns."""
    from vs30 import cli

    return cli.load_model_config(version)


def _grid_for(resolution: int) -> config.GridConfig:
    base = constants.FULL_NZ_GRID_CONFIG
    return config.GridConfig(
        grid_xmin=base.grid_xmin,
        grid_xmax=base.grid_xmax,
        grid_ymin=base.grid_ymin,
        grid_ymax=base.grid_ymax,
        grid_dx=resolution,
        grid_dy=resolution,
    )


def _append_row(row: dict) -> None:
    new_file = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run_cohort(
    cohort_label: str,
    version: constants.FixedModelVersion,
    resolution: int,
    nproc: int,
) -> None:
    cfg = _load_cli_config(version)
    grid = _grid_for(resolution)
    logger.info(f"=== {cohort_label} resolution={resolution}m nproc={nproc} ===")
    t0 = time.perf_counter()
    pipeline.grid_pipeline(
        grid_config=grid,
        output_dir=None,
        geology_categorical_csv=cfg["geology_categorical_csv"],
        terrain_categorical_csv=cfg["terrain_categorical_csv"],
        clustered_observations_csv=cfg["clustered_observations_csv"],
        independent_observations_csv=cfg["independent_observations_csv"],
        combination_method=constants.CombinationMethod(cfg["combination_method"]),
        combine_ratio=cfg["combine_ratio"],
        noisy=cfg["noisy"],
        do_bayesian_update=cfg["do_bayesian_update"],
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        apply_coastal_distance_mod=cfg["apply_coastal_distance_mod"],
        fill_gaps=cfg["fill_gaps"],
        geology_corr_fn=cfg["geology_corr_fn"],
        terrain_corr_fn=cfg["terrain_corr_fn"],
        dbscan_nproc=nproc,
    )
    t_total = time.perf_counter() - t0
    _append_row(
        {
            "cohort_label": cohort_label,
            "N_obs_label": str(version),
            "resolution_m": resolution,
            "nproc": nproc,
            "t_total_s": t_total,
            "timestamp_iso": dt.datetime.now().isoformat(timespec="seconds"),
        }
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Only the cheapest two cohorts at one nproc",
    )
    args = parser.parse_args()

    if args.smoke:
        cohorts_with_nproc = [(c, (1,)) for c in COHORTS[:2]]
    else:
        # Phase 1 already established nproc=8 loses everywhere by 50-100x;
        # the only meaningful end-to-end multiproc comparison is on a sparse
        # cohort (the dense cohorts auto-fall-back to nproc=1 via the
        # MULTIPROCESS_OBSERVATION_THRESHOLD guard). Run all 4 cohorts at
        # nproc=1, plus sparse_coarse at nproc=8 to confirm Phase 1's
        # finding survives the surrounding pipeline overhead.
        cohorts_with_nproc = []
        for label, version, resolution in COHORTS:
            if label == "sparse_coarse":
                cohorts_with_nproc.append(((label, version, resolution), (1, 8)))
            else:
                cohorts_with_nproc.append(((label, version, resolution), (1,)))

    for cohort, nproc_options in cohorts_with_nproc:
        label, version, resolution = cohort
        for nproc in nproc_options:
            run_cohort(label, version, resolution, nproc)

    logger.info(f"Phase 2 complete - results at {OUT_CSV}")


if __name__ == "__main__":
    main()

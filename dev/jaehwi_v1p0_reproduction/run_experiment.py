"""Run Jaehwi v1p0 reproduction experiments.

Each experiment varies the pipeline configuration (categorical CSVs, Bayesian
update, combination method, coastal modification) and compares the output
subgrid against Jaehwi's reference V1.0_26Mar.tif.
"""

import subprocess
import sys
from pathlib import Path

# Add vs30 package to path
sys.path.insert(0, str(Path("/home/arr65/src/vs30")))

from vs30 import config, constants, pipeline

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path("/home/arr65/src/vs30")
VS30_PKG = REPO / "vs30"
EXPERIMENT_DIR = REPO / "dev" / "jaehwi_v1p0_reproduction"
REFERENCE_SUBGRID = EXPERIMENT_DIR / "reference_subgrid.tif"

CATEGORICAL_DIR = VS30_PKG / "resources" / "categorical_vs30_mean_and_stddev"
OBSERVATIONS_DIR = VS30_PKG / "resources" / "observations"
FOSTER_GEOL_POST = (
    CATEGORICAL_DIR
    / "geology_model_posterior_from_foster_2019_mean_and_standard_deviation.csv"
)
FOSTER_TERR_POST = (
    CATEGORICAL_DIR
    / "terrain_model_posterior_from_foster_2019_mean_and_standard_deviation.csv"
)
GEOL_PRIOR = CATEGORICAL_DIR / "geology_model_prior_mean_and_standard_deviation.csv"
TERR_PRIOR = CATEGORICAL_DIR / "terrain_model_prior_mean_and_standard_deviation.csv"
CURRENT_OBS = OBSERVATIONS_DIR / "jaehwi_v1p0_independent_observations.csv"
RECONSTRUCTED_OBS = EXPERIMENT_DIR / "jaehwi_reconstructed_observations.csv"
COMPARE_SCRIPT = REPO / "dev" / "compare_rasters.py"

# ---------------------------------------------------------------------------
# Subgrid (Wellington region)
# ---------------------------------------------------------------------------
SUBGRID = config.GridConfig(
    grid_xmin=1555050,
    grid_xmax=1610050,
    grid_ymin=5145050,
    grid_ymax=5195050,
    grid_dx=100,
    grid_dy=100,
)

# ---------------------------------------------------------------------------
# Experiment definitions
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "exp0_baseline_current_config": dict(
        geology_categorical_csv=GEOL_PRIOR,
        terrain_categorical_csv=TERR_PRIOR,
        do_bayesian_update=True,
        combination_method=constants.CombinationMethod.STANDARD_DEVIATION_WEIGHTING,
        combine_ratio=None,
        apply_coastal_distance_mod=True,
        independent_observations_csv=CURRENT_OBS,
    ),
    "exp1_foster_posterior_ratio_no_coast": dict(
        geology_categorical_csv=FOSTER_GEOL_POST,
        terrain_categorical_csv=FOSTER_TERR_POST,
        do_bayesian_update=False,
        combination_method=constants.CombinationMethod.RATIO,
        combine_ratio=1.0,
        apply_coastal_distance_mod=False,
        independent_observations_csv=RECONSTRUCTED_OBS,
    ),
    "exp2_foster_posterior_ratio_coast_on": dict(
        geology_categorical_csv=FOSTER_GEOL_POST,
        terrain_categorical_csv=FOSTER_TERR_POST,
        do_bayesian_update=False,
        combination_method=constants.CombinationMethod.RATIO,
        combine_ratio=1.0,
        apply_coastal_distance_mod=True,
        independent_observations_csv=RECONSTRUCTED_OBS,
    ),
    "exp3_foster_posterior_stdv_no_coast": dict(
        geology_categorical_csv=FOSTER_GEOL_POST,
        terrain_categorical_csv=FOSTER_TERR_POST,
        do_bayesian_update=False,
        combination_method=constants.CombinationMethod.STANDARD_DEVIATION_WEIGHTING,
        combine_ratio=None,
        apply_coastal_distance_mod=False,
        independent_observations_csv=RECONSTRUCTED_OBS,
    ),
    "exp4_prior_bayesian_ratio_no_coast": dict(
        geology_categorical_csv=GEOL_PRIOR,
        terrain_categorical_csv=TERR_PRIOR,
        do_bayesian_update=True,
        combination_method=constants.CombinationMethod.RATIO,
        combine_ratio=1.0,
        apply_coastal_distance_mod=False,
        independent_observations_csv=RECONSTRUCTED_OBS,
    ),
    "exp5_prior_bayesian_stdv_no_coast": dict(
        geology_categorical_csv=GEOL_PRIOR,
        terrain_categorical_csv=TERR_PRIOR,
        do_bayesian_update=True,
        combination_method=constants.CombinationMethod.STANDARD_DEVIATION_WEIGHTING,
        combine_ratio=None,
        apply_coastal_distance_mod=False,
        independent_observations_csv=RECONSTRUCTED_OBS,
    ),
}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
def run_experiment(name, params):
    """Run a single experiment and compare to reference."""
    output_dir = EXPERIMENT_DIR / "experiments" / name
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print(f"Running experiment: {name}")
    print(f"{'=' * 60}")
    for k, v in params.items():
        print(f"  {k}: {v}")

    pipeline.compute_grid(
        grid_config=SUBGRID,
        output_dir=output_dir,
        n_proc=-1,
        include_intermediate=True,
        noisy=True,
        mvn=True,
        **params,
    )

    # Compare to reference
    combined_tif = output_dir / constants.COMBINED_VS30_FILENAME
    if not combined_tif.exists():
        # Try alternative filename
        tifs = list(output_dir.glob("combined*.tif"))
        if tifs:
            combined_tif = tifs[0]
        else:
            print(f"WARNING: No combined output TIF found in {output_dir}")
            return

    print(f"\nComparing {combined_tif.name} to reference_subgrid.tif...")
    subprocess.run(
        [
            sys.executable,
            str(COMPARE_SCRIPT),
            "stats",
            str(REFERENCE_SUBGRID),
            str(combined_tif),
        ],
        check=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Jaehwi v1p0 reproduction experiments",
    )
    parser.add_argument(
        "experiments",
        nargs="*",
        default=list(EXPERIMENTS.keys()),
        help="Names of experiments to run (default: all)",
    )
    args = parser.parse_args()

    for name in args.experiments:
        if name not in EXPERIMENTS:
            print(f"Unknown experiment: {name}")
            print(f"Available: {list(EXPERIMENTS.keys())}")
            sys.exit(1)
        run_experiment(name, EXPERIMENTS[name])

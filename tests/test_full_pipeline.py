"""
End-to-end tests for the VS30 full-pipeline command.

These tests verify that the full pipeline produces consistent results
for different observation type combinations and processing modes.

Test scenarios (using small domain for fast execution):
- Independent observations only
- Clustered (CPT) observations only
- Both observation types combined

Each scenario is tested with both single-process (n_proc=1) and
multi-process (n_proc=cpu_count) modes to ensure parallel processing works correctly.
"""

import os

import pytest
from conftest import BENCHMARKS_DIR, compare_output_files, load_test_config

from vs30 import config, constants, pipeline

SCENARIOS = [
    "small_independent_only",
    "small_both",
    "small_clustered_only",
]

KEY_OUTPUT_FILES = [
    "combined_vs30.tif",
    "geology_vs30_slope_and_coastal_distance_and_spatially_adjusted_with_uncertainty.tif",
    "terrain_vs30_spatially_adjusted_with_uncertainty.tif",
    "posterior_geology_model_posterior_from_foster_2019_mean_and_standard_deviation.csv",
    "posterior_terrain_model_posterior_from_foster_2019_mean_and_standard_deviation.csv",
]


def run_pipeline_scenario(tmp_path, scenario: str, n_proc: int) -> None:
    """Run the grid pipeline for a test scenario and compare outputs to benchmarks."""
    config_data = load_test_config(scenario)

    pipeline.grid_pipeline(
        grid_config=config.GridConfig.from_dict(config_data),
        output_dir=tmp_path,
        geology_categorical_csv=config_data["geology_categorical_csv"],
        terrain_categorical_csv=config_data["terrain_categorical_csv"],
        clustered_observations_csv=config_data["clustered_observations_csv"],
        independent_observations_csv=config_data["independent_observations_csv"],
        combination_method=constants.CombinationMethod(
            config_data["combination_method"]
        ),
        combine_ratio=config_data["combine_ratio"],
        noisy=config_data["noisy"],
        do_bayesian_update=config_data["do_bayesian_update"],
        include_intermediate=True,
        n_proc=n_proc,
        apply_alluvium_slope_mod=config_data["apply_alluvium_slope_mod"],
        apply_coastal_distance_mod=config_data["apply_coastal_distance_mod"],
    )
    compare_output_files(tmp_path, BENCHMARKS_DIR / scenario, KEY_OUTPUT_FILES)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_single_process(tmp_path, scenario):
    """Test pipeline with n_proc=1 for each scenario."""
    run_pipeline_scenario(tmp_path, scenario, n_proc=1)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_multiprocess(tmp_path, scenario):
    """Test pipeline with all available CPUs for each scenario."""
    # `or 1` guards against None return from os.cpu_count() to satisfy the type checker.
    run_pipeline_scenario(tmp_path, scenario, n_proc=os.cpu_count() or 1)

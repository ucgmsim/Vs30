"""
End-to-end tests for the VS30 full-pipeline command.

These tests verify that the full pipeline produces consistent results
for different observation type combinations and processing modes.

Test scenarios (using small domain for fast execution):
- Independent observations only (fast)
- Clustered (CPT) observations only (slow — ~19K observations)
- Both observation types combined (slow — ~19K observations)

Each scenario is tested with both single-process (n_proc=1) and
multi-process (n_proc=cpu_count) modes to ensure parallel processing works correctly.

Slow tests can be skipped with: pytest -m "not slow"
"""

import os

import pytest
import yaml

from conftest import BENCHMARKS_DIR
from conftest import compare_output_files
from conftest import FIXTURES_DIR

from vs30 import constants, pipeline
from vs30 import config as config_module

FAST_SCENARIOS = [
    "small_independent_only",
]

SLOW_SCENARIOS = [
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


def load_test_config(scenario: str) -> dict:
    """Load a test configuration YAML file and return its data."""
    config_file = FIXTURES_DIR / f"test_config_{scenario}.yaml"
    with open(config_file) as f:
        return yaml.safe_load(f)


def run_pipeline_scenario(tmp_path, scenario: str, n_proc: int) -> None:
    """Run the grid pipeline for a test scenario and compare outputs to benchmarks."""
    config_data = load_test_config(scenario)
    clustered_file = config_data.get("clustered_observations_file")
    independent_file = config_data.get("independent_observations_file")
    pipeline.compute_grid(
        grid_config=config_module.GridConfig.from_dict(config_data),
        output_dir=tmp_path,
        combination_method=constants.CombinationMethod.RATIO,
        combine_ratio=float(config_data["combination_method"]),
        geology_categorical_csv=constants.RESOURCE_PATH / config_data["geology_categorical_file"],
        terrain_categorical_csv=constants.RESOURCE_PATH / config_data["terrain_categorical_file"],
        clustered_observations_csv=constants.RESOURCE_PATH / clustered_file if clustered_file else None,
        independent_observations_csv=constants.RESOURCE_PATH / independent_file if independent_file else None,
        do_bayesian_update=config_data.get(
            "do_bayesian_update_of_geology_and_terrain_categorical_vs30_values", True
        ),
        noisy=config_data.get("noisy", True),
        n_proc=n_proc,
        max_spatial_boolean_array_memory_gb=config_data.get(
            "max_spatial_boolean_array_memory_gb", 1.0
        ),
    )
    compare_output_files(tmp_path, BENCHMARKS_DIR / scenario, KEY_OUTPUT_FILES)


@pytest.mark.parametrize("scenario", FAST_SCENARIOS)
def test_single_process(tmp_path, scenario):
    """Test pipeline with n_proc=1 for each scenario."""
    run_pipeline_scenario(tmp_path, scenario, n_proc=1)


@pytest.mark.parametrize("scenario", FAST_SCENARIOS)
def test_multiprocess(tmp_path, scenario):
    """Test pipeline with all available CPUs for each scenario."""
    # `or 1` guards against None return from os.cpu_count() to satisfy the type checker.
    run_pipeline_scenario(tmp_path, scenario, n_proc=os.cpu_count() or 1)


@pytest.mark.slow
@pytest.mark.parametrize("scenario", SLOW_SCENARIOS)
def test_single_process_slow(tmp_path, scenario):
    """Test pipeline with n_proc=1 for scenarios using clustered observations."""
    run_pipeline_scenario(tmp_path, scenario, n_proc=1)


@pytest.mark.slow
@pytest.mark.parametrize("scenario", SLOW_SCENARIOS)
def test_multiprocess_slow(tmp_path, scenario):
    """Test pipeline with all available CPUs for scenarios using clustered observations."""
    run_pipeline_scenario(tmp_path, scenario, n_proc=os.cpu_count() or 1)

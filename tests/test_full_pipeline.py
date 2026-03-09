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
from pathlib import Path

import pytest
import yaml

from conftest import BENCHMARKS_DIR
from conftest import compare_output_files
from conftest import FIXTURES_DIR

from vs30 import constants, pipeline
from vs30 import config as config_module

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


def load_test_config(scenario: str) -> dict:
    """Load a test configuration YAML file and return its data."""
    config_file = FIXTURES_DIR / f"test_config_{scenario}.yaml"
    with open(config_file) as f:
        return yaml.safe_load(f)


def resolve_observation_csv(config_data: dict, key: str) -> Path | None:
    """Resolve an observation CSV path from config data."""
    filename = config_data.get(key)
    if filename is None or filename == "none":
        return None
    candidate = constants.RESOURCE_PATH / filename
    if candidate.exists():
        return candidate
    return None


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_single_process(tmp_path, scenario):
    """Test pipeline with n_proc=1 for each scenario."""
    config_data = load_test_config(scenario)
    grid_config = config_module.GridConfig.from_dict(config_data)
    pipeline.compute_grid(
        grid_config=grid_config,
        output_dir=tmp_path,
        combination_method=constants.CombinationMethod.RATIO,
        combine_ratio=float(config_data["combination_method"]),
        clustered_observations_csv=resolve_observation_csv(
            config_data, "clustered_observations_file"
        ),
        independent_observations_csv=resolve_observation_csv(
            config_data, "independent_observations_file"
        ),
        do_bayesian_update=config_data.get(
            "do_bayesian_update_of_geology_and_terrain_categorical_vs30_values", True
        ),
        noisy=config_data.get("noisy", True),
        n_proc=1,
        max_spatial_boolean_array_memory_gb=config_data.get(
            "max_spatial_boolean_array_memory_gb", 1.0
        ),
        obs_subsample_step_for_clustered=config_data.get(
            "obs_subsample_step_for_clustered", 100
        ),
    )
    compare_output_files(tmp_path, BENCHMARKS_DIR / scenario, KEY_OUTPUT_FILES)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_multiprocess(tmp_path, scenario):
    """Test pipeline with all available CPUs for each scenario."""
    config_data = load_test_config(scenario)
    grid_config = config_module.GridConfig.from_dict(config_data)
    # `or 1` guards against None return from os.cpu_count() to satisfy the type checker.
    pipeline.compute_grid(
        grid_config=grid_config,
        output_dir=tmp_path,
        combination_method=constants.CombinationMethod.RATIO,
        combine_ratio=float(config_data["combination_method"]),
        clustered_observations_csv=resolve_observation_csv(
            config_data, "clustered_observations_file"
        ),
        independent_observations_csv=resolve_observation_csv(
            config_data, "independent_observations_file"
        ),
        do_bayesian_update=config_data.get(
            "do_bayesian_update_of_geology_and_terrain_categorical_vs30_values", True
        ),
        noisy=config_data.get("noisy", True),
        n_proc=os.cpu_count() or 1,
        max_spatial_boolean_array_memory_gb=config_data.get(
            "max_spatial_boolean_array_memory_gb", 1.0
        ),
        obs_subsample_step_for_clustered=config_data.get(
            "obs_subsample_step_for_clustered", 100
        ),
    )
    compare_output_files(tmp_path, BENCHMARKS_DIR / scenario, KEY_OUTPUT_FILES)

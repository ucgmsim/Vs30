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
from conftest import run_cli

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


def create_test_config(scenario: str, output_dir: Path, n_proc: int = 1) -> Path:
    """
    Create a test configuration file with the specified settings.

    Parameters
    ----------
    scenario : str
        One of "small_independent_only", "small_clustered_only", or "small_both".
    output_dir : Path
        Directory for pipeline output.
    n_proc : int
        Number of processes to use.

    Returns
    -------
    Path
        Path to the created config file.
    """
    config_file = FIXTURES_DIR / f"test_config_{scenario}.yaml"
    with open(config_file) as f:
        config = yaml.safe_load(f)

    config["n_proc"] = n_proc
    config["output_dir"] = str(output_dir)

    test_config_path = output_dir / "test_config.yaml"
    with open(test_config_path, "w") as f:
        yaml.dump(config, f)

    return test_config_path


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_single_process(tmp_path, scenario):
    """Test pipeline with n_proc=1 for each scenario."""
    config_path = create_test_config(scenario, tmp_path, n_proc=1)
    run_cli(["--config", str(config_path), "full-pipeline"])
    compare_output_files(tmp_path, BENCHMARKS_DIR / scenario, KEY_OUTPUT_FILES)


@pytest.mark.parametrize("scenario", SCENARIOS)
def test_multiprocess(tmp_path, scenario):
    """Test pipeline with all available CPUs for each scenario."""
    # `or 1` guards against None return from os.cpu_count() to satisfy the type checker.
    config_path = create_test_config(scenario, tmp_path, n_proc=os.cpu_count() or 1)
    run_cli(["--config", str(config_path), "full-pipeline"])
    compare_output_files(tmp_path, BENCHMARKS_DIR / scenario, KEY_OUTPUT_FILES)

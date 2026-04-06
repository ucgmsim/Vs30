"""
Pytest configuration and shared test utilities.

This module contains fixtures and helper functions used across multiple test files.
All shared test fixtures should be defined here to avoid duplication.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml
from pandas.testing import assert_frame_equal

from vs30 import constants


def pytest_addoption(parser):
    """Register the --runslow command-line option."""
    parser.addoption(
        "--runslow", action="store_true", default=False, help="run slow tests"
    )


def pytest_collection_modifyitems(config, items):
    """Skip tests marked @pytest.mark.slow unless --runslow is given."""
    if config.getoption("--runslow"):
        return
    skip_slow = pytest.mark.skip(reason="use --runslow to run")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip_slow)

TESTS_DIR: Path = Path(__file__).parent
FIXTURES_DIR: Path = TESTS_DIR / "fixtures"
BENCHMARKS_DIR: Path = TESTS_DIR / "benchmarks"

TEST_RTOL: float = 1e-5
TEST_ATOL: float = 1e-8


def compare_output_files(
    output_dir: Path, benchmark_dir: Path, filenames: list[str]
) -> None:
    """
    Compare a list of output files against their benchmarks.

    Routes .tif files to compare_rasters; .csv files are compared with pandas
    assert_frame_equal after loading.

    Parameters
    ----------
    output_dir : Path
        Directory containing actual output files.
    benchmark_dir : Path
        Directory containing expected benchmark files.
    filenames : list[str]
        Names of files to compare.
    """
    for filename in filenames:
        actual = output_dir / filename
        expected = benchmark_dir / filename
        assert actual.exists(), f"Missing output file: {filename}"

        if filename.endswith(".tif"):
            compare_rasters(actual, expected)
        elif filename.endswith(".csv"):
            actual_df = pd.read_csv(actual)
            expected_df = pd.read_csv(expected)
            assert_frame_equal(actual_df, expected_df)


def compare_rasters(
    actual_path: Path,
    expected_path: Path,
) -> None:
    """
    Compare two raster files for equality within tolerance.

    Parameters
    ----------
    actual_path : Path
        Path to the actual output raster.
    expected_path : Path
        Path to the expected benchmark raster.

    Raises
    ------
    AssertionError
        If the rasters differ beyond tolerance.
    """
    with rasterio.open(actual_path) as actual, rasterio.open(expected_path) as expected:
        assert actual.count == expected.count, (
            f"Band count mismatch: {actual.count} vs {expected.count}"
        )
        assert actual.width == expected.width, (
            f"Width mismatch: {actual.width} vs {expected.width}"
        )
        assert actual.height == expected.height, (
            f"Height mismatch: {actual.height} vs {expected.height}"
        )

        for band_idx in range(1, actual.count + 1):
            actual_data = actual.read(band_idx)
            expected_data = expected.read(band_idx)
            nodata = actual.nodata

            if nodata is not None:
                actual_valid = actual_data != nodata
                expected_valid = expected_data != nodata
                assert np.array_equal(actual_valid, expected_valid), (
                    f"Band {band_idx}: Valid data masks differ"
                )
                valid_actual = actual_data[actual_valid]
                valid_expected = expected_data[expected_valid]
            else:
                valid_actual = actual_data
                valid_expected = expected_data

            if np.any(valid_actual):
                assert valid_actual == pytest.approx(
                    valid_expected,
                    rel=TEST_RTOL,
                    abs=TEST_ATOL,
                ), f"Band {band_idx}: Data values differ beyond tolerance"


def load_test_config(scenario: str) -> dict:
    """
    Load a test configuration YAML and resolve CSV paths.

    Observation CSVs are resolved relative to FIXTURES_DIR. Categorical model
    CSVs are resolved relative to the package resource directory.

    Parameters
    ----------
    scenario : str
        Name of the test scenario (used to find the YAML config file).

    Returns
    -------
    dict
        Loaded config with all CSV paths resolved to absolute Paths.
    """
    config_file = FIXTURES_DIR / f"test_config_{scenario}.yaml"
    with open(config_file) as f:
        config_data = yaml.safe_load(f)

    for key, subdir in constants.RESOURCE_SUBDIRS.items():
        if config_data[key]:
            if subdir == "observations":
                config_data[key] = FIXTURES_DIR / config_data[key]
            else:
                config_data[key] = (
                    constants.RESOURCE_PATH / subdir / config_data[key]
                )

    return config_data


def load_fixed_model_config(version: constants.FixedModelVersion) -> dict:
    """
    Load and resolve a fixed model version's YAML config for testing.

    Mirrors cli.load_model_config but without the typer dependency.
    Resolves CSV paths relative to the resources directory and builds
    correlation function callables.

    Parameters
    ----------
    version : FixedModelVersion
        Model version to load.

    Returns
    -------
    dict
        Resolved config dict ready to pass to pipeline functions.
    """
    from vs30.cli import resolve_correlation_function

    config_path = constants.MODEL_VERSION_TO_CONFIG[version]
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    # Resolve CSV paths
    for key, subdir in constants.RESOURCE_SUBDIRS.items():
        if config_data.get(key):
            config_data[key] = constants.RESOURCE_PATH / subdir / config_data[key]

    # Build correlation functions
    config_data["geology_corr_fn"] = resolve_correlation_function(
        config_data["geology_correlation"]
    )
    config_data["terrain_corr_fn"] = resolve_correlation_function(
        config_data["terrain_correlation"]
    )

    return config_data

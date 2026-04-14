"""
Pytest configuration and shared test utilities.

This module contains fixtures and helper functions used across multiple test files.
All shared test fixtures should be defined here to avoid duplication.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio
import yaml

from vs30 import constants, cli 


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

TEST_RTOL: float = 1e-3

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
    

    config_path = constants.MODEL_VERSION_TO_CONFIG[version]
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    # Resolve CSV paths
    for key, subdir in constants.RESOURCE_SUBDIRS.items():
        if config_data.get(key):
            config_data[key] = constants.RESOURCE_PATH / subdir / config_data[key]

    # Build correlation functions
    config_data["geology_corr_fn"] = cli.resolve_correlation_function(
        config_data["geology_correlation"]
    )
    config_data["terrain_corr_fn"] = cli.resolve_correlation_function(
        config_data["terrain_correlation"]
    )

    return config_data


def assert_arrays_match_raster_benchmark(
    vs30_array: np.ndarray,
    stdv_array: np.ndarray,
    benchmark_path: Path,
) -> None:
    """
    Compare in-memory vs30 and stdv arrays against a two-band benchmark raster.

    Parameters
    ----------
    vs30_array : np.ndarray
        Combined Vs30 result array (matches band 1 of benchmark).
    stdv_array : np.ndarray
        Combined standard deviation array (matches band 2 of benchmark).
    benchmark_path : Path
        Path to the benchmark .tif file.
    """
    with rasterio.open(benchmark_path) as benchmark:
        nodata = benchmark.nodata
        for band_idx, actual_data in enumerate([vs30_array, stdv_array], start=1):
            expected_data = benchmark.read(band_idx)
            # Treat NaN as nodata too: the legacy pipeline writes NaN into
            # nodata pixels even though the tif metadata declares -32767, so
            # both representations must be masked out to compare only valid
            # pixels.
            actual_valid = ~np.isnan(actual_data)
            expected_valid = ~np.isnan(expected_data)
            if nodata is not None:
                actual_valid &= actual_data != nodata
                expected_valid &= expected_data != nodata
            assert np.array_equal(actual_valid, expected_valid), (
                f"Band {band_idx}: Valid data masks differ"
            )
            valid_actual = actual_data[actual_valid]
            valid_expected = expected_data[expected_valid]

            if np.any(valid_actual):
                assert valid_actual == pytest.approx(
                    valid_expected,
                    rel=TEST_RTOL,
                    abs=0,
                ), f"Band {band_idx}: Data values differ beyond tolerance"

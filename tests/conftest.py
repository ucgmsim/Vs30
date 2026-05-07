"""
Pytest configuration and shared test utilities.

This module contains fixtures and helper functions used across multiple test files.
All shared test fixtures should be defined here to avoid duplication.
"""

from pathlib import Path

import numpy as np
import pytest
import rasterio

from vs30 import config, constants


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


@pytest.fixture(scope="session", autouse=True)
def require_extracted_shapefiles():
    """Fail fast if shapefiles.tar.xz has not been extracted by setup.py."""
    required = [
        constants.GEOSPATIAL_DIR / constants.GEOLOGY_SHAPEFILE_PATH,
        constants.GEOSPATIAL_DIR / constants.COASTLINE_SHAPEFILE_PATH,
    ]
    missing = [p for p in required if not p.exists()]
    if missing:
        pretty = "\n  ".join(str(p) for p in missing)
        pytest.exit(
            f"Required shapefile(s) not extracted:\n  {pretty}\n"
            f"Run `pip install -e .` in the active environment to extract "
            f"{constants.SHAPEFILES_ARCHIVE_FILENAME}.",
            returncode=1,
        )


TESTS_DIR: Path = Path(__file__).parent
FIXTURES_DIR: Path = TESTS_DIR / "fixtures"

TEST_RTOL: float = 1e-3


def load_fixed_model_config(version: constants.FixedModelVersion) -> dict:
    """
    Load and resolve a fixed model version's YAML config for testing.

    Thin wrapper around ``config.load_config_from_yaml`` so tests pick up any
    validation changes there.

    Parameters
    ----------
    version : FixedModelVersion
        Model version to load.

    Returns
    -------
    dict
        Resolved config dict ready to pass to pipeline functions.
    """
    return config.load_config_from_yaml(constants.MODEL_VERSION_TO_CONFIG[version])


def assert_arrays_match_raster_benchmark(
    vs30_array: np.ndarray,
    stdv_array: np.ndarray,
    benchmark_path: Path,
    rtol: float | None = None,
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
    rtol : float, optional
        Relative tolerance for the value comparison. If None, uses
        ``TEST_RTOL`` (default ``1e-3``). Pass a larger value for
        benchmarks where sub-meter coordinate drift between the legacy
        observation CSV and the refactored CSV produces small bounded
        deviations independent of any code change.
    """
    if rtol is None:
        rtol = TEST_RTOL
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
                    rel=rtol,
                    abs=0,
                ), f"Band {band_idx}: Data values differ beyond tolerance"

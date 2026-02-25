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
from pandas.testing import assert_frame_equal

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

"""
Pytest configuration and shared test utilities.

This module contains fixtures and helper functions used across multiple test files.
All shared test fixtures should be defined here to avoid duplication.
"""

import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pandas.testing import assert_frame_equal
from typer.testing import CliRunner

from vs30 import cli

TESTS_DIR: Path = Path(__file__).parent
FIXTURES_DIR: Path = TESTS_DIR / "fixtures"
BENCHMARKS_DIR: Path = TESTS_DIR / "benchmarks"

TEST_RTOL: float = 1e-5
TEST_ATOL: float = 1e-8

runner = CliRunner()

def run_cli(args: list[str]) -> None:
    """
    Run a vs30 CLI command via CliRunner, raising on failure.

    Parameters
    ----------
    args : list[str]
        CLI arguments to pass to the vs30 app.

    Raises
    ------
    RuntimeError
        If the CLI command exits with a non-zero code.
    """
    result = runner.invoke(cli.app, args)
    if result.exit_code != 0:
        print(f"Output:\n{result.stdout}")
        if result.exception:
            print(
                f"Exception:\n{''.join(traceback.format_exception(type(result.exception), result.exception, result.exception.__traceback__))}"
            )
        raise RuntimeError(f"CLI command failed with exit code {result.exit_code}")


def compare_output_files(output_dir: Path, benchmark_dir: Path, filenames: list[str]) -> None:
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
    rtol: float = TEST_RTOL,
    atol: float = TEST_ATOL,
) -> None:
    """
    Compare two raster files for equality within tolerance.

    Parameters
    ----------
    actual_path : Path
        Path to the actual output raster.
    expected_path : Path
        Path to the expected benchmark raster.
    rtol : float
        Relative tolerance for numpy.allclose.
    atol : float
        Absolute tolerance for numpy.allclose.

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
                if np.any(actual_valid):
                    assert np.allclose(
                        actual_data[actual_valid],
                        expected_data[expected_valid],
                        rtol=rtol,
                        atol=atol,
                    ), f"Band {band_idx}: Data values differ beyond tolerance"
            else:
                assert np.allclose(
                    actual_data, expected_data, rtol=rtol, atol=atol
                ), f"Band {band_idx}: Data values differ beyond tolerance"

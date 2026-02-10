"""
Pytest configuration and shared test utilities.

This module contains fixtures and helper functions used across multiple test files.
All shared test fixtures should be defined here to avoid duplication.
"""

import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from rasterio import transform

from vs30 import constants

TESTS_DIR: Path = Path(__file__).parent
FIXTURES_DIR: Path = TESTS_DIR / "fixtures"
BENCHMARKS_DIR: Path = TESTS_DIR / "benchmarks"

TEST_RASTER_WIDTH: int = 10
TEST_RASTER_HEIGHT: int = 10
TEST_RASTER_XMIN: int = 1500000
TEST_RASTER_XMAX: int = 1505000
TEST_RASTER_YMIN: int = 5100000
TEST_RASTER_YMAX: int = 5105000
TEST_RASTER_NODATA: float = -9999.0

TEST_RTOL: float = 1e-5
TEST_ATOL: float = 1e-8

@pytest.fixture
def temp_dir() -> Path:
    """
    Create a temporary directory for test outputs.

    Yields
    ------
    Path
        Path to temporary directory. Automatically cleaned up after test.
    """
    tmpdir = tempfile.mkdtemp(prefix="vs30_test_")
    yield Path(tmpdir)
    shutil.rmtree(tmpdir)


@pytest.fixture
def sample_vs30_raster(temp_dir: Path) -> Path:
    """
    Create a sample 2-band VS30 raster for testing.

    Parameters
    ----------
    temp_dir : Path
        Temporary directory for the raster file.

    Returns
    -------
    Path
        Path to the created test raster.
    """
    raster_path = temp_dir / "test_vs30.tif"

    raster_transform = transform.from_bounds(
        TEST_RASTER_XMIN,
        TEST_RASTER_YMIN,
        TEST_RASTER_XMAX,
        TEST_RASTER_YMAX,
        TEST_RASTER_WIDTH,
        TEST_RASTER_HEIGHT,
    )

    vs30_data = np.random.uniform(
        200, 600, (TEST_RASTER_HEIGHT, TEST_RASTER_WIDTH)
    ).astype(np.float32)
    stdv_data = np.random.uniform(
        20, 60, (TEST_RASTER_HEIGHT, TEST_RASTER_WIDTH)
    ).astype(np.float32)

    with rasterio.open(
        raster_path,
        "w",
        driver=constants.GEOTIFF_DRIVER,
        height=TEST_RASTER_HEIGHT,
        width=TEST_RASTER_WIDTH,
        count=2,
        dtype="float32",
        crs=constants.NZTM_CRS,
        transform=raster_transform,
        nodata=TEST_RASTER_NODATA,
    ) as dst:
        dst.write(vs30_data, constants.RASTER_BAND_VS30)
        dst.write(stdv_data, constants.RASTER_BAND_STDV)

    return raster_path

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


def compare_csvs(
    actual_path: Path,
    expected_path: Path,
    rtol: float = TEST_RTOL,
    atol: float = TEST_ATOL,
) -> None:
    """
    Compare two CSV files for equality within tolerance.

    Parameters
    ----------
    actual_path : Path
        Path to the actual output CSV.
    expected_path : Path
        Path to the expected benchmark CSV.
    rtol : float
        Relative tolerance for numeric comparison.
    atol : float
        Absolute tolerance for numeric comparison.

    Raises
    ------
    AssertionError
        If the CSVs differ beyond tolerance.
    """
    actual_df = pd.read_csv(actual_path)
    expected_df = pd.read_csv(expected_path)

    assert set(actual_df.columns) == set(expected_df.columns), (
        f"Column mismatch: {set(actual_df.columns)} vs {set(expected_df.columns)}"
    )
    assert len(actual_df) == len(expected_df), (
        f"Row count mismatch: {len(actual_df)} vs {len(expected_df)}"
    )

    for col in actual_df.columns:
        if pd.api.types.is_numeric_dtype(actual_df[col]):
            assert np.allclose(
                actual_df[col].values,
                expected_df[col].values,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            ), f"Column '{col}' values differ beyond tolerance"
        else:
            assert actual_df[col].equals(expected_df[col]), (
                f"Column '{col}' values differ"
            )

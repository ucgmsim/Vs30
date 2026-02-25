"""
Tests for the VS30 compute-at-locations functionality.

These tests verify that compute_at_locations produces consistent Vs30 values
for known locations (major NZ cities).
"""

import pandas as pd
from pandas.testing import assert_frame_equal

from conftest import BENCHMARKS_DIR
from conftest import FIXTURES_DIR
from vs30 import pipeline
from vs30.config import Vs30Config

EXPECTED_CSV = BENCHMARKS_DIR / "nz_cities_vs30.csv"


def test_single_process(tmp_path):
    """Test compute_at_locations with n_proc=1."""
    output_csv = tmp_path / "output.csv"
    pipeline.compute_at_locations(
        locations_csv=FIXTURES_DIR / "nz_cities.csv",
        output_csv=output_csv,
        cfg=Vs30Config.default(),
        lon_column="longitude",
        lat_column="latitude",
        include_intermediate=True,
        n_proc=1,
    )
    actual_df = pd.read_csv(output_csv)
    expected_df = pd.read_csv(EXPECTED_CSV)
    assert_frame_equal(actual_df, expected_df)


def test_multiprocess(tmp_path):
    """Test that multiprocess results match single-process benchmark."""
    output_csv = tmp_path / "output.csv"
    pipeline.compute_at_locations(
        locations_csv=FIXTURES_DIR / "nz_cities.csv",
        output_csv=output_csv,
        cfg=Vs30Config.default(),
        lon_column="longitude",
        lat_column="latitude",
        include_intermediate=True,
        n_proc=-1,
    )
    actual_df = pd.read_csv(output_csv)
    expected_df = pd.read_csv(EXPECTED_CSV)
    assert_frame_equal(actual_df, expected_df)

"""
Tests for the VS30 compute-at-locations functionality.

These tests verify that compute_at_locations produces consistent Vs30 values
for known locations (major NZ cities).
"""

import pandas as pd
from pandas.testing import assert_frame_equal

from conftest import BENCHMARKS_DIR
from conftest import FIXTURES_DIR
from vs30 import constants, pipeline

EXPECTED_CSV = BENCHMARKS_DIR / "nz_cities_vs30.csv"
LOCATIONS_CSV = FIXTURES_DIR / "nz_cities.csv"

# Default files from resources (matching the foster_2019 config)
GEOLOGY_CATEGORICAL_CSV = (
    constants.RESOURCE_PATH
    / "categorical_vs30_mean_and_stddev/geology_model_posterior_from_foster_2019_mean_and_standard_deviation.csv"
)
TERRAIN_CATEGORICAL_CSV = (
    constants.RESOURCE_PATH
    / "categorical_vs30_mean_and_stddev/terrain_model_posterior_from_foster_2019_mean_and_standard_deviation.csv"
)
INDEPENDENT_OBS_CSV = (
    constants.RESOURCE_PATH / "observations/foster_2019_measured_vs30_independent_observations.csv"
)
CLUSTERED_OBS_CSV = FIXTURES_DIR / "test_viktor_cpt_subset.csv"


def run_and_compare(n_proc: int):
    """Run compute_at_locations and compare against benchmark."""
    locations_df = pd.read_csv(LOCATIONS_CSV)
    result_df = pipeline.compute_at_locations(
        longitudes=locations_df["longitude"].values,
        latitudes=locations_df["latitude"].values,
        combination_method=constants.CombinationMethod.RATIO,
        combine_ratio=1.0,
        geology_categorical_csv=GEOLOGY_CATEGORICAL_CSV,
        terrain_categorical_csv=TERRAIN_CATEGORICAL_CSV,
        include_intermediate=True,
        clustered_observations_csv=CLUSTERED_OBS_CSV,
        independent_observations_csv=INDEPENDENT_OBS_CSV,
        noisy=True,
        n_proc=n_proc,
    )
    # Prepend original columns to match benchmark format
    original_cols = [c for c in locations_df.columns if c not in result_df.columns]
    actual_df = pd.concat(
        [locations_df[original_cols].reset_index(drop=True), result_df], axis=1
    )
    expected_df = pd.read_csv(EXPECTED_CSV)
    assert_frame_equal(actual_df, expected_df, check_dtype=False)


def test_single_process():
    """Test compute_at_locations with n_proc=1."""
    run_and_compare(n_proc=1)


def test_multiprocess():
    """Test that multiprocess results match single-process benchmark."""
    run_and_compare(n_proc=-1)

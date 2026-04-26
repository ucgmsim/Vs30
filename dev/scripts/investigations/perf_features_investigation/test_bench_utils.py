"""Unit tests for the perf-features-investigation benchmarking harness."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vs30 import constants

sys.path.insert(0, str(Path(__file__).parent))

import bench_utils


def test_subsample_observations_count() -> None:
    df = bench_utils.subsample_observations(100, seed=42)
    assert len(df) == 100


def test_subsample_observations_determinism() -> None:
    df1 = bench_utils.subsample_observations(100, seed=42)
    df2 = bench_utils.subsample_observations(100, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_subsample_observations_required_columns() -> None:
    df = bench_utils.subsample_observations(50, seed=42)
    required = {"easting", "northing", "vs30", "uncertainty"}
    assert required.issubset(df.columns)


def test_subsample_observations_too_many_raises() -> None:
    with pytest.raises(ValueError, match="exceeds available"):
        bench_utils.subsample_observations(10**9, seed=42)


def test_make_raster_data_returns_valid_raster_data() -> None:
    raster_data, profile = bench_utils.make_raster_data(n_target=1000)
    # Real RasterData with a non-empty valid mask
    assert raster_data.valid_flat_indices.size > 0
    # Profile carries transform and crs
    assert "transform" in profile
    assert profile["transform"] is not None


def test_make_raster_data_n_target_scales() -> None:
    # Larger n_target should produce more valid pixels
    rd_small, _ = bench_utils.make_raster_data(n_target=1000)
    rd_large, _ = bench_utils.make_raster_data(n_target=100_000)
    assert rd_large.valid_flat_indices.size > rd_small.valid_flat_indices.size


def test_make_full_bbox_result_marks_all_valid_pixels() -> None:
    raster_data, _ = bench_utils.make_raster_data(n_target=1000)
    bbox = bench_utils.make_full_bbox_result(raster_data, n_obs=10)
    assert bbox.mask.shape == (raster_data.vs30.size,)
    # Every valid pixel should be marked affected.
    assert bool(np.all(bbox.mask[raster_data.valid_flat_indices]))
    assert bbox.n_affected_pixels == raster_data.valid_flat_indices.size


def test_bypass_observation_threshold_restores_original() -> None:
    original = constants.MULTIPROCESS_OBSERVATION_THRESHOLD
    with bench_utils.bypass_observation_threshold():
        assert constants.MULTIPROCESS_OBSERVATION_THRESHOLD == 10**12
    assert constants.MULTIPROCESS_OBSERVATION_THRESHOLD == original


def test_bypass_observation_threshold_restores_on_exception() -> None:
    original = constants.MULTIPROCESS_OBSERVATION_THRESHOLD
    with pytest.raises(RuntimeError):
        with bench_utils.bypass_observation_threshold():
            raise RuntimeError("boom")
    assert constants.MULTIPROCESS_OBSERVATION_THRESHOLD == original


def test_time_one_run_returns_expected_keys() -> None:
    raster_data, _ = bench_utils.make_raster_data(n_target=1000)
    obs_df = bench_utils.subsample_observations(50, seed=42)
    obs_data = bench_utils.prepare_terrain_obs_data(obs_df, raster_data)
    row = bench_utils.time_one_run(
        raster_data=raster_data,
        obs_data=obs_data,
        nproc=1,
        ffap=True,
        rep=0,
    )
    expected_keys = {
        "N_obs",
        "N_grid_actual",
        "N_affected",
        "nproc",
        "ffap",
        "rep",
        "t_bbox_s",
        "t_spatial_s",
        "t_total_s",
        "peak_rss_mb",
        "timestamp_iso",
    }
    assert expected_keys.issubset(row.keys())
    assert row["t_total_s"] >= row["t_bbox_s"]
    assert row["t_spatial_s"] > 0
    assert row["t_bbox_s"] >= 0


def test_numerical_equivalence_check_passes_on_small_case() -> None:
    """All four (nproc × ffap) combinations should produce identical output."""
    bench_utils.run_numerical_equivalence_check(
        n_obs=200,
        n_target=1000,
        nproc_options=(1, 2),  # use 2 instead of 8 to keep test fast
    )

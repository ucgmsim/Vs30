"""Unit tests for the perf-features-investigation benchmarking harness."""

import sys
from pathlib import Path

import pandas as pd
import pytest

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

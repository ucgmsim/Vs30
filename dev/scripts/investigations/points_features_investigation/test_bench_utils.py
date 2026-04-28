"""Unit tests for the points-perf-investigation harness."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))

import bench_utils


def test_generate_nz_land_points_count() -> None:
    lons, lats = bench_utils.generate_nz_land_points(100, seed=42)
    assert len(lons) == 100
    assert len(lats) == 100


def test_generate_nz_land_points_determinism() -> None:
    lons1, lats1 = bench_utils.generate_nz_land_points(50, seed=42)
    lons2, lats2 = bench_utils.generate_nz_land_points(50, seed=42)
    np.testing.assert_array_equal(lons1, lons2)
    np.testing.assert_array_equal(lats1, lats2)


def test_generate_nz_land_points_within_nz_bbox() -> None:
    lons, lats = bench_utils.generate_nz_land_points(100, seed=42)
    # Approximate NZ bounding box (WGS84)
    assert np.all(lons >= 165.0) and np.all(lons <= 180.0)
    assert np.all(lats >= -48.0) and np.all(lats <= -34.0)


def test_generate_nz_land_points_all_on_land() -> None:
    """Every returned point should land on a valid IwahashiPike pixel."""
    from qcore import coordinates

    from vs30 import category, constants

    lons, lats = bench_utils.generate_nz_land_points(100, seed=42)
    nztm = coordinates.wgs_depth_to_nztm(np.column_stack([lats, lons]))
    eastings = nztm[:, 1]
    northings = nztm[:, 0]
    points = np.column_stack([eastings, northings])
    terrain_ids = category.assign_to_category_terrain(points)
    assert np.all(terrain_ids != constants.RASTER_ID_NODATA_VALUE), (
        "Some sampled points fall on terrain nodata (i.e., off NZ land)"
    )


def test_materialize_obs_csvs_creates_files(tmp_path) -> None:
    paths = bench_utils.materialize_obs_csvs(
        out_dir=tmp_path, n_obs_values=[50, 100], seed=42
    )
    assert set(paths.keys()) == {50, 100}
    for n_obs, path in paths.items():
        assert path.exists()
        df = pd.read_csv(path)
        # Required columns must survive the round-trip
        required = {"easting", "northing", "vs30", "uncertainty"}
        assert required.issubset(df.columns)
        assert len(df) == n_obs


def test_time_one_run_returns_expected_keys(tmp_path) -> None:
    # Materialise a small obs CSV
    paths = bench_utils.materialize_obs_csvs(
        out_dir=tmp_path, n_obs_values=[100], seed=42
    )
    # 5 query points, just enough to exercise the pipeline end-to-end
    lons, lats = bench_utils.generate_nz_land_points(5, seed=42)
    cfg = bench_utils.load_modified_foster_2019_config()
    row = bench_utils.time_one_run(
        lons=lons,
        lats=lats,
        obs_csv_path=paths[100],
        nproc=1,
        rep=0,
        cfg=cfg,
    )
    expected_keys = {
        "N_query",
        "N_obs",
        "nproc",
        "rep",
        "t_total_s",
        "peak_rss_mb",
        "timestamp_iso",
    }
    assert expected_keys.issubset(row.keys())
    assert row["t_total_s"] > 0
    assert row["N_query"] == 5
    assert row["N_obs"] == 100

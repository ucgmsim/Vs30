"""Helpers for the points-perf-investigation harness."""

import datetime as _dt  # noqa: F401
import importlib.util
import resource  # noqa: F401
import time  # noqa: F401
from pathlib import Path

import numpy as np
import pandas as pd  # noqa: F401
from qcore import coordinates

from vs30 import category, constants, pipeline  # noqa: F401

# Reuse subsample_observations from the grid harness via path import.
# This avoids duplicating the canonical subsampling routine.
_GRID_HARNESS_DIR = Path(__file__).resolve().parents[1] / "perf_features_investigation"
_grid_bench_spec = importlib.util.spec_from_file_location(
    "perf_bench_utils", _GRID_HARNESS_DIR / "bench_utils.py"
)
_grid_bench_mod = importlib.util.module_from_spec(_grid_bench_spec)
_grid_bench_spec.loader.exec_module(_grid_bench_mod)
subsample_observations = _grid_bench_mod.subsample_observations  # noqa: F401  -- re-exported


# WGS84 bounding box that comfortably covers all NZ land. Slightly looser than
# tight to allow for the rejection sampler's land-mask filtering to do its job.
_NZ_LON_MIN, _NZ_LON_MAX = 165.0, 180.0
_NZ_LAT_MIN, _NZ_LAT_MAX = -48.0, -34.0


def generate_nz_land_points(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return ``n`` random (lon, lat) points uniformly distributed over NZ land.

    Rejection samples WGS84 lon/lat within the NZ bounding box and keeps only
    points that land on a valid (non-nodata) IwahashiPike terrain raster pixel.

    Parameters
    ----------
    n
        Number of points to return.
    seed
        Seed for the numpy random generator.

    Returns
    -------
    lons, lats : np.ndarray
        Two ``(n,)`` arrays of WGS84 longitudes and latitudes.
    """
    rng = np.random.default_rng(seed)
    kept_lons: list[float] = []
    kept_lats: list[float] = []
    while len(kept_lons) < n:
        # Over-sample by ~3x; about 30-40% of the NZ bbox is land.
        batch_size = max(3 * n, 1000)
        lons = rng.uniform(_NZ_LON_MIN, _NZ_LON_MAX, batch_size)
        lats = rng.uniform(_NZ_LAT_MIN, _NZ_LAT_MAX, batch_size)
        nztm = coordinates.wgs_depth_to_nztm(np.column_stack([lats, lons]))
        eastings = nztm[:, 1]
        northings = nztm[:, 0]
        points = np.column_stack([eastings, northings])
        terrain_ids = category.assign_to_category_terrain(points)
        land_mask = terrain_ids != constants.RASTER_ID_NODATA_VALUE
        kept_lons.extend(lons[land_mask].tolist())
        kept_lats.extend(lats[land_mask].tolist())
    return np.array(kept_lons[:n]), np.array(kept_lats[:n])

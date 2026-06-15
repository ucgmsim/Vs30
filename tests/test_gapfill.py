"""Tests for the gap-fill module."""

import numpy as np
import rasterio

from vs30 import config, constants, gapfill


def test_classify_nodata_excludes_water_and_offshore():
    """classify_nodata should exclude GID=0 (water) and offshore pixels."""
    # Wellington CBD: known on-land location
    onland_location = np.array([[1749000.0, 5427000.0]])
    # Far offshore: known ocean location east of NZ
    offshore_location = np.array([[2200000.0, 5400000.0]])

    result = gapfill.classify_nodata(
        combined_vs30=np.array([np.nan]),
        geology_ids=np.array([5]),
        locations=onland_location,
    )
    assert result[0], "On-land nodata pixel with valid GID should be fillable"

    result = gapfill.classify_nodata(
        combined_vs30=np.array([np.nan]),
        geology_ids=np.array([0]),
        locations=onland_location,
    )
    assert not result[0], "Water pixel (GID=0) should not be fillable"

    result = gapfill.classify_nodata(
        combined_vs30=np.array([300.0]),
        geology_ids=np.array([5]),
        locations=onland_location,
    )
    assert not result[0], "Valid (non-NaN) pixel should not be fillable"

    result = gapfill.classify_nodata(
        combined_vs30=np.array([np.nan]),
        geology_ids=np.array([5]),
        locations=offshore_location,
    )
    assert not result[0], "Offshore nodata pixel should not be fillable"


def test_fill_nodata_grid_nearest_neighbor():
    """fill_nodata_grid fills an on-land NaN gap with the nearest valid value."""
    # 3x3 grid at 100m spacing, placed so pixel (1,1) center = (1749050, 5427050)
    # Origin is the top-left corner of pixel (0,0).
    # pixel center = origin + (index + 0.5) * pixel_size
    # For col 1: origin_x + 1.5 * 100 = 1749050 -> origin_x = 1748900
    # For row 1: origin_y + 1.5 * (-100) = 5427050 -> origin_y = 5427200
    pixel_size = 100
    transform = rasterio.transform.Affine(pixel_size, 0, 1748900, 0, -pixel_size, 5427200)
    profile = {"transform": transform}

    vs30 = np.array(
        [
            [200.0, 250.0, 275.0],
            [300.0, np.nan, 350.0],
            [375.0, 400.0, 450.0],
        ]
    )
    stdv = np.array(
        [
            [0.50, 0.60, 0.65],
            [0.70, np.nan, 0.80],
            [0.75, 0.90, 0.95],
        ]
    )
    # All non-zero GIDs so the center pixel passes the water filter
    geology_ids = np.full((3, 3), 5, dtype=int)

    filled_vs30, filled_stdv = gapfill.fill_nodata_grid(
        vs30, stdv, geology_ids, profile
    )

    # All four edge neighbors are equidistant; KDTree picks the first in row-major order.
    assert filled_vs30[1, 1] == 250.0
    assert filled_stdv[1, 1] == 0.6

    # All other pixels should be unchanged
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    np.testing.assert_array_equal(filled_vs30[mask], vs30[mask])
    np.testing.assert_array_equal(filled_stdv[mask], stdv[mask])


def test_create_local_grid_config_expansion():
    """create_local_grid_config: bigger half_width → bigger grid, same centre, same pixel lattice as the full NZ grid."""
    dx = config.FULL_NZ_GRID_CONFIG.grid_dx
    dy = config.FULL_NZ_GRID_CONFIG.grid_dy

    # Pick an arbitrary pixel CENTRE on the full NZ grid (100 pixels from the
    # origin). Pixel centres are at grid_xmin + dx/2 + n*dx under the
    # pixel-edge bounds convention.
    n_pixels = 100
    easting = (
        config.FULL_NZ_GRID_CONFIG.grid_xmin
        + dx / 2
        + n_pixels * dx
    )
    northing = (
        config.FULL_NZ_GRID_CONFIG.grid_ymin
        + dy / 2
        + n_pixels * dy
    )

    # Production constants must satisfy create_local_grid_config's pixel-edge
    # constraint (half_width = k*dx + dx/2).
    assert (constants.GAPFILL_INITIAL_HALF_WIDTH_M - dx / 2) % dx == 0, (
        f"GAPFILL_INITIAL_HALF_WIDTH_M = {constants.GAPFILL_INITIAL_HALF_WIDTH_M} "
        f"must equal k*dx + dx/2 for create_local_grid_config to produce "
        f"pixel-aligned local grids."
    )
    assert constants.GAPFILL_HALF_WIDTH_EXPANSION_M % dx == 0, (
        f"GAPFILL_HALF_WIDTH_EXPANSION_M = {constants.GAPFILL_HALF_WIDTH_EXPANSION_M} "
        f"must be a multiple of dx so successive expansions stay aligned."
    )
    initial_half_width = constants.GAPFILL_INITIAL_HALF_WIDTH_M
    expanded_half_width = initial_half_width + constants.GAPFILL_HALF_WIDTH_EXPANSION_M

    initial_grid = gapfill.create_local_grid_config(
        easting,
        northing,
        config.FULL_NZ_GRID_CONFIG,
        initial_half_width,
    )
    expanded_grid = gapfill.create_local_grid_config(
        easting,
        northing,
        config.FULL_NZ_GRID_CONFIG,
        expanded_half_width,
    )

    # The expanded grid should be larger
    assert (expanded_grid.grid_xmax - expanded_grid.grid_xmin) > (
        initial_grid.grid_xmax - initial_grid.grid_xmin
    )

    # Both grids should still be centered on the snapped pixel centre
    assert (initial_grid.grid_xmin + initial_grid.grid_xmax) / 2 == (
        expanded_grid.grid_xmin + expanded_grid.grid_xmax
    ) / 2
    assert (initial_grid.grid_ymin + initial_grid.grid_ymax) / 2 == (
        expanded_grid.grid_ymin + expanded_grid.grid_ymax
    ) / 2

    # Both grids share the full NZ grid's pixel-centre lattice
    # (xmin offsets are an integer number of dx away from the full grid's xmin).
    assert initial_grid.grid_dx == dx
    assert expanded_grid.grid_dx == dx
    assert (
        initial_grid.grid_xmin - config.FULL_NZ_GRID_CONFIG.grid_xmin
    ) % dx == 0
    assert (
        expanded_grid.grid_xmin - config.FULL_NZ_GRID_CONFIG.grid_xmin
    ) % dx == 0

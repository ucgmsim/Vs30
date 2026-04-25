"""
Tests for gap-fill nodata classification and nearest-neighbor filling.

Validates that classify_nodata correctly distinguishes on-land gaps (fillable)
from water pixels (GID=0) and offshore pixels (outside coastline), and that
fill_nodata_grid copies the nearest valid neighbor's values.
"""

import numpy as np
import rasterio

from vs30 import constants, gapfill


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
    """fill_nodata_grid should fill an on-land gap with the nearest valid value.

    Constructs a 3x3 grid centered on Wellington CBD with the center pixel
    set to NaN. The four edge-adjacent pixels have distinct values; the
    nearest-neighbor fill should copy from one of them (all equidistant,
    so cKDTree picks the first match).
    """
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

    assert filled_vs30[1, 1] == 250.0
    assert filled_stdv[1, 1] == 0.6

    # All other pixels should be unchanged
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    np.testing.assert_array_equal(filled_vs30[mask], vs30[mask])
    np.testing.assert_array_equal(filled_stdv[mask], stdv[mask])


def test_create_local_grid_config_expansion():
    """When gap-fill can't find a valid donor in the initial local grid, it
    retries with a larger grid. create_local_grid_config builds these grids.
    This test checks that the expanded grid is larger but stays centered on
    the same point and keeps its pixels aligned to the full NZ grid.
    """
    # Pick an arbitrary point on the full NZ grid (100 pixels from the origin)
    n_pixels = 100
    easting = (
        constants.FULL_NZ_GRID_CONFIG.grid_xmin
        + n_pixels * constants.FULL_NZ_GRID_CONFIG.grid_dx
    )
    northing = (
        constants.FULL_NZ_GRID_CONFIG.grid_ymin
        + n_pixels * constants.FULL_NZ_GRID_CONFIG.grid_dy
    )

    # Simulate the first attempt (5 km half-width -> 10 km x 10 km grid)
    initial_half_width = constants.GAPFILL_LOCAL_GRID_SIZE_M
    initial_grid = gapfill.create_local_grid_config(
        easting,
        northing,
        constants.FULL_NZ_GRID_CONFIG,
        initial_half_width,
    )

    # Simulate the retry after expansion (10 km half-width -> 20 km x 20 km grid)
    expanded_half_width = initial_half_width + constants.GAPFILL_LOCAL_GRID_EXPANSION_M
    expanded_grid = gapfill.create_local_grid_config(
        easting,
        northing,
        constants.FULL_NZ_GRID_CONFIG,
        expanded_half_width,
    )

    # The expanded grid should be larger
    assert (expanded_grid.grid_xmax - expanded_grid.grid_xmin) > (
        initial_grid.grid_xmax - initial_grid.grid_xmin
    )

    # Both grids should still be centered on the same point
    assert (initial_grid.grid_xmin + initial_grid.grid_xmax) / 2 == (
        expanded_grid.grid_xmin + expanded_grid.grid_xmax
    ) / 2
    assert (initial_grid.grid_ymin + initial_grid.grid_ymax) / 2 == (
        expanded_grid.grid_ymin + expanded_grid.grid_ymax
    ) / 2

    # Both grids should have pixels aligned to the full NZ grid
    assert initial_grid.grid_dx == constants.FULL_NZ_GRID_CONFIG.grid_dx
    assert expanded_grid.grid_dx == constants.FULL_NZ_GRID_CONFIG.grid_dx
    assert (
        initial_grid.grid_xmin - constants.FULL_NZ_GRID_CONFIG.grid_xmin
    ) % constants.FULL_NZ_GRID_CONFIG.grid_dx == 0
    assert (
        expanded_grid.grid_xmin - constants.FULL_NZ_GRID_CONFIG.grid_xmin
    ) % constants.FULL_NZ_GRID_CONFIG.grid_dx == 0

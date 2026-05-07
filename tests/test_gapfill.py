"""
Tests for gap-fill nodata classification and nearest-neighbor filling.

Validates that classify_nodata correctly distinguishes on-land gaps (fillable)
from water pixels (GID=0) and offshore pixels (outside coastline), and that
fill_nodata_grid copies the nearest valid neighbor's values.
"""

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

    # Use the production constants directly so this test fails if a future
    # constant change violates the pixel-edge constraint that
    # create_local_grid_config requires (half_width must equal k*dx + dx/2).
    half_dx = dx / 2
    assert (constants.GAPFILL_INITIAL_HALF_WIDTH_M - half_dx) % dx == 0, (
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

    # Both grids should share the same pixel-centre lattice as the full NZ
    # grid. Pixel centres of a grid are at xmin + dx/2 + n*dx. Two grids
    # share the same lattice iff their xmin values are an integer number of
    # dx apart. (With pixel-edge bounds and a pixel-centre snap, xmin is
    # always snap_e - half_width where snap_e is a full NZ pixel centre and
    # half_width = n*dx + dx/2, so xmin is also a full NZ pixel centre —
    # i.e. (xmin - full_xmin) % dx == 0.)
    assert initial_grid.grid_dx == dx
    assert expanded_grid.grid_dx == dx
    assert (
        initial_grid.grid_xmin - config.FULL_NZ_GRID_CONFIG.grid_xmin
    ) % dx == 0
    assert (
        expanded_grid.grid_xmin - config.FULL_NZ_GRID_CONFIG.grid_xmin
    ) % dx == 0

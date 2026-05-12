"""Tests that canonical grid configs are pixel-aligned with IwahashiPike (avoids nondeterministic GDAL nearest-neighbour tie-breaking)."""

import rasterio

from test_benchmarks import BENCHMARK_NZ_GRID
from vs30 import config, constants


def assert_grid_aligned_with_iwahashipike(grid: config.GridConfig) -> None:
    """Assert every pixel centre of the grid lands on an IwahashiPike pixel centre."""
    iw_path = constants.GEOSPATIAL_DIR / constants.TERRAIN_RASTER_FILENAME
    with rasterio.open(iw_path) as src:
        iw_t = src.transform
        iw_dx = iw_t.a
        iw_dy = abs(iw_t.e)

    nx = round((grid.grid_xmax - grid.grid_xmin) / grid.grid_dx)
    ny = round((grid.grid_ymax - grid.grid_ymin) / grid.grid_dy)
    grid_t = rasterio.transform.from_bounds(
        grid.grid_xmin, grid.grid_ymin, grid.grid_xmax, grid.grid_ymax, nx, ny
    )

    # Pixel (0,0) centre in real-world coords.
    ul_centre_x = grid_t.c + grid_t.a / 2
    ul_centre_y = grid_t.f + grid_t.e / 2

    # Express that centre as a fractional pixel coordinate within IwahashiPike.
    # An IwahashiPike pixel centre has fractional coord = integer + 0.5.
    iw_col_f = (ul_centre_x - iw_t.c) / iw_dx
    iw_row_f = (iw_t.f - ul_centre_y) / iw_dy

    col_offset_pixels = abs(iw_col_f - (round(iw_col_f - 0.5) + 0.5))
    row_offset_pixels = abs(iw_row_f - (round(iw_row_f - 0.5) + 0.5))

    assert col_offset_pixels < 1e-6, (
        f"Grid pixel (0,0) CENTRE x = {ul_centre_x} is offset by "
        f"{col_offset_pixels * iw_dx:.1f} m from the nearest IwahashiPike pixel CENTRE. "
        f"This causes GDAL's nearest-neighbour resampling to tie at every pixel."
    )
    assert row_offset_pixels < 1e-6, (
        f"Grid pixel (0,0) CENTRE y = {ul_centre_y} is offset by "
        f"{row_offset_pixels * iw_dy:.1f} m from the nearest IwahashiPike pixel CENTRE. "
        f"This causes GDAL's nearest-neighbour resampling to tie at every pixel."
    )


def test_full_nz_grid_config_aligned_with_iwahashipike():
    """The production NZ-wide grid must be IwahashiPike-aligned."""
    assert_grid_aligned_with_iwahashipike(config.FULL_NZ_GRID_CONFIG)


def test_benchmark_nz_grid_aligned_with_iwahashipike():
    """The 5 km benchmark grid must also be IwahashiPike-aligned."""
    assert_grid_aligned_with_iwahashipike(BENCHMARK_NZ_GRID)

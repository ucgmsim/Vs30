"""
Gap-fill nodata pixels in the combined VS30 output.

Fills on-land nodata gaps using nearest-neighbor interpolation. Pixels are
classified into three categories:
1. Water (GID=0): remain nodata.
2. Offshore (outside coastline polygon): remain nodata.
3. On-land gaps (inside coastline, GID != 0): filled with nearest valid value.

Only category 3 is filled. The coastline shapefile distinguishes categories 2
and 3. This reproduces the gap-fill logic from Jaehwi's Vs30_extraction_26Mar.py.
"""

import logging

import numpy as np
import scipy.ndimage
import scipy.spatial
import shapely

from vs30 import config, constants, raster

logger = logging.getLogger(__name__)


def pixel_coords_float32(
    rows: np.ndarray,
    cols: np.ndarray,
    transform,
) -> np.ndarray:
    """
    Compute float32 NZTM pixel center coordinates from row/col indices.

    Computes in float64 from the affine transform, then downcasts to float32.
    Float32 precision at NZTM magnitudes (~6.25M meters) gives worst-case
    error of ~0.7m — negligible on a 100m grid for nearest-neighbor lookup.
    """
    eastings = transform.c + transform.a * (cols + constants.PIXEL_CENTER_OFFSET)
    northings = transform.f + transform.e * (rows + constants.PIXEL_CENTER_OFFSET)
    return np.column_stack([
        eastings.astype(np.float32),
        northings.astype(np.float32),
    ])


def points_inside_coastline(locations: np.ndarray) -> np.ndarray:
    """
    Return a boolean mask indicating which NZTM points lie inside the NZ coastline.

    Parameters
    ----------
    locations : ndarray
        (N, 2) array of [easting, northing] NZTM coordinates.

    Returns
    -------
    ndarray
        Boolean mask of length N, True where the point is inside the coastline polygon.
    """
    if len(locations) == 0:
        return np.zeros(0, dtype=bool)

    # union_all() is expensive — raster.load_coast_union() memoises it.
    coast_union = raster.load_coast_union()
    return shapely.within(shapely.points(locations), coast_union)


def classify_nodata(
    combined_vs30: np.ndarray,
    geology_ids: np.ndarray,
    locations: np.ndarray,
) -> np.ndarray:
    """
    Identify nodata pixels eligible for gap-filling.

    Returns a boolean mask where True means the pixel is: (1) NaN in the
    combined VS30 output, (2) not water (geology_id != 0), and (3) inside
    the NZ coastline polygon.

    Parameters
    ----------
    combined_vs30 : ndarray
        1D array of combined VS30 values (NaN for nodata).
    geology_ids : ndarray
        1D array of geology category IDs.
    locations : ndarray
        (N, 2) array of [easting, northing] NZTM coordinates.

    Returns
    -------
    ndarray
        Boolean mask where True = eligible for filling.
    """
    candidate_mask = np.isnan(combined_vs30) & (geology_ids != 0)
    candidate_indices = np.where(candidate_mask)[0]
    fillable_mask = np.zeros(len(combined_vs30), dtype=bool)
    fillable_mask[candidate_indices] = points_inside_coastline(
        locations[candidate_indices]
    )

    if np.count_nonzero(fillable_mask) > 0:
        logger.info(
            f"  Gap-fill: {np.count_nonzero(fillable_mask)} on-land nodata pixel(s) identified for filling"
        )

    return fillable_mask


def fill_nodata_grid(
    vs30: np.ndarray,
    stdv: np.ndarray,
    geology_ids: np.ndarray,
    profile: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Fill nodata gaps in a Vs30 grid using nearest-neighbor.

    Parameters
    ----------
    vs30 : ndarray
        2D array of Vs30 values (NaN for nodata).
    stdv : ndarray
        2D array of standard deviation values (NaN for nodata).
    geology_ids : ndarray
        2D array of geology category IDs.
    profile : dict
        Rasterio profile with affine transform.

    Returns
    -------
    tuple[ndarray, ndarray]
        Gap-filled copies of (vs30, stdv).
    """

    candidate_2d = np.isnan(vs30) & (geology_ids != 0)
    transform = profile["transform"]
    candidate_rows, candidate_cols = np.where(candidate_2d)
    candidate_locations = pixel_coords_float32(
        candidate_rows, candidate_cols, transform
    )
    fillable_of_candidates = points_inside_coastline(candidate_locations)
    if not np.any(fillable_of_candidates):
        return vs30, stdv

    fillable_rows = candidate_rows[fillable_of_candidates]
    fillable_cols = candidate_cols[fillable_of_candidates]
    fillable_2d = np.zeros(vs30.shape, dtype=bool)
    fillable_2d[fillable_rows, fillable_cols] = True
    fillable_locations = candidate_locations[fillable_of_candidates]

    dx = abs(transform.a)
    buffer_pixels = round(constants.GAPFILL_LOCAL_GRID_SIZE_M / dx)
    expansion_pixels = round(constants.GAPFILL_LOCAL_GRID_EXPANSION_M / dx)
    max_buffer_pixels = round(constants.GAPFILL_MAX_LOCAL_GRID_HALF_WIDTH_M / dx)

    filled_vs30 = vs30.copy()
    filled_stdv = stdv.copy()

    valid_2d = ~np.isnan(vs30)
    while buffer_pixels <= max_buffer_pixels:
        # Dilate only the fillable mask (not the full nodata mask) to define
        # the donor search neighborhood. maximum_filter with a square kernel
        # is separable and runs in O(N) regardless of kernel size.
        struct_size = 2 * buffer_pixels + 1
        neighborhood_2d = scipy.ndimage.maximum_filter(fillable_2d, size=struct_size)

        valid_in_neighborhood = valid_2d & neighborhood_2d
        if not np.any(valid_in_neighborhood):
            logger.info(
                f"  Gap-fill: no valid donors within {buffer_pixels}-pixel "
                f"({buffer_pixels * dx:.0f}m) buffer, expanding"
            )
            buffer_pixels += expansion_pixels
            continue

        valid_rows, valid_cols = np.where(valid_in_neighborhood)
        valid_locations = pixel_coords_float32(
            valid_rows, valid_cols, transform
        )

        tree = scipy.spatial.KDTree(valid_locations)
        _, nn_indices = tree.query(fillable_locations)

        donor_rows = valid_rows[nn_indices]
        donor_cols = valid_cols[nn_indices]
        filled_vs30[fillable_rows, fillable_cols] = vs30[donor_rows, donor_cols]
        filled_stdv[fillable_rows, fillable_cols] = stdv[donor_rows, donor_cols]

        logger.info(
            f"  Gap-fill: filled {len(fillable_rows)} pixel(s) with nearest-neighbor values"
        )
        return filled_vs30, filled_stdv

    # Exhausted all buffer expansions without finding valid donors
    logger.warning(
        f"  Gap-fill: no valid donors found within maximum buffer of "
        f"{max_buffer_pixels} pixels ({max_buffer_pixels * dx:.0f}m). "
        f"{len(fillable_rows)} pixel(s) remain unfilled."
    )
    return filled_vs30, filled_stdv


def create_local_grid_config(
    easting: float,
    northing: float,
    gapfill_grid_config: config.GridConfig,
    half_width: int,
) -> config.GridConfig:
    """
    Create a local grid config for gap-filling a single point.

    Snaps the point to the nearest pixel center in the reference grid
    config, then creates a local grid of size (2 * half_width) centered on
    that pixel. The local grid uses the same spacing and origin alignment as
    gapfill_grid_config to ensure pixel centers match the full grid.

    Parameters
    ----------
    easting : float
        Query point easting (NZTM).
    northing : float
        Query point northing (NZTM).
    gapfill_grid_config : GridConfig
        Reference grid config defining the pixel alignment.
    half_width : int
        Half-width of the local grid in meters.

    Returns
    -------
    GridConfig
        Local grid config aligned to the reference grid.
    """

    snap_e = (
        gapfill_grid_config.grid_xmin
        + round((easting - gapfill_grid_config.grid_xmin) / gapfill_grid_config.grid_dx) * gapfill_grid_config.grid_dx
    )
    snap_n = (
        gapfill_grid_config.grid_ymin
        + round((northing - gapfill_grid_config.grid_ymin) / gapfill_grid_config.grid_dy) * gapfill_grid_config.grid_dy
    )

    return config.GridConfig(
        grid_xmin=snap_e - half_width,
        grid_xmax=snap_e + half_width,
        grid_ymin=snap_n - half_width,
        grid_ymax=snap_n + half_width,
        grid_dx=gapfill_grid_config.grid_dx,
        grid_dy=gapfill_grid_config.grid_dy,
    )

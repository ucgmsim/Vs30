"""Gap-fill nodata pixels in rasters."""

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
    """Return (N, 2) float32 NZTM pixel-centre coordinates for the given row/col indices.

    Float32 at NZTM magnitudes (~6.25e6 m) has worst-case error ~0.7 m,
    negligible on a 100 m grid for nearest-neighbor lookup.

    Parameters
    ----------
    rows, cols : ndarray
        Row and column indices.
    transform : rasterio.transform.Affine
        Pixel-to-world affine transform.

    Returns
    -------
    ndarray
        (N, 2) array of [easting, northing] float32 NZTM coordinates.
    """
    eastings = transform.c + transform.a * (cols + constants.PIXEL_CENTER_OFFSET)
    northings = transform.f + transform.e * (rows + constants.PIXEL_CENTER_OFFSET)
    return np.column_stack([
        eastings.astype(np.float32),
        northings.astype(np.float32),
    ])


def points_inside_coastline(locations: np.ndarray) -> np.ndarray:
    """Boolean mask of which NZTM points lie inside the NZ coastline polygon.

    Parameters
    ----------
    locations : ndarray
        (N, 2) array of [easting, northing] NZTM coordinates.

    Returns
    -------
    ndarray
        Length-N boolean mask.
    """
    if len(locations) == 0:
        return np.zeros(0, dtype=bool)

    # Use contains_xy (not shapely.within on point objects) to minimize memory.
    return shapely.contains_xy(
        raster.load_coast_union(), locations[:, 0], locations[:, 1]
    )


def classify_nodata(
    combined_vs30: np.ndarray,
    geology_ids: np.ndarray,
    locations: np.ndarray,
) -> np.ndarray:
    """Boolean mask of nodata pixels eligible for gap-filling.

    A pixel is eligible when it is NaN in ``combined_vs30``, has a
    non-water geology ID, and lies inside the NZ coastline polygon.

    Parameters
    ----------
    combined_vs30 : ndarray
        1D array of combined Vs30 values (NaN for nodata).
    geology_ids : ndarray
        1D array of geology category IDs.
    locations : ndarray
        (N, 2) array of [easting, northing] NZTM coordinates.

    Returns
    -------
    ndarray
        Length-N boolean mask.
    """
    candidate_indices = np.where(np.isnan(combined_vs30) & (geology_ids != 0))[0]
    fillable_mask = np.zeros(len(combined_vs30), dtype=bool)
    fillable_mask[candidate_indices] = points_inside_coastline(
        locations[candidate_indices]
    )

    n_fillable = int(fillable_mask.sum())
    if n_fillable > 0:
        logger.info(
            f"  Gap-fill: {n_fillable} on-land nodata pixel(s) identified for filling"
        )

    return fillable_mask


def fill_nodata_grid(
    vs30: np.ndarray,
    stdv: np.ndarray,
    geology_ids: np.ndarray,
    profile: dict,
) -> tuple[np.ndarray, np.ndarray]:
    """Fill on-land nodata gaps in a Vs30 grid using nearest-neighbor donors.

    Parameters
    ----------
    vs30, stdv : ndarray
        2D Vs30 / standard-deviation arrays (NaN for nodata).
    geology_ids : ndarray
        2D array of geology category IDs.
    profile : dict
        Rasterio profile with affine transform.

    Returns
    -------
    tuple[ndarray, ndarray]
        Gap-filled copies of (vs30, stdv).
    """
    transform = profile["transform"]
    candidate_rows, candidate_cols = np.where(np.isnan(vs30) & (geology_ids != 0))
    candidate_locations = pixel_coords_float32(
        candidate_rows, candidate_cols, transform
    )
    fillable_of_candidates = points_inside_coastline(candidate_locations)
    if not np.any(fillable_of_candidates):
        return vs30, stdv

    fillable_rows = candidate_rows[fillable_of_candidates]
    fillable_cols = candidate_cols[fillable_of_candidates]
    fillable_locations = candidate_locations[fillable_of_candidates]
    fillable_2d = np.zeros(vs30.shape, dtype=bool)
    fillable_2d[fillable_rows, fillable_cols] = True

    dx = abs(transform.a)
    half_width_pixels = round(constants.GAPFILL_INITIAL_HALF_WIDTH_M / dx)
    half_width_expansion_pixels = round(constants.GAPFILL_HALF_WIDTH_EXPANSION_M / dx)
    max_half_width_pixels = round(constants.GAPFILL_MAX_HALF_WIDTH_M / dx)

    filled_vs30 = vs30.copy()
    filled_stdv = stdv.copy()
    valid_2d = ~np.isnan(vs30)

    while half_width_pixels <= max_half_width_pixels:
        # Dilate the fillable mask (not the full nodata mask) to define the
        # donor search neighborhood. maximum_filter with a square kernel is
        # separable: O(N) regardless of kernel size.
        neighborhood_2d = scipy.ndimage.maximum_filter(
            fillable_2d, size=2 * half_width_pixels + 1
        )
        valid_in_neighborhood = valid_2d & neighborhood_2d
        if not np.any(valid_in_neighborhood):
            logger.info(
                f"  Gap-fill: no valid donors within {half_width_pixels}-pixel "
                f"({half_width_pixels * dx:.0f}m) half-width, expanding"
            )
            half_width_pixels += half_width_expansion_pixels
            continue

        valid_rows, valid_cols = np.where(valid_in_neighborhood)
        valid_locations = pixel_coords_float32(valid_rows, valid_cols, transform)
        _, nn_indices = scipy.spatial.KDTree(valid_locations).query(fillable_locations)

        filled_vs30[fillable_rows, fillable_cols] = vs30[
            valid_rows[nn_indices], valid_cols[nn_indices]
        ]
        filled_stdv[fillable_rows, fillable_cols] = stdv[
            valid_rows[nn_indices], valid_cols[nn_indices]
        ]
        logger.info(
            f"  Gap-fill: filled {len(fillable_rows)} pixel(s) with nearest-neighbor values"
        )
        return filled_vs30, filled_stdv

    logger.warning(
        f"  Gap-fill: no valid donors found within maximum half-width of "
        f"{max_half_width_pixels} pixels ({max_half_width_pixels * dx:.0f}m). "
        f"{len(fillable_rows)} pixel(s) remain unfilled."
    )
    return filled_vs30, filled_stdv


def create_local_grid_config(
    easting: float,
    northing: float,
    gapfill_grid_config: config.GridConfig,
    half_width: int,
) -> config.GridConfig:
    """Build a local grid for gap-filling a single point, aligned to the reference grid.

    Snaps the point to the nearest pixel centre of ``gapfill_grid_config``,
    then builds a (2 * half_width) box around it. The result shares the
    reference grid's pixel-centre lattice.

    Parameters
    ----------
    easting, northing : float
        Query point coordinates (NZTM).
    gapfill_grid_config : GridConfig
        Reference grid defining the pixel alignment.
    half_width : int
        Half-width of the local grid in metres. Must equal ``k * grid_dx +
        grid_dx / 2`` for integer k (e.g. 150 m, 250 m, … for dx=100 m) so
        the local grid's outer bounds remain pixel edges.

    Returns
    -------
    GridConfig
        Local grid aligned to the reference grid.
    """
    dx = gapfill_grid_config.grid_dx
    dy = gapfill_grid_config.grid_dy
    first_centre_x = gapfill_grid_config.grid_xmin + dx / 2
    first_centre_y = gapfill_grid_config.grid_ymin + dy / 2

    snap_e = first_centre_x + round((easting - first_centre_x) / dx) * dx
    snap_n = first_centre_y + round((northing - first_centre_y) / dy) * dy

    return config.GridConfig(
        grid_xmin=snap_e - half_width,
        grid_xmax=snap_e + half_width,
        grid_ymin=snap_n - half_width,
        grid_ymax=snap_n + half_width,
        grid_dx=dx,
        grid_dy=dy,
    )

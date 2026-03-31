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

import geopandas as gpd
import numpy as np
import shapely
from scipy.spatial import cKDTree

from vs30 import config as config_module
from vs30 import constants, raster

logger = logging.getLogger(__name__)


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
    nodata_mask = np.isnan(combined_vs30)
    if not np.any(nodata_mask):
        return np.zeros(len(combined_vs30), dtype=bool)

    # Exclude water pixels (GID=0)
    candidate_mask = nodata_mask & (geology_ids != 0)
    if not np.any(candidate_mask):
        return np.zeros(len(combined_vs30), dtype=bool)

    # Load coastline and test point-in-polygon for candidates only
    coastline_path = constants.GEOSPATIAL_DIR / constants.COASTLINE_SHAPEFILE_PATH
    raster.ensure_shapefile_extracted(coastline_path, "coast")
    coast_gdf = gpd.read_file(coastline_path)
    coast_union = coast_gdf.geometry.union_all()

    candidate_indices = np.where(candidate_mask)[0]
    candidate_points = shapely.points(locations[candidate_indices])
    inside = shapely.within(candidate_points, coast_union)

    fillable_mask = np.zeros(len(combined_vs30), dtype=bool)
    fillable_mask[candidate_indices] = inside

    n_fillable = int(np.sum(fillable_mask))
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
    """
    Fill nodata gaps in a combined VS30 grid using nearest-neighbor.

    Derives pixel center locations from the rasterio profile, identifies
    fillable pixels via classify_nodata, builds a cKDTree from all valid
    (non-NaN) pixel coordinates, and copies both vs30 and stdv values from
    the nearest valid donor pixel.

    Parameters
    ----------
    vs30 : ndarray
        2D array of combined VS30 values (NaN for nodata).
    stdv : ndarray
        2D array of combined standard deviation values (NaN for nodata).
    geology_ids : ndarray
        2D array of geology category IDs.
    profile : dict
        Rasterio profile with affine transform.

    Returns
    -------
    tuple[ndarray, ndarray]
        Gap-filled copies of (vs30, stdv).
    """
    nrows, ncols = vs30.shape
    vs30_flat = vs30.ravel()

    # Fast path: no nodata pixels
    nodata_mask = np.isnan(vs30_flat)
    if not np.any(nodata_mask):
        return vs30.copy(), stdv.copy()

    # Compute pixel center coordinates
    transform = profile["transform"]
    x_centers = transform.c + transform.a * (np.arange(ncols) + 0.5)
    y_centers = transform.f + transform.e * (np.arange(nrows) + 0.5)

    # Build full locations array for classify_nodata
    cols_2d, rows_2d = np.meshgrid(np.arange(ncols), np.arange(nrows))
    locations = np.column_stack([
        x_centers[cols_2d.ravel()],
        y_centers[rows_2d.ravel()],
    ])

    fillable_mask = classify_nodata(vs30_flat, geology_ids.ravel(), locations)
    if not np.any(fillable_mask):
        return vs30.copy(), stdv.copy()

    # Check for valid donor pixels
    valid_mask = ~nodata_mask
    if not np.any(valid_mask):
        logger.warning(
            "Gap-fill: no valid donor pixels found, returning arrays unchanged"
        )
        return vs30.copy(), stdv.copy()

    # Nearest-neighbor fill using cKDTree
    tree = cKDTree(locations[valid_mask])
    _, nn_indices = tree.query(locations[fillable_mask])

    filled_vs30 = vs30.copy()
    filled_stdv = stdv.copy()
    filled_vs30_flat = filled_vs30.ravel()
    filled_stdv_flat = filled_stdv.ravel()

    filled_vs30_flat[fillable_mask] = vs30_flat[valid_mask][nn_indices]
    filled_stdv_flat[fillable_mask] = stdv.ravel()[valid_mask][nn_indices]

    n_filled = int(np.sum(fillable_mask))
    logger.info(f"  Gap-fill: filled {n_filled} pixel(s) with nearest-neighbor values")

    return filled_vs30, filled_stdv


def create_local_grid_config(
    easting: float,
    northing: float,
    gapfill_grid_config: config_module.GridConfig,
    half_width: int,
) -> config_module.GridConfig:
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
    dx = gapfill_grid_config.grid_dx
    dy = gapfill_grid_config.grid_dy

    # Snap to nearest pixel center
    snap_e = gapfill_grid_config.grid_xmin + round(
        (easting - gapfill_grid_config.grid_xmin) / dx
    ) * dx
    snap_n = gapfill_grid_config.grid_ymin + round(
        (northing - gapfill_grid_config.grid_ymin) / dy
    ) * dy

    return config_module.GridConfig(
        grid_xmin=snap_e - half_width,
        grid_xmax=snap_e + half_width,
        grid_ymin=snap_n - half_width,
        grid_ymax=snap_n + half_width,
        grid_dx=dx,
        grid_dy=dy,
    )

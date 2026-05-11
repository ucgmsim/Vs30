"""Grid-pipeline MVN spatial adjustment."""

import logging
import time
from collections.abc import Callable

import numpy as np
import pandas as pd

from vs30 import constants, spatial, utils

logger = logging.getLogger(__name__)


def compute_spatial_adjustment_on_grid(
    vs30_array: np.ndarray,
    stdv_array: np.ndarray,
    profile: dict,
    observations_df: pd.DataFrame,
    model_values_df: pd.DataFrame,
    model_type: constants.ModelType,
    corr_fn: Callable,
    apply_alluvium_slope_mod: bool,
    apply_coastal_distance_mod: bool,
    noisy: bool = True,
    max_spatial_boolean_array_memory_gb: float = constants.MAX_SPATIAL_BOOLEAN_ARRAY_MEMORY_GB,
    slope_array: np.ndarray | None = None,
    coast_dist_array: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute MVN spatial adjustment on a grid.

    Parameters
    ----------
    vs30_array : np.ndarray
        Input Vs30 array (2D).
    stdv_array : np.ndarray
        Input standard deviation array (2D).
    profile : dict
        Rasterio profile with transform, crs, nodata.
    observations_df : pd.DataFrame
        Measured Vs30 values. Must contain columns: easting, northing, vs30,
        uncertainty.
    model_values_df : pd.DataFrame
        Updated categorical Vs30 values.
    model_type : ModelType
        Either GEOLOGY or TERRAIN.
    corr_fn : Callable
        Correlation function for spatial adjustment.
    apply_alluvium_slope_mod : bool
        Whether to apply slope-based interpolation for GID 4 (alluvium).
    apply_coastal_distance_mod : bool
        Whether to apply coastal-distance modification for GID 4 and GID 10.
    noisy : bool, optional
        Whether to apply noise weighting in spatial adjustment.
    max_spatial_boolean_array_memory_gb : float, optional
        Memory cap for spatial boolean arrays.
    slope_array : np.ndarray, optional
        Pre-computed slope array (for geology observation data preparation).
    coast_dist_array : np.ndarray, optional
        Pre-computed coast distance array.

    Returns
    -------
    adjusted_vs30 : np.ndarray
        Spatially-adjusted Vs30 array.
    adjusted_stdv : np.ndarray
        Spatially-adjusted standard deviation array.
    """
    logger.info(f"Starting spatial adjustment for {model_type} model")

    raster_data = spatial.RasterData.from_arrays(
        vs30=vs30_array,
        stdv=stdv_array,
        transform=profile["transform"],
        nodata=constants.NODATA_VALUE,
    )
    spatial.validate_raster_data(raster_data)
    spatial.validate_observations(observations_df)

    # Convert 1-indexed model IDs to 0-indexed array rows (0..max_id-1).
    mean_col, std_col = utils.select_vs30_columns_by_priority(
        list(model_values_df.columns)
    )
    max_id = model_values_df[constants.STANDARD_ID_COLUMN].max()
    updated_model_table = np.full((max_id, 2), np.nan)
    ids = model_values_df[constants.STANDARD_ID_COLUMN].to_numpy().astype(int) - 1
    valid = (ids >= 0) & (ids < max_id)
    updated_model_table[ids[valid], 0] = model_values_df[mean_col].to_numpy()[valid]
    updated_model_table[ids[valid], 1] = model_values_df[std_col].to_numpy()[valid]

    logger.info("Preparing observation data for spatial adjustment...")
    obs_data = spatial.prepare_observation_data(
        observations_df,
        raster_data,
        updated_model_table,
        model_type,
        apply_alluvium_slope_mod=apply_alluvium_slope_mod,
        apply_coastal_distance_mod=apply_coastal_distance_mod,
        noisy=noisy,
        slope_array=slope_array,
        coast_dist_array=coast_dist_array,
    )
    n_obs = len(obs_data.locations)
    logger.info(f"Prepared {n_obs} valid observations")

    if n_obs == 0:
        logger.warning(
            "No valid observations found within model bounds. "
            "Returning input arrays unchanged."
        )
        return vs30_array.copy(), stdv_array.copy()

    t_bbox_start = time.perf_counter()
    bbox_mask, grid_locs = spatial.find_affected_pixels(
        raster_data,
        obs_data,
        max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
        model_type=model_type,
        max_dist_m=constants.MAX_DIST_M,
    )
    logger.info(
        f"Found {int(bbox_mask.sum()):,} affected pixels in "
        f"{time.perf_counter() - t_bbox_start:.1f}s"
    )

    t_spatial_start = time.perf_counter()
    adjusted_vs30, adjusted_stdv = spatial.compute_spatial_adjustments(
        raster_data,
        obs_data,
        bbox_mask,
        grid_locs,
        corr_fn,
        max_dist_m=constants.MAX_DIST_M,
        max_points=constants.MAX_POINTS,
        noisy=noisy,
        cov_reduc=constants.COV_REDUC,
    )
    logger.info(
        f"Spatial adjustments completed in "
        f"{time.perf_counter() - t_spatial_start:.1f}s"
    )

    return adjusted_vs30, adjusted_stdv



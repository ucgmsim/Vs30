"""Grid-pipeline stage helpers (initial arrays, hybrid mods, spatial adjustment, combination, raster writing)."""

import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

from vs30 import config, constants, raster, spatial, utils

logger = logging.getLogger(__name__)


def create_initial_vs30_arrays(
    grid_config: config.GridConfig,
    model_type: constants.ModelType,
    model_values_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Create initial VS30 arrays from categorical model in memory.

    Generates category ID arrays by rasterizing terrain or geology data to the
    target grid, then maps category IDs to VS30 mean and standard deviation
    values from the model DataFrame.

    Parameters
    ----------
    grid_config : GridConfig
        Grid domain and resolution parameters.
    model_type : ModelType
        Either GEOLOGY or TERRAIN.
    model_values_df : pd.DataFrame
        DataFrame with categorical model values (must have 'id' column and
        mean/stdv columns recognized by ``select_vs30_columns_by_priority``).

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, dict]
        A tuple containing:
        - vs30_array (float32): VS30 mean values.
        - stdv_array (float32): VS30 standard deviation values.
        - id_array (uint8): Category ID array.
        - profile (dict): Rasterio profile describing the grid.
    """
    logger.info(f"Using grid parameters: {grid_config}")

    logger.info(f"Creating {model_type} category ID array...")
    id_array, profile = raster.create_category_id_array(
        model_type,
        xmin=grid_config.grid_xmin,
        xmax=grid_config.grid_xmax,
        ymin=grid_config.grid_ymin,
        ymax=grid_config.grid_ymax,
        dx=grid_config.grid_dx,
        dy=grid_config.grid_dy,
    )

    logger.info(f"Creating {model_type} VS30 arrays from IDs...")
    vs30_array, stdv_array = raster.create_vs30_arrays_from_ids(
        id_array, model_values_df
    )

    return vs30_array, stdv_array, id_array, profile


def compute_hybrid_geology_arrays(
    vs30_array: np.ndarray,
    stdv_array: np.ndarray,
    id_array: np.ndarray,
    profile: dict,
    apply_alluvium_slope_mod: bool,
    apply_coastal_distance_mod: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply hybrid geology modifications in memory.

    Computes slope and coastal distance arrays for the grid, then applies
    slope-based and coast-distance-based modifications to the geology VS30 model.

    Slope and coast distance arrays are returned because they are also needed
    later by ``prepare_observation_data`` to compute residuals at observation
    locations.

    Parameters
    ----------
    vs30_array : np.ndarray
        Initial geology VS30 array (2D, float32).
    stdv_array : np.ndarray
        Initial geology standard deviation array (2D, float32).
    id_array : np.ndarray
        Category ID array (2D, uint8).
    profile : dict
        Rasterio profile for the grid (needed for slope/coast computation).
    apply_alluvium_slope_mod : bool
        Whether to apply slope-based interpolation for GID 4 (alluvium).
        When False, GID 4 keeps its categorical Vs30 value.
    apply_coastal_distance_mod : bool
        Whether to apply coastal distance modification for GID 4 and GID 10.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        A tuple containing:
        - hybrid_vs30 (float32): Modified VS30 array.
        - hybrid_stdv (float32): Modified standard deviation array.
        - slope_array: Slope values for the grid.
        - coast_dist_array (float32): Distance to coast for the grid.
    """
    logger.info("Computing slope array...")
    slope_array = raster.compute_slope_array(profile)

    if apply_coastal_distance_mod:
        logger.info("Computing coast distance array...")
        coast_dist_array = raster.compute_coast_distance_array(profile)
    else:
        logger.info("Skipping coast distance computation (disabled in config)")
        coast_dist_array = np.zeros_like(vs30_array)

    hybrid_vs30, hybrid_stdv = raster.apply_hybrid_geology_modifications(
        vs30_array,
        stdv_array,
        id_array,
        slope_array,
        coast_dist_array,
        apply_alluvium_slope_mod=apply_alluvium_slope_mod,
        apply_coastal_distance_mod=apply_coastal_distance_mod,
    )

    return hybrid_vs30, hybrid_stdv, slope_array, coast_dist_array


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
    max_spatial_boolean_array_memory_gb: float = 1.0,
    slope_array: np.ndarray | None = None,
    coast_dist_array: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute MVN spatial adjustment on a grid in memory.

    Performs a spatial adjustment of VS30 arrays by:

    1. Constructing RasterData from in-memory arrays.
    2. Loading measurements and mapping them to categories.
    3. Computing spatial fits to update pixels affected by measurements.
    4. Returning the updated arrays.

    Parameters
    ----------
    vs30_array : np.ndarray
        Input VS30 array (2D).
    stdv_array : np.ndarray
        Input standard deviation array (2D).
    profile : dict
        Rasterio profile with transform, crs, nodata.
    observations_df : pd.DataFrame
        DataFrame with measured VS30 values. Must contain columns:
        easting, northing, vs30, uncertainty.
    model_values_df : pd.DataFrame
        DataFrame with updated categorical Vs30 values.
    model_type : ModelType
        Model type: either GEOLOGY or TERRAIN.
    corr_fn : Callable
        Correlation function for spatial adjustment.
    apply_alluvium_slope_mod : bool
        Whether to apply slope-based interpolation for GID 4 (alluvium).
    apply_coastal_distance_mod : bool
        Whether to apply coastal distance modification for GID 4 and GID 10.
    noisy : bool, optional
        Whether to apply noise weighting in spatial adjustment.
    max_spatial_boolean_array_memory_gb : float, optional
        Maximum memory for spatial boolean arrays.
    slope_array : np.ndarray, optional
        Pre-computed slope array (for geology observation data preparation).
    coast_dist_array : np.ndarray, optional
        Pre-computed coast distance array.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (adjusted_vs30, adjusted_stdv) arrays.
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

    # Model IDs are 1-indexed; convert to 0-indexed array indices.
    mean_col, std_col = raster.select_vs30_columns_by_priority(
        list(model_values_df.columns)
    )
    max_id = model_values_df[constants.STANDARD_ID_COLUMN].max()
    # Indices are 1-based ids minus 1, so we need max_id rows (covering 0..max_id-1).
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

    logger.info("Finding pixels affected by observations...")
    t_bbox_start = time.perf_counter()
    bbox_mask, grid_locs = spatial.find_affected_pixels(
        raster_data,
        obs_data,
        max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
        model_type=model_type,
        max_dist_m=constants.MAX_DIST_M,
    )
    t_bbox_elapsed = time.perf_counter() - t_bbox_start
    logger.info(
        f"Found {int(bbox_mask.sum()):,} affected pixels "
        f"in {t_bbox_elapsed:.1f}s"
    )

    logger.info("Computing spatial updates...")
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
    t_spatial_elapsed = time.perf_counter() - t_spatial_start
    logger.info(f"Spatial adjustments completed in {t_spatial_elapsed:.1f}s")

    return adjusted_vs30, adjusted_stdv


def combine_model_arrays(
    geol_vs30: np.ndarray,
    geol_stdv: np.ndarray,
    terr_vs30: np.ndarray,
    terr_stdv: np.ndarray,
    combination_method: constants.CombinationMethod,
    combine_ratio: float | None = None,
    nodata: float = constants.NODATA_VALUE,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combine geology and terrain VS30 arrays in memory.

    Combines the two model outputs in log-space using the specified weighting
    method.

    Parameters
    ----------
    geol_vs30 : np.ndarray
        Geology model VS30 array.
    geol_stdv : np.ndarray
        Geology model standard deviation array.
    terr_vs30 : np.ndarray
        Terrain model VS30 array.
    terr_stdv : np.ndarray
        Terrain model standard deviation array.
    combination_method : CombinationMethod
        Method for combining models: STANDARD_DEVIATION_WEIGHTING for
        variance-based weighting, or RATIO for fixed-ratio weighting.
    combine_ratio : float, optional
        Geology-to-terrain weight ratio. Required when combination_method is RATIO.
    nodata : float, optional
        No-data value.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (combined_vs30, combined_stdv) arrays.
    """
    geol_vs30 = geol_vs30.astype(np.float32, copy=True)
    geol_stdv = geol_stdv.astype(np.float32, copy=True)
    terr_vs30 = terr_vs30.astype(np.float32, copy=True)
    terr_stdv = terr_stdv.astype(np.float32, copy=True)
    for arr in (geol_vs30, geol_stdv, terr_vs30, terr_stdv):
        arr[arr == nodata] = np.nan

    return utils.combine_vs30_models(
        geol_vs30=geol_vs30,
        geol_stdv=geol_stdv,
        terr_vs30=terr_vs30,
        terr_stdv=terr_stdv,
        combination_method=combination_method,
        combine_ratio=combine_ratio,
    )


def write_raster(
    output_path: Path,
    profile: dict,
    bands: list[np.ndarray],
    band_descriptions: tuple[str, ...],
    *,
    dtype: str = "float32",
    nodata: float | None = constants.NODATA_VALUE,
) -> None:
    """
    Write a multi-band raster to a GeoTIFF file.

    Parameters
    ----------
    output_path : Path
        Output file path.
    profile : dict
        Rasterio profile with CRS, transform, dimensions. Caller-supplied
        values for dtype/count/nodata/compress are overridden.
    bands : list[np.ndarray]
        2D arrays to write, one per band.
    band_descriptions : tuple[str, ...]
        Per-band description strings; must match ``len(bands)``.
    dtype : str, optional
        Output dtype (e.g. ``"float32"`` or ``"uint8"``). Default ``"float32"``.
    nodata : float or None, optional
        No-data value. Pass ``None`` to omit nodata metadata.
        Default ``constants.NODATA_VALUE``.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_profile = profile.copy()
    write_profile.update(
        {
            "dtype": dtype,
            "count": len(bands),
            "nodata": nodata,
            "compress": constants.GEOTIFF_COMPRESSION,
        }
    )

    with rasterio.open(output_path, "w", **write_profile) as dst:
        for i, band in enumerate(bands, start=1):
            data = band if band.dtype.name == dtype else band.astype(dtype)
            dst.write(data, i)
        dst.descriptions = band_descriptions

    logger.info(f"Wrote raster: {output_path}")

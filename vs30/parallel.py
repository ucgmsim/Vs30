"""
Multiprocessing support for VS30 computation.

This module provides parallel processing capabilities for the computationally
expensive spatial adjustment. It divides input data into chunks and
processes them in parallel using Python's multiprocessing module.

The core spatial adjustment functions in spatial.py remain unchanged - each
worker process simply calls them with a smaller input, unaware it's part of
a parallel job.
"""

import contextlib
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import threadpoolctl
from qcore import coordinates
from tqdm import tqdm

from vs30 import category, constants, raster, spatial, utils


@contextlib.contextmanager
def single_threaded_blas():
    """Restrict BLAS to single-threaded operation to prevent oversubscription during multiprocessing."""
    with threadpoolctl.threadpool_limits(limits=1, user_api="blas"):
        yield


def resolve_n_proc(n_proc: int | None) -> int:
    """
    Convert user input to actual process count.

    Parameters
    ----------
    n_proc : int or None
        User-specified number of processes.
        None or 1 = single-threaded
        -1 = use all available CPU cores
        > 1 = use that many processes

    Returns
    -------
    int
        Actual number of processes to use (always >= 1)

    Raises
    ------
    ValueError
        If n_proc is 0 or less than -1
    """
    if n_proc is None or n_proc == 1:
        return 1
    if n_proc == -1:
        return mp.cpu_count()
    if n_proc < -1 or n_proc == 0:
        raise ValueError(f"n_proc must be -1, 1, or > 1, got {n_proc}")
    return min(n_proc, mp.cpu_count())


# Use spawn context to avoid GDAL fork issues
# GDAL is not fork-safe; using spawn starts fresh processes without inheriting
# the parent's GDAL state, which prevents deadlocks
_spawn_context = mp.get_context("spawn")


def process_geology_at_points(
    points: np.ndarray,
    model_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    coast_distance_raster: Path | None = None,
    noisy: bool = False,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Process geology model at points, including hybrid modifications and spatial adjustment.

    This function encapsulates the full geology processing pipeline:
    1. Get initial Vs30 values from categorical model
    2. Apply hybrid modifications (slope and coastal distance) if coast raster provided
    3. Apply spatial adjustment using observations

    Parameters
    ----------
    points : ndarray
        Array of shape (n_points, 2) with (easting, northing) coordinates.
    model_df : DataFrame
        Categorical geology model with Vs30 mean and standard deviation per category.
    observations_df : DataFrame
        Observation data with columns: easting, northing, vs30, uncertainty.
    coast_distance_raster : Path, optional
        Path to coastal distance raster. If None, hybrid modifications are skipped.
    noisy : bool
        Whether to apply noise weighting in spatial adjustment.

    Returns
    -------
    geol_ids : ndarray
        Geology category IDs at each point.
    geol_vs30 : ndarray
        Initial geology Vs30 values (before hybrid mods).
    geol_stdv : ndarray
        Initial geology standard deviation (before hybrid mods).
    geol_vs30_hybrid : ndarray
        Geology Vs30 after hybrid modifications.
    geol_stdv_hybrid : ndarray
        Geology standard deviation after hybrid modifications.
    geol_mvn_vs30 : ndarray
        Final geology Vs30 after spatial adjustment.
    geol_mvn_stdv : ndarray
        Final geology standard deviation after spatial adjustment.
    """
    # Assign geology category IDs to points
    geol_ids = category.assign_to_category_geology(points)

    # Get initial Vs30 values from categorical model
    geol_vs30_df = category.get_vs30_for_ids(geol_ids, model_df)

    # Apply hybrid modifications (slope and coastal distance)
    if coast_distance_raster is not None and coast_distance_raster.exists():
        geol_vs30_hybrid, geol_stdv_hybrid = (
            raster.apply_hybrid_modifications_at_points(
                points,
                geol_vs30_df[constants.COL_CATEGORY_VS30_MEAN].values,
                geol_vs30_df[constants.COL_CATEGORY_VS30_STDV].values,
                geol_ids,
                slope_raster_path=None,  # Uses default from constants
                coast_distance_raster_path=coast_distance_raster,
            )
        )
    else:
        geol_vs30_hybrid = geol_vs30_df[constants.COL_CATEGORY_VS30_MEAN].values
        geol_stdv_hybrid = geol_vs30_df[constants.COL_CATEGORY_VS30_STDV].values

    # Apply spatial adjustment if observations are available
    if len(observations_df) > 0:
        obs_locs = observations_df[
            [constants.COL_EASTING, constants.COL_NORTHING]
        ].values
        obs_geol_ids = category.assign_to_category_geology(obs_locs)
        obs_geol_vs30_df = category.get_vs30_for_ids(obs_geol_ids, model_df)
        geol_mvn_vs30, geol_mvn_stdv = spatial.compute_spatial_adjustment_at_points(
            points=points,
            model_vs30=geol_vs30_hybrid,
            model_stdv=geol_stdv_hybrid,
            obs_locations=obs_locs,
            obs_vs30=observations_df[constants.COL_VS30].values,
            obs_model_vs30=obs_geol_vs30_df[constants.COL_CATEGORY_VS30_MEAN].values,
            obs_model_stdv=obs_geol_vs30_df[constants.COL_CATEGORY_VS30_STDV].values,
            obs_uncertainty=observations_df[constants.COL_UNCERTAINTY].values,
            model_type=constants.ModelType.GEOLOGY,
            noisy=noisy,
        )
    else:
        geol_mvn_vs30 = geol_vs30_hybrid
        geol_mvn_stdv = geol_stdv_hybrid

    return (
        geol_ids,
        geol_vs30_df[constants.COL_CATEGORY_VS30_MEAN].values,
        geol_vs30_df[constants.COL_CATEGORY_VS30_STDV].values,
        geol_vs30_hybrid,
        geol_stdv_hybrid,
        geol_mvn_vs30,
        geol_mvn_stdv,
    )


def process_terrain_at_points(
    points: np.ndarray,
    model_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    noisy: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process terrain model at points, including spatial adjustment.

    This function encapsulates the full terrain processing pipeline:
    1. Get initial Vs30 values from categorical model
    2. Apply spatial adjustment using observations (no hybrid modifications for terrain)

    Parameters
    ----------
    points : ndarray
        Array of shape (n_points, 2) with (easting, northing) coordinates.
    model_df : DataFrame
        Categorical terrain model with Vs30 mean and standard deviation per category.
    observations_df : DataFrame
        Observation data with columns: easting, northing, vs30, uncertainty.
    noisy : bool
        Whether to apply noise weighting in spatial adjustment.

    Returns
    -------
    terr_ids : ndarray
        Terrain category IDs at each point.
    terr_vs30 : ndarray
        Initial terrain Vs30 values.
    terr_stdv : ndarray
        Initial terrain standard deviation.
    terr_mvn_vs30 : ndarray
        Final terrain Vs30 after spatial adjustment.
    terr_mvn_stdv : ndarray
        Final terrain standard deviation after spatial adjustment.
    """
    # Assign terrain category IDs to points
    terr_ids = category.assign_to_category_terrain(points)

    # Get initial Vs30 values from categorical model
    terr_vs30_df = category.get_vs30_for_ids(terr_ids, model_df)

    # Apply spatial adjustment if observations are available
    if len(observations_df) > 0:
        obs_locs = observations_df[
            [constants.COL_EASTING, constants.COL_NORTHING]
        ].values
        obs_terr_ids = category.assign_to_category_terrain(obs_locs)
        obs_terr_vs30_df = category.get_vs30_for_ids(obs_terr_ids, model_df)
        terr_mvn_vs30, terr_mvn_stdv = spatial.compute_spatial_adjustment_at_points(
            points=points,
            model_vs30=terr_vs30_df[constants.COL_CATEGORY_VS30_MEAN].values,
            model_stdv=terr_vs30_df[constants.COL_CATEGORY_VS30_STDV].values,
            obs_locations=obs_locs,
            obs_vs30=observations_df[constants.COL_VS30].values,
            obs_model_vs30=obs_terr_vs30_df[constants.COL_CATEGORY_VS30_MEAN].values,
            obs_model_stdv=obs_terr_vs30_df[constants.COL_CATEGORY_VS30_STDV].values,
            obs_uncertainty=observations_df[constants.COL_UNCERTAINTY].values,
            model_type=constants.ModelType.TERRAIN,
            noisy=noisy,
        )
    else:
        terr_mvn_vs30 = terr_vs30_df[constants.COL_CATEGORY_VS30_MEAN].values
        terr_mvn_stdv = terr_vs30_df[constants.COL_CATEGORY_VS30_STDV].values

    return (
        terr_ids,
        terr_vs30_df[constants.COL_CATEGORY_VS30_MEAN].values,
        terr_vs30_df[constants.COL_CATEGORY_VS30_STDV].values,
        terr_mvn_vs30,
        terr_mvn_stdv,
    )


@dataclass
class LocationsChunkConfig:
    """
    Configuration parameters for processing a locations chunk.

    Attributes
    ----------
    lon_column : str
        Name of longitude column in input CSV.
    lat_column : str
        Name of latitude column in input CSV.
    include_intermediate : bool
        Whether to include intermediate values in output.
    combination_method : str or float
        Method for combining geology and terrain models.
    coast_distance_raster : Path or None
        Path to coastal distance raster for hybrid modifications.
    noisy : bool
        Whether to apply noise weighting in spatial adjustment.
    """

    lon_column: str
    lat_column: str
    include_intermediate: bool
    combination_method: str | float
    coast_distance_raster: Path | None
    noisy: bool


def process_locations_chunk(
    args: tuple,
) -> tuple[int, pd.DataFrame]:  # pragma: no cover
    """
    Worker function: process a chunk of locations through the full pipeline.

    This function runs in a separate process and processes a subset of
    locations through the complete VS30 pipeline (geology + terrain + combine).

    Note: This function is excluded from coverage because it runs in a
    spawned subprocess which cannot be tracked by pytest-cov.

    Parameters
    ----------
    args : tuple
        (chunk_df, chunk_id, observations_df, geol_model_df, terr_model_df, config)

    Returns
    -------
    tuple
        (chunk_id, result_df) where result_df has all computed columns
    """
    (
        chunk_df,
        chunk_id,
        observations_df,
        geol_model_df,
        terr_model_df,
        config,
    ) = args

    # Make a copy to avoid modifying the original
    chunk_df = chunk_df.copy()

    # Convert coordinates to NZTM
    nztm_coords = coordinates.wgs_depth_to_nztm(
        np.column_stack(
            [chunk_df[config.lat_column].values, chunk_df[config.lon_column].values]
        )
    )
    northing, easting = nztm_coords[:, 0], nztm_coords[:, 1]
    chunk_df[constants.COL_EASTING] = easting
    chunk_df[constants.COL_NORTHING] = northing
    points = np.column_stack([easting, northing])

    # Process geology model
    (
        geol_ids,
        geol_vs30,
        geol_stdv,
        geol_vs30_hybrid,
        geol_stdv_hybrid,
        geol_mvn_vs30,
        geol_mvn_stdv,
    ) = process_geology_at_points(
        points,
        geol_model_df,
        observations_df,
        config.coast_distance_raster,
        config.noisy,
    )

    chunk_df[constants.COL_GEOLOGY_ID] = geol_ids
    if config.include_intermediate:
        chunk_df[constants.COL_GEOLOGY_VS30] = geol_vs30
        chunk_df[constants.COL_GEOLOGY_STDV] = geol_stdv
        chunk_df[constants.COL_GEOLOGY_VS30_HYBRID] = geol_vs30_hybrid
        chunk_df[constants.COL_GEOLOGY_STDV_HYBRID] = geol_stdv_hybrid
    chunk_df[constants.COL_GEOLOGY_MVN_VS30] = geol_mvn_vs30
    chunk_df[constants.COL_GEOLOGY_MVN_STDV] = geol_mvn_stdv

    # Process terrain model
    (
        terr_ids,
        terr_vs30,
        terr_stdv,
        terr_mvn_vs30,
        terr_mvn_stdv,
    ) = process_terrain_at_points(points, terr_model_df, observations_df, config.noisy)

    chunk_df[constants.COL_TERRAIN_ID] = terr_ids
    if config.include_intermediate:
        chunk_df[constants.COL_TERRAIN_VS30] = terr_vs30
        chunk_df[constants.COL_TERRAIN_STDV] = terr_stdv
    chunk_df[constants.COL_TERRAIN_MVN_VS30] = terr_mvn_vs30
    chunk_df[constants.COL_TERRAIN_MVN_STDV] = terr_mvn_stdv

    # Combine models
    combined_vs30, combined_stdv = utils.combine_vs30_models(
        geol_mvn_vs30,
        geol_mvn_stdv,
        terr_mvn_vs30,
        terr_mvn_stdv,
        config.combination_method,
    )

    chunk_df[constants.COL_VS30] = combined_vs30
    chunk_df[constants.COL_COMBINED_STDV] = combined_stdv

    return chunk_id, chunk_df


def process_pixels_chunk(
    args: tuple,
) -> tuple[int, list[spatial.SpatialAdjustmentResult]]:  # pragma: no cover
    """
    Worker function: compute spatial adjustments for a chunk of affected pixels.

    This function runs in a separate process and computes spatial adjustments
    for a subset of affected pixels.

    Note: This function is excluded from coverage because it runs in a
    spawned subprocess which cannot be tracked by pytest-cov.

    Parameters
    ----------
    args : tuple
        (pixel_indices, chunk_id, pixel_data_dict, obs_data_dict, config_params)

    Returns
    -------
    tuple
        (chunk_id, list of spatial.SpatialAdjustmentResult)
    """
    pixel_indices, chunk_id, pixel_data_dict, obs_data_dict, config_params = args

    # Reconstruct ObservationData from dict (dataclasses can't always be pickled cleanly)
    obs_data = spatial.ObservationData(
        locations=obs_data_dict[constants.KEY_LOCATIONS],
        vs30=obs_data_dict[constants.COL_VS30],
        model_vs30=obs_data_dict[constants.KEY_MODEL_VS30],
        model_stdv=obs_data_dict[constants.KEY_MODEL_STDV],
        residuals=obs_data_dict[constants.KEY_RESIDUALS],
        omega=obs_data_dict[constants.KEY_OMEGA],
        uncertainty=obs_data_dict[constants.COL_UNCERTAINTY],
    )

    updates = []
    for idx in pixel_indices:
        # Get pixel data from the prepared dict
        pixel_info = pixel_data_dict[idx]
        pixel = spatial.PixelData(
            location=pixel_info[constants.KEY_LOCATION],
            vs30=pixel_info[constants.COL_VS30],
            stdv=pixel_info[constants.KEY_STDV],
            index=pixel_info[constants.KEY_INDEX],
        )

        update = spatial.compute_spatial_adjustment_for_pixel(
            pixel,
            obs_data,
            config_params[constants.KEY_MODEL_TYPE],
            max_dist_m=config_params[constants.KEY_MAX_DIST_M],
            max_points=config_params[constants.KEY_MAX_POINTS],
            noisy=config_params[constants.KEY_NOISY],
            cov_reduc=config_params[constants.KEY_COV_REDUC],
        )

        if update is not None:
            updates.append(update)

    return chunk_id, updates


def run_parallel_locations(
    locations_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    geol_model_df: pd.DataFrame,
    terr_model_df: pd.DataFrame,
    config: LocationsChunkConfig,
    n_proc: int,
) -> pd.DataFrame:
    """
    Process locations in parallel.

    Divides the locations DataFrame into chunks and processes each chunk
    in a separate process using the full VS30 pipeline.

    Parameters
    ----------
    locations_df : DataFrame
        Input locations with lon/lat columns (not yet converted to NZTM)
    observations_df : DataFrame
        Observation data for spatial adjustment (must have easting, northing, vs30, uncertainty)
    geol_model_df : DataFrame
        Geology categorical model
    terr_model_df : DataFrame
        Terrain categorical model
    config : LocationsChunkConfig
        Configuration parameters for processing
    n_proc : int
        Number of processes to use (must be > 1)

    Returns
    -------
    DataFrame
        Results with vs30, stdv, and intermediate columns (if requested)
    """
    split_indices = np.array_split(range(len(locations_df)), n_proc)
    chunk_args = [
        (
            locations_df.iloc[idx].reset_index(drop=True),
            i,
            observations_df,
            geol_model_df,
            terr_model_df,
            config,
        )
        for i, idx in enumerate(split_indices)
        if len(idx) > 0
    ]

    actual_n_proc = len(chunk_args)

    # Process in parallel using spawn context (avoids GDAL fork issues)
    # Use single_threaded_blas to prevent BLAS oversubscription
    with single_threaded_blas():
        with _spawn_context.Pool(processes=actual_n_proc) as pool:
            results = list(
                tqdm(
                    pool.imap(process_locations_chunk, chunk_args),
                    total=actual_n_proc,
                    desc=f"Processing locations ({actual_n_proc} workers)",
                    unit="chunk",
                )
            )

    # Merge: concatenate in order
    results.sort(key=lambda x: x[0])
    return pd.concat([r[1] for r in results], ignore_index=True)


def run_parallel_spatial_fit(
    affected_flat_indices: np.ndarray,
    raster_data,  # RasterData - avoid import cycle
    obs_data: spatial.ObservationData,
    model_type: str,
    max_dist_m: float,
    max_points: int,
    noisy: bool,
    cov_reduc: float,
    n_proc: int,
) -> list[spatial.SpatialAdjustmentResult]:
    """
    Compute spatial adjustments for affected pixels in parallel.

    Divides the affected pixels into chunks and processes each chunk
    in a separate process.

    Parameters
    ----------
    affected_flat_indices : ndarray
        Flat indices of affected pixels in the raster
    raster_data : RasterData
        Raster data object with vs30, stdv, and coordinate info
    obs_data : ObservationData
        Observation data for spatial adjustment
    model_type : constants.ModelType
        Model type (ModelType.GEOLOGY or ModelType.TERRAIN)
    max_dist_m : float
        Maximum distance for considering observations
    max_points : int
        Maximum number of observations per pixel
    noisy : bool
        Whether to apply noise weighting
    cov_reduc : float
        Covariance reduction factor
    n_proc : int
        Number of processes to use (must be > 1)

    Returns
    -------
    list[spatial.SpatialAdjustmentResult]
        Updates for all affected pixels
    """
    # Prepare pixel data as a dict (for pickling)
    grid_locs = raster_data.get_coordinates()
    pixel_data_dict = {}
    for i, flat_idx in enumerate(affected_flat_indices):
        # Map flat index to valid index for coordinates
        valid_idx = np.searchsorted(raster_data.valid_flat_indices, flat_idx)
        if valid_idx < len(grid_locs):
            pixel_data_dict[i] = {
                constants.KEY_LOCATION: grid_locs[valid_idx],
                constants.COL_VS30: float(raster_data.vs30.flat[flat_idx]),
                constants.KEY_STDV: float(raster_data.stdv.flat[flat_idx]),
                constants.KEY_INDEX: int(flat_idx),
            }

    # Convert ObservationData to dict for pickling
    obs_data_dict = {
        constants.KEY_LOCATIONS: obs_data.locations,
        constants.COL_VS30: obs_data.vs30,
        constants.KEY_MODEL_VS30: obs_data.model_vs30,
        constants.KEY_MODEL_STDV: obs_data.model_stdv,
        constants.KEY_RESIDUALS: obs_data.residuals,
        constants.KEY_OMEGA: obs_data.omega,
        constants.COL_UNCERTAINTY: obs_data.uncertainty,
    }

    # Config params
    config_params = {
        constants.KEY_MODEL_TYPE: model_type,
        constants.KEY_MAX_DIST_M: max_dist_m,
        constants.KEY_MAX_POINTS: max_points,
        constants.KEY_NOISY: noisy,
        constants.KEY_COV_REDUC: cov_reduc,
    }

    # Divide pixel indices into chunks
    pixel_indices_list = list(range(len(affected_flat_indices)))
    chunks = np.array_split(pixel_indices_list, n_proc)
    chunk_args = [
        (list(chunk), i, pixel_data_dict, obs_data_dict, config_params)
        for i, chunk in enumerate(chunks)
        if len(chunk) > 0
    ]

    actual_n_proc = len(chunk_args)

    # Process in parallel using spawn context (avoids GDAL fork issues)
    # Use single_threaded_blas to prevent BLAS oversubscription
    with single_threaded_blas():
        with _spawn_context.Pool(processes=actual_n_proc) as pool:
            results = list(
                tqdm(
                    pool.imap(process_pixels_chunk, chunk_args),
                    total=actual_n_proc,
                    desc=f"Processing pixels ({actual_n_proc} workers)",
                    unit="chunk",
                )
            )

    # Merge: concatenate update lists (order does not matter; each update carries its pixel_index)
    all_updates = []
    for _, chunk_updates in results:
        all_updates.extend(chunk_updates)

    return all_updates

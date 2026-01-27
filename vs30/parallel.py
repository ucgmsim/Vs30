"""
Multiprocessing support for VS30 computation.

This module provides parallel processing capabilities for the computationally
expensive spatial adjustment. It divides input data into chunks and
processes them in parallel using Python's multiprocessing module.

The core spatial adjustment functions in spatial.py remain unchanged - each
worker process simply calls them with a smaller input, unaware it's part of
a parallel job.
"""

from contextlib import contextmanager
from dataclasses import dataclass
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from vs30.category import get_vs30_for_points
from vs30.raster import _get_slope_raster_path, apply_hybrid_modifications_at_points
from vs30.spatial import (
    SpatialAdjustmentResult,
    ObservationData,
    PixelData,
    compute_spatial_adjustment_at_points,
    compute_spatial_adjustment_for_pixel,
)
from vs30.utils import combine_vs30_models


@contextmanager
def single_threaded_blas():
    """
    Context manager to restrict BLAS to single-threaded operation.

    Use this when running multiprocessing to prevent oversubscription.
    For example, 8 processes x 8 BLAS threads = 64 threads competing
    for 8 cores, which is slower than 8 single-threaded processes.

    This replaces the previous approach of setting OMP_NUM_THREADS etc.
    at import time, allowing BLAS thread control to happen at runtime
    after CLI arguments have been parsed.

    Usage
    -----
        with single_threaded_blas():
            with mp.Pool(n_proc) as pool:
                results = pool.map(worker_func, chunks)
    """
    with threadpool_limits(limits=1, user_api="blas"):
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
_spawn_context = mp.get_context('spawn')


# =============================================================================
# Point Processing Helper Functions
# =============================================================================


def process_geology_at_points(
    points: np.ndarray,
    model_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    coast_distance_raster: Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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
    # Get initial Vs30 values at points
    geol_vs30, geol_stdv, geol_ids = get_vs30_for_points(
        points, "geology", model_df
    )

    # Apply hybrid modifications (slope and coastal distance)
    if coast_distance_raster is not None and coast_distance_raster.exists():
        geol_vs30_hybrid, geol_stdv_hybrid = apply_hybrid_modifications_at_points(
            points,
            geol_vs30,
            geol_stdv,
            geol_ids,
            slope_raster_path=_get_slope_raster_path(),
            coast_distance_raster_path=coast_distance_raster,
        )
    else:
        geol_vs30_hybrid = geol_vs30
        geol_stdv_hybrid = geol_stdv

    # Apply spatial adjustment if observations are available
    if len(observations_df) > 0:
        obs_locs = observations_df[["easting", "northing"]].values
        obs_geol_vs30, obs_geol_stdv, _ = get_vs30_for_points(
            obs_locs, "geology", model_df
        )
        geol_mvn_vs30, geol_mvn_stdv = compute_spatial_adjustment_at_points(
            points=points,
            model_vs30=geol_vs30_hybrid,
            model_stdv=geol_stdv_hybrid,
            obs_locations=obs_locs,
            obs_vs30=observations_df["vs30"].values,
            obs_model_vs30=obs_geol_vs30,
            obs_model_stdv=obs_geol_stdv,
            obs_uncertainty=observations_df["uncertainty"].values,
            model_type="geology",
        )
    else:
        geol_mvn_vs30 = geol_vs30_hybrid
        geol_mvn_stdv = geol_stdv_hybrid

    return (
        geol_ids,
        geol_vs30,
        geol_stdv,
        geol_vs30_hybrid,
        geol_stdv_hybrid,
        geol_mvn_vs30,
        geol_mvn_stdv,
    )


def process_terrain_at_points(
    points: np.ndarray,
    model_df: pd.DataFrame,
    observations_df: pd.DataFrame,
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
    # Get initial Vs30 values at points
    terr_vs30, terr_stdv, terr_ids = get_vs30_for_points(
        points, "terrain", model_df
    )

    # Apply spatial adjustment if observations are available
    if len(observations_df) > 0:
        obs_locs = observations_df[["easting", "northing"]].values
        obs_terr_vs30, obs_terr_stdv, _ = get_vs30_for_points(
            obs_locs, "terrain", model_df
        )
        terr_mvn_vs30, terr_mvn_stdv = compute_spatial_adjustment_at_points(
            points=points,
            model_vs30=terr_vs30,
            model_stdv=terr_stdv,
            obs_locations=obs_locs,
            obs_vs30=observations_df["vs30"].values,
            obs_model_vs30=obs_terr_vs30,
            obs_model_stdv=obs_terr_stdv,
            obs_uncertainty=observations_df["uncertainty"].values,
            model_type="terrain",
        )
    else:
        terr_mvn_vs30 = terr_vs30
        terr_mvn_stdv = terr_stdv

    return terr_ids, terr_vs30, terr_stdv, terr_mvn_vs30, terr_mvn_stdv


# =============================================================================
# Data Classes for Parallel Processing
# =============================================================================


@dataclass
class LocationsChunkConfig:
    """Configuration parameters for processing a locations chunk."""

    lon_column: str
    lat_column: str
    include_intermediate: bool
    combination_method: str | float
    coast_distance_raster: Path | None
    k_value: float = 3.0
    epsilon: float = 1e-10


# =============================================================================
# Worker Functions
# =============================================================================


def _process_locations_chunk(args: tuple) -> tuple[int, pd.DataFrame]:  # pragma: no cover
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
    from qcore import coordinates

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
    lat_col = config.lat_column
    lon_col = config.lon_column
    nztm_coords = coordinates.wgs_depth_to_nztm(
        np.column_stack([chunk_df[lat_col].values, chunk_df[lon_col].values])
    )
    northing, easting = nztm_coords[:, 0], nztm_coords[:, 1]
    chunk_df["easting"] = easting
    chunk_df["northing"] = northing
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
        points, geol_model_df, observations_df, config.coast_distance_raster
    )

    chunk_df["geology_id"] = geol_ids
    if config.include_intermediate:
        chunk_df["geology_vs30"] = geol_vs30
        chunk_df["geology_stdv"] = geol_stdv
        chunk_df["geology_vs30_hybrid"] = geol_vs30_hybrid
        chunk_df["geology_stdv_hybrid"] = geol_stdv_hybrid
    chunk_df["geology_mvn_vs30"] = geol_mvn_vs30
    chunk_df["geology_mvn_stdv"] = geol_mvn_stdv

    # Process terrain model
    (
        terr_ids,
        terr_vs30,
        terr_stdv,
        terr_mvn_vs30,
        terr_mvn_stdv,
    ) = process_terrain_at_points(points, terr_model_df, observations_df)

    chunk_df["terrain_id"] = terr_ids
    if config.include_intermediate:
        chunk_df["terrain_vs30"] = terr_vs30
        chunk_df["terrain_stdv"] = terr_stdv
    chunk_df["terrain_mvn_vs30"] = terr_mvn_vs30
    chunk_df["terrain_mvn_stdv"] = terr_mvn_stdv

    # Combine models
    combined_vs30, combined_stdv = combine_vs30_models(
        geol_mvn_vs30,
        geol_mvn_stdv,
        terr_mvn_vs30,
        terr_mvn_stdv,
        config.combination_method,
        k_value=config.k_value,
        epsilon=config.epsilon,
    )

    chunk_df["vs30"] = combined_vs30
    chunk_df["stdv"] = combined_stdv

    return chunk_id, chunk_df


def _process_pixels_chunk(args: tuple) -> tuple[int, list[SpatialAdjustmentResult]]:  # pragma: no cover
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
        (chunk_id, list of SpatialAdjustmentResult)
    """
    pixel_indices, chunk_id, pixel_data_dict, obs_data_dict, config_params = args

    # Reconstruct ObservationData from dict (dataclasses can't always be pickled cleanly)
    obs_data = ObservationData(
        locations=obs_data_dict["locations"],
        vs30=obs_data_dict["vs30"],
        model_vs30=obs_data_dict["model_vs30"],
        model_stdv=obs_data_dict["model_stdv"],
        residuals=obs_data_dict["residuals"],
        omega=obs_data_dict["omega"],
        uncertainty=obs_data_dict["uncertainty"],
    )

    updates = []
    for idx in pixel_indices:
        # Get pixel data from the prepared dict
        pixel_info = pixel_data_dict[idx]
        pixel = PixelData(
            location=pixel_info["location"],
            vs30=pixel_info["vs30"],
            stdv=pixel_info["stdv"],
            index=pixel_info["index"],
        )

        update = compute_spatial_adjustment_for_pixel(
            pixel,
            obs_data,
            config_params["model_type"],
            phi=config_params["phi"],
            max_dist_m=config_params["max_dist_m"],
            max_points=config_params["max_points"],
            noisy=config_params["noisy"],
            cov_reduc=config_params["cov_reduc"],
        )

        if update is not None:
            updates.append(update)

    return chunk_id, updates


# =============================================================================
# Orchestration Functions
# =============================================================================


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
    chunks = np.array_split(locations_df, n_proc)
    chunk_args = [
        (chunk.reset_index(drop=True), i, observations_df, geol_model_df, terr_model_df, config)
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
                    pool.imap(_process_locations_chunk, chunk_args),
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
    obs_data: ObservationData,
    model_type: str,
    phi: float,
    max_dist_m: float,
    max_points: int,
    noisy: bool,
    cov_reduc: float,
    n_proc: int,
) -> list[SpatialAdjustmentResult]:
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
    model_type : str
        Model type ("geology" or "terrain")
    phi : float
        Correlation length parameter
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
    list[SpatialAdjustmentResult]
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
                "location": grid_locs[valid_idx],
                "vs30": float(raster_data.vs30.flat[flat_idx]),
                "stdv": float(raster_data.stdv.flat[flat_idx]),
                "index": int(flat_idx),
            }

    # Convert ObservationData to dict for pickling
    obs_data_dict = {
        "locations": obs_data.locations,
        "vs30": obs_data.vs30,
        "model_vs30": obs_data.model_vs30,
        "model_stdv": obs_data.model_stdv,
        "residuals": obs_data.residuals,
        "omega": obs_data.omega,
        "uncertainty": obs_data.uncertainty,
    }

    # Config params
    config_params = {
        "model_type": model_type,
        "phi": phi,
        "max_dist_m": max_dist_m,
        "max_points": max_points,
        "noisy": noisy,
        "cov_reduc": cov_reduc,
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
                    pool.imap(_process_pixels_chunk, chunk_args),
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

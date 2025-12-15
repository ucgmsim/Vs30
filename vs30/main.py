"""
MVN updates for geology and terrain models.

This module implements multivariate normal (MVN) distribution updates for pixels
in geology.tif and terrain.tif that are within range of observations.
"""

import logging
import time
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import typer
from matplotlib import pyplot as plt
from scipy.spatial.distance import cdist, euclidean
from tqdm import tqdm
from typer import Option

from vs30 import categorical_raster
from vs30.categorical_values import find_posterior
from vs30.model import ID_NODATA
from vs30.model_registry import get_model_info
from vs30.mvn_data import (
    BoundingBoxResult,
    MVNUpdateResult,
    ObservationData,
    PixelData,
    RasterData,
)
from vs30.vs30_map_config_handler import Vs30MapConfig

logger = logging.getLogger(__name__)

# Create Typer app for CLI
app = typer.Typer(
    name="vs30map",
    help="VS30 map generation and categorical model updates",
    add_completion=False,
)


def grid_points_in_bbox(
    grid_locs,  # Mx2 array of grid point coordinates in NZTM
    obs_eastings_min,  # (n_obs, 1) precomputed obs_eastings - max_dist
    obs_eastings_max,  # (n_obs, 1) precomputed obs_eastings + max_dist
    obs_northings_min,  # (n_obs, 1) precomputed obs_northings - max_dist
    obs_northings_max,  # (n_obs, 1) precomputed obs_northings + max_dist
    start_grid_idx=0,  # Starting index of grid_locs in the full grid
):
    """
    Find grid points within bounding boxes of observations using fully vectorized NumPy.

    Uses broadcasting to compute all observation-grid pairs simultaneously.
    Returns collapsed mask and per-observation grid indices.

    Parameters
    ----------
    grid_locs : array_like, shape (M, 2)
        Grid point coordinates as (easting, northing) in NZTM.
    obs_eastings_min : ndarray, shape (N, 1)
        Precomputed obs_eastings - max_dist.
    obs_eastings_max : ndarray, shape (N, 1)
        Precomputed obs_eastings + max_dist.
    obs_northings_min : ndarray, shape (N, 1)
        Precomputed obs_northings - max_dist.
    obs_northings_max : ndarray, shape (N, 1)
        Precomputed obs_northings + max_dist.
    start_grid_idx : int, optional
        Starting index of grid_locs in the full grid (for offsetting indices).
        Default is 0.

    Returns
    -------
    chunk_mask : ndarray, shape (M,), dtype=bool
        Boolean array indicating which grid points in this chunk are affected
        by any observation (collapsed with np.any(axis=0)).
    obs_to_grid_indices : list of ndarray
        List of length n_obs. Each element is an array of grid point indices
        (in the full grid) that are within that observation's bounding box.
    """
    # Extract coordinates
    grid_eastings = grid_locs[:, 0]  # (n_grid,)
    grid_northings = grid_locs[:, 1]  # (n_grid,)

    # Vectorized bounding box check using broadcasting
    # (n_obs, 1) operation against (n_grid,) -> (n_obs, n_grid)
    # Checks: for each observation, which grid points are in its bounding box
    in_bbox = (
        (grid_eastings >= obs_eastings_min)
        & (grid_eastings <= obs_eastings_max)
        & (grid_northings >= obs_northings_min)
        & (grid_northings <= obs_northings_max)
    )

    # Collapse to single mask: which grid points are affected by any observation
    chunk_mask = np.any(in_bbox, axis=0)

    # For each observation, get the grid point indices within its bounding box
    # in_bbox shape: (n_obs, n_grid_chunk)
    n_obs = in_bbox.shape[0]
    obs_to_grid_indices = []

    for obs_idx in range(n_obs):
        # Get grid indices within this observation's bounding box
        grid_indices_in_chunk = np.where(in_bbox[obs_idx])[0]
        # Convert to full grid indices by adding start_grid_idx
        full_grid_indices = grid_indices_in_chunk + start_grid_idx
        obs_to_grid_indices.append(full_grid_indices)

    return chunk_mask, obs_to_grid_indices


def calculate_chunk_size(n_obs, total_memory_gb):
    """
    Calculate the maximum number of grid points per chunk based on total available memory.

    The memory-intensive operation is the boolean array of shape (n_obs, n_grid_chunk).
    NumPy boolean arrays use 1 byte per element (not 1 bit or 8 bytes).
    Memory usage per chunk: n_obs * n_grid_chunk * 1 byte (for boolean array)

    Parameters
    ----------
    n_obs : int
        Number of observations.
    total_memory_gb : float
        Total available memory in GB.

    Returns
    -------
    chunk_size : int
        Maximum number of grid points per chunk.
    """
    memory_per_chunk_bytes = total_memory_gb * 1024 * 1024 * 1024
    # Memory needed: n_obs * n_grid_chunk * 1 byte (NumPy boolean arrays are 1 byte/element)
    # Solve for n_grid_chunk: n_grid_chunk = memory_per_chunk_bytes / n_obs
    chunk_size = int(memory_per_chunk_bytes / n_obs)
    # Ensure at least 1 grid point per chunk
    return max(1, chunk_size)


# ============================================================================
# Helper Functions for Distance and Correlation
# ============================================================================


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two points.

    Parameters
    ----------
    p1 : ndarray
        First point coordinates (2,) array [easting, northing].
    p2 : ndarray
        Second point coordinates (2,) array [easting, northing].

    Returns
    -------
    float
        Euclidean distance in meters.
    """
    return euclidean(p1, p2)


def euclidean_distance_matrix(points: np.ndarray) -> np.ndarray:
    """
    Calculate full distance matrix between points.

    Parameters
    ----------
    points : ndarray
        Array of point coordinates (N, 2) where each row is [easting, northing].

    Returns
    -------
    ndarray
        Distance matrix (N, N) where element [i, j] is distance between points i and j.
    """
    return cdist(points, points, metric="euclidean")


def correlation_function(distances: np.ndarray, phi: float) -> np.ndarray:
    """
    Calculate correlation function from distances.

    Parameters
    ----------
    distances : ndarray
        Array of distances in meters. Can be scalar, 1D, or 2D (distance matrix).
    phi : float
        Correlation length parameter in meters.

    Returns
    -------
    correlations : ndarray
        Correlation values between 0 and 1. Same shape as distances.

    Notes
    -----
    Uses exponential correlation function: 1 / exp(max(0.1, distance) / phi)
    Minimum distance of 0.1 meters is enforced to prevent division issues.
    """
    return 1 / np.exp(np.maximum(0.1, distances) / phi)


# ============================================================================
# Observation Data Preparation
# ============================================================================


def interpolate_raster_at_points(
    raster_data: RasterData, points: np.ndarray, band: int = 1
) -> np.ndarray:
    """
    Interpolate raster values at point locations using nearest neighbor.

    Parameters
    ----------
    raster_data : RasterData
        Raster data object.
    points : ndarray
        Point coordinates (N, 2) array of [easting, northing].
    band : int, optional
        Band number to read (1 for vs30, 2 for stdv). Default is 1.

    Returns
    -------
    ndarray
        Interpolated values (N,) array.
    """
    # Get raster band
    if band == 1:
        raster_array = raster_data.vs30
    elif band == 2:
        raster_array = raster_data.stdv
    else:
        raise ValueError(f"Invalid band: {band}")

    # Convert coordinates to row/col indices
    transform = raster_data.transform
    rows = np.floor((points[:, 1] - transform[5]) / transform[4]).astype(np.int32)
    cols = np.floor((points[:, 0] - transform[0]) / transform[1]).astype(np.int32)

    # Check bounds
    valid = (
        (rows >= 0)
        & (rows < raster_array.shape[0])
        & (cols >= 0)
        & (cols < raster_array.shape[1])
    )

    # Initialize output with NaN
    values = np.full(len(points), np.nan, dtype=raster_array.dtype)

    # Extract values for valid points
    values[valid] = raster_array[rows[valid], cols[valid]]

    # Handle NoData
    if raster_data.nodata is not None:
        values[values == raster_data.nodata] = np.nan

    return values


def prepare_observation_data(
    observations: pd.DataFrame,
    raster_data: RasterData,
    updated_model_table: np.ndarray,
    model_name: str,
    mvn_config: Vs30MapConfig,
) -> ObservationData:
    """
    Prepare observation data for MVN processing.

    Parameters
    ----------
    observations : DataFrame
        Observations with vs30, uncertainty, easting, northing, and model ID.
    raster_data : RasterData
        Raster data object.
    updated_model_table : ndarray
        Updated model table (n_categories, 2) array of [vs30, stdv].
    model_name : str
        Model name ("geology" or "terrain").
    mvn_config : Vs30MapConfig
        VS30 map configuration.

    Returns
    -------
    ObservationData
        Prepared observation data object.
    """
    # Get observation locations
    obs_locs = observations[["easting", "northing"]].values

    # Interpolate model values at observation locations
    # Get model IDs using registry
    model_info = get_model_info(model_name)
    model_id_func = model_info.get_model_id_func()
    model_ids = model_id_func(obs_locs)

    # Get model vs30 and stdv from updated model table
    # Model IDs are 1-indexed in the raster, but 0-indexed in the model table
    # ID_NODATA is 255, valid IDs are 1-15 for geology, 1-16 for terrain
    valid_mask = (
        (model_ids != ID_NODATA)
        & (model_ids > 0)
        & (model_ids <= len(updated_model_table))
    )
    model_vs30 = np.full(len(observations), np.nan)
    model_stdv = np.full(len(observations), np.nan)

    # Convert 1-indexed model IDs to 0-indexed array indices
    valid_model_ids = model_ids[valid_mask] - 1
    model_vs30[valid_mask] = updated_model_table[valid_model_ids, 0]
    model_stdv[valid_mask] = updated_model_table[valid_model_ids, 1]

    # Filter out observations where model values are NaN/NoData
    valid_obs_mask = ~np.isnan(model_vs30) & ~np.isnan(model_stdv)
    obs_locs = obs_locs[valid_obs_mask]
    vs30_obs = observations.vs30.values[valid_obs_mask]
    model_vs30 = model_vs30[valid_obs_mask]
    model_stdv = model_stdv[valid_obs_mask]
    uncertainty = observations.uncertainty.values[valid_obs_mask]

    # Calculate log residuals
    residuals = np.log(vs30_obs / model_vs30)

    # Apply noise weighting (if noisy=True)
    if mvn_config.noisy:
        omega = np.sqrt(model_stdv**2 / (model_stdv**2 + uncertainty**2))
        residuals *= omega
    else:
        omega = np.ones(len(residuals))

    return ObservationData(
        locations=obs_locs,
        vs30=vs30_obs,
        model_vs30=model_vs30,
        model_stdv=model_stdv,
        residuals=residuals,
        omega=omega,
        uncertainty=uncertainty,
    )


# ============================================================================
# Covariance Matrix Building
# ============================================================================


def build_covariance_matrix(
    pixel: PixelData,
    selected_observations: ObservationData,
    model_name: str,
    mvn_config: Vs30MapConfig,
) -> np.ndarray:
    """
    Build covariance matrix through clear pipeline of steps.

    Parameters
    ----------
    pixel : PixelData
        Pixel data for the pixel being updated.
    selected_observations : ObservationData
        Selected observations for this pixel.
    model_name : str
        Model name ("geology" or "terrain").
    mvn_config : Vs30MapConfig
        VS30 map configuration.

    Returns
    -------
    ndarray
        Covariance matrix (n_selected_obs + 1, n_selected_obs + 1).
        First row/column is for the pixel, rest are for observations.
    """
    # Step 1: Compute distances
    all_points = np.vstack([pixel.location, selected_observations.locations])
    distance_matrix = euclidean_distance_matrix(all_points)

    # Step 2: Apply correlation function
    phi = mvn_config.get_phi(model_name)
    corr = correlation_function(distance_matrix, phi)

    # Step 3: Scale by standard deviations
    stdvs = np.insert(selected_observations.model_stdv, 0, pixel.stdv)
    cov = corr * np.outer(stdvs, stdvs)

    # Step 4: Apply noise weighting (if enabled)
    if mvn_config.noisy:
        omega = np.insert(selected_observations.omega, 0, 1.0)
        omega_matrix = np.outer(omega, omega)
        np.fill_diagonal(omega_matrix, 1.0)
        cov *= omega_matrix

    # Step 5: Apply covariance reduction (if enabled)
    if mvn_config.cov_reduc > 0:
        log_vs30s = np.insert(
            np.log(selected_observations.model_vs30), 0, np.log(pixel.vs30)
        )
        log_dist_matrix = euclidean_distance_matrix(log_vs30s.reshape(-1, 1))
        cov *= np.exp(-mvn_config.cov_reduc * log_dist_matrix)

    return cov


# ============================================================================
# MVN Update Computation
# ============================================================================


def select_observations_for_pixel(
    pixel: PixelData,
    obs_data: ObservationData,
    obs_to_grid_indices: list[np.ndarray],
    chunk_grid_to_obs: dict[int, list[int]],
    mvn_config: Vs30MapConfig,
) -> ObservationData:
    """
    Select observations for a pixel using chunk cache and distance filtering.

    Parameters
    ----------
    pixel : PixelData
        Pixel data.
    obs_data : ObservationData
        Full observation data.
    obs_to_grid_indices : list
        List of grid indices per observation.
    chunk_grid_to_obs : dict
        Chunk cache mapping pixel indices to observation indices.
    mvn_config : Vs30MapConfig
        VS30 map configuration.

    Returns
    -------
    ObservationData
        Selected observations (subset of obs_data).
    """
    # Get candidate observations from chunk cache
    candidate_obs_indices = chunk_grid_to_obs.get(pixel.index, [])

    if len(candidate_obs_indices) == 0:
        # Return empty ObservationData
        return ObservationData(
            locations=np.empty((0, 2)),
            vs30=np.empty(0),
            model_vs30=np.empty(0),
            model_stdv=np.empty(0),
            residuals=np.empty(0),
            omega=np.empty(0),
            uncertainty=np.empty(0),
        )

    # Calculate actual Euclidean distances
    candidate_locs = obs_data.locations[candidate_obs_indices]
    distances = cdist(
        pixel.location.reshape(1, -1), candidate_locs, metric="euclidean"
    )[0]

    # Filter to only observations within max_dist_m
    within_dist_mask = distances <= mvn_config.max_dist_m
    filtered_indices = np.array(candidate_obs_indices)[within_dist_mask]
    filtered_distances = distances[within_dist_mask]

    if len(filtered_indices) == 0:
        # Return empty ObservationData
        return ObservationData(
            locations=np.empty((0, 2)),
            vs30=np.empty(0),
            model_vs30=np.empty(0),
            model_stdv=np.empty(0),
            residuals=np.empty(0),
            omega=np.empty(0),
            uncertainty=np.empty(0),
        )

    # Limit to closest max_points
    if len(filtered_indices) > mvn_config.max_points:
        # Sort by distance and take closest max_points
        sorted_idx = np.argsort(filtered_distances)
        selected_idx = sorted_idx[: mvn_config.max_points]
        filtered_indices = filtered_indices[selected_idx]
        filtered_distances = filtered_distances[selected_idx]

    # Create subset of ObservationData
    return ObservationData(
        locations=obs_data.locations[filtered_indices],
        vs30=obs_data.vs30[filtered_indices],
        model_vs30=obs_data.model_vs30[filtered_indices],
        model_stdv=obs_data.model_stdv[filtered_indices],
        residuals=obs_data.residuals[filtered_indices],
        omega=obs_data.omega[filtered_indices],
        uncertainty=obs_data.uncertainty[filtered_indices],
    )


def compute_mvn_update_for_pixel(
    pixel: PixelData,
    obs_data: ObservationData,
    obs_to_grid_indices: list[np.ndarray],
    chunk_grid_to_obs: dict[int, list[int]],
    model_name: str,
    mvn_config: Vs30MapConfig,
) -> MVNUpdateResult | None:
    """
    Compute MVN update for a single pixel.

    Parameters
    ----------
    pixel : PixelData
        Pixel data.
    obs_data : ObservationData
        Full observation data.
    obs_to_grid_indices : list
        List of grid indices per observation.
    chunk_grid_to_obs : dict
        Chunk cache mapping pixel indices to observation indices.
    model_name : str
        Model name ("geology" or "terrain").
    mvn_config : Vs30MapConfig
        VS30 map configuration.

    Returns
    -------
    MVNUpdateResult or None
        Update result, or None if pixel should be skipped.
    """
    # Handle NaN/NoData pixels
    if np.isnan(pixel.vs30) or np.isnan(pixel.stdv):
        return None

    # Select observations for this pixel
    selected_obs = select_observations_for_pixel(
        pixel, obs_data, obs_to_grid_indices, chunk_grid_to_obs, mvn_config
    )

    if len(selected_obs.locations) == 0:
        # No observations nearby, return unchanged values
        return MVNUpdateResult(
            updated_vs30=pixel.vs30,
            updated_stdv=pixel.stdv,
            n_observations_used=0,
            min_distance=np.inf,
            pixel_index=pixel.index,
        )

    # Build covariance matrix
    cov_matrix = build_covariance_matrix(pixel, selected_obs, model_name, mvn_config)

    # Invert covariance matrix (observations only)
    # Note: This uses BLAS/LAPACK and can cause CPU spikes when many observations
    # are present (up to max_points=500, resulting in 500x500 matrix inversion)
    inv_cov = np.linalg.inv(cov_matrix[1:, 1:])

    # Calculate prediction update
    pred_update = np.dot(np.dot(cov_matrix[0, 1:], inv_cov), selected_obs.residuals)

    # Calculate variance
    var = cov_matrix[0, 0] - np.dot(
        np.dot(cov_matrix[0, 1:], inv_cov), cov_matrix[1:, 0]
    )

    # Update vs30 and stdv
    new_vs30 = pixel.vs30 * np.exp(pred_update)
    new_stdv = sqrt(var)

    # Calculate minimum distance
    distances = cdist(
        pixel.location.reshape(1, -1),
        selected_obs.locations,
        metric="euclidean",
    )[0]
    min_distance = np.min(distances)

    return MVNUpdateResult(
        updated_vs30=new_vs30,
        updated_stdv=new_stdv,
        n_observations_used=len(selected_obs.locations),
        min_distance=min_distance,
        pixel_index=pixel.index,
    )


# ============================================================================
# Find Affected Pixels
# ============================================================================


def find_affected_pixels(
    raster_data: RasterData, obs_data: ObservationData, mvn_config: Vs30MapConfig
) -> BoundingBoxResult:
    """
    Find pixels affected by observations using bounding boxes.

    Parameters
    ----------
    raster_data : RasterData
        Raster data object.
    obs_data : ObservationData
        Observation data.
    mvn_config : Vs30MapConfig
        VS30 map configuration.

    Returns
    -------
    BoundingBoxResult
        Result containing mask and observation-to-grid mappings.
    """
    # Get coordinates for valid pixels
    grid_locs = raster_data.get_coordinates()

    # Calculate chunk size
    chunk_size = calculate_chunk_size(
        len(obs_data.locations), mvn_config.total_memory_gb
    )
    n_chunks = int(np.ceil(len(grid_locs) / chunk_size))

    # Precompute observation bounds
    obs_eastings = obs_data.locations[:, 0:1]  # (n_obs, 1)
    obs_northings = obs_data.locations[:, 1:2]  # (n_obs, 1)
    obs_eastings_min = obs_eastings - mvn_config.max_dist_m
    obs_eastings_max = obs_eastings + mvn_config.max_dist_m
    obs_northings_min = obs_northings - mvn_config.max_dist_m
    obs_northings_max = obs_northings + mvn_config.max_dist_m

    # Initialize mask and obs_to_grid_indices
    valid_points_in_bbox_mask = np.zeros(len(grid_locs), dtype=bool)
    obs_to_grid_indices = [
        np.array([], dtype=np.int64) for _ in range(len(obs_data.locations))
    ]

    # Process each chunk
    logger.info(f"Processing {n_chunks} chunks of {chunk_size:,} pixels each")
    for chunk_idx in tqdm(
        range(n_chunks),
        desc="Finding affected pixels",
        unit="chunk",
    ):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(grid_locs))
        grid_locs_chunk = grid_locs[start_idx:end_idx]

        chunk_mask, chunk_obs_to_grid = grid_points_in_bbox(
            grid_locs=grid_locs_chunk,
            obs_eastings_min=obs_eastings_min,
            obs_eastings_max=obs_eastings_max,
            obs_northings_min=obs_northings_min,
            obs_northings_max=obs_northings_max,
            start_grid_idx=start_idx,
        )

        valid_points_in_bbox_mask[start_idx:end_idx] = chunk_mask

        # Accumulate grid indices for each observation
        for obs_idx, valid_indices in enumerate(chunk_obs_to_grid):
            if len(valid_indices) > 0:
                full_raster_indices = raster_data.valid_flat_indices[valid_indices]
                obs_to_grid_indices[obs_idx] = np.concatenate(
                    [obs_to_grid_indices[obs_idx], full_raster_indices]
                )

    # Create full-size mask
    grid_points_in_bbox_mask = np.zeros(raster_data.vs30.size, dtype=bool)
    grid_points_in_bbox_mask[raster_data.valid_flat_indices] = valid_points_in_bbox_mask

    n_affected = np.sum(valid_points_in_bbox_mask)
    logger.info(
        f"Bounding box search complete: {n_affected:,} pixels affected "
        f"({n_affected / len(grid_locs) * 100:.1f}% of valid pixels)"
    )

    return BoundingBoxResult(
        mask=grid_points_in_bbox_mask,
        obs_to_grid_indices=obs_to_grid_indices,
        n_affected_pixels=n_affected,
    )


# ============================================================================
# Main MVN Updates Computation
# ============================================================================


def compute_mvn_updates(
    raster_data: RasterData,
    obs_data: ObservationData,
    bbox_result: BoundingBoxResult,
    model_name: str,
    mvn_config: Vs30MapConfig,
) -> list[MVNUpdateResult]:
    """
    Compute MVN updates for all affected pixels.

    Parameters
    ----------
    raster_data : RasterData
        Raster data object.
    obs_data : ObservationData
        Observation data.
    bbox_result : BoundingBoxResult
        Bounding box result.
    model_name : str
        Model name ("geology" or "terrain").
    mvn_config : Vs30MapConfig
        VS30 map configuration.

    Returns
    -------
    list
        List of MVNUpdateResult objects.
    """
    # Get affected pixel indices
    affected_flat_indices = np.where(bbox_result.mask)[0]
    affected_valid_indices = np.where(bbox_result.mask[raster_data.valid_flat_indices])[
        0
    ]

    # Get coordinates for affected pixels
    grid_locs = raster_data.get_coordinates()
    affected_locs = grid_locs[affected_valid_indices]

    # Get model values for affected pixels
    affected_vs30 = raster_data.vs30.flat[affected_flat_indices]
    affected_stdv = raster_data.stdv.flat[affected_flat_indices]

    # Process in chunks for memory efficiency
    chunk_size = calculate_chunk_size(
        len(obs_data.locations), mvn_config.total_memory_gb
    )
    n_chunks = int(np.ceil(len(affected_flat_indices) / chunk_size))

    all_updates = []

    logger.info(
        f"Processing {n_chunks} chunks of up to {chunk_size:,} pixels each "
        f"({len(affected_flat_indices):,} total pixels to update)"
    )

    # Create a single progress bar that tracks pixels across all chunks
    total_pixels = len(affected_flat_indices)
    pixel_pbar = tqdm(
        total=total_pixels,
        desc="Computing MVN updates",
        unit="pixel",
        mininterval=0.1,  # Update at least every 0.1 seconds for responsiveness
        maxinterval=1.0,  # But don't update more than once per second
    )

    pixels_processed = 0

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(affected_flat_indices))

        chunk_flat_indices = affected_flat_indices[start_idx:end_idx]
        chunk_valid_indices = affected_valid_indices[start_idx:end_idx]

        # Build chunk cache: reverse mapping from grid pixels to observations
        # Use set for faster lookup
        pixel_pbar.set_postfix(
            {"status": "building cache", "chunk": f"{chunk_idx + 1}/{n_chunks}"}
        )
        chunk_flat_indices_set = set(chunk_flat_indices)
        chunk_grid_to_obs: dict[int, list[int]] = {}
        for obs_idx, grid_indices in enumerate(bbox_result.obs_to_grid_indices):
            for grid_idx in grid_indices:
                if grid_idx in chunk_flat_indices_set:
                    if grid_idx not in chunk_grid_to_obs:
                        chunk_grid_to_obs[grid_idx] = []
                    chunk_grid_to_obs[grid_idx].append(obs_idx)

        # Process each pixel in chunk
        chunk_affected_locs = affected_locs[start_idx:end_idx]
        chunk_affected_vs30 = affected_vs30[start_idx:end_idx]
        chunk_affected_stdv = affected_stdv[start_idx:end_idx]

        pixel_pbar.set_postfix(
            {
                "status": "processing",
                "chunk": f"{chunk_idx + 1}/{n_chunks}",
                "updated": len(all_updates),
            }
        )

        for i, (flat_idx, valid_idx) in enumerate(
            zip(chunk_flat_indices, chunk_valid_indices)
        ):
            pixel = PixelData(
                location=chunk_affected_locs[i],
                vs30=chunk_affected_vs30[i],
                stdv=chunk_affected_stdv[i],
                index=flat_idx,
            )

            update_result = compute_mvn_update_for_pixel(
                pixel,
                obs_data,
                bbox_result.obs_to_grid_indices,
                chunk_grid_to_obs,
                model_name,
                mvn_config,
            )

            if update_result is not None:
                all_updates.append(update_result)

            pixels_processed += 1
            pixel_pbar.update(1)

            # Update progress bar with stats every 1000 pixels
            if pixels_processed % 1000 == 0:
                rate = pixel_pbar.format_dict.get("rate", 0)
                pixel_pbar.set_postfix(
                    {
                        "status": "processing",
                        "chunk": f"{chunk_idx + 1}/{n_chunks}",
                        "updated": f"{len(all_updates):,}",
                        "rate": f"{rate:.0f} pix/s" if rate else "calculating...",
                    }
                )

    pixel_pbar.close()

    logger.info(f"Completed processing all chunks: {len(all_updates):,} pixels updated")

    return all_updates


# ============================================================================
# Apply Updates and Write Output
# ============================================================================


def apply_and_write_updates(
    raster_data: RasterData,
    updates: list[MVNUpdateResult],
    model_name: str,
    mvn_config: Vs30MapConfig,
    base_path: Path,
) -> None:
    """
    Apply updates to raster and write output file.

    Parameters
    ----------
    raster_data : RasterData
        Raster data object.
    updates : list
        List of MVNUpdateResult objects.
    model_name : str
        Model name ("geology" or "terrain").
    mvn_config : Vs30MapConfig
        VS30 map configuration.
    base_path : Path
        Base path for output files.
    """
    # Initialize output arrays with original values
    updated_vs30 = raster_data.vs30.copy()
    updated_stdv = raster_data.stdv.copy()

    # Apply updates
    for update in updates:
        updated_vs30.flat[update.pixel_index] = update.updated_vs30
        updated_stdv.flat[update.pixel_index] = update.updated_stdv

    # Create output directory if it doesn't exist
    output_dir = base_path / mvn_config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write output using filename from config
    output_filename = mvn_config.get_output_filename(model_name)
    output_path = output_dir / output_filename

    raster_data.write_updated(output_path, updated_vs30, updated_stdv)

    logger.info(
        f"Wrote updated raster to {output_path} "
        f"({len(updates):,} pixels updated out of {np.sum(raster_data.valid_mask):,} valid)"
    )


# ============================================================================
# Validation Functions
# ============================================================================


def validate_raster_data(raster_data: RasterData) -> None:
    """
    Validate raster data before processing.

    Parameters
    ----------
    raster_data : RasterData
        Raster data object.

    Raises
    ------
    AssertionError
        If raster data is invalid.
    """
    assert raster_data.vs30.shape == raster_data.stdv.shape, "Band shapes must match"
    assert np.all(np.isfinite(raster_data.vs30[raster_data.valid_mask])), (
        "Valid pixels must be finite"
    )
    assert np.all(raster_data.vs30[raster_data.valid_mask] > 0), "Vs30 must be positive"
    assert np.all(raster_data.stdv[raster_data.valid_mask] > 0), "Stdv must be positive"


def validate_observations(observations: pd.DataFrame) -> None:
    """
    Validate observation data.

    Parameters
    ----------
    observations : DataFrame
        Observation data.

    Raises
    ------
    AssertionError
        If observation data is invalid.
    """
    assert "easting" in observations.columns, "Missing 'easting' column"
    assert "northing" in observations.columns, "Missing 'northing' column"
    assert "vs30" in observations.columns, "Missing 'vs30' column"
    assert "uncertainty" in observations.columns, "Missing 'uncertainty' column"
    assert np.all(observations["vs30"] > 0), "Vs30 must be positive"
    assert np.all(observations["uncertainty"] > 0), "Uncertainty must be positive"


def process_model_updates(config: Vs30MapConfig, base_path: Path) -> None:
    """
    Main pipeline for processing MVN updates.

    Stages:
    1. Load and validate inputs
    2. Bayesian update of categorical priors (optional)
    3. Prepare observation data
    4. Find affected pixels
    5. Compute MVN updates
    6. Apply updates to raster
    7. Write output

    Parameters
    ----------
    config : Vs30MapConfig
        VS30 map configuration.
    base_path : Path
        Base path for input/output files.
    """
    for model_name in config.models_to_process:
        logger.info(f"Processing {model_name} model...")

        # Stage 1: Load raster data
        raster_path = config.get_model_raster_path(model_name, base_path)
        raster_data = RasterData.from_file(raster_path)
        validate_raster_data(raster_data)

        # Stage 2: Load observations
        logger.info("Loading observations...")
        observations = load_observations(base_path, config.observations_file)
        validate_observations(observations)
        logger.info(f"Loaded {len(observations)} observations")

        # Stage 3: Load model values and optionally apply Bayesian update
        if config.compute_bayesian_update:
            logger.info(
                "Loading model values and computing Bayesian update from observations..."
            )
        else:
            logger.info("Loading model values from CSV (bypassing Bayesian update)...")
        start_time_stage = time.time()
        updated_model_table = update_categorical_priors(
            model_name, observations, config
        )
        logger.info(
            f"Loaded/updated model values in {time.time() - start_time_stage:.2f}s"
        )

        # Stage 4: Prepare observation data for MVN
        logger.info("Preparing observation data for MVN processing...")
        start_time_stage = time.time()
        obs_data = prepare_observation_data(
            observations, raster_data, updated_model_table, model_name, config
        )
        logger.info(
            f"Prepared {len(obs_data.locations)} valid observations "
            f"(filtered from {len(observations)}) in {time.time() - start_time_stage:.2f}s"
        )

        # Stage 5: Find affected pixels
        logger.info("Finding pixels affected by observations...")
        start_time_stage = time.time()
        bbox_result = find_affected_pixels(raster_data, obs_data, config)
        logger.info(
            f"Found {bbox_result.n_affected_pixels:,} affected pixels "
            f"(out of {np.sum(raster_data.valid_mask):,} valid pixels) "
            f"in {time.time() - start_time_stage:.2f}s"
        )

        # Stage 6: Compute MVN updates
        logger.info("Computing MVN updates for affected pixels...")
        start_time_stage = time.time()
        updates = compute_mvn_updates(
            raster_data, obs_data, bbox_result, model_name, config
        )
        elapsed_stage = time.time() - start_time_stage
        if len(updates) > 0:
            logger.info(
                f"Computed {len(updates):,} MVN updates in {elapsed_stage:.2f}s "
                f"({elapsed_stage / len(updates) * 1000:.2f}ms per pixel)"
            )
        else:
            logger.info(
                f"Computed {len(updates):,} MVN updates in {elapsed_stage:.2f}s"
            )

        # Log statistics about updates
        if len(updates) > 0:
            n_obs_used = [u.n_observations_used for u in updates]
            min_dists = [u.min_distance for u in updates if u.min_distance != np.inf]
            stats_parts = [
                f"avg observations used: {np.mean(n_obs_used):.1f} "
                f"(min: {np.min(n_obs_used)}, max: {np.max(n_obs_used)})"
            ]
            if len(min_dists) > 0:
                stats_parts.append(
                    f"avg min distance: {np.mean(min_dists):.1f}m "
                    f"(min: {np.min(min_dists):.1f}m, max: {np.max(min_dists):.1f}m)"
                )
            logger.info(f"Update statistics: {', '.join(stats_parts)}")

        # Stage 7: Apply updates and write output
        logger.info("Applying updates and writing output raster...")
        start_time_stage = time.time()
        apply_and_write_updates(raster_data, updates, model_name, config, base_path)
        logger.info(f"Wrote output raster in {time.time() - start_time_stage:.2f}s")

        logger.info(f"Completed processing {model_name} model")


# ============================================================================
# CLI Functions
# ============================================================================


def _resolve_base_path(config_path: Path) -> Path:
    """
    Resolve base path from config file location.

    The base path is the parent directory of the vs30 package directory.
    For example, if config is at vs30/config.yaml, base_path is the workspace root.

    Parameters
    ----------
    config_path : Path
        Path to config.yaml file.

    Returns
    -------
    Path
        Base path for input/output files.
    """
    if config_path.name == "config.yaml" and config_path.parent.name == "vs30":
        return config_path.parent.parent
    else:
        return config_path.parent


def _find_config_file(config: Path | None) -> Path:
    """
    Find config.yaml file if not specified.

    Parameters
    ----------
    config : Path | None
        Config path if provided, None otherwise.

    Returns
    -------
    Path
        Path to config.yaml file.

    Raises
    ------
    FileNotFoundError
        If config file cannot be found.
    """
    if config is not None:
        return config

    cwd = Path.cwd()
    # Try common locations
    candidates = [
        cwd / "vs30" / "config.yaml",
        cwd / "config.yaml",
        cwd.parent / "vs30" / "config.yaml",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find config.yaml. Tried:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + "\nPlease specify --config PATH"
    )


@app.command()
def update_categorical_vs30_models(
    categorical_model_csv: Path = Option(
        ...,
        "--categorical-model-csv",
        "-m",
        help="Path to CSV file with categorical vs30 mean and standard deviation values (e.g., geology_model_prior_mean_and_standard_deviation.csv)",
    ),
    observations_csv: Path = Option(
        ...,
        "--observations-csv",
        "-o",
        help="Path to CSV file with observations (e.g., measured_vs30_original_filtered.csv)",
    ),
    output_dir: Path = Option(
        ...,
        "--output-dir",
        "-d",
        help="Path to output directory (will be created if it does not exist)",
    ),
    model_type: str = Option(
        ...,
        "--model-type",
        "-t",
        help="Model type: either 'geology' or 'terrain'",
    ),
    n_prior: int = Option(
        3,
        "--n-prior",
        help="Effective number of prior observations (default: 3)",
    ),
    min_sigma: float = Option(
        0.5,
        "--min-sigma",
        help="Minimum standard deviation allowed (default: 0.5)",
    ),
) -> None:
    """
    Update categorical model values using Bayesian updates and save to CSV files.

    This command loads observations and categorical model values, applies Bayesian
    updates to the categorical model values (mean and standard deviation per category),
    and writes the updated values back to CSV files.
    """
    try:
        # Validate input files exist
        if not categorical_model_csv.exists():
            typer.echo(
                f"Error: Categorical model CSV file not found: {categorical_model_csv}",
                err=True,
            )
            raise typer.Exit(1)

        if not observations_csv.exists():
            typer.echo(
                f"Error: Observations CSV file not found: {observations_csv}", err=True
            )
            raise typer.Exit(1)

        # Validate model_type parameter
        if model_type not in ["geology", "terrain"]:
            typer.echo(
                f"Error: model_type must be 'geology' or 'terrain', got '{model_type}'",
                err=True,
            )
            raise typer.Exit(1)

        logger.info(f"Model type: {model_type}")
        logger.info(f"Loading categorical model from: {categorical_model_csv}")
        logger.info(f"Loading observations from: {observations_csv}")

        # Load categorical model CSV (absolute path)
        categorical_model_df = pd.read_csv(categorical_model_csv, skipinitialspace=True)

        # Drop rows with invalid placeholder values (e.g., -9999 for water category)
        categorical_model_df = categorical_model_df[
            categorical_model_df["mean_vs30_km_per_s"] != -9999
        ]

        # Validate required columns
        required_cols = ["mean_vs30_km_per_s", "standard_deviation_vs30_km_per_s"]
        missing_cols = [
            col for col in required_cols if col not in categorical_model_df.columns
        ]
        if missing_cols:
            typer.echo(
                f"Error: CSV file missing required columns: {missing_cols}",
                err=True,
            )
            raise typer.Exit(1)

        # Load observations CSV
        observations_df = pd.read_csv(observations_csv, skipinitialspace=True)

        # Validate observations have required columns
        obs_required_cols = ["easting", "northing", "vs30", "uncertainty"]
        missing_obs_cols = [
            col for col in obs_required_cols if col not in observations_df.columns
        ]
        if missing_obs_cols:
            typer.echo(
                f"Error: Observations CSV missing required columns: {missing_obs_cols}",
                err=True,
            )
            raise typer.Exit(1)

        logger.info(f"Loaded {len(observations_df)} observations")

        # Apply Bayesian update
        logger.info("Applying Bayesian update...")

        # Import assignment functions and constants from categorical_values
        from vs30.categorical_values import (
            STANDARD_ID_COLUMN,
            _assign_to_category_geology,
            _assign_to_category_terrain,
        )

        # Assign model categorical model IDs to observations based on their location
        # taking into account whether we are considering geology or terrain.
        obs_locs = observations_df[["easting", "northing"]].values
        if model_type == "geology":
            model_ids = _assign_to_category_geology(obs_locs)
        elif model_type == "terrain":
            model_ids = _assign_to_category_terrain(obs_locs)
        else:
            typer.echo(
                f"Error: model_type must be 'geology' or 'terrain', got '{model_type}'",
                err=True,
            )
            raise typer.Exit(1)

        # Add to DataFrame using STANDARD column name (same for all models!)
        observations_df[STANDARD_ID_COLUMN] = model_ids

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        updated_categorical_model_df = find_posterior(
            categorical_model_df, observations_df, n_prior, min_sigma
        )

        output_filename = "posterior_" + categorical_model_csv.name
        output_path = output_dir / output_filename

        updated_categorical_model_df.to_csv(output_path, index=False)

        typer.echo("✓ Successfully updated categorical model values")
        typer.echo(f"  Output saved to: {output_path}")

    except Exception as e:
        logger.exception("Error updating category values")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command(name="make-initial-vs30-raster")
def make_initial_vs30_raster(
    terrain: bool = Option(False, "--terrain", help="Create terrain VS30 raster"),
    geology: bool = Option(False, "--geology", help="Create geology VS30 raster"),
    config: Path = Option(
        None,
        "--config",
        "-c",
        help="Path to config.yaml file (default: vs30/config.yaml relative to workspace root)",
    ),
    output_dir: Path = Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory (default: output_dir from config.yaml)",
    ),
) -> None:
    """
    Create initial VS30 mean and standard deviation rasters from category IDs.

    This command generates initial VS30 rasters by:
    1. Creating category ID rasters (from terrain raster or geology shapefile)
    2. Mapping category IDs to VS30 mean and standard deviation values from CSV files
    3. Writing 2-band GeoTIFFs with VS30 mean (band 1) and standard deviation (band 2)

    Output files are saved as terrain_initial_vs30.tif and/or geology_initial_vs30.tif.
    """
    try:
        # Validate that at least one model type is specified
        if not terrain and not geology:
            typer.echo(
                "Error: At least one of --terrain or --geology must be specified",
                err=True,
            )
            raise typer.Exit(1)

        # Load config
        config_path = _find_config_file(config)
        if not config_path.exists():
            typer.echo(f"Error: Config file not found: {config_path}", err=True)
            raise typer.Exit(1)

        logger.info(f"Loading configuration from {config_path}")
        # Load config, filtering out unknown keys
        import yaml

        with open(config_path) as f:
            config_data = yaml.safe_load(f)
        # Get only the fields that Vs30MapConfig expects
        config_fields = {
            field.name for field in Vs30MapConfig.__dataclass_fields__.values()
        }
        filtered_config = {k: v for k, v in config_data.items() if k in config_fields}
        mvn_config = Vs30MapConfig(**filtered_config)

        base_path = _resolve_base_path(config_path)
        logger.info(f"Using base path: {base_path}")

        # Determine output directory
        if output_dir is None:
            output_dir = base_path / mvn_config.output_dir
        else:
            output_dir = Path(output_dir)

        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Process terrain if requested
        if terrain:
            logger.info("Processing terrain model...")
            csv_path = mvn_config.terrain_mean_and_standard_deviation_per_category_file
            logger.info(f"Using terrain model values from {csv_path}")

            logger.info("Creating terrain category ID raster...")
            id_raster = categorical_raster.create_category_id_raster(
                "terrain", output_dir
            )

            logger.info("Creating terrain VS30 raster...")
            vs30_raster = output_dir / "terrain_initial_vs30.tif"
            categorical_raster.create_vs30_raster_from_ids(
                id_raster, csv_path, vs30_raster
            )
            typer.echo(f"✓ Created terrain VS30 raster: {vs30_raster}")

        # Process geology if requested
        if geology:
            logger.info("Processing geology model...")
            csv_path = mvn_config.geology_mean_and_standard_deviation_per_category_file
            logger.info(f"Using geology model values from {csv_path}")

            logger.info("Creating geology category ID raster...")
            id_raster = categorical_raster.create_category_id_raster(
                "geology", output_dir
            )

            logger.info("Creating geology VS30 raster...")
            vs30_raster = output_dir / "geology_initial_vs30.tif"
            categorical_raster.create_vs30_raster_from_ids(
                id_raster, csv_path, vs30_raster
            )
            typer.echo(f"✓ Created geology VS30 raster: {vs30_raster}")

        typer.echo("✓ Successfully created initial VS30 rasters")

    except Exception as e:
        logger.exception("Error creating initial VS30 rasters")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def make_map(
    config: Path = Option(
        None,
        "--config",
        "-c",
        help="Path to config.yaml file (default: vs30/config.yaml relative to workspace root)",
    ),
) -> None:
    """
    Generate VS30 map by applying MVN updates to 2D rasters.

    This command runs the full MVN update pipeline:
    1. Loads raster data (geology.tif and/or terrain.tif)
    2. Optionally applies Bayesian updates to categorical priors
    3. Prepares observation data
    4. Finds affected pixels
    5. Computes MVN updates
    6. Applies updates to raster and writes output

    The output rasters are written to the directory specified in config.yaml
    (default: new_vs30map/).
    """
    try:
        config_path = _find_config_file(config)
        if not config_path.exists():
            typer.echo(f"Error: Config file not found: {config_path}", err=True)
            raise typer.Exit(1)

        logger.info(f"Loading configuration from {config_path}")
        mvn_config = Vs30MapConfig.from_yaml(config_path)

        base_path = _resolve_base_path(config_path)
        logger.info(f"Using base path: {base_path}")

        process_model_updates(mvn_config, base_path)

        typer.echo("✓ Successfully generated VS30 map")

    except Exception as e:
        logger.exception("Error generating map")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def plot_posterior_values(
    csv_path: Path = Option(
        ...,
        "--csv-path",
        "-c",
        help="Path to CSV file with prior and posterior values",
    ),
    output_dir: Path = Option(
        ...,
        "--output-dir",
        "-o",
        help="Path to output directory for plots (will be created if it does not exist)",
    ),
) -> None:
    """
    Plot prior and posterior vs30 mean values with error bars.

    Creates a plot showing vs30 mean values (y-axis) vs category ID (x-axis)
    for both prior and posterior values, with error bars showing standard deviation.
    """
    try:
        # Validate input file exists
        if not csv_path.exists():
            typer.echo(f"Error: CSV file not found: {csv_path}", err=True)
            raise typer.Exit(1)

        logger.info(f"Loading data from: {csv_path}")

        # Load CSV
        df = pd.read_csv(csv_path, skipinitialspace=True)

        # Filter out invalid rows (e.g., -9999 placeholder values)
        df = df[df["prior_mean_vs30_km_per_s"] != -9999].copy()

        # Validate required columns
        required_cols = [
            "id",
            "prior_mean_vs30_km_per_s",
            "prior_standard_deviation_vs30_km_per_s",
            "posterior_mean_vs30_km_per_s",
            "posterior_standard_deviation_vs30_km_per_s",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            typer.echo(
                f"Error: CSV file missing required columns: {missing_cols}",
                err=True,
            )
            raise typer.Exit(1)

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract data
        category_ids = df["id"].values
        prior_mean = df["prior_mean_vs30_km_per_s"].values
        prior_std = df["prior_standard_deviation_vs30_km_per_s"].values
        posterior_mean = df["posterior_mean_vs30_km_per_s"].values
        posterior_std = df["posterior_standard_deviation_vs30_km_per_s"].values

        # Create figure
        fig, ax = plt.subplots(figsize=(12, 8))

        # Offset for x positions to separate prior and posterior
        offset = 0.2
        prior_x = category_ids - offset
        posterior_x = category_ids + offset

        # Plot prior values with error bars
        ax.errorbar(
            prior_x,
            prior_mean,
            yerr=prior_std,
            fmt="o",
            label="Prior",
            capsize=5,
            capthick=1.5,
            markersize=6,
            alpha=0.7,
        )

        # Plot posterior values with error bars
        ax.errorbar(
            posterior_x,
            posterior_mean,
            yerr=posterior_std,
            fmt="s",
            label="Posterior",
            capsize=5,
            capthick=1.5,
            markersize=6,
            alpha=0.7,
        )

        # Set labels and title
        ax.set_xlabel("Category ID", fontsize=12)
        ax.set_ylabel("Vs30 (m/s)", fontsize=12)
        ax.set_title("Prior vs Posterior Vs30 Values by Category", fontsize=14)
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)

        # Set x-axis to show category IDs
        ax.set_xticks(category_ids)
        ax.set_xticklabels(category_ids.astype(int))

        # Generate output filename from input CSV
        output_filename = csv_path.stem + "_plot.png"
        output_path = output_dir / output_filename

        # Save plot
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Plot saved to: {output_path}")
        typer.echo(f"✓ Plot saved to: {output_path}")

    except Exception as e:
        logger.exception("Error creating plot")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


def main() -> None:
    """Entry point for vs30map command."""
    app()


if __name__ == "__main__":
    # When run as script, use CLI
    main()

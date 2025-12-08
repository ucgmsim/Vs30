"""
MVN updates for geology and terrain models.

This module implements multivariate normal (MVN) distribution updates for pixels
in geology.tif and terrain.tif that are within range of observations.
"""

import logging
import time
from math import exp, log, sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean
from tqdm import tqdm

from vs30.model import ID_NODATA
from vs30.mvn_config import ModelConfig, MVNConfig
from vs30.mvn_data import (
    BoundingBoxResult,
    MVNUpdateResult,
    ObservationData,
    PixelData,
    RasterData,
)

logger = logging.getLogger(__name__)


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
# Bayesian Update Functions
# ============================================================================


def compute_category_statistics(
    sites: pd.DataFrame, idcol: str
) -> dict[int, dict[str, float]]:
    """
    Group observations by category and compute aggregated statistics.

    Parameters
    ----------
    sites : DataFrame
        Observations with vs30, uncertainty, and category ID column.
        Model IDs in the DataFrame are 1-indexed (from model_id() function).
    idcol : str
        Column name for model ID in sites.

    Returns
    -------
    dict
        Dictionary mapping category ID (1-indexed) to statistics dict with keys:
        - 'mean': mean log(vs30)
        - 'variance': variance of log(vs30)
        - 'count': number of observations
        - 'mean_uncertainty': mean uncertainty
    """
    stats = {}
    for cat_id in sites[idcol].unique():
        if cat_id == ID_NODATA:
            continue
        cat_sites = sites[sites[idcol] == cat_id]
        if len(cat_sites) == 0:
            continue

        log_vs30 = np.log(cat_sites.vs30.values)
        stats[cat_id] = {
            "mean": np.mean(log_vs30),
            "variance": np.var(log_vs30, ddof=0),
            "count": len(cat_sites),
            "mean_uncertainty": np.mean(cat_sites.uncertainty.values),
        }
    return stats


def apply_bayesian_update(
    prior_model: np.ndarray,
    category_stats: dict[int, dict[str, float]],
    n_prior: int,
    min_sigma: float,
) -> np.ndarray:
    """
    Apply Bayesian update formulas using pre-computed statistics.

    Parameters
    ----------
    prior_model : ndarray
        Prior model values (n_categories, 2) array of [vs30, stdv].
    category_stats : dict
        Statistics per category from compute_category_statistics.
    n_prior : int
        Assume prior model made up of n_prior measurements.
    min_sigma : float
        Minimum standard deviation allowed.

    Returns
    -------
    ndarray
        Updated model values (n_categories, 2) array of [vs30, stdv].
    """
    updated_model = prior_model.copy()
    vs30 = updated_model[:, 0]
    stdv = np.maximum(updated_model[:, 1], min_sigma)

    n0 = np.repeat(n_prior, len(updated_model))

    for cat_id, stats in category_stats.items():
        # Model IDs in observations are 1-indexed, but model table is 0-indexed
        # Convert to 0-indexed for array access
        if cat_id == ID_NODATA:
            continue
        array_idx = cat_id - 1  # Convert 1-indexed to 0-indexed
        if array_idx < 0 or array_idx >= len(updated_model):
            continue

        # Use aggregated statistics
        n_obs = stats["count"]
        mean_log_vs30 = stats["mean"]
        mean_uncertainty = stats["mean_uncertainty"]

        # Update variance
        var_new = (n0[array_idx] * stdv[array_idx] ** 2 + mean_uncertainty**2) / (
            n0[array_idx] + n_obs
        )

        # Update mean (in log space)
        log_mu_0 = log(vs30[array_idx])
        log_mu_new = (n0[array_idx] / var_new * log_mu_0 + mean_log_vs30 / var_new) / (
            n0[array_idx] / var_new + n_obs / var_new
        )

        vs30[array_idx] = exp(log_mu_new)
        stdv[array_idx] = sqrt(var_new)
        n0[array_idx] += n_obs

    return np.column_stack((vs30, stdv))


def update_categorical_priors(
    model_config: ModelConfig,
    observations: pd.DataFrame,
    mvn_config: MVNConfig,
) -> np.ndarray:
    """
    Unified function for updating categorical priors.

    Parameters
    ----------
    model_config : ModelConfig
        Model configuration.
    observations : DataFrame
        Observations with vs30, uncertainty, and model ID column.
    mvn_config : MVNConfig
        MVN configuration.

    Returns
    -------
    ndarray
        Updated model table (n_categories, 2) array of [vs30, stdv].
    """
    # Load prior model
    if model_config.name == "geology":
        import vs30.model_geology as model_module
    elif model_config.name == "terrain":
        import vs30.model_terrain as model_module
    else:
        raise ValueError(f"Unknown model: {model_config.name}")

    prior_model = model_module.model_prior()

    if mvn_config.update_mode == "prior":
        # Return prior unchanged (bypasses Bayesian update)
        return prior_model
    elif mvn_config.update_mode == "posterior_paper":
        # Load hardcoded posterior values
        return model_module.model_posterior_paper()
    elif mvn_config.update_mode == "computed":
        # Compute batch Bayesian update using observations
        # First, compute model IDs from observation locations and add to DataFrame
        obs_locs = observations[["easting", "northing"]].values
        model_ids = model_module.model_id(obs_locs)
        observations = observations.copy()  # Avoid modifying original
        observations[model_config.id_column] = model_ids

        category_stats = compute_category_statistics(
            observations, model_config.id_column
        )
        return apply_bayesian_update(
            prior_model, category_stats, mvn_config.n_prior, mvn_config.min_sigma
        )
    elif mvn_config.update_mode == "custom":
        # Use custom values if provided (for now, return prior)
        # TODO: Add support for custom values from config
        return prior_model
    else:
        raise ValueError(f"Unknown update_mode: {mvn_config.update_mode}")


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
    model_config: ModelConfig,
    mvn_config: MVNConfig,
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
    model_config : ModelConfig
        Model configuration.
    mvn_config : MVNConfig
        MVN configuration.

    Returns
    -------
    ObservationData
        Prepared observation data object.
    """
    # Get observation locations
    obs_locs = observations[["easting", "northing"]].values

    # Interpolate model values at observation locations
    # First get model IDs
    if model_config.name == "geology":
        import vs30.model_geology as model_module
    elif model_config.name == "terrain":
        import vs30.model_terrain as model_module
    else:
        raise ValueError(f"Unknown model: {model_config.name}")

    model_ids = model_module.model_id(obs_locs)

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
    model_config: ModelConfig,
    mvn_config: MVNConfig,
) -> np.ndarray:
    """
    Build covariance matrix through clear pipeline of steps.

    Parameters
    ----------
    pixel : PixelData
        Pixel data for the pixel being updated.
    selected_observations : ObservationData
        Selected observations for this pixel.
    model_config : ModelConfig
        Model configuration.
    mvn_config : MVNConfig
        MVN configuration.

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
    corr = correlation_function(distance_matrix, model_config.phi)

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
    mvn_config: MVNConfig,
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
    mvn_config : MVNConfig
        MVN configuration.

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
    model_config: ModelConfig,
    mvn_config: MVNConfig,
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
    model_config : ModelConfig
        Model configuration.
    mvn_config : MVNConfig
        MVN configuration.

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
    cov_matrix = build_covariance_matrix(pixel, selected_obs, model_config, mvn_config)

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
    raster_data: RasterData, obs_data: ObservationData, mvn_config: MVNConfig
) -> BoundingBoxResult:
    """
    Find pixels affected by observations using bounding boxes.

    Parameters
    ----------
    raster_data : RasterData
        Raster data object.
    obs_data : ObservationData
        Observation data.
    mvn_config : MVNConfig
        MVN configuration.

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
    model_config: ModelConfig,
    mvn_config: MVNConfig,
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
    model_config : ModelConfig
        Model configuration.
    mvn_config : MVNConfig
        MVN configuration.

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
                model_config,
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
    model_config: ModelConfig,
    mvn_config: MVNConfig,
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
    model_config : ModelConfig
        Model configuration.
    mvn_config : MVNConfig
        MVN configuration.
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

    # Write output
    output_path = base_path / "vs30map" / f"{model_config.name}_mvn.tif"
    output_path.parent.mkdir(parents=True, exist_ok=True)

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


# ============================================================================
# Main Pipeline
# ============================================================================


def load_observations(base_path: Path) -> pd.DataFrame:
    """
    Load observation CSV file.

    Parameters
    ----------
    base_path : Path
        Base path for data files.

    Returns
    -------
    DataFrame
        Observations with vs30, uncertainty, easting, northing.
    """
    obs_path = (
        base_path
        / "vs30"
        / "resources"
        / "data"
        / "measured_vs30_original_filtered.csv"
    )
    return pd.read_csv(obs_path)


def process_model_updates(config: MVNConfig, base_path: Path) -> None:
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
    config : MVNConfig
        MVN configuration.
    base_path : Path
        Base path for input/output files.
    """
    for model_name in config.models_to_process:
        logger.info(f"Processing {model_name} model...")

        # Stage 1: Load raster data
        model_config = ModelConfig.from_mvn_config(config, model_name, base_path)
        raster_data = RasterData.from_file(model_config.raster_path)
        validate_raster_data(raster_data)

        # Stage 2: Load observations
        logger.info("Loading observations...")
        observations = load_observations(base_path)
        validate_observations(observations)
        logger.info(f"Loaded {len(observations)} observations")

        # Stage 3: Bayesian update (optional)
        logger.info(f"Updating categorical priors (mode: {config.update_mode})...")
        start_time_stage = time.time()
        updated_model_table = update_categorical_priors(
            model_config, observations, config
        )
        logger.info(
            f"Updated categorical priors in {time.time() - start_time_stage:.2f}s"
        )

        # Stage 4: Prepare observation data for MVN
        logger.info("Preparing observation data for MVN processing...")
        start_time_stage = time.time()
        obs_data = prepare_observation_data(
            observations, raster_data, updated_model_table, model_config, config
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
            raster_data, obs_data, bbox_result, model_config, config
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
        apply_and_write_updates(raster_data, updates, model_config, config, base_path)
        logger.info(f"Wrote output raster in {time.time() - start_time_stage:.2f}s")

        logger.info(f"Completed processing {model_name} model")


if __name__ == "__main__":
    import logging

    logging.basicConfig(level=logging.INFO)
    start_time = time.time()

    base_path = Path(__file__).parent

    # Load configuration
    config_path = base_path / "config.yaml"
    config = MVNConfig.from_yaml(config_path)

    # Run main pipeline
    process_model_updates(config, base_path.parent)

    elapsed_total = time.time() - start_time
    logger.info(
        f"Total processing time: {elapsed_total:.2f}s ({elapsed_total / 60:.1f} minutes)"
    )

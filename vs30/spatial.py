"""
MVN (multivariate normal distribution)
for modifying vs30 values based on proximity to measured values.
"""

import logging
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio import transform as rasterio_transform
from scipy.spatial.distance import cdist
from tqdm import tqdm

from vs30 import constants, utils
from vs30 import spatial as mvn

logger = logging.getLogger(__name__)

"""
Data classes for MVN processing.
"""


@dataclass
class ObservationData:
    """Bundled observation data for MVN processing."""

    locations: np.ndarray  # (n_obs, 2) array of [easting, northing]
    vs30: np.ndarray  # (n_obs,) measured vs30 values
    model_vs30: np.ndarray  # (n_obs,) model vs30 at observation locations
    model_stdv: np.ndarray  # (n_obs,) model stdv at observation locations
    residuals: np.ndarray  # (n_obs,) log residuals: log(vs30 / model_vs30)
    omega: np.ndarray  # (n_obs,) noise weights (if noisy=True)
    uncertainty: np.ndarray  # (n_obs,) observation uncertainties


@dataclass
class PixelData:
    """Data for a single pixel."""

    location: np.ndarray  # [easting, northing]
    vs30: float  # prior vs30 value
    stdv: float  # prior stdv value
    index: int  # flat index in the raster


@dataclass
class MVNUpdateResult:
    """Result of an MVN update for a single pixel."""

    updated_vs30: float
    updated_stdv: float
    n_observations_used: int
    min_distance: float
    pixel_index: int


@dataclass
class RasterData:
    """Raster data and metadata."""

    vs30: np.ndarray  # Band 1: Vs30 mean
    stdv: np.ndarray  # Band 2: Vs30 stdv
    transform: rasterio.transform.Affine
    crs: rasterio.crs.CRS
    nodata: float | None
    valid_mask: np.ndarray  # Boolean mask of non-nodata pixels
    valid_flat_indices: np.ndarray  # Flat indices of non-nodata pixels

    @classmethod
    def from_file(cls, path: Path) -> "RasterData":
        """Load 2-band VS30 raster."""
        with rasterio.open(path) as src:
            vs30 = src.read(1)
            stdv = src.read(2)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata

            # Create mask of valid pixels (not nodata, not nan, and positive)
            valid_mask = (
                (vs30 != nodata)
                & (~np.isnan(vs30))
                & (~np.isnan(stdv))
                & (vs30 > 0)
                & (stdv > 0)
            )
            valid_flat_indices = np.where(valid_mask.flatten())[0]

            return cls(
                vs30=vs30,
                stdv=stdv,
                transform=transform,
                crs=crs,
                nodata=nodata,
                valid_mask=valid_mask,
                valid_flat_indices=valid_flat_indices,
            )

    def get_coordinates(self) -> np.ndarray:
        """
        Get coordinates for all valid pixels.

        Returns
        -------
        ndarray
            (N_valid, 2) array of [easting, northing] coordinates.
        """
        valid_rows, valid_cols = np.where(self.valid_mask)
        rows_valid = valid_rows.astype(float) + 0.5
        cols_valid = valid_cols.astype(float) + 0.5
        xs, ys = rasterio_transform.xy(self.transform, rows_valid, cols_valid)
        return np.column_stack((np.array(xs), np.array(ys))).astype(np.float64)

    def write_updated(
        self, path: Path, updated_vs30: np.ndarray, updated_stdv: np.ndarray
    ) -> None:
        """
        Write updated raster to file.

        Parameters
        ----------
        path : Path
            Output file path.
        updated_vs30 : ndarray
            Updated vs30 values (same shape as self.vs30).
        updated_stdv : ndarray
            Updated stdv values (same shape as self.stdv).
        """
        # Write using rasterio with compression
        with rasterio.open(
            path,
            "w",
            driver="GTiff",
            height=self.vs30.shape[0],
            width=self.vs30.shape[1],
            count=2,
            dtype=self.vs30.dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=self.nodata,
            compress="deflate",
            tiled=True,
            bigtiff="yes",
        ) as dst:
            dst.write(updated_vs30, 1)
            dst.write(updated_stdv, 2)


@dataclass
class BoundingBoxResult:
    """Result of bounding box search for affected pixels."""

    mask: np.ndarray  # Boolean mask of pixels in any observation's bounding box
    obs_to_grid_indices: list[
        np.ndarray
    ]  # For each observation, flat indices of pixels in its bounding box
    n_affected_pixels: int


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
# Covariance Matrix Building
# ============================================================================


def build_covariance_matrix(
    pixel: PixelData,
    selected_observations: ObservationData,
    model_name: str,
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

    Returns
    -------
    ndarray
        Covariance matrix (n_selected_obs + 1, n_selected_obs + 1).
        First row/column is for the pixel, rest are for observations.
    """
    # Step 1: Compute distances
    all_points = np.vstack([pixel.location, selected_observations.locations])
    distance_matrix = utils.euclidean_distance_matrix(all_points)

    # Step 2: Apply correlation function
    phi = constants.PHI[model_name]
    corr = utils.correlation_function(distance_matrix, phi)

    # Step 3: Scale by standard deviations
    stdvs = np.insert(selected_observations.model_stdv, 0, pixel.stdv)
    cov = corr * np.outer(stdvs, stdvs)

    # Step 4: Apply noise weighting (if enabled)
    if constants.NOISY:
        omega = np.insert(selected_observations.omega, 0, 1.0)
        omega_matrix = np.outer(omega, omega)
        np.fill_diagonal(omega_matrix, 1.0)
        cov *= omega_matrix

    # Step 5: Apply covariance reduction (if enabled)
    if constants.COV_REDUC > 0:
        log_vs30s = np.insert(
            np.log(selected_observations.model_vs30), 0, np.log(pixel.vs30)
        )
        log_dist_matrix = utils.euclidean_distance_matrix(log_vs30s.reshape(-1, 1))
        cov *= np.exp(-constants.COV_REDUC * log_dist_matrix)

    return cov


# ============================================================================
# MVN Update Computation
# ============================================================================


def select_observations_for_pixel(
    pixel: mvn.PixelData,
    obs_data: mvn.ObservationData,
    obs_to_grid_indices: list[np.ndarray],
    chunk_grid_to_obs: dict[int, list[int]],
) -> mvn.ObservationData:
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

    Returns
    -------
    ObservationData
        Selected observations (subset of obs_data).
    """
    # Get candidate observations from chunk cache
    candidate_obs_indices = chunk_grid_to_obs.get(pixel.index, [])

    if len(candidate_obs_indices) == 0:
        # Return empty ObservationData
        return mvn.ObservationData(
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
    within_dist_mask = distances <= constants.MAX_DIST_M
    filtered_indices = np.array(candidate_obs_indices)[within_dist_mask]
    filtered_distances = distances[within_dist_mask]

    if len(filtered_indices) == 0:
        # Return empty ObservationData
        return mvn.ObservationData(
            locations=np.empty((0, 2)),
            vs30=np.empty(0),
            model_vs30=np.empty(0),
            model_stdv=np.empty(0),
            residuals=np.empty(0),
            omega=np.empty(0),
            uncertainty=np.empty(0),
        )

    # Limit to closest max_points
    if len(filtered_indices) > constants.MAX_POINTS:
        # Sort by distance and take closest max_points
        sorted_idx = np.argsort(filtered_distances)
        selected_idx = sorted_idx[: constants.MAX_POINTS]
        filtered_indices = filtered_indices[selected_idx]
        filtered_distances = filtered_distances[selected_idx]

    # Create subset of ObservationData
    return mvn.ObservationData(
        locations=obs_data.locations[filtered_indices],
        vs30=obs_data.vs30[filtered_indices],
        model_vs30=obs_data.model_vs30[filtered_indices],
        model_stdv=obs_data.model_stdv[filtered_indices],
        residuals=obs_data.residuals[filtered_indices],
        omega=obs_data.omega[filtered_indices],
        uncertainty=obs_data.uncertainty[filtered_indices],
    )


def compute_mvn_update_for_pixel(
    pixel: mvn.PixelData,
    obs_data: mvn.ObservationData,
    obs_to_grid_indices: list[np.ndarray],
    chunk_grid_to_obs: dict[int, list[int]],
    model_name: str,
) -> mvn.MVNUpdateResult | None:
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
        pixel, obs_data, obs_to_grid_indices, chunk_grid_to_obs
    )

    if len(selected_obs.locations) == 0:
        # No observations nearby, return unchanged values
        return mvn.MVNUpdateResult(
            updated_vs30=pixel.vs30,
            updated_stdv=pixel.stdv,
            n_observations_used=0,
            min_distance=np.inf,
            pixel_index=pixel.index,
        )

    # Build covariance matrix
    cov_matrix = build_covariance_matrix(pixel, selected_obs, model_name)

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

    return mvn.MVNUpdateResult(
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
    raster_data: mvn.RasterData,
    obs_data: mvn.ObservationData,
) -> mvn.BoundingBoxResult:
    """
    Find pixels affected by observations using bounding boxes.

    Parameters
    ----------
    raster_data : RasterData
        Raster data object.
    obs_data : ObservationData
        Observation data.

    Returns
    -------
    BoundingBoxResult
        Result containing mask and observation-to-grid mappings.
    """
    # Get coordinates for valid pixels
    grid_locs = raster_data.get_coordinates()

    # Calculate chunk size
    chunk_size = calculate_chunk_size(
        len(obs_data.locations), constants.TOTAL_MEMORY_GB
    )
    n_chunks = int(np.ceil(len(grid_locs) / chunk_size))

    # Precompute observation bounds
    obs_eastings = obs_data.locations[:, 0:1]  # (n_obs, 1)
    obs_northings = obs_data.locations[:, 1:2]  # (n_obs, 1)
    obs_eastings_min = obs_eastings - constants.MAX_DIST_M
    obs_eastings_max = obs_eastings + constants.MAX_DIST_M
    obs_northings_min = obs_northings - constants.MAX_DIST_M
    obs_northings_max = obs_northings + constants.MAX_DIST_M

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

    return mvn.BoundingBoxResult(
        mask=grid_points_in_bbox_mask,
        obs_to_grid_indices=obs_to_grid_indices,
        n_affected_pixels=n_affected,
    )


# ============================================================================
# Main MVN Updates Computation
# ============================================================================


def compute_mvn_updates(
    raster_data: mvn.RasterData,
    obs_data: mvn.ObservationData,
    bbox_result: mvn.BoundingBoxResult,
    model_name: str,
) -> list[mvn.MVNUpdateResult]:
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
        len(obs_data.locations), constants.TOTAL_MEMORY_GB
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
            pixel = mvn.PixelData(
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
    output_dir: Path,
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
    output_dir : Path
        Directory where output raster will be saved.
    """
    # Initialize output arrays with original values
    updated_vs30 = raster_data.vs30.copy()
    updated_stdv = raster_data.stdv.copy()

    # Apply updates
    for update in updates:
        updated_vs30.flat[update.pixel_index] = update.updated_vs30
        updated_stdv.flat[update.pixel_index] = update.updated_stdv

    # Write output using filename from constants
    output_filename = constants.OUTPUT_FILENAMES[model_name]
    output_path = output_dir / output_filename

    raster_data.write_updated(output_path, updated_vs30, updated_stdv)

    logger.info(
        f"Wrote updated raster to {output_path} "
        f"({len(updates):,} pixels updated out of {np.sum(raster_data.valid_mask):,} valid)"
    )


#######################################################################################
#######################################################################################
# Original mvn.py functions
#######################################################################################
#######################################################################################


def _corr_func(distances, model):
    """
    Correlation function by distance.
    phi is 993 for terrain model, 1407 for geology
    """
    if model == "geology":
        phi = 1407
    elif model == "terrain":
        phi = 993
    else:
        raise ValueError("unknown model")
    # r code linearly interpolated from logarithmically spaced distances:
    # d = np.exp(np.linspace(np.log(0.1), np.log(2000e3), 128))
    # c = 1 / np.e ** (d / phi)
    # return np.interp(distances, d, c)
    # minimum distance of 0.1 metres enforced
    return 1 / np.e ** (np.maximum(0.1, distances) / phi)


def _tcrossprod(x):
    """
    Matrix cross product (or outer product) from a 1d numpy array.
    Same functionality as the R function tcrossprod(x) with y = NULL.
    https://stat.ethz.ch/R-manual/R-devel/library/base/html/crossprod.html
    """
    return x[:, np.newaxis] * x


def _dists(x):
    """
    Euclidean distance from 2d diff array.
    """
    # return np.linalg.norm(x, axis=1)
    # alternative, may be faster
    return np.sqrt(np.einsum("ij,ij->i", x, x))


def _xy2complex(x):
    """
    Convert array of 2D coordinates to array of 1D complex numbers.
    """
    c = x[:, 0].astype(np.complex64)
    c.imag += x[:, 1]
    return c


def _dist_mat(x):
    """
    Distance matrix between coordinates (complex numbers) or simple values.
    """
    return np.abs(x[:, np.newaxis] - x)


def _mvn(
    model_locs,
    model_vs30,
    model_stdv,
    sites,
    model_name,
    cov_reduc=1.5,
    noisy=True,
    max_dist=10000,
    max_points=500,
):
    """
    Modify model with observed locations.
    noisy: whether measurements are noisy True/False
    max_dist: only consider observed locations within this many metres
    max_points: limit observed locations to closest N points within max_dist
    """
    # cut not-applicable sites to prevent nan propagation
    sites = sites[~np.isnan(sites[f"{model_name}_vs30"])]

    obs_locs = np.column_stack((sites.easting.values, sites.northing.values))
    obs_model_stdv = sites[f"{model_name}_stdv"].values
    obs_residuals = np.log(sites.vs30.values / sites[f"{model_name}_vs30"].values)

    # Wea equation 33, 40, 41
    if noisy:
        omega_obs = np.sqrt(
            obs_model_stdv**2 / (obs_model_stdv**2 + sites.uncertainty.values**2)
        )
        obs_residuals *= omega_obs

    # default outputs if no sites closeby
    pred = np.log(model_vs30)
    var = model_stdv**2 * _corr_func(0, model_name)

    # model point to observations
    for i, model_loc in enumerate(model_locs):
        if np.isnan(model_vs30[i]):
            # input of nan is always output of nan
            continue
        # don't recalculate distances if delta distance is too small anyway
        # useful when calculating accross grids or close points
        try:
            movement = _dists(np.atleast_2d(model_loc - prev_model_loc))[0]
            if min_dist - movement > max_dist:
                continue
        except NameError:
            pass
        distances = _dists(obs_locs - model_loc)
        max_points_i = min(max_points, len(distances)) - 1
        min_dist, cutoff_dist = np.partition(distances, [0, max_points_i])[
            [0, max_points_i]
        ]
        if min_dist > max_dist:
            # not close enough to any observed locations
            continue
        loc_mask = distances <= min(max_dist, cutoff_dist)

        # distances between interesting points
        cov_matrix = _dist_mat(_xy2complex(np.vstack((model_loc, obs_locs[loc_mask]))))
        # correlation
        cov_matrix = _corr_func(cov_matrix, model_name)
        # uncertainties
        cov_matrix *= _tcrossprod(np.insert(obs_model_stdv[loc_mask], 0, model_stdv[i]))

        if noisy:
            omega = _tcrossprod(np.insert(omega_obs[loc_mask], 0, 1))
            np.fill_diagonal(omega, 1)
            cov_matrix *= omega

        # Invert covariance matrix (observations only)
        # Note: This uses BLAS/LAPACK and can cause CPU spikes when many observations
        # are present (up to max_points=500, resulting in 500x500 matrix inversion)
        inv_cov = np.linalg.inv(cov_matrix[1:, 1:])

        # Calculate prediction update
        pred_update = np.dot(
            np.dot(cov_matrix[0, 1:], inv_cov), obs_residuals[loc_mask]
        )

        # Calculate variance
        var = cov_matrix[0, 0] - np.dot(
            np.dot(cov_matrix[0, 1:], inv_cov), cov_matrix[1:, 0]
        )

        pred[i] += pred_update
        model_stdv[i] = np.sqrt(var)

    return np.exp(pred), model_stdv


def mvn_tiff(
    tiff_path,
    obs_sites,
    model_name,
    output_path,
    cov_reduc=1.5,
    noisy=True,
    max_dist=10000,
    max_points=500,
):
    """
    Update vs30 tiff using observations.
    tiff_path: path to input tiff
    obs_sites: dataframe with observations
    model_name: "geology" or "terrain"
    output_path: path to output tiff
    """
    with rasterio.open(tiff_path) as src:
        vs30 = src.read(1)
        model_stdv = src.read(2)
        transform = src.transform
        crs = src.crs
        nodata = src.nodata

    # mask out nodata
    mask = vs30 != nodata
    if np.any(mask):
        rows, cols = np.where(mask)
        # center of pixel
        rows_f = rows.astype(float) + 0.5
        cols_f = cols.astype(float) + 0.5
        xs, ys = rasterio_transform.xy(transform, rows_f, cols_f)
        model_locs = np.column_stack((np.array(xs), np.array(ys))).astype(np.float64)

        # compute updates
        updated_vs30, updated_stdv = _mvn(
            model_locs,
            vs30[mask],
            model_stdv[mask],
            obs_sites,
            model_name,
            cov_reduc=cov_reduc,
            noisy=noisy,
            max_dist=max_dist,
            max_points=max_points,
        )

        # apply updates
        vs30[mask] = updated_vs30
        model_stdv[mask] = updated_stdv

    # write output
    with rasterio.open(
        output_path,
        "w",
        driver="GTiff",
        height=vs30.shape[0],
        width=vs30.shape[1],
        count=2,
        dtype=vs30.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        bigtiff="yes",
    ) as dst:
        dst.write(vs30, 1)
        dst.write(model_stdv, 2)

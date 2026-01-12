"""
Multivariate Normal (MVN) distribution-based spatial adjustment for Vs30 values.

This module implements spatial interpolation using MVN conditioning, which
adjusts model Vs30 predictions based on nearby measurements. The approach:

1. For each pixel near observations, builds a covariance matrix relating
   the pixel to nearby measurements
2. Uses MVN conditioning to compute the posterior (updated) Vs30 and
   uncertainty given the observations
3. Applies updates to create spatially-adjusted Vs30 maps

The covariance structure uses an exponential correlation function, with
separate correlation lengths (phi) for geology and terrain models.

Key Parameters (from config.yaml)
---------------------------------
- phi: Correlation length controlling spatial smoothness (different for geology/terrain)
- max_dist_m: Maximum distance (meters) to consider observations
- max_points: Maximum number of observations per pixel update
- noisy: Whether to apply noise weighting based on observation uncertainty
- cov_reduc: Covariance reduction factor for dissimilar Vs30 values

Scientific Background
---------------------
The MVN conditioning approach assumes that Vs30 residuals (log(measured/model))
are spatially correlated following an exponential correlation model. Near
observation points, the model prediction is adjusted toward the measured value,
with the adjustment magnitude depending on distance and correlation structure.
"""

import logging
from dataclasses import dataclass
from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from scipy.spatial.distance import cdist
from tqdm import tqdm

from vs30 import constants, utils
from vs30.category import (
    ID_NODATA,
    _assign_to_category_geology,
    _assign_to_category_terrain,
)

logger = logging.getLogger(__name__)


@dataclass
class ObservationData:
    """Bundled observation data for spatial processing."""

    locations: np.ndarray  # (n_obs, 2) array of [easting, northing]
    vs30: np.ndarray  # (n_obs,) measured vs30 values
    model_vs30: np.ndarray  # (n_obs,) model vs30 at observation locations
    model_stdv: np.ndarray  # (n_obs,) model stdv at observation locations
    residuals: np.ndarray  # (n_obs,) log residuals: log(vs30 / model_vs30)
    omega: np.ndarray  # (n_obs,) noise weights (if noisy=True)
    uncertainty: np.ndarray  # (n_obs,) observation uncertainties

    @classmethod
    def empty(cls) -> "ObservationData":
        """Create an empty ObservationData object with zero observations."""
        return cls(
            locations=np.empty((0, 2)),
            vs30=np.empty(0),
            model_vs30=np.empty(0),
            model_stdv=np.empty(0),
            residuals=np.empty(0),
            omega=np.empty(0),
            uncertainty=np.empty(0),
        )


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
        Get coordinates for all valid pixels using GDAL affine transform.

        Returns
        -------
        ndarray
            (N_valid, 2) array of [easting, northing] coordinates.
        """
        valid_rows, valid_cols = np.where(self.valid_mask)

        # Rasterio Affine: [a, b, c, d, e, f] = [x_scale, x_shear, x_origin, y_shear, y_scale, y_origin]
        x_scale, x_shear, x_origin = (
            self.transform[0],
            self.transform[1],
            self.transform[2],
        )
        y_shear, y_scale, y_origin = (
            self.transform[3],
            self.transform[4],
            self.transform[5],
        )

        # Pixel centers: add 0.5 to row/col indices (following legacy implementation)
        rows_center = valid_rows.astype(float) + 0.5
        cols_center = valid_cols.astype(float) + 0.5

        # Apply affine transformation (matching legacy GDAL calculation)
        xs = x_origin + cols_center * x_scale  # col controls easting
        ys = y_origin + rows_center * y_scale  # row controls northing

        return np.column_stack((xs, ys)).astype(np.float32)

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


def prepare_observation_data(
    observations: pd.DataFrame,
    raster_data: RasterData,
    updated_model_table: np.ndarray,
    model_name: str,
    output_dir: Path,
    noisy: bool = constants.NOISY,
    # Hybrid Modification Parameters (optional, for geology)
    hybrid_mod6_dist_min: float | None = None,
    hybrid_mod6_dist_max: float | None = None,
    hybrid_mod6_vs30_min: float | None = None,
    hybrid_mod6_vs30_max: float | None = None,
    hybrid_mod13_dist_min: float | None = None,
    hybrid_mod13_dist_max: float | None = None,
    hybrid_mod13_vs30_min: float | None = None,
    hybrid_mod13_vs30_max: float | None = None,
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

    Returns
    -------
    ObservationData
        Prepared observation data object.
    """
    # Get observation locations
    obs_locs = observations[["easting", "northing"]].values

    # Interpolate model values at observation locations
    if model_name == "geology":
        model_ids = _assign_to_category_geology(obs_locs)
    elif model_name == "terrain":
        model_ids = _assign_to_category_terrain(obs_locs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

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
    # For geology, we must apply hybrid modifications to model values at observation points
    if model_name == "geology":
        from vs30.raster import (
            apply_hybrid_geology_modifications,
            create_coast_distance_raster,
            create_slope_raster,
        )

        # 1. Get slope and coast distance at points
        # Use existing rasters in output_dir if possible, otherwise create temporary ones
        slope_path = output_dir / constants.SLOPE_RASTER_FILENAME
        coast_path = output_dir / constants.COAST_DISTANCE_RASTER_FILENAME

        profile = {
            "transform": raster_data.transform,
            "width": raster_data.vs30.shape[1],
            "height": raster_data.vs30.shape[0],
            "crs": rasterio.crs.CRS.from_string(constants.NZTM_CRS),
        }

        if not slope_path.exists():
            create_slope_raster(slope_path, profile)
        if not coast_path.exists():
            create_coast_distance_raster(coast_path, profile)

        # Sample rasters at observation locations
        with rasterio.open(slope_path) as src:
            slope_obs = np.array([v[0] for v in src.sample(obs_locs)])
        with rasterio.open(coast_path) as src:
            coast_obs = np.array([v[0] for v in src.sample(obs_locs)])

        # Apply modifications to model_vs30 and model_stdv at points
        # Note: we only need to modify where category IDs match hybrid types
        model_vs30, model_stdv = apply_hybrid_geology_modifications(
            model_vs30,
            model_stdv,
            model_ids[valid_obs_mask],
            slope_obs,
            coast_obs,
            mod6=True,
            mod13=True,
            hybrid=True,
            # Pass hybrid parameters
            hybrid_mod6_dist_min=hybrid_mod6_dist_min,
            hybrid_mod6_dist_max=hybrid_mod6_dist_max,
            hybrid_mod6_vs30_min=hybrid_mod6_vs30_min,
            hybrid_mod6_vs30_max=hybrid_mod6_vs30_max,
            hybrid_mod13_dist_min=hybrid_mod13_dist_min,
            hybrid_mod13_dist_max=hybrid_mod13_dist_max,
            hybrid_mod13_vs30_min=hybrid_mod13_vs30_min,
            hybrid_mod13_vs30_max=hybrid_mod13_vs30_max,
        )

    residuals = np.log(vs30_obs / model_vs30)

    # Apply noise weighting (if noisy=True)
    if noisy:
        omega = np.sqrt(model_stdv**2 / (model_stdv**2 + uncertainty**2))
        residuals *= omega
    else:
        omega = np.ones(len(residuals))  # Defaults to float64

    return ObservationData(
        locations=obs_locs,
        vs30=vs30_obs,
        model_vs30=model_vs30,
        model_stdv=model_stdv,
        residuals=residuals,
        omega=omega,
        uncertainty=uncertainty,
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


def calculate_chunk_size(n_obs, max_spatial_boolean_array_memory_gb):
    """
    Calculate the maximum number of grid points per chunk based on available memory for spatial boolean arrays.

    The memory-intensive operation is the boolean array of shape (n_obs, n_grid_chunk)
    used to determine which grid points are within bounding boxes of observations.
    NumPy boolean arrays use 1 byte per element.
    Memory usage per chunk: n_obs * n_grid_chunk * 1 byte

    Parameters
    ----------
    n_obs : int
        Number of observations.
    max_spatial_boolean_array_memory_gb : float
        Maximum memory in GB allocated for spatial boolean arrays during chunked processing.

    Returns
    -------
    chunk_size : int
        Maximum number of grid points per chunk.
    """
    memory_per_chunk_bytes = max_spatial_boolean_array_memory_gb * 1024 * 1024 * 1024
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
    phi: float | None = None,
    noisy: bool = constants.NOISY,
    cov_reduc: float = constants.COV_REDUC,
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
    # Step 1: Compute Euclidean distance matrix
    all_points = np.vstack([pixel.location, selected_observations.locations]).astype(
        np.float64
    )
    distance_matrix = utils.euclidean_distance_matrix(all_points)

    # Step 2: Apply correlation function
    if phi is None:
        phi = constants.PHI[model_name]
    corr = utils.correlation_function(distance_matrix, phi)

    # Step 3: Scale by standard deviations
    stdvs = np.insert(selected_observations.model_stdv, 0, pixel.stdv)
    cov = corr * np.outer(stdvs, stdvs)

    # Step 4: Apply noise weighting (if enabled)
    if noisy:
        omega = np.insert(selected_observations.omega, 0, 1.0)
        omega_matrix = np.outer(omega, omega)
        np.fill_diagonal(omega_matrix, 1.0)
        cov *= omega_matrix

    # Step 5: Apply covariance reduction (if enabled)
    if cov_reduc > 0:
        log_vs30s = np.insert(
            np.log(selected_observations.model_vs30), 0, np.log(pixel.vs30)
        )
        log_dist_matrix = np.abs(log_vs30s[:, np.newaxis] - log_vs30s)
        cov *= np.exp(-cov_reduc * log_dist_matrix)

    return cov


# ============================================================================
# MVN Update Computation
# ============================================================================


def select_observations_for_pixel(
    pixel: PixelData,
    obs_data: ObservationData,
    max_dist_m: float = constants.MAX_DIST_M,
    max_points: int = constants.MAX_POINTS,
) -> ObservationData:
    """
    Select observations for a pixel using distance filtering.

    Uses accurate distance-based selection to find the closest observations
    within the maximum distance limit.

    Parameters
    ----------
    pixel : PixelData
        Pixel data.
    obs_data : ObservationData
        Full observation data.
    max_dist_m : float
        Maximum distance in meters to consider observations.
    max_points : int
        Maximum number of observations to select.

    Returns
    -------
    ObservationData
        Selected observations (subset of obs_data).
    """
    # Calculate distances from pixel to all observations (O(N) efficient calculation)
    distances = np.sqrt(np.sum((obs_data.locations - pixel.location) ** 2, axis=1))

    candidate_obs_indices = np.arange(len(obs_data.locations))  # All observations

    # Select observations using accurate distance-based filtering
    max_points_i = min(max_points, len(distances)) - 1
    if max_points_i >= 0:
        min_dist, cutoff_dist = np.partition(distances, [0, max_points_i])[
            [0, max_points_i]
        ]
        if min_dist > max_dist_m:
            # Not close enough to any observed locations
            return ObservationData.empty()
        # Include all observations within cutoff distance (may exceed max_points for accuracy)
        loc_mask = distances <= min(max_dist_m, cutoff_dist)
        filtered_indices = np.array(candidate_obs_indices)[loc_mask]
    else:
        # No observations available
        return ObservationData.empty()

    # Create subset of ObservationData
    selected_obs = ObservationData(
        locations=obs_data.locations[filtered_indices],
        vs30=obs_data.vs30[filtered_indices],
        model_vs30=obs_data.model_vs30[filtered_indices],
        model_stdv=obs_data.model_stdv[filtered_indices],
        residuals=obs_data.residuals[filtered_indices],
        omega=obs_data.omega[filtered_indices],
        uncertainty=obs_data.uncertainty[filtered_indices],
    )

    return selected_obs


def compute_mvn_update_for_pixel(
    pixel: PixelData,
    obs_data: ObservationData,
    model_name: str,
    phi: float | None = None,
    max_dist_m: float = constants.MAX_DIST_M,
    max_points: int = constants.MAX_POINTS,
    noisy: bool = constants.NOISY,
    cov_reduc: float = constants.COV_REDUC,
) -> MVNUpdateResult | None:
    """
    Compute MVN update for a single pixel.

    Parameters
    ----------
    pixel : PixelData
        Pixel data.
    obs_data : ObservationData
        Full observation data.
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

    phi_val = phi if phi is not None else constants.PHI[model_name]
    corr_zero = utils.correlation_function(np.array([0.0]), phi_val)[0]
    initial_var = (pixel.stdv**2) * corr_zero

    # Select observations for this pixel
    selected_obs = select_observations_for_pixel(
        pixel,
        obs_data,
        max_dist_m=max_dist_m,
        max_points=max_points,
    )

    if len(selected_obs.locations) == 0:
        # No observations nearby, return unchanged values (but with shrunk stdv matching legacy)
        return MVNUpdateResult(
            updated_vs30=pixel.vs30,
            updated_stdv=sqrt(initial_var),
            n_observations_used=0,
            min_distance=np.inf,
            pixel_index=pixel.index,
        )

    # Build covariance matrix
    cov_matrix = build_covariance_matrix(
        pixel,
        selected_obs,
        model_name,
        phi=phi,
        noisy=noisy,
        cov_reduc=cov_reduc,
    )

    # Invert covariance matrix (observations only)
    inv_cov = np.linalg.inv(cov_matrix[1:, 1:])

    # Calculate prediction update
    pred_update = np.dot(
        np.dot(cov_matrix[0, 1:], inv_cov),
        selected_obs.residuals,
    )

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
    min_distance = np.min(distances) if len(distances) > 0 else np.inf

    return MVNUpdateResult(
        updated_vs30=float(new_vs30),
        updated_stdv=float(new_stdv),
        n_observations_used=len(selected_obs.locations),
        min_distance=float(min_distance),
        pixel_index=pixel.index,
    )


# ============================================================================
# Find Affected Pixels
# ============================================================================


def find_affected_pixels(
    raster_data: RasterData,
    obs_data: ObservationData,
    max_dist_m: float = constants.MAX_DIST_M,
) -> BoundingBoxResult:
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
        len(obs_data.locations), constants.MAX_SPATIAL_BOOLEAN_ARRAY_MEMORY_GB
    )
    n_chunks = int(np.ceil(len(grid_locs) / chunk_size))

    # Precompute observation bounds
    obs_eastings = obs_data.locations[:, 0:1]  # (n_obs, 1)
    obs_northings = obs_data.locations[:, 1:2]  # (n_obs, 1)
    obs_eastings_min = obs_eastings - max_dist_m
    obs_eastings_max = obs_eastings + max_dist_m
    obs_northings_min = obs_northings - max_dist_m
    obs_northings_max = obs_northings + max_dist_m

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
    phi: float | None = None,
    max_dist_m: float = constants.MAX_DIST_M,
    max_points: int = constants.MAX_POINTS,
    noisy: bool = constants.NOISY,
    cov_reduc: float = constants.COV_REDUC,
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
        If True, enables legacy behavior with known bugs (exceeding max_points).
        Only kept temporarily for validation purposes.

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
        len(obs_data.locations), constants.MAX_SPATIAL_BOOLEAN_ARRAY_MEMORY_GB
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
        desc="Computing spatial updates",
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
                vs30=float(chunk_affected_vs30[i]),
                stdv=float(chunk_affected_stdv[i]),
                index=flat_idx,
            )

            update_result = compute_mvn_update_for_pixel(
                pixel,
                obs_data,
                model_name,
                phi=phi,
                max_dist_m=max_dist_m,
                max_points=max_points,
                noisy=noisy,
                cov_reduc=cov_reduc,
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


# ============================================================================
# Point-Based MVN Computation
# ============================================================================


def compute_mvn_at_points(
    points: np.ndarray,
    model_vs30: np.ndarray,
    model_stdv: np.ndarray,
    obs_locations: np.ndarray,
    obs_vs30: np.ndarray,
    obs_model_vs30: np.ndarray,
    obs_model_stdv: np.ndarray,
    obs_uncertainty: np.ndarray,
    model_type: str,
    phi: float | None = None,
    max_dist_m: float = constants.MAX_DIST_M,
    max_points: int = constants.MAX_POINTS,
    noisy: bool = constants.NOISY,
    cov_reduc: float = constants.COV_REDUC,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MVN spatial adjustment at specific query points.

    This is the point-based equivalent of compute_mvn_updates(). It applies
    the same MVN conditioning algorithm but for arbitrary query points instead
    of raster pixels.

    Parameters
    ----------
    points : np.ndarray
        (N, 2) array of [easting, northing] query locations in NZTM.
    model_vs30 : np.ndarray
        (N,) array of model Vs30 values at query points (before MVN adjustment).
    model_stdv : np.ndarray
        (N,) array of model standard deviation at query points.
    obs_locations : np.ndarray
        (M, 2) array of observation [easting, northing] locations.
    obs_vs30 : np.ndarray
        (M,) array of measured Vs30 values at observations.
    obs_model_vs30 : np.ndarray
        (M,) array of model Vs30 values at observation locations.
    obs_model_stdv : np.ndarray
        (M,) array of model standard deviation at observation locations.
    obs_uncertainty : np.ndarray
        (M,) array of observation uncertainties.
    model_type : str
        Either "geology" or "terrain" (determines phi correlation length).
    phi : float, optional
        Correlation length parameter. If None, uses value from constants.PHI[model_type].
    max_dist_m : float, optional
        Maximum distance (meters) to consider observations. Default from constants.
    max_points : int, optional
        Maximum number of observations per point. Default from constants.
    noisy : bool, optional
        Whether to apply noise weighting. Default from constants.
    cov_reduc : float, optional
        Covariance reduction factor. Default from constants.

    Returns
    -------
    mvn_vs30 : np.ndarray
        (N,) array of spatially adjusted Vs30 values.
    mvn_stdv : np.ndarray
        (N,) array of spatially adjusted standard deviation values.

    Notes
    -----
    The algorithm for each query point:
    1. Find observations within max_dist_m (limited to max_points closest)
    2. Compute log residuals: log(obs_vs30 / obs_model_vs30)
    3. Build covariance matrix using exponential correlation function
    4. Apply MVN conditioning to get posterior mean and variance
    5. Convert back from log-space to linear Vs30
    """
    if phi is None:
        phi = constants.PHI[model_type]

    n_points = len(points)
    n_obs = len(obs_locations)

    # Initialize output arrays with prior values
    mvn_vs30 = model_vs30.copy()
    mvn_stdv = model_stdv.copy()

    if n_obs == 0:
        logger.warning("No observations provided for MVN adjustment")
        return mvn_vs30, mvn_stdv

    # Compute log residuals at observation locations
    # Filter out invalid observations (NaN model values)
    valid_obs_mask = (
        ~np.isnan(obs_model_vs30)
        & ~np.isnan(obs_model_stdv)
        & (obs_model_vs30 > 0)
        & (obs_model_stdv > 0)
    )

    if not np.any(valid_obs_mask):
        logger.warning("No valid observations for MVN adjustment")
        return mvn_vs30, mvn_stdv

    # Filter to valid observations
    obs_locs = obs_locations[valid_obs_mask]
    obs_vs30_valid = obs_vs30[valid_obs_mask]
    obs_model_vs30_valid = obs_model_vs30[valid_obs_mask]
    obs_model_stdv_valid = obs_model_stdv[valid_obs_mask]
    obs_uncertainty_valid = obs_uncertainty[valid_obs_mask]

    # Compute residuals
    obs_residuals = np.log(obs_vs30_valid / obs_model_vs30_valid)

    # Apply noise weighting if enabled
    if noisy:
        omega_obs = np.sqrt(
            obs_model_stdv_valid**2
            / (obs_model_stdv_valid**2 + obs_uncertainty_valid**2)
        )
        obs_residuals = obs_residuals * omega_obs
    else:
        omega_obs = np.ones(len(obs_residuals))

    # Compute correlation at distance 0 for default variance
    corr_zero = utils.correlation_function(np.array([0.0]), phi)[0]

    # Process each query point
    for i in tqdm(range(n_points), desc="Computing MVN at points", unit="point"):
        point = points[i]
        prior_vs30 = model_vs30[i]
        prior_stdv = model_stdv[i]

        # Skip invalid points
        if np.isnan(prior_vs30) or np.isnan(prior_stdv) or prior_vs30 <= 0 or prior_stdv <= 0:
            continue

        # Calculate distances from this point to all observations
        distances = np.sqrt(np.sum((obs_locs - point) ** 2, axis=1))

        # Find nearby observations
        nearby_mask = distances <= max_dist_m

        if not np.any(nearby_mask):
            # No nearby observations - apply default variance shrinkage (matching legacy)
            mvn_stdv[i] = sqrt(prior_stdv**2 * corr_zero)
            continue

        # Limit to max_points closest observations
        nearby_indices = np.where(nearby_mask)[0]
        if len(nearby_indices) > max_points:
            nearby_distances = distances[nearby_indices]
            closest_indices = np.argsort(nearby_distances)[:max_points]
            nearby_indices = nearby_indices[closest_indices]

        # Extract data for nearby observations
        nearby_locs = obs_locs[nearby_indices]
        nearby_residuals = obs_residuals[nearby_indices]
        nearby_model_stdv = obs_model_stdv_valid[nearby_indices]
        nearby_omega = omega_obs[nearby_indices]
        nearby_model_vs30 = obs_model_vs30_valid[nearby_indices]

        # Build covariance matrix
        # First element is query point, rest are nearby observations
        all_locs = np.vstack([point, nearby_locs])
        dist_matrix = utils.euclidean_distance_matrix(all_locs)

        # Apply correlation function
        corr_matrix = utils.correlation_function(dist_matrix, phi)

        # Scale by standard deviations to get covariance
        stdv_vector = np.concatenate([[prior_stdv], nearby_model_stdv])
        cov_matrix = corr_matrix * np.outer(stdv_vector, stdv_vector)

        # Apply noise weighting if enabled
        if noisy:
            omega_vector = np.concatenate([[1.0], nearby_omega])
            omega_matrix = np.outer(omega_vector, omega_vector)
            np.fill_diagonal(omega_matrix, 1.0)
            cov_matrix *= omega_matrix

        # Apply covariance reduction for dissimilar Vs30 values
        if cov_reduc > 0:
            vs30_vector = np.concatenate([[prior_vs30], nearby_model_vs30])
            log_vs30_dist = np.abs(
                np.log(vs30_vector[:, np.newaxis]) - np.log(vs30_vector)
            )
            cov_matrix *= np.exp(-cov_reduc * log_vs30_dist)

        # MVN conditioning
        # Partition: C = [[C_pp, C_po], [C_op, C_oo]]
        C_pp = cov_matrix[0, 0]  # point-point covariance
        C_po = cov_matrix[0, 1:]  # point-observation covariance
        C_oo = cov_matrix[1:, 1:]  # observation-observation covariance

        try:
            C_oo_inv = np.linalg.inv(C_oo)

            # Posterior mean adjustment: C_po @ inv(C_oo) @ residuals
            pred_adjustment = C_po @ C_oo_inv @ nearby_residuals

            # Posterior variance: C_pp - C_po @ inv(C_oo) @ C_op
            var_reduction = C_po @ C_oo_inv @ C_po
            posterior_var = C_pp - var_reduction

            # Update vs30 in log-space, then convert back
            log_vs30_posterior = np.log(prior_vs30) + pred_adjustment
            mvn_vs30[i] = np.exp(log_vs30_posterior)
            mvn_stdv[i] = sqrt(max(0, posterior_var))

        except np.linalg.LinAlgError:
            # Singular matrix - keep prior values with default variance shrinkage
            mvn_stdv[i] = sqrt(prior_stdv**2 * corr_zero)
            logger.debug(f"Singular covariance matrix at point {i}, keeping prior values")

    return mvn_vs30, mvn_stdv

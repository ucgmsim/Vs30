"""Multivariate Normal (MVN) distribution-based spatial adjustment of Vs30 using nearby observations."""

import logging
import multiprocessing as mp
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import scipy
from tqdm import tqdm

from vs30 import category, constants, raster, utils

# Use spawn context to avoid GDAL fork issues
_spawn_context = mp.get_context("spawn")

logger = logging.getLogger(__name__)


@dataclass
class ObservationData:
    """
    Bundled observation data for spatial processing.

    Attributes
    ----------
    locations : ndarray
        (n_obs, 2) array of [easting, northing] coordinates.
    vs30 : ndarray
        (n_obs,) measured Vs30 values.
    model_vs30 : ndarray
        (n_obs,) model Vs30 at observation locations.
    model_stdv : ndarray
        (n_obs,) model standard deviation at observation locations.
    residuals : ndarray
        (n_obs,) log residuals: log(vs30 / model_vs30).
    omega : ndarray
        (n_obs,) noise weights (if noisy=True).
    uncertainty : ndarray
        (n_obs,) observation uncertainties.
    cluster_labels : ndarray or None
        (n_obs,) cluster labels from DBSCAN, -1 = unclustered.
    """

    locations: np.ndarray
    vs30: np.ndarray
    model_vs30: np.ndarray
    model_stdv: np.ndarray
    residuals: np.ndarray
    omega: np.ndarray
    uncertainty: np.ndarray
    cluster_labels: np.ndarray | None = None

    @classmethod
    def empty(cls) -> "ObservationData":
        """
        Create an empty ObservationData object with zero observations.

        Returns
        -------
        ObservationData
            An ObservationData instance with zero-length arrays.
        """
        return cls(
            locations=np.empty((0, 2)),
            vs30=np.empty(0),
            model_vs30=np.empty(0),
            model_stdv=np.empty(0),
            residuals=np.empty(0),
            omega=np.empty(0),
            uncertainty=np.empty(0),
            cluster_labels=None,
        )


@dataclass
class PixelData:
    """
    Data for a single pixel.

    Attributes
    ----------
    location : ndarray
        [easting, northing] coordinates.
    vs30 : float
        Prior Vs30 value.
    stdv : float
        Prior standard deviation value.
    index : int
        Flat index in the raster.
    """

    location: np.ndarray
    vs30: float
    stdv: float
    index: int


@dataclass
class SpatialAdjustmentResult:
    """
    Result of an MVN update for a single pixel.

    Attributes
    ----------
    updated_vs30 : float
        Updated Vs30 value after spatial adjustment.
    updated_stdv : float
        Updated standard deviation after spatial adjustment.
    n_observations_used : int
        Number of observations used in the MVN conditioning.
    pixel_index : int
        Flat index of this pixel in the raster.
    """

    updated_vs30: float
    updated_stdv: float
    n_observations_used: int
    pixel_index: int


def _compute_valid_mask(
    vs30: np.ndarray, stdv: np.ndarray, nodata: float | None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the boolean mask of valid pixels and their flat indices.

    Valid pixels are those that are not nodata, not NaN, and positive
    in both the VS30 and standard deviation arrays.

    Parameters
    ----------
    vs30 : ndarray
        2D array of Vs30 mean values.
    stdv : ndarray
        2D array of Vs30 standard deviation values.
    nodata : float or None
        No-data sentinel value.

    Returns
    -------
    tuple[ndarray, ndarray]
        (valid_mask, valid_flat_indices) where valid_mask is a boolean 2D
        array and valid_flat_indices is a 1D array of flat indices.
    """
    valid_mask = (
        (vs30 != nodata)
        & (~np.isnan(vs30))
        & (~np.isnan(stdv))
        & (vs30 > 0)
        & (stdv > 0)
    )
    valid_flat_indices = np.where(valid_mask.flatten())[0]
    return valid_mask, valid_flat_indices


@dataclass
class RasterData:
    """
    Raster data and metadata.

    Attributes
    ----------
    vs30 : ndarray
        Band 1: Vs30 mean values (2D array).
    stdv : ndarray
        Band 2: Vs30 standard deviation values (2D array).
    transform : rasterio.transform.Affine
        Affine transformation for coordinate conversion.
    crs : rasterio.crs.CRS
        Coordinate reference system.
    nodata : float or None
        No-data value used in the raster.
    valid_mask : ndarray
        Boolean mask of non-nodata pixels (2D array).
    valid_flat_indices : ndarray
        Flat indices of non-nodata pixels (1D array).
    """

    vs30: np.ndarray
    stdv: np.ndarray
    transform: rasterio.transform.Affine
    crs: rasterio.crs.CRS
    nodata: float | None
    valid_mask: np.ndarray
    valid_flat_indices: np.ndarray

    @classmethod
    def from_file(cls, path: Path) -> "RasterData":
        """
        Load 2-band VS30 raster.

        Parameters
        ----------
        path : Path
            Path to the 2-band raster file (band 1: mean Vs30, band 2: standard deviation of Vs30).

        Returns
        -------
        RasterData
            Loaded raster data with valid pixel mask.
        """
        with rasterio.open(path) as src:
            vs30 = src.read(1)
            stdv = src.read(2)
            transform = src.transform
            crs = src.crs
            nodata = src.nodata

            valid_mask, valid_flat_indices = _compute_valid_mask(vs30, stdv, nodata)

            return cls(
                vs30=vs30,
                stdv=stdv,
                transform=transform,
                crs=crs,
                nodata=nodata,
                valid_mask=valid_mask,
                valid_flat_indices=valid_flat_indices,
            )

    @classmethod
    def from_arrays(
        cls,
        vs30: np.ndarray,
        stdv: np.ndarray,
        transform: rasterio.transform.Affine,
        crs=constants.NZTM_CRS,
        nodata: float = constants.NODATA_VALUE,
    ) -> "RasterData":
        """
        Create RasterData from in-memory arrays.

        Parameters
        ----------
        vs30 : ndarray
            2D array of Vs30 mean values.
        stdv : ndarray
            2D array of Vs30 standard deviation values.
        transform : rasterio.transform.Affine
            Affine transformation for coordinate conversion.
        crs : str or rasterio.crs.CRS, optional
            Coordinate reference system. Default is NZTM (EPSG:2193).
            If a string is provided, it is converted to a CRS object.
        nodata : float, optional
            No-data value. Default from constants.NODATA_VALUE.

        Returns
        -------
        RasterData
            Raster data with valid pixel mask computed from the arrays.
        """
        if isinstance(crs, str):
            crs = rasterio.crs.CRS.from_string(crs)

        valid_mask, valid_flat_indices = _compute_valid_mask(vs30, stdv, nodata)

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
        x_scale = self.transform[0]
        x_origin = self.transform[2]
        y_scale = self.transform[4]
        y_origin = self.transform[5]

        # Pixel centers: add offset to row/col indices (following legacy implementation)
        cols_center = valid_cols.astype(float) + constants.PIXEL_CENTER_OFFSET
        rows_center = valid_rows.astype(float) + constants.PIXEL_CENTER_OFFSET

        xs = x_origin + cols_center * x_scale
        ys = y_origin + rows_center * y_scale

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
            driver=constants.GEOTIFF_DRIVER,
            height=self.vs30.shape[0],
            width=self.vs30.shape[1],
            count=2,
            dtype=self.vs30.dtype,
            crs=self.crs,
            transform=self.transform,
            nodata=self.nodata,
            compress=constants.GEOTIFF_COMPRESSION,
            tiled=constants.GEOTIFF_TILED,
            bigtiff=constants.GEOTIFF_BIGTIFF,
        ) as dst:
            dst.write(updated_vs30, 1)
            dst.write(updated_stdv, 2)


@dataclass
class BoundingBoxResult:
    """
    Result of bounding box search for affected pixels.

    Attributes
    ----------
    mask : ndarray
        Boolean mask of pixels in any observation's bounding box.
    obs_to_grid_indices : list[ndarray]
        For each observation, flat indices of pixels in its bounding box.
    n_affected_pixels : int
        Total number of pixels affected by at least one observation.
    """

    mask: np.ndarray
    obs_to_grid_indices: list[np.ndarray]
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
    ValueError
        If raster data is invalid.
    """
    if raster_data.vs30.shape != raster_data.stdv.shape:
        raise ValueError("Band shapes must match")
    if not np.all(np.isfinite(raster_data.vs30[raster_data.valid_mask])):
        raise ValueError("Valid pixels must be finite")
    if not np.all(raster_data.vs30[raster_data.valid_mask] > 0):
        raise ValueError("Vs30 must be positive")
    if not np.all(raster_data.stdv[raster_data.valid_mask] > 0):
        raise ValueError("Stdv must be positive")


def validate_observations(observations: pd.DataFrame) -> None:
    """
    Validate observation data.

    Parameters
    ----------
    observations : DataFrame
        Observation data.

    Raises
    ------
    ValueError
        If observation data is invalid.
    """
    missing = [
        col
        for col in constants.REQUIRED_OBSERVATION_COLUMNS
        if col not in observations.columns
    ]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    if not np.all(observations[constants.COL_VS30] > 0):
        raise ValueError("Vs30 must be positive")
    if not np.all(observations[constants.COL_UNCERTAINTY] > 0):
        raise ValueError("Uncertainty must be positive")


def prepare_observation_data(
    observations: pd.DataFrame,
    raster_data: RasterData,
    updated_model_table: np.ndarray,
    model_type: constants.ModelType,
    output_dir: Path | None = None,
    noisy: bool = False,
    slope_array: np.ndarray | None = None,
    coast_dist_array: np.ndarray | None = None,
) -> ObservationData:
    """
    Prepare observation data for MVN processing.

    For geology models, hybrid modifications (slope and coastal distance)
    are applied to model values at observation locations before computing
    residuals.

    Slope and coastal distance data can be provided either as in-memory arrays
    (slope_array, coast_dist_array) or read from files in output_dir. When
    arrays are provided, observation locations are sampled using the raster
    transform instead of reading from disk.

    Parameters
    ----------
    observations : DataFrame
        Observations with vs30, uncertainty, easting, northing columns.
    raster_data : RasterData
        Raster data object (used for transform/profile info).
    updated_model_table : ndarray
        Updated model table (n_categories, 2) array of [vs30, stdv].
    model_type : constants.ModelType
        Model type (ModelType.GEOLOGY or ModelType.TERRAIN).
    output_dir : Path or None, optional
        Output directory for intermediate rasters (slope, coast distance).
        Required for geology models when slope_array and coast_dist_array
        are not provided.
    noisy : bool, optional
        Whether to apply noise weighting. Default is False.
    slope_array : ndarray or None, optional
        In-memory 2D slope array. When provided together with
        coast_dist_array, observation values are sampled from these arrays
        using the raster transform instead of reading from files.
    coast_dist_array : ndarray or None, optional
        In-memory 2D coastal distance array. When provided together with
        slope_array, observation values are sampled from these arrays
        using the raster transform instead of reading from files.

    Returns
    -------
    ObservationData
        Prepared observation data object.

    Raises
    ------
    ValueError
        If model_type is GEOLOGY and neither in-memory arrays nor output_dir
        are provided.
    """
    # Validate inputs for geology models
    has_in_memory_arrays = slope_array is not None and coast_dist_array is not None
    if model_type == constants.ModelType.GEOLOGY and not has_in_memory_arrays and output_dir is None:
        raise ValueError(
            "For geology models, either provide both slope_array and "
            "coast_dist_array, or provide output_dir for file-based access."
        )

    # Get observation locations
    obs_locs = observations[[constants.COL_EASTING, constants.COL_NORTHING]].values

    # Interpolate model values at observation locations
    if model_type == constants.ModelType.GEOLOGY:
        model_ids = category.assign_to_category_geology(obs_locs)
    elif model_type == constants.ModelType.TERRAIN:
        model_ids = category.assign_to_category_terrain(obs_locs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Get model vs30 and stdv from updated model table
    # Model IDs are 1-indexed in the raster, but 0-indexed in the model table
    # constants.RASTER_ID_NODATA_VALUE is 255, valid IDs are 1-15 for geology, 1-16 for terrain
    valid_mask = (
        (model_ids != constants.RASTER_ID_NODATA_VALUE)
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
    vs30_obs = observations[constants.COL_VS30].values[valid_obs_mask]
    model_vs30 = model_vs30[valid_obs_mask]
    model_stdv = model_stdv[valid_obs_mask]
    uncertainty = observations[constants.COL_UNCERTAINTY].values[valid_obs_mask]

    # Calculate log residuals
    # For geology, we must apply hybrid modifications to model values at observation points
    if model_type == constants.ModelType.GEOLOGY:
        if has_in_memory_arrays:
            # Convert observation coordinates to grid pixel indices
            rows, cols = rasterio.transform.rowcol(
                raster_data.transform, obs_locs[:, 0], obs_locs[:, 1]
            )
            rows = np.asarray(rows)
            cols = np.asarray(cols)

            # Determine which observations fall within the grid domain
            within_grid = (
                (rows >= 0)
                & (rows < slope_array.shape[0])
                & (cols >= 0)
                & (cols < slope_array.shape[1])
            )

            # Observations within the grid domain: sample slope and coastal
            # distance from the grid arrays for consistency with the
            # grid-resampled values used in pixel updates.
            # Observations outside the grid domain: sample slope and coastal
            # distance from the original source rasters, since there are no
            # corresponding grid pixels to be consistent with.
            slope_obs = np.empty(len(rows), dtype=np.float64)
            coast_obs = np.empty(len(rows), dtype=np.float64)

            slope_obs[within_grid] = slope_array[rows[within_grid], cols[within_grid]]
            coast_obs[within_grid] = coast_dist_array[
                rows[within_grid], cols[within_grid]
            ]

            if not np.all(within_grid):
                outside_grid_points = obs_locs[~within_grid]
                slope_obs[~within_grid] = raster.sample_slope_at_points(
                    outside_grid_points
                )
                coast_obs[~within_grid] = (
                    raster.compute_coastal_distance_at_points(outside_grid_points)
                )
        else:
            # File-based path: read slope and coast distance from rasters
            slope_path = output_dir / constants.SLOPE_RASTER_FILENAME
            coast_path = output_dir / constants.COAST_DISTANCE_RASTER_FILENAME

            profile = {
                "transform": raster_data.transform,
                "width": raster_data.vs30.shape[1],
                "height": raster_data.vs30.shape[0],
                "crs": rasterio.crs.CRS.from_string(constants.NZTM_CRS),
            }

            if not slope_path.exists():
                raster.create_slope_raster(slope_path, profile)
            if not coast_path.exists():
                raster.create_coast_distance_raster(coast_path, profile)

            # Sample resampled rasters at observation locations.
            # Note: this introduces minor precision loss because the rasters are
            # resampled to grid resolution, while observations are at arbitrary
            # coordinates. For higher precision, these could be replaced with
            # raster.sample_slope_at_points() and
            # raster.compute_coastal_distance_at_points(), which use the source
            # data directly. The effect on Vs30 is small (<2%), but posterior
            # stdv can differ more (~25%) due to sensitivity in MVN conditioning.
            # We keep it this way so that the grid pipeline consistently uses the
            # same resampled raster values for both pixel updates and observation
            # residuals.
            with rasterio.open(slope_path) as src:
                slope_obs = np.array([v[0] for v in src.sample(obs_locs)])
            with rasterio.open(coast_path) as src:
                coast_obs = np.array([v[0] for v in src.sample(obs_locs)])

        # Apply modifications to model_vs30 and model_stdv at points
        model_vs30, model_stdv = raster.apply_hybrid_geology_modifications(
            model_vs30,
            model_stdv,
            model_ids[valid_obs_mask],
            slope_obs,
            coast_obs,
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
    grid_locs: np.ndarray,
    obs_eastings_min: np.ndarray,
    obs_eastings_max: np.ndarray,
    obs_northings_min: np.ndarray,
    obs_northings_max: np.ndarray,
    start_grid_idx: int = 0,
) -> tuple[np.ndarray, list[np.ndarray]]:
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
    obs_to_grid_indices = []

    for obs_idx in range(in_bbox.shape[0]):
        # Get grid indices within this observation's bounding box
        # Convert to full grid indices by adding start_grid_idx
        obs_to_grid_indices.append(np.where(in_bbox[obs_idx])[0] + start_grid_idx)

    return chunk_mask, obs_to_grid_indices


def calculate_chunk_size(n_obs: int, max_spatial_boolean_array_memory_gb: float) -> int:
    """
    Calculate the max grid points per chunk given a memory budget for the mask of grid points inside observation bounding boxes.

    Parameters
    ----------
    n_obs : int
        Number of observations.
    max_spatial_boolean_array_memory_gb : float
        Maximum memory in GB allocated for spatial boolean arrays during chunked processing.

    Returns
    -------
    int
        Maximum number of grid points per chunk.
    """
    return max(1, int(max_spatial_boolean_array_memory_gb * 1024**3 / n_obs))


def process_bbox_chunk(args: tuple) -> tuple[int, np.ndarray, list[np.ndarray]]:
    """
    Worker function for parallel bounding box processing.

    Processes a single chunk of grid points to find which are affected by observations.

    Parameters
    ----------
    args : tuple
        (chunk_idx, grid_locs_chunk, start_idx, obs_bounds)
        where obs_bounds is (obs_eastings_min, obs_eastings_max,
                            obs_northings_min, obs_northings_max)

    Returns
    -------
    tuple
        (chunk_idx, chunk_mask, obs_to_grid_indices)
    """
    chunk_idx, grid_locs_chunk, start_idx, obs_bounds = args
    obs_eastings_min, obs_eastings_max, obs_northings_min, obs_northings_max = (
        obs_bounds
    )

    chunk_mask, obs_to_grid_indices = grid_points_in_bbox(
        grid_locs=grid_locs_chunk,
        obs_eastings_min=obs_eastings_min,
        obs_eastings_max=obs_eastings_max,
        obs_northings_min=obs_northings_min,
        obs_northings_max=obs_northings_max,
        start_grid_idx=start_idx,
    )

    return chunk_idx, chunk_mask, obs_to_grid_indices


def build_covariance_matrix(
    pixel: PixelData,
    selected_observations: ObservationData,
    model_type: constants.ModelType,
    noisy: bool = False,
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
    model_type : constants.ModelType
        Model type (ModelType.GEOLOGY or ModelType.TERRAIN).
    noisy : bool, optional
        Whether to apply noise weighting based on observation uncertainty.
    cov_reduc : float, optional
        Covariance reduction factor for dissimilar Vs30 values.

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
    distance_matrix = scipy.spatial.distance.cdist(
        all_points, all_points, metric="euclidean"
    )

    # Step 2: Apply correlation function
    corr = utils.correlation_function(distance_matrix, constants.PHI[model_type])

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
    # Calculate distances from pixel to all observations
    distances = np.sqrt(np.sum((obs_data.locations - pixel.location) ** 2, axis=1))

    # Select observations using distance-based filtering
    max_points_i = min(max_points, len(distances)) - 1
    if max_points_i < 0:
        return ObservationData.empty()

    min_dist, cutoff_dist = np.partition(distances, [0, max_points_i])[
        [0, max_points_i]
    ]
    if min_dist > max_dist_m:
        # Not close enough to any observed locations
        return ObservationData.empty()

    # Include all observations within cutoff distance (may exceed max_points for accuracy)
    loc_mask = distances <= min(max_dist_m, cutoff_dist)
    filtered_indices = np.where(loc_mask)[0]

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


def compute_spatial_adjustment_for_pixel(
    pixel: PixelData,
    obs_data: ObservationData,
    model_type: constants.ModelType,
    max_dist_m: float = constants.MAX_DIST_M,
    max_points: int = constants.MAX_POINTS,
    noisy: bool = False,
    cov_reduc: float = constants.COV_REDUC,
    corr_zero: float | None = None,
) -> SpatialAdjustmentResult | None:
    """
    Compute MVN update for a single pixel.

    Parameters
    ----------
    pixel : PixelData
        Pixel data.
    obs_data : ObservationData
        Full observation data.
    model_type : constants.ModelType
        Model type (ModelType.GEOLOGY or ModelType.TERRAIN).
    max_dist_m : float, optional
        Maximum distance in meters to consider observations.
    max_points : int, optional
        Maximum number of observations to select per pixel.
    noisy : bool, optional
        Whether to apply noise weighting based on observation uncertainty.
    cov_reduc : float, optional
        Covariance reduction factor for dissimilar Vs30 values.
    corr_zero : float or None, optional
        Pre-computed correlation at zero distance. If None, computed from
        model_type. Pass this when calling in a loop to avoid recomputing.

    Returns
    -------
    SpatialAdjustmentResult or None
        Update result, or None if pixel should be skipped.
    """
    # Handle NaN/NoData/invalid pixels
    if np.isnan(pixel.vs30) or np.isnan(pixel.stdv) or pixel.vs30 <= 0 or pixel.stdv <= 0:
        return None

    # Correlation at zero distance is slightly less than 1.0 due to the
    # enforced minimum distance (nugget effect). This shrinks the prior
    # variance to match the legacy implementation's behavior.
    if corr_zero is None:
        corr_zero = utils.correlation_function(
            np.array([0.0]), constants.PHI[model_type]
        )[0]
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
        return SpatialAdjustmentResult(
            updated_vs30=pixel.vs30,
            updated_stdv=np.sqrt(initial_var),
            n_observations_used=0,
            pixel_index=pixel.index,
        )

    # Build covariance matrix
    cov_matrix = build_covariance_matrix(
        pixel,
        selected_obs,
        model_type,
        noisy=noisy,
        cov_reduc=cov_reduc,
    )

    try:
        inv_cov = np.linalg.inv(cov_matrix[1:, 1:])

        pred_update = np.dot(
            np.dot(cov_matrix[0, 1:], inv_cov),
            selected_obs.residuals,
        )

        var = cov_matrix[0, 0] - np.dot(
            np.dot(cov_matrix[0, 1:], inv_cov), cov_matrix[1:, 0]
        )

        return SpatialAdjustmentResult(
            updated_vs30=float(pixel.vs30 * np.exp(pred_update)),
            updated_stdv=float(np.sqrt(max(0, var))),
            n_observations_used=len(selected_obs.locations),
            pixel_index=pixel.index,
        )
    except np.linalg.LinAlgError:
        # Singular covariance matrix — keep prior values with default variance shrinkage
        logger.debug(
            f"Singular covariance matrix at pixel {pixel.index}, keeping prior values"
        )
        return SpatialAdjustmentResult(
            updated_vs30=pixel.vs30,
            updated_stdv=np.sqrt(initial_var),
            n_observations_used=0,
            pixel_index=pixel.index,
        )


def find_affected_pixels(
    raster_data: RasterData,
    obs_data: ObservationData,
    max_spatial_boolean_array_memory_gb: float,
    model_type: constants.ModelType,
    max_dist_m: float = constants.MAX_DIST_M,
    n_proc: int = 1,
) -> BoundingBoxResult:
    """
    Find pixels affected by observations using bounding boxes.

    Parameters
    ----------
    raster_data : RasterData
        Raster data object.
    obs_data : ObservationData
        Observation data.
    max_spatial_boolean_array_memory_gb : float
        Memory limit (GB) for boolean arrays in spatial processing.
    model_type : constants.ModelType
        Model type (ModelType.GEOLOGY or ModelType.TERRAIN), used for
        progress bar labelling.
    max_dist_m : float, optional
        Maximum distance for considering observations.
    n_proc : int, optional
        Number of parallel processes. 1 for sequential (default),
        >1 for parallel processing.

    Returns
    -------
    BoundingBoxResult
        Result containing mask and observation-to-grid mappings.
    """
    # Get coordinates for valid pixels
    grid_locs = raster_data.get_coordinates()

    n_obs = len(obs_data.locations)

    # Calculate chunk size based on observation count
    chunk_size = calculate_chunk_size(n_obs, max_spatial_boolean_array_memory_gb)
    n_chunks = int(np.ceil(len(grid_locs) / chunk_size))

    # Precompute observation bounds
    obs_eastings = obs_data.locations[:, 0:1]  # (n_obs, 1)
    obs_northings = obs_data.locations[:, 1:2]  # (n_obs, 1)
    obs_eastings_min = obs_eastings - max_dist_m
    obs_eastings_max = obs_eastings + max_dist_m
    obs_northings_min = obs_northings - max_dist_m
    obs_northings_max = obs_northings + max_dist_m

    # Bundle observation bounds for passing to workers
    obs_bounds = (
        obs_eastings_min,
        obs_eastings_max,
        obs_northings_min,
        obs_northings_max,
    )

    # Initialize mask and obs_to_grid_indices
    valid_points_in_bbox_mask = np.zeros(len(grid_locs), dtype=bool)
    obs_to_grid_indices = [np.array([], dtype=np.int64) for _ in range(n_obs)]

    logger.info(f"Processing {n_chunks} chunks of {chunk_size:,} pixels each")

    label = str(model_type).capitalize()

    # Prepare chunk arguments
    chunk_args = []
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(grid_locs))
        grid_locs_chunk = grid_locs[start_idx:end_idx]
        chunk_args.append((chunk_idx, grid_locs_chunk, start_idx, obs_bounds))

    if n_proc > 1 and n_chunks > 1:
        # Parallel processing
        actual_n_proc = min(n_proc, n_chunks)
        logger.info(f"Using {actual_n_proc} parallel workers")
        with _spawn_context.Pool(processes=actual_n_proc) as pool:
            results = list(
                tqdm(
                    pool.imap(process_bbox_chunk, chunk_args),
                    total=n_chunks,
                    desc=f"{label}: checking pixels for nearby observations ({n_chunks} chunks)",
                    unit="chunk",
                )
            )

    elif n_chunks > 1:
        # Sequential processing with multiple chunks
        results = []
        for chunk_idx in tqdm(
            range(n_chunks),
            desc=f"{label}: checking pixels for nearby observations ({n_chunks} chunks)",
            unit="chunk",
        ):
            results.append(process_bbox_chunk(chunk_args[chunk_idx]))

    else:
        # Single chunk — no progress bar needed
        print(
            f"{label}: checking {len(grid_locs):,} pixels for nearby observations... ",
            end="",
            flush=True,
        )
        results = [process_bbox_chunk(chunk_args[0])]
        print("done")

    # Merge results from either parallel or sequential processing
    for chunk_idx, chunk_mask, chunk_obs_to_grid in results:
        start_idx = chunk_idx * chunk_size
        valid_points_in_bbox_mask[start_idx : start_idx + len(chunk_mask)] = chunk_mask
        for obs_idx, grid_indices in enumerate(chunk_obs_to_grid):
            if len(grid_indices) > 0:
                full_raster_indices = raster_data.valid_flat_indices[grid_indices]
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


def compute_spatial_adjustments(
    raster_data: RasterData,
    obs_data: ObservationData,
    bbox_result: BoundingBoxResult,
    model_type: constants.ModelType,
    max_spatial_boolean_array_memory_gb: float,
    max_dist_m: float = constants.MAX_DIST_M,
    max_points: int = constants.MAX_POINTS,
    noisy: bool = False,
    cov_reduc: float = constants.COV_REDUC,
) -> list[SpatialAdjustmentResult]:
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
    model_type : constants.ModelType
        Model type (ModelType.GEOLOGY or ModelType.TERRAIN).
    max_spatial_boolean_array_memory_gb : float
        Memory limit (GB) for boolean arrays in spatial processing.
    max_dist_m : float, optional
        Maximum distance in meters to consider observations.
    max_points : int, optional
        Maximum number of observations to select per pixel.
    noisy : bool, optional
        Whether to apply noise weighting based on observation uncertainty.
    cov_reduc : float, optional
        Covariance reduction factor for dissimilar Vs30 values.

    Returns
    -------
    list
        List of SpatialAdjustmentResult objects.
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
        len(obs_data.locations), max_spatial_boolean_array_memory_gb
    )
    n_chunks = int(np.ceil(len(affected_flat_indices) / chunk_size))

    all_updates = []

    logger.info(
        f"Processing {n_chunks} chunks of up to {chunk_size:,} pixels each "
        f"({len(affected_flat_indices):,} total pixels to update)"
    )

    # Pre-compute correlation at zero distance (constant for all pixels)
    corr_zero = utils.correlation_function(
        np.array([0.0]), constants.PHI[model_type]
    )[0]

    # Process all affected pixels with a single progress bar
    label = str(model_type).capitalize()
    with tqdm(
        total=len(affected_flat_indices),
        desc=f"{label}: spatial adjustment",
        unit="pixel",
    ) as pbar:
        for chunk_idx in range(n_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, len(affected_flat_indices))

            chunk_flat_indices = affected_flat_indices[start_idx:end_idx]
            chunk_affected_locs = affected_locs[start_idx:end_idx]
            chunk_affected_vs30 = affected_vs30[start_idx:end_idx]
            chunk_affected_stdv = affected_stdv[start_idx:end_idx]

            for i, flat_idx in enumerate(chunk_flat_indices):
                pixel = PixelData(
                    location=chunk_affected_locs[i],
                    vs30=float(chunk_affected_vs30[i]),
                    stdv=float(chunk_affected_stdv[i]),
                    index=flat_idx,
                )

                update_result = compute_spatial_adjustment_for_pixel(
                    pixel,
                    obs_data,
                    model_type,
                    max_dist_m=max_dist_m,
                    max_points=max_points,
                    noisy=noisy,
                    cov_reduc=cov_reduc,
                    corr_zero=corr_zero,
                )

                if update_result is not None:
                    all_updates.append(update_result)

                pbar.update(1)

    logger.info(f"Completed processing all chunks: {len(all_updates):,} pixels updated")

    return all_updates


def apply_updates(
    raster_data: RasterData,
    updates: list[SpatialAdjustmentResult],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply spatial adjustment updates to raster arrays without writing to disk.

    Parameters
    ----------
    raster_data : RasterData
        Raster data object.
    updates : list
        List of SpatialAdjustmentResult objects.

    Returns
    -------
    tuple of ndarray
        (updated_vs30, updated_stdv) arrays with updates applied.
    """
    # Initialize output arrays with original values
    updated_vs30 = raster_data.vs30.copy()
    updated_stdv = raster_data.stdv.copy()

    # Apply updates
    for update in updates:
        updated_vs30.flat[update.pixel_index] = update.updated_vs30
        updated_stdv.flat[update.pixel_index] = update.updated_stdv

    return updated_vs30, updated_stdv


def apply_and_write_updates(
    raster_data: RasterData,
    updates: list[SpatialAdjustmentResult],
    model_type: constants.ModelType,
    output_dir: Path,
) -> None:
    """
    Apply updates to raster and write output file.

    Parameters
    ----------
    raster_data : RasterData
        Raster data object.
    updates : list
        List of SpatialAdjustmentResult objects.
    model_type : constants.ModelType
        Model type (ModelType.GEOLOGY or ModelType.TERRAIN).
    output_dir : Path
        Directory where output raster will be saved.
    """
    updated_vs30, updated_stdv = apply_updates(raster_data, updates)

    # Write output using filename from constants
    output_path = output_dir / constants.OUTPUT_FILENAMES[model_type]

    raster_data.write_updated(output_path, updated_vs30, updated_stdv)

    logger.info(
        f"Wrote updated raster to {output_path} "
        f"({len(updates):,} pixels updated out of {np.sum(raster_data.valid_mask):,} valid)"
    )


def compute_spatial_adjustment_at_points(
    points: np.ndarray,
    model_vs30: np.ndarray,
    model_stdv: np.ndarray,
    obs_locations: np.ndarray,
    obs_vs30: np.ndarray,
    obs_model_vs30: np.ndarray,
    obs_model_stdv: np.ndarray,
    obs_uncertainty: np.ndarray,
    model_type: constants.ModelType,
    max_dist_m: float = constants.MAX_DIST_M,
    max_points: int = constants.MAX_POINTS,
    noisy: bool = False,
    cov_reduc: float = constants.COV_REDUC,
    progress_bar: tqdm | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MVN spatial adjustment at specific query points.

    This is the point-based equivalent of compute_spatial_adjustments(). It
    delegates to compute_spatial_adjustment_for_pixel() for each point, sharing
    the same MVN conditioning algorithm used by the grid pipeline.

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
    model_type : constants.ModelType
        Either ModelType.GEOLOGY or ModelType.TERRAIN (determines phi correlation length).
    max_dist_m : float, optional
        Maximum distance (meters) to consider observations. Default from constants.
    max_points : int, optional
        Maximum number of observations per point. Default from constants.
    noisy : bool
        Whether to apply noise weighting.
    cov_reduc : float
        Covariance reduction factor. Default from constants.
    progress_bar : tqdm, optional
        External progress bar to update per point. If None, no progress is shown.

    Returns
    -------
    mvn_vs30 : np.ndarray
        (N,) array of spatially adjusted Vs30 values.
    mvn_stdv : np.ndarray
        (N,) array of spatially adjusted standard deviation values.
    """
    # Initialize output arrays with prior values
    mvn_vs30 = model_vs30.copy()
    mvn_stdv = model_stdv.copy()

    if len(obs_locations) == 0:
        logger.warning("No observations provided for MVN adjustment")
        return mvn_vs30, mvn_stdv

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

    # Build ObservationData from valid observations
    valid_model_vs30 = obs_model_vs30[valid_obs_mask]
    valid_model_stdv = obs_model_stdv[valid_obs_mask]
    valid_vs30 = obs_vs30[valid_obs_mask]
    valid_uncertainty = obs_uncertainty[valid_obs_mask]

    residuals = np.log(valid_vs30 / valid_model_vs30)
    if noisy:
        omega = np.sqrt(
            valid_model_stdv**2 / (valid_model_stdv**2 + valid_uncertainty**2)
        )
        residuals *= omega
    else:
        omega = np.ones(len(residuals))

    obs_data = ObservationData(
        locations=obs_locations[valid_obs_mask],
        vs30=valid_vs30,
        model_vs30=valid_model_vs30,
        model_stdv=valid_model_stdv,
        residuals=residuals,
        omega=omega,
        uncertainty=valid_uncertainty,
    )

    # Pre-compute correlation at zero distance
    corr_zero = utils.correlation_function(
        np.array([0.0]), constants.PHI[model_type]
    )[0]

    # Process each query point using the shared per-pixel MVN function
    for i in range(len(points)):
        pixel = PixelData(
            location=points[i],
            vs30=float(model_vs30[i]),
            stdv=float(model_stdv[i]),
            index=i,
        )

        result = compute_spatial_adjustment_for_pixel(
            pixel,
            obs_data,
            model_type,
            max_dist_m=max_dist_m,
            max_points=max_points,
            noisy=noisy,
            cov_reduc=cov_reduc,
            corr_zero=corr_zero,
        )

        if result is not None:
            mvn_vs30[i] = result.updated_vs30
            mvn_stdv[i] = result.updated_stdv

        if progress_bar is not None:
            progress_bar.update(1)

    return mvn_vs30, mvn_stdv

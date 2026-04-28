"""Multivariate Normal (MVN) distribution-based spatial adjustment of Vs30 using nearby observations."""

import logging
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rasterio
import scipy.spatial.distance
from tqdm import tqdm

from vs30 import category, constants, multiprocess, raster, utils

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

    @staticmethod
    def compute_valid_mask(
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

        valid_mask, valid_flat_indices = cls.compute_valid_mask(vs30, stdv, nodata)

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


@dataclass
class BoundingBoxResult:
    """
    Result of bounding box search for affected pixels.

    Attributes
    ----------
    mask : ndarray
        Boolean mask of pixels in any observation's bounding box.
    n_affected_pixels : int
        Total number of pixels affected by at least one observation.
    """

    mask: np.ndarray
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
    utils.validate_csv_columns(
        observations, constants.ObservationColumn.REQUIRED, "Observations"
    )
    if not np.all(observations[constants.ObservationColumn.VS30] > 0):
        raise ValueError("Vs30 must be positive")
    if not np.all(observations[constants.ObservationColumn.UNCERTAINTY] > 0):
        raise ValueError("Uncertainty must be positive")


def prepare_observation_data(
    observations: pd.DataFrame,
    raster_data: RasterData,
    updated_model_table: np.ndarray,
    model_type: constants.ModelType,
    apply_alluvium_slope_mod: bool,
    apply_coastal_distance_mod: bool,
    slope_array: np.ndarray | None = None,
    coast_dist_array: np.ndarray | None = None,
    noisy: bool = False,
) -> ObservationData:
    """
    Prepare observation data for MVN processing.

    For geology models, hybrid modifications (slope and coastal distance)
    are applied to model values at observation locations before computing
    residuals. Terrain models do not use slope_array or coast_dist_array.

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
    apply_alluvium_slope_mod : bool
        Whether to apply slope-based interpolation for GID 4 (alluvium).
    apply_coastal_distance_mod : bool
        Whether to apply coastal distance modification for GID 4 and GID 10.
    slope_array : ndarray, optional
        In-memory 2D slope array. Required for geology models; ignored
        for terrain. Observation values are sampled from this array using
        the raster transform.
    coast_dist_array : ndarray, optional
        In-memory 2D coastal distance array. Required for geology models;
        ignored for terrain.
    noisy : bool, optional
        Whether to apply noise weighting. Default is False.

    Returns
    -------
    ObservationData
        Prepared observation data object.

    Raises
    ------
    ValueError
        If model_type is GEOLOGY and slope_array or coast_dist_array is None.
    """
    if model_type == constants.ModelType.GEOLOGY and (
        slope_array is None or coast_dist_array is None
    ):
        raise ValueError(
            "slope_array and coast_dist_array are required for geology models."
        )
    obs_locs = observations[
        [constants.ObservationColumn.EASTING, constants.ObservationColumn.NORTHING]
    ].values

    model_ids = category.assign_to_category(obs_locs, model_type)

    # Model IDs are 1-indexed in the raster, but 0-indexed in the model table.
    # constants.RASTER_ID_NODATA_VALUE is 255; valid IDs are 1-15 for geology, 1-16 for terrain.
    valid_mask = (
        (model_ids != constants.RASTER_ID_NODATA_VALUE)
        & (model_ids > 0)
        & (model_ids <= len(updated_model_table))
    )
    model_vs30 = np.full(len(observations), np.nan)
    model_stdv = np.full(len(observations), np.nan)

    valid_model_ids = model_ids[valid_mask] - 1
    model_vs30[valid_mask] = updated_model_table[valid_model_ids, 0]
    model_stdv[valid_mask] = updated_model_table[valid_model_ids, 1]

    valid_obs_mask = ~np.isnan(model_vs30) & ~np.isnan(model_stdv)
    obs_locs = obs_locs[valid_obs_mask]
    vs30_obs = observations[constants.ObservationColumn.VS30].values[valid_obs_mask]
    model_vs30 = model_vs30[valid_obs_mask]
    model_stdv = model_stdv[valid_obs_mask]
    uncertainty = observations[constants.ObservationColumn.UNCERTAINTY].values[
        valid_obs_mask
    ]

    if model_type == constants.ModelType.GEOLOGY:
        rows, cols = rasterio.transform.rowcol(
            raster_data.transform, obs_locs[:, 0], obs_locs[:, 1]
        )
        rows = np.asarray(rows)
        cols = np.asarray(cols)

        within_grid = (
            (rows >= 0)
            & (rows < slope_array.shape[0])
            & (cols >= 0)
            & (cols < slope_array.shape[1])
        )

        # Observations within the grid domain: sample slope and coastal
        # distance from the grid arrays for consistency with the
        # grid-resampled values used in pixel updates. Observations outside
        # the grid domain: sample from the original source rasters since
        # there are no corresponding grid pixels to be consistent with.
        slope_obs = np.empty(len(rows), dtype=np.float64)
        coast_obs = np.empty(len(rows), dtype=np.float64)

        slope_obs[within_grid] = slope_array[rows[within_grid], cols[within_grid]]
        coast_obs[within_grid] = coast_dist_array[rows[within_grid], cols[within_grid]]

        if not np.all(within_grid):
            outside_grid_points = obs_locs[~within_grid]
            slope_obs[~within_grid] = raster.sample_slope_at_points(outside_grid_points)
            coast_obs[~within_grid] = raster.compute_coastal_distance_at_points(
                outside_grid_points
            )

        # Legacy parity: the legacy interpolate_raster replaces tif-NODATA
        # slope samples with ID_NODATA=255 for observations, which slips past
        # the _hyb_calc (slope == 0) | (slope == -9999) check, causing
        # log10(255) ≈ 2.41 to feed np.interp and return the MAX Vs30 for
        # the gid. The equivalent grid-pixel NODATA handling uses 1e-9 and
        # returns the MIN Vs30. Reproduce the legacy obs behaviour.
        slope_obs = np.where(
            slope_obs < 0, constants.LEGACY_OBS_SLOPE_NODATA_SENTINEL, slope_obs
        )

        model_vs30, model_stdv = raster.apply_hybrid_geology_modifications(
            model_vs30,
            model_stdv,
            model_ids[valid_obs_mask],
            slope_obs,
            coast_obs,
            apply_alluvium_slope_mod=apply_alluvium_slope_mod,
            apply_coastal_distance_mod=apply_coastal_distance_mod,
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
) -> np.ndarray:
    """
    Find grid points within bounding boxes of observations using fully vectorized NumPy.

    Uses broadcasting to compute all observation-grid pairs simultaneously.
    Returns a collapsed boolean mask of which grid points fall inside any
    observation's bounding box.

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

    Returns
    -------
    chunk_mask : ndarray, shape (M,), dtype=bool
        Boolean array indicating which grid points in this chunk are affected
        by any observation (collapsed with np.any(axis=0)).
    """
    grid_eastings = grid_locs[:, 0]
    grid_northings = grid_locs[:, 1]

    # Broadcasting (n_obs, 1) against (n_grid,) gives an (n_obs, n_grid) mask
    # of which grid points fall in each observation's bounding box.
    in_bbox = (
        (grid_eastings >= obs_eastings_min)
        & (grid_eastings <= obs_eastings_max)
        & (grid_northings >= obs_northings_min)
        & (grid_northings <= obs_northings_max)
    )

    return np.any(in_bbox, axis=0)


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


def process_bbox_chunk(args: tuple) -> tuple[int, np.ndarray]:
    """
    Worker function for parallel bounding box processing.

    Processes a single chunk of grid points to find which are affected by observations.

    Parameters
    ----------
    args : tuple
        ``(chunk_idx, grid_locs_chunk, obs_bounds)`` where ``obs_bounds`` is
        ``(obs_eastings_min, obs_eastings_max, obs_northings_min, obs_northings_max)``.

    Returns
    -------
    tuple
        ``(chunk_idx, chunk_mask)``.
    """
    chunk_idx, grid_locs_chunk, obs_bounds = args
    obs_eastings_min, obs_eastings_max, obs_northings_min, obs_northings_max = (
        obs_bounds
    )

    chunk_mask = grid_points_in_bbox(
        grid_locs=grid_locs_chunk,
        obs_eastings_min=obs_eastings_min,
        obs_eastings_max=obs_eastings_max,
        obs_northings_min=obs_northings_min,
        obs_northings_max=obs_northings_max,
    )

    return chunk_idx, chunk_mask


def build_covariance_matrix(
    pixel: PixelData,
    obs_data: ObservationData,
    obs_indices: np.ndarray,
    corr_fn: Callable[[np.ndarray], np.ndarray],
    noisy: bool = False,
    cov_reduc: float = constants.COV_REDUC,
) -> np.ndarray:
    """
    Build covariance matrix through clear pipeline of steps.

    Parameters
    ----------
    pixel : PixelData
        Pixel data for the pixel being updated.
    obs_data : ObservationData
        Full observation data.
    obs_indices : ndarray
        Integer indices into obs_data for the selected observations.
    corr_fn : callable
        Correlation function mapping distances (ndarray) to correlations (ndarray).
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

    all_points = np.vstack([pixel.location, obs_data.locations[obs_indices]]).astype(
        np.float64
    )
    distance_matrix = scipy.spatial.distance.cdist(
        all_points, all_points, metric="euclidean"
    )

    corr = corr_fn(distance_matrix)

    stdvs = np.insert(obs_data.model_stdv[obs_indices], 0, pixel.stdv)
    cov = corr * np.outer(stdvs, stdvs)

    if noisy:
        omega = np.insert(obs_data.omega[obs_indices], 0, 1.0)
        omega_matrix = np.outer(omega, omega)
        np.fill_diagonal(omega_matrix, 1.0)
        cov *= omega_matrix

    if cov_reduc > 0:
        log_vs30s = np.insert(
            np.log(obs_data.model_vs30[obs_indices]), 0, np.log(pixel.vs30)
        )
        log_dist_matrix = np.abs(log_vs30s[:, np.newaxis] - log_vs30s)
        cov *= np.exp(-cov_reduc * log_dist_matrix)

    return cov


def select_observations_for_pixel(
    pixel: PixelData,
    obs_data: ObservationData,
    max_dist_m: float = constants.MAX_DIST_M,
    max_points: int = constants.MAX_POINTS,
) -> np.ndarray:
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
    ndarray
        Integer indices into obs_data for the selected observations.
        Empty array if no observations are within range.
    """
    # Euclidean distance from pixel to each observation.
    # einsum("ij,ij->i", diff, diff) computes the row-wise dot product,
    # i.e. sum of squared differences per row — equivalent to
    # np.sum(diff**2, axis=1) but avoids creating intermediate arrays.
    diff = obs_data.locations - pixel.location
    distances = np.sqrt(np.einsum("ij,ij->i", diff, diff))

    # Select observations using distance-based filtering
    max_points_i = min(max_points, len(distances)) - 1
    if max_points_i < 0:
        return np.array([], dtype=np.intp)

    min_dist, cutoff_dist = np.partition(distances, [0, max_points_i])[
        [0, max_points_i]
    ]
    if min_dist > max_dist_m:
        # Not close enough to any observed locations
        return np.array([], dtype=np.intp)

    # Include all observations within cutoff distance (may exceed max_points for accuracy)
    loc_mask = distances <= min(max_dist_m, cutoff_dist)
    return np.where(loc_mask)[0]


def compute_spatial_adjustment_for_pixel(
    pixel: PixelData,
    obs_data: ObservationData,
    corr_fn: Callable[[np.ndarray], np.ndarray],
    max_dist_m: float = constants.MAX_DIST_M,
    max_points: int = constants.MAX_POINTS,
    noisy: bool = False,
    cov_reduc: float = constants.COV_REDUC,
    corr_zero: float | None = None,
) -> tuple[float, float, int] | None:
    """
    Compute MVN update for a single pixel.

    Parameters
    ----------
    pixel : PixelData
        Pixel data.
    obs_data : ObservationData
        Full observation data.
    corr_fn : callable
        Correlation function mapping distances (ndarray) to correlations (ndarray).
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
        corr_fn. Pass this when calling in a loop to avoid recomputing.

    Returns
    -------
    tuple of (float, float, int) or None
        (updated_vs30, updated_stdv, n_observations_used), or None if the
        pixel should be skipped (NaN/invalid input).
    """
    if (
        np.isnan(pixel.vs30)
        or np.isnan(pixel.stdv)
        or pixel.vs30 <= 0
        or pixel.stdv <= 0
    ):
        return None

    # Correlation at zero distance is ≈1 (minus a tiny epsilon from the
    # enforced minimum distance). Matches the legacy R code, which evaluates
    # the correlation at distances >= 0.1 m and sets corr(0) = 1 explicitly.
    if corr_zero is None:
        corr_zero = corr_fn(np.array([0.0]))[0]
    initial_var = (pixel.stdv**2) * corr_zero

    obs_indices = select_observations_for_pixel(
        pixel,
        obs_data,
        max_dist_m=max_dist_m,
        max_points=max_points,
    )

    n_obs = len(obs_indices)
    if n_obs == 0:
        # No observations nearby, return unchanged values (but with shrunk stdv matching legacy)
        return (pixel.vs30, float(np.sqrt(initial_var)), 0)

    cov_matrix = build_covariance_matrix(
        pixel,
        obs_data,
        obs_indices,
        corr_fn,
        noisy=noisy,
        cov_reduc=cov_reduc,
    )

    try:
        inv_cov = np.linalg.inv(cov_matrix[1:, 1:])
        pred_update = np.dot(
            np.dot(cov_matrix[0, 1:], inv_cov),
            obs_data.residuals[obs_indices],
        )
        var = cov_matrix[0, 0] - np.dot(
            np.dot(cov_matrix[0, 1:], inv_cov), cov_matrix[1:, 0]
        )
        return (
            float(pixel.vs30 * np.exp(pred_update)),
            float(np.sqrt(max(0, var))),
            n_obs,
        )
    except np.linalg.LinAlgError:
        # Singular covariance matrix — keep prior values with default variance shrinkage
        logger.debug(
            f"Singular covariance matrix at pixel {pixel.index}, keeping prior values"
        )
        return (pixel.vs30, float(np.sqrt(initial_var)), 0)


def find_affected_pixels(
    raster_data: RasterData,
    obs_data: ObservationData,
    max_spatial_boolean_array_memory_gb: float,
    model_type: constants.ModelType,
    max_dist_m: float = constants.MAX_DIST_M,
    nproc: int = 1,
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
    nproc : int, optional
        Number of parallel processes. 1 for sequential (default),
        >1 for parallel processing.

    Returns
    -------
    BoundingBoxResult
        Result containing the affected-pixel mask and pixel count.
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

    valid_points_in_bbox_mask = np.zeros(len(grid_locs), dtype=bool)

    logger.info(f"Processing {n_chunks} chunks of {chunk_size:,} pixels each")

    label = str(model_type).capitalize()

    # Prepare chunk arguments
    chunk_args = []
    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(grid_locs))
        grid_locs_chunk = grid_locs[start_idx:end_idx]
        chunk_args.append((chunk_idx, grid_locs_chunk, obs_bounds))

    if nproc > 1 and n_chunks > 1:
        # Parallel processing
        actual_nproc = min(nproc, n_chunks)
        logger.info(f"Using {actual_nproc} parallel workers")
        with multiprocess.spawn_context.Pool(processes=actual_nproc) as pool:
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
        logger.info(
            f"{label}: checking {len(grid_locs):,} pixels for nearby observations"
        )
        results = [process_bbox_chunk(chunk_args[0])]

    # Merge results from either parallel or sequential processing
    for chunk_idx, chunk_mask in results:
        start_idx = chunk_idx * chunk_size
        valid_points_in_bbox_mask[start_idx : start_idx + len(chunk_mask)] = chunk_mask

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
        n_affected_pixels=n_affected,
    )


def compute_spatial_adjustments(
    raster_data: RasterData,
    obs_data: ObservationData,
    bbox_result: BoundingBoxResult,
    corr_fn: Callable[[np.ndarray], np.ndarray],
    max_dist_m: float = constants.MAX_DIST_M,
    max_points: int = constants.MAX_POINTS,
    noisy: bool = False,
    cov_reduc: float = constants.COV_REDUC,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute MVN updates for all affected pixels and return updated arrays.

    Parameters
    ----------
    raster_data : RasterData
        Raster data object.
    obs_data : ObservationData
        Observation data.
    bbox_result : BoundingBoxResult
        Bounding box result.
    corr_fn : callable
        Correlation function mapping distances (ndarray) to correlations (ndarray).
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
    tuple of ndarray
        (updated_vs30, updated_stdv) arrays with spatial adjustments applied.
    """
    affected_flat_indices = np.where(bbox_result.mask)[0]
    affected_valid_indices = np.where(bbox_result.mask[raster_data.valid_flat_indices])[
        0
    ]

    grid_locs = raster_data.get_coordinates()
    affected_locs = grid_locs[affected_valid_indices]
    affected_vs30 = raster_data.vs30.flat[affected_flat_indices]
    affected_stdv = raster_data.stdv.flat[affected_flat_indices]

    updated_vs30 = raster_data.vs30.copy()
    updated_stdv = raster_data.stdv.copy()

    corr_zero = corr_fn(np.array([0.0]))[0]
    n_updated = 0

    for i, flat_idx in enumerate(
        tqdm(affected_flat_indices, desc="Spatial adjustment", unit="pixel")
    ):
        pixel = PixelData(
            location=affected_locs[i],
            vs30=float(affected_vs30[i]),
            stdv=float(affected_stdv[i]),
            index=flat_idx,
        )
        result = compute_spatial_adjustment_for_pixel(
            pixel,
            obs_data,
            corr_fn,
            max_dist_m=max_dist_m,
            max_points=max_points,
            noisy=noisy,
            cov_reduc=cov_reduc,
            corr_zero=corr_zero,
        )
        if result is not None:
            vs30, stdv, _ = result
            updated_vs30.flat[flat_idx] = vs30
            updated_stdv.flat[flat_idx] = stdv
            n_updated += 1

    logger.info(f"Spatial adjustment complete: {n_updated:,} pixels updated")

    return updated_vs30, updated_stdv


def compute_spatial_adjustment_at_points(
    points: np.ndarray,
    model_vs30: np.ndarray,
    model_stdv: np.ndarray,
    obs_locations: np.ndarray,
    obs_vs30: np.ndarray,
    obs_model_vs30: np.ndarray,
    obs_model_stdv: np.ndarray,
    obs_uncertainty: np.ndarray,
    corr_fn: Callable[[np.ndarray], np.ndarray],
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
    corr_fn : callable
        Correlation function mapping distances (ndarray) to correlations (ndarray).
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
    corr_zero = corr_fn(np.array([0.0]))[0]

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
            corr_fn,
            max_dist_m=max_dist_m,
            max_points=max_points,
            noisy=noisy,
            cov_reduc=cov_reduc,
            corr_zero=corr_zero,
        )

        if result is not None:
            mvn_vs30[i] = result[0]
            mvn_stdv[i] = result[1]

        if progress_bar is not None:
            progress_bar.update(1)

    return mvn_vs30, mvn_stdv

"""Multivariate Normal (MVN) distribution-based spatial adjustment of Vs30 using nearby observations."""

import logging
import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
import rasterio
import scipy.spatial
import scipy.spatial.distance
from tqdm import tqdm

from vs30 import category, constants, raster, utils

logger = logging.getLogger(__name__)


@dataclass
class ObservationData:
    """
    Prepared observation data for spatial processing.

    Attributes
    ----------
    locations : ndarray
        (n_obs, 2) array of [easting, northing] coordinates.
    model_stdv : ndarray
        (n_obs,) model standard deviation at observation locations.
    log_model_vs30 : ndarray
        (n_obs,) precomputed ``np.log(model_vs30)``.
    residuals : ndarray
        (n_obs,) log residuals: log(vs30 / model_vs30). Pre-scaled by
        ``noise_weights`` when constructed with ``noisy=True``.
    noise_weights : ndarray
        (n_obs,) per-observation weights when constructed with
        ``noisy=True``; all ones otherwise.
    tree : scipy.spatial.KDTree or None
        Spatial index over ``locations`` for nearest-neighbour observation
        queries in the MVN pixel loop. None when there are no observations.
    """

    locations: np.ndarray
    model_stdv: np.ndarray
    log_model_vs30: np.ndarray
    residuals: np.ndarray
    noise_weights: np.ndarray
    tree: scipy.spatial.KDTree | None = None

    def __post_init__(self):
        """Build the spatial index from ``locations`` if not supplied."""
        if self.tree is None and len(self.locations) > 0:
            self.tree = scipy.spatial.KDTree(self.locations)

    @classmethod
    def empty(cls) -> "ObservationData":
        """
        Construct an instance with zero-length arrays.

        Returns
        -------
        ObservationData
            Instance with all arrays empty.
        """
        return cls(
            locations=np.empty((0, 2)),
            model_stdv=np.empty(0),
            log_model_vs30=np.empty(0),
            residuals=np.empty(0),
            noise_weights=np.empty(0),
        )


@dataclass
class PixelData:
    """
    Per-pixel inputs for MVN spatial adjustment.

    Attributes
    ----------
    location : ndarray
        (2,) [easting, northing] coordinates.
    vs30 : float
        Prior Vs30 value.
    stdv : float
        Prior standard deviation.
    """

    location: np.ndarray
    vs30: float
    stdv: float


@dataclass
class RasterData:
    """
    Vs30 raster data with precomputed valid-pixel mask.

    Attributes
    ----------
    vs30 : ndarray
        2D array of Vs30 mean values.
    stdv : ndarray
        2D array of Vs30 standard deviation values.
    transform : rasterio.transform.Affine
        Affine transformation for pixel<->world coordinate conversion.
    valid_mask : ndarray
        2D boolean mask: True where Vs30 and stdv are non-nodata, non-NaN,
        and positive.
    valid_flat_indices : ndarray
        1D flat indices of pixels where valid_mask is True.
    """

    vs30: np.ndarray
    stdv: np.ndarray
    transform: rasterio.transform.Affine
    valid_mask: np.ndarray
    valid_flat_indices: np.ndarray

    @staticmethod
    def compute_valid_mask(
        vs30: np.ndarray, stdv: np.ndarray, nodata: float | None
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Compute the boolean mask of valid pixels and their flat indices.

        Valid pixels are non-nodata, non-NaN, and positive in both arrays.

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
        valid_mask : ndarray
            2D boolean mask.
        valid_flat_indices : ndarray
            1D flat indices of True entries in valid_mask.
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
        nodata: float = constants.NODATA_VALUE,
    ) -> "RasterData":
        """
        Create RasterData from arrays, computing the valid-pixel mask.

        Parameters
        ----------
        vs30 : ndarray
            2D array of Vs30 mean values.
        stdv : ndarray
            2D array of Vs30 standard deviation values.
        transform : rasterio.transform.Affine
            Affine transformation for coordinate conversion.
        nodata : float, optional
            No-data sentinel for valid-mask computation.

        Returns
        -------
        RasterData
            Instance with computed valid-pixel mask and flat indices.
        """
        valid_mask, valid_flat_indices = cls.compute_valid_mask(vs30, stdv, nodata)

        return cls(
            vs30=vs30,
            stdv=stdv,
            transform=transform,
            valid_mask=valid_mask,
            valid_flat_indices=valid_flat_indices,
        )

    def get_coordinates(self) -> np.ndarray:
        """
        Pixel-center coordinates of all valid pixels.

        Returns
        -------
        ndarray
            (N_valid, 2) array of [easting, northing] coordinates.
        """
        valid_rows, valid_cols = np.where(self.valid_mask)
        xs, ys = rasterio.transform.xy(self.transform, valid_rows, valid_cols)
        return np.column_stack((xs, ys)).astype(np.float32)


def validate_raster_data(raster_data: RasterData) -> None:
    """
    Check shape match and finiteness/positivity of Vs30/stdv at valid pixels.

    Parameters
    ----------
    raster_data : RasterData
        Raster data to validate.

    Raises
    ------
    ValueError
        If any check fails.
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
    Check observations have the required columns and positive Vs30/uncertainty.

    Parameters
    ----------
    observations : DataFrame
        Observations to validate.

    Raises
    ------
    ValueError
        If any check fails.
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
        Provides the transform used to locate observations within the grid.
    updated_model_table : ndarray
        Updated model table (n_categories, 2) array of [vs30, stdv].
    model_type : constants.ModelType
        Either GEOLOGY or TERRAIN.
    apply_alluvium_slope_mod : bool
        Whether to apply slope-based interpolation for GID 4 (alluvium).
    apply_coastal_distance_mod : bool
        Whether to apply coastal distance modification for GID 4 and GID 10.
    slope_array : ndarray, optional
        2D slope array. Required for geology models; ignored for terrain.
    coast_dist_array : ndarray, optional
        2D coastal distance array. Required for geology models; ignored
        for terrain.
    noisy : bool, optional
        Whether to apply noise weighting.

    Returns
    -------
    ObservationData
        Prepared observation data.

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
    ].to_numpy()

    model_ids = category.assign_to_category(obs_locs, model_type)

    # Model IDs are 1-indexed in the raster, but 0-indexed in the model table.
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
    vs30_obs = observations[constants.ObservationColumn.VS30].to_numpy()[valid_obs_mask]
    model_vs30 = model_vs30[valid_obs_mask]
    model_stdv = model_stdv[valid_obs_mask]
    uncertainty = observations[constants.ObservationColumn.UNCERTAINTY].to_numpy()[
        valid_obs_mask
    ]

    if model_type == constants.ModelType.GEOLOGY:
        assert slope_array is not None and coast_dist_array is not None  # type guard: entry check raises if either is None when GEOLOGY
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

        # Within-grid observations: sample from grid arrays for consistency
        # with pixel updates. Outside-grid observations: sample from source
        # rasters since there are no corresponding grid pixels to be
        # consistent with.
        slope_obs = np.empty(len(rows), dtype=np.float64)
        coast_obs = np.full(len(rows), np.nan, dtype=np.float64)

        slope_obs[within_grid] = slope_array[rows[within_grid], cols[within_grid]]
        coast_obs[within_grid] = coast_dist_array[rows[within_grid], cols[within_grid]]

        if not np.all(within_grid):
            outside_grid_points = obs_locs[~within_grid]
            slope_obs[~within_grid] = raster.sample_slope_at_points(outside_grid_points)
            if apply_coastal_distance_mod:
                coast_obs[~within_grid] = raster.compute_coast_distance_at_points(
                    outside_grid_points
                )

        # NODATA slope at obs locations gets replaced with OBS_SLOPE_NODATA_SENTINEL
        # so np.interp returns MAX Vs30 (distinct from grid pixels' NODATA path,
        # which yields MIN Vs30).
        slope_obs = np.where(
            slope_obs < 0, constants.OBS_SLOPE_NODATA_SENTINEL, slope_obs
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

    residuals, noise_weights = compute_residuals(
        vs30_obs, model_vs30, model_stdv, uncertainty, noisy
    )

    return ObservationData(
        locations=obs_locs,
        model_stdv=model_stdv,
        log_model_vs30=np.log(model_vs30),
        residuals=residuals,
        noise_weights=noise_weights,
    )


def compute_residuals(
    vs30: np.ndarray,
    model_vs30: np.ndarray,
    model_stdv: np.ndarray,
    uncertainty: np.ndarray,
    noisy: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute log residuals and per-observation noise weights.

    The noise weights are ``ω`` from Worden et al. (2018), Eq. 32. They
    shrink the contribution of high-uncertainty observations toward zero.

    Parameters
    ----------
    vs30 : ndarray
        Measured Vs30 values.
    model_vs30 : ndarray
        Model Vs30 values at observation locations.
    model_stdv : ndarray
        Model standard deviations at observation locations.
    uncertainty : ndarray
        Per-observation Vs30 uncertainties (in log space).
    noisy : bool
        If True, scale residuals by ``noise_weights`` and return the weights
        for downstream covariance shaping. If False, ``noise_weights`` is
        all ones and residuals are unscaled.

    Returns
    -------
    residuals : ndarray
        Log residuals, pre-scaled by ``noise_weights`` when ``noisy=True``.
    noise_weights : ndarray
        Per-observation weights when ``noisy=True``; all ones otherwise.

    References
    ----------
    Worden, C. B., Thompson, E. M., Baker, J. W., Bradley, B. A., Luco, N.,
    and Wald, D. J. (2018). Spatial and Spectral Interpolation of Ground-Motion
    Intensity Measure Observations. Bulletin of the Seismological Society of
    America, 108(2), 866–875. doi:10.1785/0120170201. Local copy at
    ``reference_papers/worden_2018_mvn_interpolation.pdf``.
    """
    residuals = np.log(vs30 / model_vs30)
    if noisy:
        noise_weights = np.sqrt(model_stdv**2 / (model_stdv**2 + uncertainty**2))
        residuals *= noise_weights
    else:
        noise_weights = np.ones(len(residuals))
    return residuals, noise_weights


def grid_points_in_bbox(
    grid_locs: np.ndarray,
    obs_eastings_min: np.ndarray,
    obs_eastings_max: np.ndarray,
    obs_northings_min: np.ndarray,
    obs_northings_max: np.ndarray,
) -> np.ndarray:
    """
    Mark grid points that fall inside any observation's bounding box.

    Parameters
    ----------
    grid_locs : array_like, shape (M, 2)
        Grid point coordinates as (easting, northing) in NZTM.
    obs_eastings_min : ndarray, shape (N, 1)
        Per-observation lower easting bound.
    obs_eastings_max : ndarray, shape (N, 1)
        Per-observation upper easting bound.
    obs_northings_min : ndarray, shape (N, 1)
        Per-observation lower northing bound.
    obs_northings_max : ndarray, shape (N, 1)
        Per-observation upper northing bound.

    Returns
    -------
    ndarray
        Shape ``(M,)`` boolean mask; True for grid points inside at least one
        observation bbox.
    """
    grid_eastings = grid_locs[:, 0]
    grid_northings = grid_locs[:, 1]

    in_bbox = (
        (grid_eastings >= obs_eastings_min)
        & (grid_eastings <= obs_eastings_max)
        & (grid_northings >= obs_northings_min)
        & (grid_northings <= obs_northings_max)
    )

    return np.any(in_bbox, axis=0)


def build_covariance_matrix(
    pixel: PixelData,
    obs_data: ObservationData,
    obs_indices: np.ndarray,
    corr_fn: Callable[[np.ndarray], np.ndarray],
    noisy: bool = False,
    cov_reduc: float = constants.COV_REDUC,
) -> np.ndarray:
    """
    Build the covariance matrix for one pixel and its nearby observations.

    Parameters
    ----------
    pixel : PixelData
        Pixel being updated.
    obs_data : ObservationData
        Prepared observation data; the rows used here are selected by
        ``obs_indices``.
    obs_indices : ndarray
        Integer indices into ``obs_data`` for the selected observations.
    corr_fn : callable
        Correlation function mapping distances (ndarray) to correlations
        (ndarray).
    noisy : bool, optional
        If True, down-weight uncertain observations by ``obs_data.noise_weights``.
    cov_reduc : float, optional
        If > 0, shrink covariance between points with dissimilar Vs30 values.

    Returns
    -------
    ndarray
        Covariance matrix of shape ``(n_selected_obs + 1, n_selected_obs + 1)``.
        Row/column 0 is the pixel; rows/columns 1..n are the selected
        observations.
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
        noise_weights = np.insert(obs_data.noise_weights[obs_indices], 0, 1.0)
        noise_weight_matrix = np.outer(noise_weights, noise_weights)
        np.fill_diagonal(noise_weight_matrix, 1.0)
        cov *= noise_weight_matrix

    if cov_reduc > 0:
        log_vs30s = np.insert(
            obs_data.log_model_vs30[obs_indices], 0, np.log(pixel.vs30)
        )
        log_dist_matrix = np.abs(log_vs30s[:, np.newaxis] - log_vs30s)
        cov *= np.exp(-cov_reduc * log_dist_matrix)

    return cov


def select_observations_for_pixel_batch(
    pixel_locations: np.ndarray,
    obs_data: ObservationData,
    max_dist_m: float = constants.MAX_DIST_M,
    max_points: int = constants.MAX_POINTS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Select observations for a batch of pixels via one KDTree query.

    Parameters
    ----------
    pixel_locations : ndarray
        (N_pix, 2) array of [easting, northing] coordinates in NZTM.
    obs_data : ObservationData
        Prepared observation data. ``obs_data.tree`` carries the spatial index.
    max_dist_m : float
        Maximum distance in meters to consider observations.
    max_points : int
        Maximum number of observations to return per pixel.

    Returns
    -------
    indices : ndarray, shape (N_pix, k), dtype intp
        Indices into ``obs_data.locations``. Slots with no neighbour within
        ``max_dist_m`` are set to ``len(obs_data.locations)`` (scipy sentinel).
        ``k = min(max_points, n_obs)``.
    distances : ndarray, shape (N_pix, k), dtype float64
        Euclidean distances. Sentinel slots are ``np.inf``.
    """
    if obs_data.tree is None:
        return (
            np.empty((len(pixel_locations), 0), dtype=np.intp),
            np.empty((len(pixel_locations), 0), dtype=np.float64),
        )
    k = min(max_points, len(obs_data.locations))
    distances, indices = obs_data.tree.query(
        pixel_locations, k=k, distance_upper_bound=max_dist_m
    )
    # scipy.spatial.KDTree.query returns shape (N_pix, k) when k > 1 but
    # squeezes to (N_pix,) when k == 1 (the trailing axis is dropped). Reshape
    # to (N_pix, k) so callers can always index batch_indices[i] to get this
    # pixel's row regardless of k.
    return (
        np.asarray(indices).reshape(len(pixel_locations), k),
        np.asarray(distances).reshape(len(pixel_locations), k),
    )


def compute_spatial_adjustment_for_pixel(
    pixel: PixelData,
    obs_data: ObservationData,
    obs_indices: np.ndarray,
    corr_fn: Callable[[np.ndarray], np.ndarray],
    corr_zero: float,
    noisy: bool = False,
    cov_reduc: float = constants.COV_REDUC,
) -> tuple[float, float] | None:
    """
    Compute MVN update for a single pixel given pre-selected observations.

    Parameters
    ----------
    pixel : PixelData
        Pixel being updated.
    obs_data : ObservationData
        Prepared observation data.
    obs_indices : ndarray
        Integer indices into ``obs_data`` for the observations conditioning
        this pixel. Pass an empty array when there are no nearby observations;
        the prior vs30 is returned unchanged and stdv is shrunk by ``corr_zero``.
    corr_fn : callable
        Correlation function mapping distances (ndarray) to correlations
        (ndarray).
    corr_zero : float
        Pre-computed ``corr_fn(np.array([0.0]))[0]``.
    noisy : bool, optional
        If True, down-weight uncertain observations by ``obs_data.noise_weights``.
    cov_reduc : float, optional
        If > 0, shrink covariance between points with dissimilar Vs30 values.

    Returns
    -------
    tuple of (float, float) or None
        ``(updated_vs30, updated_stdv)``, or None if the pixel should be
        skipped (NaN/invalid input).
    """
    if (
        np.isnan(pixel.vs30)
        or np.isnan(pixel.stdv)
        or pixel.vs30 <= 0
        or pixel.stdv <= 0
    ):
        return None

    # corr_zero ≈ 1 (slightly less than 1 because correlations.exponential enforces MIN_DIST_ENFORCED before the exp).
    initial_var = (pixel.stdv**2) * corr_zero

    if len(obs_indices) == 0:
        return (pixel.vs30, float(np.sqrt(initial_var)))

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
        )
    except np.linalg.LinAlgError:
        logger.debug("Singular covariance matrix, keeping prior values")
        return (pixel.vs30, float(np.sqrt(initial_var)))


def find_affected_pixels(
    raster_data: RasterData,
    obs_data: ObservationData,
    max_spatial_intermediate_array_memory_gb: float,
    model_type: constants.ModelType,
    max_dist_m: float = constants.MAX_DIST_M,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Find pixels affected by observations using bounding boxes.

    Parameters
    ----------
    raster_data : RasterData
        Raster data with valid-pixel mask and grid transform.
    obs_data : ObservationData
        Prepared observation data.
    max_spatial_intermediate_array_memory_gb : float
        Memory cap (GB) for spatial intermediate arrays produced during MVN chunking.
    model_type : constants.ModelType
        Either GEOLOGY or TERRAIN; used for progress-bar labelling.
    max_dist_m : float, optional
        Maximum distance in meters for considering observations.

    Returns
    -------
    affected_flat_indices : ndarray
        (N_affected,) flat indices into ``raster_data.vs30`` for pixels inside
        at least one observation's bounding box.
    affected_locs : ndarray
        (N_affected, 2) NZTM pixel-center coordinates, aligned with
        ``affected_flat_indices``.
    """
    grid_locs = raster_data.get_coordinates()
    chunk_size = max(
        1,
        int(
            max_spatial_intermediate_array_memory_gb
            * constants.BYTES_PER_GB
            / len(obs_data.locations)
        ),
    )
    n_chunks = int(np.ceil(len(grid_locs) / chunk_size))

    obs_eastings = obs_data.locations[:, 0:1]
    obs_northings = obs_data.locations[:, 1:2]
    e_min, e_max = obs_eastings - max_dist_m, obs_eastings + max_dist_m
    n_min, n_max = obs_northings - max_dist_m, obs_northings + max_dist_m

    valid_in_bbox = np.zeros(len(grid_locs), dtype=bool)

    chunk_indices = range(n_chunks)
    if n_chunks > 1:
        chunk_indices = tqdm(
            chunk_indices,
            desc=f"{str(model_type).capitalize()}: checking pixels for nearby observations ({n_chunks} chunks)",
            unit="chunk",
        )
    else:
        logger.info(
            f"{str(model_type).capitalize()}: checking {len(grid_locs):,} pixels for nearby observations"
        )

    for chunk_idx in chunk_indices:
        start = chunk_idx * chunk_size
        end = min(start + chunk_size, len(grid_locs))
        valid_in_bbox[start:end] = grid_points_in_bbox(
            grid_locs[start:end], e_min, e_max, n_min, n_max,
        )

    n_affected = int(valid_in_bbox.sum())
    logger.info(
        f"Bounding box search complete: {n_affected:,} pixels affected "
        f"({n_affected / len(grid_locs) * 100:.1f}% of valid pixels)"
    )
    return raster_data.valid_flat_indices[valid_in_bbox], grid_locs[valid_in_bbox]


def compute_spatial_pixel_adjustments(
    raster_data: RasterData,
    obs_data: ObservationData,
    affected_flat_indices: np.ndarray,
    affected_locs: np.ndarray,
    corr_fn: Callable[[np.ndarray], np.ndarray],
    max_dist_m: float = constants.MAX_DIST_M,
    max_points: int = constants.MAX_POINTS,
    noisy: bool = False,
    cov_reduc: float = constants.COV_REDUC,
    max_spatial_intermediate_array_memory_gb: float = constants.MAX_SPATIAL_INTERMEDIATE_ARRAY_MEMORY_GB,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute MVN updates for all affected pixels and return updated arrays.

    Parameters
    ----------
    raster_data : RasterData
        Raster data with vs30/stdv arrays and valid-pixel mask.
    obs_data : ObservationData
        Prepared observation data.
    affected_flat_indices : ndarray
        (N_affected,) flat indices into ``raster_data.vs30`` for pixels to
        update, as returned by ``find_affected_pixels``.
    affected_locs : ndarray
        (N_affected, 2) NZTM pixel-center coordinates aligned with
        ``affected_flat_indices``.
    corr_fn : callable
        Correlation function mapping distances (ndarray) to correlations
        (ndarray).
    max_dist_m : float, optional
        Maximum distance in meters to consider observations.
    max_points : int, optional
        Maximum number of observations per pixel.
    noisy : bool, optional
        If True, down-weight uncertain observations by ``obs_data.noise_weights``.
    cov_reduc : float, optional
        If > 0, shrink covariance between points with dissimilar Vs30 values.
    max_spatial_intermediate_array_memory_gb : float, optional
        Memory cap (GB) for the per-chunk (indices, distances) arrays.

    Returns
    -------
    updated_vs30 : ndarray
        Vs30 array with spatial adjustments applied at affected pixels.
    updated_stdv : ndarray
        Standard deviation array with spatial adjustments applied.
    """
    affected_vs30 = raster_data.vs30.flat[affected_flat_indices]
    affected_stdv = raster_data.stdv.flat[affected_flat_indices]

    updated_vs30 = raster_data.vs30.copy()
    updated_stdv = raster_data.stdv.copy()

    corr_zero = corr_fn(np.array([0.0]))[0]
    n_updated = 0

    chunk_size = max(
        1,
        int(
            max_spatial_intermediate_array_memory_gb
            * constants.BYTES_PER_GB
            / (max_points * constants.BYTES_PER_KDTREE_QUERY_SLOT)
        ),
    )

    with tqdm(
        total=len(affected_flat_indices), desc="Spatial adjustment", unit="pixel"
    ) as pbar:
        for chunk_start in range(0, len(affected_flat_indices), chunk_size):
            chunk_end = min(chunk_start + chunk_size, len(affected_flat_indices))
            batch_indices, batch_distances = select_observations_for_pixel_batch(
                affected_locs[chunk_start:chunk_end],
                obs_data,
                max_dist_m=max_dist_m,
                max_points=max_points,
            )
            for i in range(chunk_end - chunk_start):
                local_idx = chunk_start + i
                flat_idx = affected_flat_indices[local_idx]
                result = compute_spatial_adjustment_for_pixel(
                    PixelData(
                        location=affected_locs[local_idx],
                        vs30=float(affected_vs30[local_idx]),
                        stdv=float(affected_stdv[local_idx]),
                    ),
                    obs_data,
                    batch_indices[i][np.isfinite(batch_distances[i])],
                    corr_fn,
                    corr_zero=corr_zero,
                    noisy=noisy,
                    cov_reduc=cov_reduc,
                )
                if result is not None:
                    vs30, stdv = result
                    updated_vs30.flat[flat_idx] = vs30
                    updated_stdv.flat[flat_idx] = stdv
                    n_updated += 1
                pbar.update(1)

    logger.info(f"Spatial adjustment complete: {n_updated:,} pixels updated")

    return updated_vs30, updated_stdv


def compute_spatial_point_adjustments(
    points: np.ndarray,
    model_vs30: np.ndarray,
    model_stdv: np.ndarray,
    obs_data: ObservationData,
    corr_fn: Callable[[np.ndarray], np.ndarray],
    max_dist_m: float = constants.MAX_DIST_M,
    max_points: int = constants.MAX_POINTS,
    noisy: bool = False,
    cov_reduc: float = constants.COV_REDUC,
    progress_bar: tqdm | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute MVN spatial adjustment at specific query points.

    Point-based equivalent of ``compute_spatial_pixel_adjustments``, using the
    same MVN conditioning algorithm.

    Parameters
    ----------
    points : ndarray
        (N, 2) array of [easting, northing] query locations in NZTM.
    model_vs30 : ndarray
        (N,) array of model Vs30 values at query points (before MVN adjustment).
    model_stdv : ndarray
        (N,) array of model standard deviation at query points.
    obs_data : ObservationData
        Prepared observation data.
    corr_fn : callable
        Correlation function mapping distances (ndarray) to correlations
        (ndarray).
    max_dist_m : float, optional
        Maximum distance in meters to consider observations.
    max_points : int, optional
        Maximum number of observations per point.
    noisy : bool, optional
        If True, down-weight uncertain observations by ``obs_data.noise_weights``.
        Must match the value used when ``obs_data`` was constructed.
    cov_reduc : float, optional
        If > 0, shrink covariance between points with dissimilar Vs30 values.
    progress_bar : tqdm, optional
        External progress bar to advance per point. If None, no progress is shown.

    Returns
    -------
    mvn_vs30 : ndarray
        (N,) array of spatially adjusted Vs30 values.
    mvn_stdv : ndarray
        (N,) array of spatially adjusted standard deviation values.
    """
    mvn_vs30 = model_vs30.copy()
    mvn_stdv = model_stdv.copy()

    if len(obs_data.locations) == 0:
        logger.warning("No valid observations for MVN adjustment")
        return mvn_vs30, mvn_stdv

    corr_zero = corr_fn(np.array([0.0]))[0]

    batch_indices, batch_distances = select_observations_for_pixel_batch(
        points, obs_data, max_dist_m=max_dist_m, max_points=max_points,
    )

    for i in range(len(points)):
        result = compute_spatial_adjustment_for_pixel(
            PixelData(
                location=points[i],
                vs30=float(model_vs30[i]),
                stdv=float(model_stdv[i]),
            ),
            obs_data,
            batch_indices[i][np.isfinite(batch_distances[i])],
            corr_fn,
            corr_zero=corr_zero,
            noisy=noisy,
            cov_reduc=cov_reduc,
        )
        if result is not None:
            mvn_vs30[i] = result[0]
            mvn_stdv[i] = result[1]
        if progress_bar is not None:
            progress_bar.update(1)

    return mvn_vs30, mvn_stdv


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
    max_spatial_intermediate_array_memory_gb: float = constants.MAX_SPATIAL_INTERMEDIATE_ARRAY_MEMORY_GB,
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
        Categorical Vs30 values; DataFrame with an ID column plus mean/stdv
        columns.
    model_type : ModelType
        Either GEOLOGY or TERRAIN.
    corr_fn : Callable
        Correlation function mapping distances (ndarray) to correlations
        (ndarray).
    apply_alluvium_slope_mod : bool
        Whether to apply slope-based interpolation for GID 4 (alluvium).
    apply_coastal_distance_mod : bool
        Whether to apply coastal-distance modification for GID 4 and GID 10.
    noisy : bool, optional
        If True, apply noise-aware weighting to residuals and the covariance
        matrix (via ``obs_data.noise_weights``).
    max_spatial_intermediate_array_memory_gb : float, optional
        Memory cap (GB) for spatial intermediate arrays produced during MVN chunking.
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

    raster_data = RasterData.from_arrays(
        vs30=vs30_array,
        stdv=stdv_array,
        transform=profile["transform"],
        nodata=constants.NODATA_VALUE,
    )
    validate_raster_data(raster_data)
    validate_observations(observations_df)

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
    obs_data = prepare_observation_data(
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
    logger.info(f"Prepared {len(obs_data.locations)} valid observations")

    if len(obs_data.locations) == 0:
        logger.warning(
            "No valid observations found within model bounds. "
            "Returning input arrays unchanged."
        )
        return vs30_array.copy(), stdv_array.copy()

    t_bbox_start = time.perf_counter()
    affected_flat_indices, affected_locs = find_affected_pixels(
        raster_data,
        obs_data,
        max_spatial_intermediate_array_memory_gb=max_spatial_intermediate_array_memory_gb,
        model_type=model_type,
        max_dist_m=constants.MAX_DIST_M,
    )
    logger.info(
        f"Found {len(affected_flat_indices):,} affected pixels in "
        f"{time.perf_counter() - t_bbox_start:.1f}s"
    )

    t_spatial_start = time.perf_counter()
    adjusted_vs30, adjusted_stdv = compute_spatial_pixel_adjustments(
        raster_data,
        obs_data,
        affected_flat_indices,
        affected_locs,
        corr_fn,
        max_dist_m=constants.MAX_DIST_M,
        max_points=constants.MAX_POINTS,
        noisy=noisy,
        cov_reduc=constants.COV_REDUC,
        max_spatial_intermediate_array_memory_gb=max_spatial_intermediate_array_memory_gb,
    )
    logger.info(
        f"Spatial adjustments completed in "
        f"{time.perf_counter() - t_spatial_start:.1f}s"
    )

    return adjusted_vs30, adjusted_stdv

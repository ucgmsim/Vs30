"""
Pipeline functions for generating Vs30 models.
"""

import functools
import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from qcore import coordinates
from tqdm import tqdm

from vs30 import category, config, constants, gapfill, parallel, raster, spatial, utils

logger = logging.getLogger(__name__)


def _default_correlation_functions(
    geology_corr_fn: Callable | None,
    terrain_corr_fn: Callable | None,
) -> tuple[Callable, Callable]:
    """
    Fill in default exponential correlation functions where None is given.

    Parameters
    ----------
    geology_corr_fn : callable or None
        Geology correlation function. If None, uses exponential with phi=1407.
    terrain_corr_fn : callable or None
        Terrain correlation function. If None, uses exponential with phi=993.

    Returns
    -------
    tuple[callable, callable]
        (geology_corr_fn, terrain_corr_fn) with defaults filled in.
    """
    if geology_corr_fn is None:
        geology_corr_fn = functools.partial(
            utils.exponential_correlation_function, phi=constants.DEFAULT_GEOLOGY_PHI
        )
    if terrain_corr_fn is None:
        terrain_corr_fn = functools.partial(
            utils.exponential_correlation_function, phi=constants.DEFAULT_TERRAIN_PHI
        )
    return geology_corr_fn, terrain_corr_fn


def _collect_observation_csvs(
    clustered_observations_csv: Path | None,
    independent_observations_csv: Path | None,
) -> pd.DataFrame:
    """
    Collect and concatenate available observation CSV files into a single DataFrame.

    Parameters
    ----------
    clustered_observations_csv : Path or None
        Path to clustered observations CSV.
    independent_observations_csv : Path or None
        Path to independent observations CSV.

    Returns
    -------
    pd.DataFrame
        Concatenated observations, or an empty DataFrame if no files are available.
    """
    csvs = [
        csv
        for csv in [clustered_observations_csv, independent_observations_csv]
        if csv is not None and csv.exists()
    ]
    if csvs:
        return pd.concat(
            [pd.read_csv(csv, comment="#") for csv in csvs],
            ignore_index=True,
        )
    return pd.DataFrame(columns=constants.ObservationColumn.REQUIRED)  # ty: ignore[invalid-argument-type]


# ============================================================================
# Stage 1: Bayesian update of categorical model values
# ============================================================================


def compute_categorical_vs30_updates(
    categorical_model_csv: Path,
    model_type: constants.ModelType,
    clustered_observations_csv: Path | None = None,
    independent_observations_csv: Path | None = None,
    n_proc: int = 1,
) -> pd.DataFrame:
    """
    Compute Bayesian updates to categorical model values and return as DataFrame.

    Loads observations and categorical model values, applies Bayesian updates to the
    categorical model values (mean and standard deviation per category), and returns
    the updated DataFrame.

    Can process clustered observations (with spatial clustering) and/or independent
    observations (without clustering). If both are provided, clustered observations
    are processed first, and the resulting posterior is used as the prior for
    independent observations.

    The order (clustered first, then independent) is scientifically motivated:
    clustered observations (e.g., CPT data) may have spatial sampling biases from
    geotechnical investigations, so clustering corrects for over-weighting dense
    samples. Independent observations (e.g., direct Vs30 measurements) are typically
    higher-quality and more representative, so they refine the bias-corrected model
    from clustered data.

    Parameters
    ----------
    categorical_model_csv : Path
        Path to CSV file with categorical Vs30 mean and standard deviation values
        (e.g., geology_model_prior_mean_and_standard_deviation.csv).
    model_type : ModelType
        Model type: either GEOLOGY or TERRAIN.
    clustered_observations_csv : Path, optional
        Path to CSV file with clustered observations (e.g., viktor_inferred_vs30_from_cpt.csv).
        These will be processed with spatial clustering.
    independent_observations_csv : Path, optional
        Path to CSV file with independent observations
        (e.g., modified_foster_2019_measured_vs30_independent_observations.csv).
        These will be processed without clustering.
    n_proc : int, optional
        Number of processes for DBSCAN clustering. Use -1 for all available cores.

    Returns
    -------
    pd.DataFrame
        Updated categorical model with posterior mean and stdv columns.

    Raises
    ------
    ValueError
        If neither observations CSV is provided, if model_type is invalid, or if
        required CSV columns are missing.
    """
    if clustered_observations_csv is None and independent_observations_csv is None:
        raise ValueError(
            "At least one of clustered_observations_csv or "
            "independent_observations_csv must be provided"
        )

    if model_type not in constants.ModelType:
        raise ValueError(f"model_type must be a valid ModelType, got '{model_type}'")

    logger.info(f"Model type: {model_type}")
    logger.info(f"Loading categorical model from: {categorical_model_csv}")

    categorical_model_df = pd.read_csv(categorical_model_csv, skipinitialspace=True)

    # Drop rows with placeholder values for excluded categories (e.g., water)
    categorical_model_df = categorical_model_df[
        categorical_model_df[constants.COL_MEAN] != constants.NODATA_VALUE
    ]

    utils.validate_csv_columns(
        categorical_model_df,
        [constants.COL_MEAN, constants.COL_STDV],
        "Categorical model CSV",
    )

    # Current prior (will be updated as we process observations)
    current_prior_df = categorical_model_df.copy()

    # Load clustered observations if provided
    clustered_observations_df = None
    if clustered_observations_csv is not None:
        logger.info(
            f"Loading clustered observations from: {clustered_observations_csv}"
        )
        clustered_observations_df = pd.read_csv(
            clustered_observations_csv, skipinitialspace=True, comment="#"
        )

        utils.validate_csv_columns(
            clustered_observations_df,
            constants.ObservationColumn.REQUIRED,
            "Clustered observations CSV",
        )

        logger.info(f"Loaded {len(clustered_observations_df)} clustered observations")

        # Assign category IDs
        obs_locs = clustered_observations_df[
            [constants.ObservationColumn.EASTING, constants.ObservationColumn.NORTHING]
        ].values
        if model_type == constants.ModelType.GEOLOGY:
            model_ids = category.assign_to_category_geology(obs_locs)
        else:  # terrain
            model_ids = category.assign_to_category_terrain(obs_locs)

        clustered_observations_df[constants.STANDARD_ID_COLUMN] = model_ids

        # Log assignment statistics
        unique_assigned_ids = clustered_observations_df[
            constants.STANDARD_ID_COLUMN
        ].unique()
        n_valid = np.count_nonzero(
            clustered_observations_df[constants.STANDARD_ID_COLUMN]
            != constants.RASTER_ID_NODATA_VALUE
        )
        logger.info(
            f"Assigned category IDs: {n_valid} valid observations "
            f"(out of {len(clustered_observations_df)} total)"
        )
        logger.info(
            f"Unique category IDs in observations: {sorted(unique_assigned_ids[unique_assigned_ids != constants.RASTER_ID_NODATA_VALUE])[:20]}"
        )
        logger.info(
            f"Category IDs in prior model: {sorted(current_prior_df[constants.STANDARD_ID_COLUMN].unique())}"
        )

        # Perform clustering
        logger.info("Performing spatial clustering...")
        clustered_observations_df = category.perform_clustering(
            clustered_observations_df, n_proc
        )

    # Load independent observations if provided
    independent_observations_df = None
    if independent_observations_csv is not None:
        logger.info(
            f"Loading independent observations from: {independent_observations_csv}"
        )
        independent_observations_df = pd.read_csv(
            independent_observations_csv, skipinitialspace=True, comment="#"
        )

        utils.validate_csv_columns(
            independent_observations_df,
            constants.ObservationColumn.REQUIRED,
            "Independent observations CSV",
        )

        logger.info(
            f"Loaded {len(independent_observations_df)} independent observations"
        )

        # Assign category IDs
        obs_locs = independent_observations_df[
            [constants.ObservationColumn.EASTING, constants.ObservationColumn.NORTHING]
        ].values
        if model_type == constants.ModelType.GEOLOGY:
            model_ids = category.assign_to_category_geology(obs_locs)
        else:  # terrain
            model_ids = category.assign_to_category_terrain(obs_locs)

        independent_observations_df[constants.STANDARD_ID_COLUMN] = model_ids

    # Perform Bayesian update(s)
    logger.info("Applying Bayesian updates...")
    if clustered_observations_df is not None:
        current_prior_df = category.update_with_clustered_data(
            current_prior_df, clustered_observations_df
        )

    if independent_observations_df is not None:
        current_prior_df = category.update_with_independent_data(
            current_prior_df, independent_observations_df
        )

    return current_prior_df


# ============================================================================
# Stage 2: Create initial VS30 arrays from categorical model
# ============================================================================


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
    grid_params = {
        "xmin": grid_config.grid_xmin,
        "xmax": grid_config.grid_xmax,
        "ymin": grid_config.grid_ymin,
        "ymax": grid_config.grid_ymax,
        "dx": grid_config.grid_dx,
        "dy": grid_config.grid_dy,
    }
    logger.info(f"Using grid parameters: {grid_params}")

    logger.info(f"Creating {model_type} category ID array...")
    id_array, profile = raster.create_category_id_array(model_type, **grid_params)

    logger.info(f"Creating {model_type} VS30 arrays from IDs...")
    vs30_array, stdv_array = raster.create_vs30_arrays_from_ids(
        id_array, model_values_df, model_type=model_type
    )

    return vs30_array, stdv_array, id_array, profile


# ============================================================================
# Stage 3: Hybrid geology modifications (slope and coastal distance)
# ============================================================================


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


# ============================================================================
# Stage 4: MVN spatial adjustment on grid
# ============================================================================


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
    n_proc: int = 1,
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
    n_proc : int, optional
        Number of parallel processes. Use -1 for all cores.
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
    n_proc_resolved = parallel.resolve_n_proc(n_proc)

    logger.info(f"Starting spatial adjustment for {model_type} model")

    # 1. Construct RasterData from arrays
    raster_data = spatial.RasterData.from_arrays(
        vs30=vs30_array,
        stdv=stdv_array,
        transform=profile["transform"],
        crs=profile.get("crs", constants.NZTM_CRS),
        nodata=constants.NODATA_VALUE,
    )
    spatial.validate_raster_data(raster_data)

    # 2. Validate observations
    spatial.validate_observations(observations_df)

    # 3. Build updated model table from DataFrame
    # Model IDs are 1-indexed; convert to 0-indexed array indices
    mean_col, std_col = raster.select_vs30_columns_by_priority(
        list(model_values_df.columns)
    )
    max_id = model_values_df[constants.STANDARD_ID_COLUMN].max()
    updated_model_table = np.full((max_id, 2), np.nan)
    ids = model_values_df[constants.STANDARD_ID_COLUMN].values.astype(int) - 1
    valid = (ids >= 0) & (ids < max_id)
    updated_model_table[ids[valid], 0] = model_values_df[mean_col].values[valid]
    updated_model_table[ids[valid], 1] = model_values_df[std_col].values[valid]

    # 4. Prepare Observation Data for Spatial Adjustment
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

    # With many observations, pixels frequently hit the MAX_POINTS cap,
    # producing large covariance matrices. In this regime, letting BLAS
    # parallelise each matrix inverse (n_proc=1) is much faster than
    # Python-level multiprocessing with single-threaded BLAS.
    if (
        n_proc_resolved > 1
        and n_obs > constants.MULTIPROCESS_OBSERVATION_THRESHOLD
    ):
        logger.info(
            f"Falling back to single-process mode: {n_obs} observations "
            f"exceeds threshold ({constants.MULTIPROCESS_OBSERVATION_THRESHOLD}). "
            f"BLAS will parallelise matrix inversions across all cores."
        )
        n_proc_resolved = 1

    # 5. Find Affected Pixels
    logger.info("Finding pixels affected by observations...")
    t_bbox_start = time.perf_counter()
    bbox_result = spatial.find_affected_pixels(
        raster_data,
        obs_data,
        max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
        model_type=model_type,
        max_dist_m=constants.MAX_DIST_M,
        n_proc=n_proc_resolved,
    )
    t_bbox_elapsed = time.perf_counter() - t_bbox_start
    print(f"  find_affected_pixels: {t_bbox_elapsed:.1f}s "
          f"({bbox_result.n_affected_pixels:,} affected pixels)")
    logger.info(
        f"Found {bbox_result.n_affected_pixels:,} affected pixels "
        f"in {t_bbox_elapsed:.1f}s"
    )

    # 6. Compute Spatial Adjustments
    logger.info("Computing spatial updates...")
    t_spatial_start = time.perf_counter()
    if n_proc_resolved > 1:
        logger.info(f"Using {n_proc_resolved} parallel workers")
        affected_flat_indices = np.where(bbox_result.mask)[0]
        updates = parallel.run_parallel_spatial_fit(
            affected_flat_indices=affected_flat_indices,
            raster_data=raster_data,
            obs_data=obs_data,
            corr_fn=corr_fn,
            model_type=model_type,
            max_dist_m=constants.MAX_DIST_M,
            max_points=constants.MAX_POINTS,
            noisy=noisy,
            cov_reduc=constants.COV_REDUC,
            n_proc=n_proc_resolved,
        )
    else:
        updates = spatial.compute_spatial_adjustments(
            raster_data,
            obs_data,
            bbox_result,
            corr_fn,
            max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
            max_dist_m=constants.MAX_DIST_M,
            max_points=constants.MAX_POINTS,
            noisy=noisy,
            cov_reduc=constants.COV_REDUC,
        )
    t_spatial_elapsed = time.perf_counter() - t_spatial_start
    print(f"  compute_spatial_adjustments: {t_spatial_elapsed:.1f}s")
    logger.info(f"Spatial adjustments completed in {t_spatial_elapsed:.1f}s")

    # 7. Apply Updates (in memory)
    logger.info("Applying updates...")
    t_apply_start = time.perf_counter()
    adjusted_vs30, adjusted_stdv = spatial.apply_updates(raster_data, updates)
    t_apply_elapsed = time.perf_counter() - t_apply_start
    print(f"  apply_updates: {t_apply_elapsed:.1f}s")
    logger.info(f"Updates applied in {t_apply_elapsed:.1f}s")

    return adjusted_vs30, adjusted_stdv


# ============================================================================
# Stage 5: Combine geology and terrain models
# ============================================================================


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
        No-data value. Default from constants.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        (combined_vs30, combined_stdv) arrays.
    """
    if (
        combination_method is constants.CombinationMethod.RATIO
        and combine_ratio is None
    ):
        raise ValueError(
            "combination_method is set to 'ratio' but combine_ratio is not provided"
        )

    # Work on copies to avoid modifying inputs
    geol_vs30 = np.array(geol_vs30, dtype=np.float32, copy=True)
    geol_stdv = np.array(geol_stdv, dtype=np.float32, copy=True)
    terr_vs30 = np.array(terr_vs30, dtype=np.float32, copy=True)
    terr_stdv = np.array(terr_stdv, dtype=np.float32, copy=True)

    # Replace nodata with NaN for calculation
    geol_vs30[geol_vs30 == nodata] = np.nan
    geol_stdv[geol_stdv == nodata] = np.nan
    terr_vs30[terr_vs30 == nodata] = np.nan
    terr_stdv[terr_stdv == nodata] = np.nan

    combined_vs30, combined_stdv = utils.combine_vs30_models(
        geol_vs30=geol_vs30,
        geol_stdv=geol_stdv,
        terr_vs30=terr_vs30,
        terr_stdv=terr_stdv,
        combination_method=combination_method,
        combine_ratio=combine_ratio,
    )

    return combined_vs30, combined_stdv


# ============================================================================
# Raster file writing helper
# ============================================================================


def write_vs30_raster(
    vs30_array: np.ndarray,
    stdv_array: np.ndarray,
    profile: dict,
    output_path: Path,
    band1_description: str = constants.BAND_DESCRIPTION_VS30,
    band2_description: str = constants.BAND_DESCRIPTION_STDV,
    nodata: float = constants.NODATA_VALUE,
) -> None:
    """
    Write a 2-band VS30 raster (mean + standard deviation) to a GeoTIFF file.

    Parameters
    ----------
    vs30_array : np.ndarray
        VS30 mean values (2D array).
    stdv_array : np.ndarray
        VS30 standard deviation values (2D array).
    profile : dict
        Rasterio profile with CRS, transform, dimensions.
    output_path : Path
        Output file path.
    band1_description : str, optional
        Description for band 1. Default from constants.
    band2_description : str, optional
        Description for band 2. Default from constants.
    nodata : float, optional
        No-data value. Default from constants.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_profile = profile.copy()
    write_profile.update(
        {
            "dtype": "float32",
            "count": 2,
            "nodata": nodata,
            "compress": "deflate",
        }
    )

    with rasterio.open(output_path, "w", **write_profile) as dst:
        dst.write(vs30_array.astype(np.float32), 1)
        dst.write(stdv_array.astype(np.float32), 2)
        dst.descriptions = (band1_description, band2_description)

    logger.info(f"Wrote raster: {output_path}")


def write_single_band_raster(
    array: np.ndarray,
    profile: dict,
    output_path: Path,
    band_description: str,
    nodata: float | None = None,
) -> None:
    """
    Write a single-band raster to a GeoTIFF file.

    Parameters
    ----------
    array : np.ndarray
        2D array to write.
    profile : dict
        Rasterio profile with CRS, transform, dimensions.
    output_path : Path
        Output file path.
    band_description : str
        Description for the band.
    nodata : float or None, optional
        No-data value. If None, no nodata value is set.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    write_profile = profile.copy()
    write_profile.update(
        {
            "dtype": "float32",
            "count": 1,
            "nodata": nodata,
            "compress": "deflate",
        }
    )

    with rasterio.open(output_path, "w", **write_profile) as dst:
        dst.write(array.astype(np.float32), 1)
        dst.descriptions = (band_description,)


def write_id_raster(
    id_array: np.ndarray,
    profile: dict,
    output_path: Path,
) -> None:
    """
    Write a category ID raster to a GeoTIFF file.

    Parameters
    ----------
    id_array : np.ndarray
        Category ID array (uint8).
    profile : dict
        Rasterio profile with CRS, transform, dimensions.
    output_path : Path
        Output file path.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(id_array, 1)
        dst.descriptions = (constants.BAND_DESCRIPTION_ID_INDEX,)


# ============================================================================
# Full pipeline for a single model type
# ============================================================================


def compute_model_grid(
    model_type: constants.ModelType,
    grid_config: config.GridConfig,
    apply_alluvium_slope_mod: bool,
    categorical_model_csv: Path | None = None,
    clustered_observations_csv: Path | None = None,
    independent_observations_csv: Path | None = None,
    do_bayesian_update: bool = True,
    mvn: bool = True,
    noisy: bool = True,
    n_proc: int = 1,
    max_spatial_boolean_array_memory_gb: float = 1.0,
    output_dir: Path | None = None,
    include_intermediate: bool = False,
    corr_fn: Callable | None = None,
    apply_coastal_distance_mod: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Run the full VS30 generation pipeline for a single model type in memory.

    Executes the complete pipeline in sequence:

    1. Bayesian update of categorical model values using observations (conditional).
    2. Create initial VS30 arrays from categorical model.
    3. Apply hybrid modifications for slope and coastal distance (geology only).
    4. Spatial adjustment using MVN conditioning with observations (conditional).

    When output_dir is provided, intermediate files are written at each stage
    (gated by include_intermediate for non-final outputs).

    Parameters
    ----------
    model_type : ModelType
        Model type: either GEOLOGY or TERRAIN.
    grid_config : GridConfig
        Grid domain and resolution parameters.
    apply_alluvium_slope_mod : bool
        Whether to apply slope-based interpolation for GID 4 (alluvium).
    categorical_model_csv : Path
        Path to CSV file with categorical Vs30 values.
    clustered_observations_csv : Path, optional
        Path to CSV file with clustered observations (e.g., CPT data).
    independent_observations_csv : Path, optional
        Path to CSV file with independent observations (e.g., measured filtered).
    do_bayesian_update : bool, optional
        Whether to perform Bayesian update of categorical model values.
    mvn : bool, optional
        Whether to perform MVN spatial adjustment. If False, spatial fit is skipped.
    noisy : bool, optional
        Whether to apply noise weighting in spatial adjustment.
    n_proc : int, optional
        Number of parallel processes. Use -1 for all cores.
    max_spatial_boolean_array_memory_gb : float, optional
        Maximum memory for spatial boolean arrays.
    output_dir : Path, optional
        Directory to write intermediate and final rasters. If None, no files
        are written.
    include_intermediate : bool, optional
        Whether to write intermediate files (ID rasters, initial VS30, slope,
        coast distance, hybrid geology). Default False.
    corr_fn : Callable, optional
        Correlation function for spatial adjustment.
    apply_coastal_distance_mod : bool
        Whether to apply coastal distance modification for GID 4 and GID 10.

    Returns
    -------
    tuple[ndarray, ndarray, ndarray, dict]
        (vs30, stdv, id_array, profile) — the final VS30 array, standard
        deviation array, category ID array, and rasterio profile.
    """
    if categorical_model_csv is None:
        raise ValueError(
            f"categorical_model_csv is required for {model_type} pipeline. "
            "Specify it in the config YAML or pass it explicitly."
        )

    if output_dir is not None:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting full pipeline for {model_type}")

    # --- Step 1: Bayesian update of categorical model values (conditional) ---
    if do_bayesian_update:
        logger.info("\n=== STEP 1: Updating Categorical Models ===")
        posterior_df = compute_categorical_vs30_updates(
            categorical_model_csv=categorical_model_csv,
            model_type=model_type,
            clustered_observations_csv=clustered_observations_csv,
            independent_observations_csv=independent_observations_csv,
            n_proc=n_proc,
        )

        if output_dir is not None and include_intermediate:
            posterior_csv_path = (
                output_dir / f"{constants.POSTERIOR_PREFIX}{categorical_model_csv.name}"
            )
            posterior_df.to_csv(posterior_csv_path, index=False)
    else:
        logger.info(
            "\n=== STEP 1: SKIPPED - Using prior categorical models directly ==="
        )
        posterior_df = pd.read_csv(categorical_model_csv, skipinitialspace=True)

    # --- Step 2: Create initial VS30 arrays from categorical model ---
    logger.info("\n=== STEP 2: Creating Initial VS30 Arrays ===")
    vs30_array, stdv_array, id_array, profile = create_initial_vs30_arrays(
        grid_config, model_type, posterior_df
    )

    if output_dir is not None and include_intermediate:
        id_filename = (
            constants.GEOLOGY_ID_FILENAME
            if model_type == constants.ModelType.GEOLOGY
            else constants.TERRAIN_ID_FILENAME
        )
        write_id_raster(id_array, profile, output_dir / id_filename)

        initial_filename = (
            constants.GEOLOGY_INITIAL_VS30_FILENAME
            if model_type == constants.ModelType.GEOLOGY
            else constants.TERRAIN_INITIAL_VS30_FILENAME
        )
        write_vs30_raster(
            vs30_array, stdv_array, profile, output_dir / initial_filename
        )

    # --- Step 3: Apply hybrid modifications (geology only) ---
    current_vs30 = vs30_array
    current_stdv = stdv_array
    slope_array = None
    coast_dist_array = None

    if model_type == constants.ModelType.GEOLOGY:
        logger.info("\n=== STEP 3: Slope and Coastal Distance Adjusted Geology ===")

        current_vs30, current_stdv, slope_array, coast_dist_array = (
            compute_hybrid_geology_arrays(
                vs30_array,
                stdv_array,
                id_array,
                profile,
                apply_coastal_distance_mod=apply_coastal_distance_mod,
                apply_alluvium_slope_mod=apply_alluvium_slope_mod,
            )
        )

        if output_dir is not None and include_intermediate:
            write_single_band_raster(
                slope_array,
                profile,
                output_dir / constants.SLOPE_RASTER_FILENAME,
                constants.BAND_DESCRIPTION_SLOPE,
                nodata=constants.NODATA_VALUE,
            )
            write_single_band_raster(
                coast_dist_array,
                profile,
                output_dir / constants.COAST_DISTANCE_RASTER_FILENAME,
                constants.BAND_DESCRIPTION_COAST_DISTANCE,
                nodata=None,
            )
            write_vs30_raster(
                current_vs30,
                current_stdv,
                profile,
                output_dir
                / constants.GEOLOGY_VS30_SLOPE_AND_COASTAL_DISTANCE_ADJUSTED_FILENAME,
                constants.BAND_DESCRIPTION_VS30_HYBRID,
                constants.BAND_DESCRIPTION_STDV_HYBRID,
            )

    # --- Step 4: MVN spatial adjustment using observations (conditional) ---
    if mvn:
        if corr_fn is None:
            raise ValueError("corr_fn must be provided for spatial adjustment.")

        logger.info("\n=== STEP 4: Spatial Adjustment ===")

        observations_df = _collect_observation_csvs(
            clustered_observations_csv, independent_observations_csv
        )
        if len(observations_df) == 0:
            raise ValueError(
                "No observation CSVs provided for spatial fit. "
                "At least one of clustered or independent observations must be specified."
            )

        current_vs30, current_stdv = compute_spatial_adjustment_on_grid(
            vs30_array=current_vs30,
            stdv_array=current_stdv,
            profile=profile,
            observations_df=observations_df,
            model_values_df=posterior_df,
            model_type=model_type,
            corr_fn=corr_fn,
            apply_alluvium_slope_mod=apply_alluvium_slope_mod,
            apply_coastal_distance_mod=apply_coastal_distance_mod,
            noisy=noisy,
            n_proc=n_proc,
            max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
            slope_array=slope_array,
            coast_dist_array=coast_dist_array,
        )
    else:
        logger.info("\n=== STEP 4: SKIPPED - MVN spatial adjustment disabled ===")

    if output_dir is not None:
        output_filename = constants.OUTPUT_FILENAMES[model_type]
        write_vs30_raster(
            current_vs30, current_stdv, profile, output_dir / output_filename
        )

    logger.info(f"\nFull pipeline for {model_type} completed successfully")

    return current_vs30, current_stdv, id_array, profile


# ============================================================================
# Full grid pipeline (orchestration)
# ============================================================================


def grid_pipeline(
    grid_config: config.GridConfig,
    apply_alluvium_slope_mod: bool,
    output_dir: Path | None = None,
    model_type: constants.ModelType = constants.ModelType.COMBINED,
    geology_categorical_csv: Path | None = None,
    terrain_categorical_csv: Path | None = None,
    clustered_observations_csv: Path | None = None,
    independent_observations_csv: Path | None = None,
    combination_method: constants.CombinationMethod = constants.CombinationMethod.STANDARD_DEVIATION_WEIGHTING,
    combine_ratio: float | None = None,
    noisy: bool = True,
    mvn: bool = True,
    do_bayesian_update: bool = True,
    include_intermediate: bool = False,
    n_proc: int = 1,
    max_spatial_boolean_array_memory_gb: float = 1.0,
    geology_corr_fn: Callable | None = None,
    terrain_corr_fn: Callable | None = None,
    apply_coastal_distance_mod: bool = True,
    fill_gaps: bool = False,
) -> dict[str, np.ndarray | dict | None]:
    """
    Run the full VS30 generation pipeline on a raster grid.

    Executes the VS30 pipeline for the requested model type(s) on a raster grid
    defined by grid_config. For COMBINED mode (default), runs both geology and
    terrain pipelines then combines the results.

    Each single-model pipeline runs the following stages:
    1. Bayesian update of categorical model values using observations (conditional).
    2. Create initial VS30 raster from categorical model.
    3. Apply hybrid modifications for slope and coastal distance (geology only).
    4. MVN spatial adjustment using observations (conditional).

    For COMBINED mode, an additional stage combines the two models:
    5. Combine geology and terrain models using weighted average.

    Parameters
    ----------
    grid_config : GridConfig
        Grid domain and resolution parameters.
    apply_alluvium_slope_mod : bool
        Whether to apply slope-based interpolation for GID 4 (alluvium).
    output_dir : Path, optional
        Directory to save all pipeline outputs (intermediate and final rasters).
        If None, no files are written.
    model_type : ModelType, optional
        Which model(s) to run: GEOLOGY, TERRAIN, or COMBINED (default).
    geology_categorical_csv : Path
        Path to geology categorical CSV.
    terrain_categorical_csv : Path
        Path to terrain categorical CSV.
    clustered_observations_csv : Path, optional
        Path to CSV file with clustered observations (e.g., CPT data).
    independent_observations_csv : Path, optional
        Path to CSV file with independent observations.
    combination_method : CombinationMethod, optional
        Method for combining models: STANDARD_DEVIATION_WEIGHTING (default)
        or RATIO.
    combine_ratio : float, optional
        Geology-to-terrain weight ratio. Required when combination_method is RATIO.
    noisy : bool, optional
        Whether to apply noise weighting in spatial adjustment.
    mvn : bool, optional
        Whether to perform MVN spatial adjustment. If False, spatial fit is skipped.
    do_bayesian_update : bool, optional
        Whether to perform Bayesian update of categorical model values.
    include_intermediate : bool, optional
        Whether to write intermediate files (ID rasters, initial VS30, slope,
        coast distance, hybrid geology). Default False.
    n_proc : int, optional
        Number of parallel processes. Use -1 for all cores.
    max_spatial_boolean_array_memory_gb : float, optional
        Maximum memory for spatial boolean arrays.
    geology_corr_fn : Callable, optional
        Correlation function for geology spatial adjustment.
    terrain_corr_fn : Callable, optional
        Correlation function for terrain spatial adjustment.
    apply_coastal_distance_mod : bool
        Whether to apply coastal distance modification for GID 4 and GID 10.
    fill_gaps : bool
        Whether to fill on-land nodata gaps in the combined output using
        nearest-neighbor interpolation.

    Returns
    -------
    dict
        Dictionary containing the computed raster data with keys:

        - ``"geology_vs30"``, ``"geology_stdv"`` : 2D arrays (when geology is computed)
        - ``"terrain_vs30"``, ``"terrain_stdv"`` : 2D arrays (when terrain is computed)
        - ``"combined_vs30"``, ``"combined_stdv"`` : 2D arrays (when both models are computed)
        - ``"profile"`` : rasterio profile dict with CRS, transform, dimensions, etc.
    """
    start_time = time.time()

    geology_corr_fn, terrain_corr_fn = _default_correlation_functions(
        geology_corr_fn, terrain_corr_fn
    )

    if output_dir is not None:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    run_geology = model_type in (
        constants.ModelType.GEOLOGY,
        constants.ModelType.COMBINED,
    )
    run_terrain = model_type in (
        constants.ModelType.TERRAIN,
        constants.ModelType.COMBINED,
    )

    result: dict[str, np.ndarray | dict | None] = {}
    profile: dict | None = None

    # 1. Run Geology Pipeline
    if run_geology:
        logger.info("\n" + "=" * 80 + "\nRUNNING GEOLOGY PIPELINE\n" + "=" * 80)
        geol_vs30, geol_stdv, geol_ids, profile = compute_model_grid(
            model_type=constants.ModelType.GEOLOGY,
            grid_config=grid_config,
            categorical_model_csv=geology_categorical_csv,
            clustered_observations_csv=clustered_observations_csv,
            independent_observations_csv=independent_observations_csv,
            do_bayesian_update=do_bayesian_update,
            mvn=mvn,
            noisy=noisy,
            n_proc=n_proc,
            max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
            output_dir=output_dir,
            include_intermediate=include_intermediate,
            corr_fn=geology_corr_fn,
            apply_coastal_distance_mod=apply_coastal_distance_mod,
            apply_alluvium_slope_mod=apply_alluvium_slope_mod,
        )
        result["geology_vs30"] = geol_vs30
        result["geology_stdv"] = geol_stdv
        result["geology_ids"] = geol_ids

    # 2. Run Terrain Pipeline
    if run_terrain:
        logger.info("\n" + "=" * 80 + "\nRUNNING TERRAIN PIPELINE\n" + "=" * 80)
        terr_vs30, terr_stdv, _, profile = compute_model_grid(
            model_type=constants.ModelType.TERRAIN,
            grid_config=grid_config,
            categorical_model_csv=terrain_categorical_csv,
            clustered_observations_csv=clustered_observations_csv,
            independent_observations_csv=independent_observations_csv,
            do_bayesian_update=do_bayesian_update,
            mvn=mvn,
            noisy=noisy,
            n_proc=n_proc,
            max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
            output_dir=output_dir,
            include_intermediate=include_intermediate,
            corr_fn=terrain_corr_fn,
            apply_alluvium_slope_mod=apply_alluvium_slope_mod,
            apply_coastal_distance_mod=apply_coastal_distance_mod,
        )
        result["terrain_vs30"] = terr_vs30
        result["terrain_stdv"] = terr_stdv

    # 3. Combine geology and terrain models using weighted average
    if run_geology and run_terrain:
        logger.info(
            "\n" + "=" * 80 + "\nCOMBINING GEOLOGY AND TERRAIN RESULTS\n" + "=" * 80
        )

        combined_vs30, combined_stdv = combine_model_arrays(
            geol_vs30=geol_vs30,
            geol_stdv=geol_stdv,
            terr_vs30=terr_vs30,
            terr_stdv=terr_stdv,
            combination_method=combination_method,
            combine_ratio=combine_ratio,
        )

        if profile is None:
            raise ValueError("profile must not be None when combining model outputs.")

        if fill_gaps:
            # Stage 6: Gap-fill on-land nodata pixels in combined output
            logger.info(
                "\n" + "=" * 80 + "\nSTAGE 6: GAP-FILLING COMBINED OUTPUT\n" + "=" * 80
            )

            if output_dir is not None and include_intermediate:
                write_vs30_raster(
                    np.where(
                        np.isnan(combined_vs30), constants.NODATA_VALUE, combined_vs30
                    ),
                    np.where(
                        np.isnan(combined_stdv), constants.NODATA_VALUE, combined_stdv
                    ),
                    profile,
                    output_dir / constants.COMBINED_VS30_BEFORE_GAPFILL_FILENAME,
                    constants.BAND_DESCRIPTION_VS30_COMBINED,
                    constants.BAND_DESCRIPTION_STDV_COMBINED,
                )

            combined_vs30, combined_stdv = gapfill.fill_nodata_grid(
                combined_vs30, combined_stdv, geol_ids, profile
            )
        result["combined_vs30"] = combined_vs30
        result["combined_stdv"] = combined_stdv

        if output_dir is not None:
            write_vs30_raster(
                np.where(
                    np.isnan(combined_vs30), constants.NODATA_VALUE, combined_vs30
                ),
                np.where(
                    np.isnan(combined_stdv), constants.NODATA_VALUE, combined_stdv
                ),
                profile,
                output_dir / constants.COMBINED_VS30_FILENAME,
                constants.BAND_DESCRIPTION_VS30_COMBINED,
                constants.BAND_DESCRIPTION_STDV_COMBINED,
            )

    result["profile"] = profile

    elapsed_time = time.time() - start_time
    logger.info(f"  Total execution time: {elapsed_time:.1f} seconds")
    if output_dir is not None:
        logger.info(f"  Output available in: {output_dir}")

    return result


# ============================================================================
# Point-based pipeline
# ============================================================================


def points_pipeline(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    apply_alluvium_slope_mod: bool,
    model_type: constants.ModelType = constants.ModelType.COMBINED,
    geology_categorical_csv: Path | None = None,
    terrain_categorical_csv: Path | None = None,
    clustered_observations_csv: Path | None = None,
    independent_observations_csv: Path | None = None,
    combination_method: constants.CombinationMethod = constants.CombinationMethod.STANDARD_DEVIATION_WEIGHTING,
    combine_ratio: float | None = None,
    noisy: bool = True,
    mvn: bool = True,
    do_bayesian_update: bool = False,
    include_intermediate: bool = False,
    n_proc: int = 1,
    geology_corr_fn: Callable | None = None,
    terrain_corr_fn: Callable | None = None,
    apply_coastal_distance_mod: bool = True,
    fill_gaps: bool = False,
    gapfill_grid_config: config.GridConfig = constants.FULL_NZ_GRID_CONFIG,
) -> pd.DataFrame:
    """
    Compute Vs30 values at specific latitude/longitude locations.

    Runs the Vs30 pipeline at the specified query points without generating
    raster grids. This is efficient for querying Vs30 at a small number of
    locations.

    The pipeline stages mirror those in grid_pipeline:
    1. Look up categorical model values at each point.
    2. Apply hybrid modifications for slope and coastal distance (geology only).
    3. MVN spatial adjustment using observations (conditional).
    4. Combine geology and terrain models using weighted average.

    Parameters
    ----------
    longitudes : ndarray
        Array of longitude values (WGS84).
    latitudes : ndarray
        Array of latitude values (WGS84).
    apply_alluvium_slope_mod : bool
        Whether to apply slope-based interpolation for GID 4 (alluvium).
    model_type : ModelType, optional
        Which model(s) to run: GEOLOGY, TERRAIN, or COMBINED (default).
    geology_categorical_csv : Path
        Path to geology categorical CSV.
    terrain_categorical_csv : Path
        Path to terrain categorical CSV.
    clustered_observations_csv : Path, optional
        Path to CSV file with clustered observations (e.g., CPT).
    independent_observations_csv : Path, optional
        Path to CSV file with independent observations.
    combination_method : CombinationMethod, optional
        Method for combining models: STANDARD_DEVIATION_WEIGHTING (default)
        or RATIO.
    combine_ratio : float, optional
        Geology-to-terrain weight ratio. Required when combination_method is RATIO.
    noisy : bool, optional
        Whether to apply noise weighting in spatial adjustment.
    mvn : bool, optional
        Whether to perform MVN spatial adjustment. If False, spatial fit is skipped.
    do_bayesian_update : bool, optional
        Whether to perform Bayesian update of categorical Vs30 values
        using observations before computing Vs30. Default False.
    include_intermediate : bool, optional
        Include intermediate values (geology/terrain separately) in output.
    n_proc : int, optional
        Number of parallel processes. Use -1 for all cores.
    geology_corr_fn : Callable, optional
        Correlation function for geology spatial adjustment.
    terrain_corr_fn : Callable, optional
        Correlation function for terrain spatial adjustment.
    apply_coastal_distance_mod : bool, optional
        Whether to apply coastal distance modification for GID 4 and GID 10.
    fill_gaps : bool
        Whether to fill on-land nodata gaps in the combined output using
        nearest-neighbor interpolation via local grid pipeline.
    gapfill_grid_config : GridConfig, optional
        Grid alignment used when generating local grids for gap-fill in
        points mode. Defaults to the standard NZ domain at 100m spacing.
        Pass explicitly to align with a specific grid mode run.

    Returns
    -------
    DataFrame
        Results with columns: easting, northing, vs30, stdv.
        If include_intermediate is True, also includes geology_id,
        geology_vs30, geology_stdv, geology_vs30_hybrid, geology_stdv_hybrid,
        geology_mvn_vs30, geology_mvn_stdv, terrain_id, terrain_vs30,
        terrain_stdv, terrain_mvn_vs30, terrain_mvn_stdv.
    """
    geology_corr_fn, terrain_corr_fn = _default_correlation_functions(
        geology_corr_fn, terrain_corr_fn
    )

    # Convert WGS84 to NZTM
    nztm_coords = coordinates.wgs_depth_to_nztm(
        np.column_stack([latitudes, longitudes])
    )
    points = nztm_coords[:, ::-1]  # (easting, northing)

    logger.info(f"Processing {len(points)} locations")

    # Load and combine all available observation files for spatial adjustment
    if mvn:
        observations_df = _collect_observation_csvs(
            clustered_observations_csv, independent_observations_csv
        )
    else:
        observations_df = pd.DataFrame(columns=constants.ObservationColumn.REQUIRED)  # ty: ignore[invalid-argument-type]

    logger.info(f"Loaded {len(observations_df)} observations for spatial adjustment")

    run_geology = model_type in (
        constants.ModelType.GEOLOGY,
        constants.ModelType.COMBINED,
    )
    run_terrain = model_type in (
        constants.ModelType.TERRAIN,
        constants.ModelType.COMBINED,
    )

    # Load categorical models (with optional Bayesian update)
    geol_model_df = None
    terr_model_df = None

    if run_geology:
        if geology_categorical_csv is None:
            raise ValueError(
                "geology_categorical_csv is required when running geology model"
            )
        if do_bayesian_update:
            logger.info(
                "Performing Bayesian update of geology categorical model values..."
            )
            geol_model_df = compute_categorical_vs30_updates(
                categorical_model_csv=geology_categorical_csv,
                model_type=constants.ModelType.GEOLOGY,
                clustered_observations_csv=clustered_observations_csv,
                independent_observations_csv=independent_observations_csv,
                n_proc=n_proc,
            )
        else:
            geol_model_df = pd.read_csv(geology_categorical_csv, skipinitialspace=True)

    if run_terrain:
        if terrain_categorical_csv is None:
            raise ValueError(
                "terrain_categorical_csv is required when running terrain model"
            )
        if do_bayesian_update:
            logger.info(
                "Performing Bayesian update of terrain categorical model values..."
            )
            terr_model_df = compute_categorical_vs30_updates(
                categorical_model_csv=terrain_categorical_csv,
                model_type=constants.ModelType.TERRAIN,
                clustered_observations_csv=clustered_observations_csv,
                independent_observations_csv=independent_observations_csv,
                n_proc=n_proc,
            )
        else:
            terr_model_df = pd.read_csv(terrain_categorical_csv, skipinitialspace=True)

    n_proc_resolved = parallel.resolve_n_proc(n_proc)

    # ================================================================
    # Parallel Processing Path
    # ================================================================
    if n_proc_resolved > 1:
        logger.info(f"\nProcessing with {n_proc_resolved} parallel workers...")

        loc_config = parallel.LocationsChunkConfig(
            include_intermediate=include_intermediate,
            model_type=model_type,
            combination_method=combination_method,
            combine_ratio=combine_ratio,
            noisy=noisy,
            geology_corr_fn=geology_corr_fn,
            terrain_corr_fn=terrain_corr_fn,
            apply_coastal_distance_mod=apply_coastal_distance_mod,
            apply_alluvium_slope_mod=apply_alluvium_slope_mod,
        )

        if geol_model_df is None:
            raise ValueError("geol_model_df must not be None for parallel processing.")
        if terr_model_df is None:
            raise ValueError("terr_model_df must not be None for parallel processing.")

        result_df = parallel.run_parallel_locations(
            points=points,
            observations_df=observations_df,
            geol_model_df=geol_model_df,
            terr_model_df=terr_model_df,
            config=loc_config,
            n_proc=n_proc_resolved,
        )

        # Add coordinate columns at the front
        result_df.insert(0, constants.ObservationColumn.EASTING, points[:, 0])
        result_df.insert(1, constants.ObservationColumn.NORTHING, points[:, 1])

        logger.info(f"  Total locations: {len(result_df)}")

    else:
        # ================================================================
        # Sequential Processing Path
        # ================================================================
        result = {}
        result[constants.ObservationColumn.EASTING] = points[:, 0]
        result[constants.ObservationColumn.NORTHING] = points[:, 1]

        # --- Stage 1-3: Geology model (categorical lookup, hybrid mods, spatial adjustment) ---
        if run_geology:
            if geol_model_df is None:
                raise ValueError(
                    "geol_model_df must not be None when running geology model."
                )
            with tqdm(
                total=len(points), desc="Geology: spatial adjustment", unit="point"
            ) as pbar:
                (
                    geol_ids,
                    geol_vs30,
                    geol_stdv,
                    geol_vs30_hybrid,
                    geol_stdv_hybrid,
                    geol_mvn_vs30,
                    geol_mvn_stdv,
                ) = parallel.process_geology_at_points(
                    points,
                    geol_model_df,
                    observations_df,
                    corr_fn=geology_corr_fn,
                    noisy=noisy,
                    progress_bar=pbar,
                    apply_coastal_distance_mod=apply_coastal_distance_mod,
                    apply_alluvium_slope_mod=apply_alluvium_slope_mod,
                )

            if include_intermediate:
                result[constants.COL_GEOLOGY_ID] = geol_ids
                result[constants.COL_GEOLOGY_VS30] = geol_vs30
                result[constants.COL_GEOLOGY_STDV] = geol_stdv
                result[constants.COL_GEOLOGY_VS30_HYBRID] = geol_vs30_hybrid
                result[constants.COL_GEOLOGY_STDV_HYBRID] = geol_stdv_hybrid
                result[constants.COL_GEOLOGY_MVN_VS30] = geol_mvn_vs30
                result[constants.COL_GEOLOGY_MVN_STDV] = geol_mvn_stdv

        # --- Stage 1, 3: Terrain model (categorical lookup, spatial adjustment — no hybrid mods) ---
        if run_terrain:
            if terr_model_df is None:
                raise ValueError(
                    "terr_model_df must not be None when running terrain model."
                )
            with tqdm(
                total=len(points), desc="Terrain: spatial adjustment", unit="point"
            ) as pbar:
                (
                    terr_ids,
                    terr_vs30,
                    terr_stdv,
                    terr_mvn_vs30,
                    terr_mvn_stdv,
                ) = parallel.process_terrain_at_points(
                    points,
                    terr_model_df,
                    observations_df,
                    corr_fn=terrain_corr_fn,
                    noisy=noisy,
                    progress_bar=pbar,
                )

            if include_intermediate:
                result[constants.COL_TERRAIN_ID] = terr_ids
                result[constants.COL_TERRAIN_VS30] = terr_vs30
                result[constants.COL_TERRAIN_STDV] = terr_stdv
                result[constants.COL_TERRAIN_MVN_VS30] = terr_mvn_vs30
                result[constants.COL_TERRAIN_MVN_STDV] = terr_mvn_stdv

        # --- Stage 4: Combine geology and terrain models or use single model result ---
        if run_geology and run_terrain:
            logger.info("Combining models...")
            combined_vs30, combined_stdv = utils.combine_vs30_models(
                geol_mvn_vs30,
                geol_mvn_stdv,
                terr_mvn_vs30,
                terr_mvn_stdv,
                combination_method,
                combine_ratio,
            )
            result[constants.ObservationColumn.VS30] = combined_vs30
            result[constants.COL_COMBINED_STDV] = combined_stdv
        elif run_geology:
            result[constants.ObservationColumn.VS30] = geol_mvn_vs30
            result[constants.COL_COMBINED_STDV] = geol_mvn_stdv
        elif run_terrain:
            result[constants.ObservationColumn.VS30] = terr_mvn_vs30
            result[constants.COL_COMBINED_STDV] = terr_mvn_stdv

        logger.info(f"  Total locations: {len(points)}")
        result_df = pd.DataFrame(result)

    # ================================================================
    # Gap-fill: fill on-land nodata points
    # ================================================================
    if fill_gaps and model_type == constants.ModelType.COMBINED:
        combined_vs30 = result_df[constants.ObservationColumn.VS30].values
        combined_stdv = result_df[constants.COL_COMBINED_STDV].values

        # Get geology IDs for query points (redundant sample, avoids
        # threading IDs through both parallel and sequential paths)
        geology_ids = category.assign_to_category_geology(points)

        fillable_mask = gapfill.classify_nodata(combined_vs30, geology_ids, points)

        if np.any(fillable_mask):
            if include_intermediate:
                result_df[constants.COL_VS30_BEFORE_GAPFILL] = combined_vs30.copy()
                result_df[constants.COL_STDV_BEFORE_GAPFILL] = combined_stdv.copy()

            fillable_indices = np.where(fillable_mask)[0]
            logger.info(
                f"  Gap-fill: filling {len(fillable_indices)} point(s) "
                f"via local grid pipeline"
            )

            for idx in fillable_indices:
                e, n = points[idx]
                half_width = constants.GAPFILL_LOCAL_GRID_SIZE_M
                fill_vs30 = np.nan
                fill_stdv = np.nan

                while half_width <= constants.GAPFILL_MAX_LOCAL_GRID_HALF_WIDTH_M:
                    local_config = gapfill.create_local_grid_config(
                        e, n, gapfill_grid_config, half_width
                    )

                    local_result = grid_pipeline(
                        grid_config=local_config,
                        output_dir=None,
                        model_type=constants.ModelType.COMBINED,
                        geology_categorical_csv=geology_categorical_csv,
                        terrain_categorical_csv=terrain_categorical_csv,
                        clustered_observations_csv=clustered_observations_csv,
                        independent_observations_csv=independent_observations_csv,
                        combination_method=combination_method,
                        combine_ratio=combine_ratio,
                        noisy=noisy,
                        mvn=mvn,
                        do_bayesian_update=do_bayesian_update,
                        include_intermediate=False,
                        n_proc=1,
                        geology_corr_fn=geology_corr_fn,
                        terrain_corr_fn=terrain_corr_fn,
                        apply_coastal_distance_mod=apply_coastal_distance_mod,
                        apply_alluvium_slope_mod=apply_alluvium_slope_mod,
                    )

                    local_profile = local_result["profile"]
                    local_vs30 = local_result["combined_vs30"]
                    local_stdv = local_result["combined_stdv"]
                    local_geol_ids = local_result["geology_ids"]

                    if local_profile is None:
                        raise ValueError(
                            "grid_pipeline returned a None profile for local gap-fill grid."
                        )
                    if not isinstance(local_vs30, np.ndarray) or not isinstance(
                        local_stdv, np.ndarray
                    ):
                        raise ValueError(
                            "grid_pipeline returned Non-array combined_vs30/combined_stdv for local gap-fill grid."
                        )

                    # Fill nodata gaps in the local grid using nearest-neighbor
                    local_vs30, local_stdv = gapfill.fill_nodata_grid(
                        local_vs30, local_stdv, local_geol_ids, local_profile
                    )

                    row, col = rasterio.transform.rowcol(
                        local_profile["transform"], e, n
                    )
                    fill_vs30 = local_vs30[row, col]
                    fill_stdv = local_stdv[row, col]

                    if not np.isnan(fill_vs30):
                        break

                    # No valid donors in the local grid — expand
                    half_width += constants.GAPFILL_LOCAL_GRID_EXPANSION_M
                    logger.info(
                        f"  Gap-fill: expanding local grid to "
                        f"{half_width * 2}m for point ({e:.0f}, {n:.0f})"
                    )
                else:
                    logger.warning(
                        f"  Gap-fill: no valid donor found for point "
                        f"({e:.0f}, {n:.0f}) after expanding to "
                        f"{half_width * 2}m, leaving as nodata"
                    )

                if not np.isnan(fill_vs30):
                    result_df.at[idx, constants.ObservationColumn.VS30] = fill_vs30
                    result_df.at[idx, constants.COL_COMBINED_STDV] = fill_stdv

    return result_df

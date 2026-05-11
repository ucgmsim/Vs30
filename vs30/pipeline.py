"""
Pipeline functions for generating Vs30 models.
"""

import logging
import time
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from qcore import coordinates
from tqdm import tqdm

from vs30 import (
    category,
    config,
    constants,
    gapfill,
    grid,
    points,
    raster,
    utils,
)

logger = logging.getLogger(__name__)


def load_observations_csv(csv_path: Path) -> pd.DataFrame:
    """
    Load and validate an observations CSV (without category assignment).

    Parameters
    ----------
    csv_path : Path
        Path to the observations CSV.

    Returns
    -------
    pd.DataFrame
        Observations DataFrame with the columns required by
        ``constants.ObservationColumn``.
    """
    logger.info(f"Loading observations from: {csv_path}")
    df = pd.read_csv(csv_path, comment="#", skipinitialspace=True)
    utils.validate_csv_columns(
        df,
        constants.ObservationColumn.REQUIRED,
        f"Observations CSV ({csv_path.name})",
    )
    logger.info(f"Loaded {len(df)} observations from {csv_path.name}")
    return df


def assign_observations_to_category(
    df: pd.DataFrame, model_type: constants.ModelType
) -> pd.DataFrame:
    """
    Return a copy of ``df`` annotated with category IDs in ``STANDARD_ID_COLUMN``.

    The category assignment is model-type specific (geology vs terrain), so
    the same loaded DataFrame can be reused across both runs.

    Parameters
    ----------
    df : pd.DataFrame
        Observations DataFrame with the easting and northing columns from
        ``constants.ObservationColumn``.
    model_type : constants.ModelType
        Model type used to select the category map (geology or terrain).

    Returns
    -------
    pd.DataFrame
        Copy of ``df`` with an added ``STANDARD_ID_COLUMN`` column holding
        the category ID for each observation.
    """
    obs_locs = df[
        [constants.ObservationColumn.EASTING, constants.ObservationColumn.NORTHING]
    ].to_numpy()
    out_df = df.copy()
    out_df[constants.STANDARD_ID_COLUMN] = category.assign_to_category(
        obs_locs, model_type
    )
    return out_df


def concat_observation_dfs(
    clustered_observations_df: pd.DataFrame | None,
    independent_observations_df: pd.DataFrame | None,
) -> pd.DataFrame:
    """
    Concatenate already-loaded observation DataFrames into a single DataFrame.

    Parameters
    ----------
    clustered_observations_df : pd.DataFrame or None
        Clustered observations (e.g., CPT data), or ``None`` if not provided.
    independent_observations_df : pd.DataFrame or None
        Independent observations (e.g., direct Vs30 measurements), or
        ``None`` if not provided.

    Returns
    -------
    pd.DataFrame
        Concatenated observations DataFrame. Returns an empty DataFrame with
        the columns required by ``constants.ObservationColumn`` when both
        inputs are ``None``.
    """
    dfs = [
        df
        for df in (clustered_observations_df, independent_observations_df)
        if df is not None
    ]
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame(columns=constants.ObservationColumn.REQUIRED)  # ty: ignore[invalid-argument-type]


def compute_categorical_vs30_updates(
    model_type: constants.ModelType,
    categorical_model_csv: Path | None = None,
    categorical_model_df: pd.DataFrame | None = None,
    clustered_observations_df: pd.DataFrame | None = None,
    independent_observations_df: pd.DataFrame | None = None,
    dbscan_nproc: int = -1,
) -> pd.DataFrame:
    """
    Compute Bayesian updates to categorical model values and return as DataFrame.

    Loads the categorical model values (from CSV or pre-loaded DataFrame) and
    applies Bayesian updates using pre-loaded observation DataFrames (mean and
    standard deviation per category), and returns the updated DataFrame. The
    observation DataFrames are expected to come from ``load_observations_csv``
    so they can be loaded once per pipeline run and reused across geology and
    terrain.

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
    model_type : ModelType
        Model type: either GEOLOGY or TERRAIN.
    categorical_model_csv : Path, optional
        Path to CSV file with categorical Vs30 mean and standard deviation values
        (e.g., geology_model_prior_mean_and_standard_deviation.csv). Mutually
        exclusive with ``categorical_model_df``; exactly one must be provided.
    categorical_model_df : pd.DataFrame, optional
        Pre-loaded categorical model DataFrame. Use this when the CSV has
        already been read by an outer caller (e.g., to avoid re-reading inside
        a per-point loop).
    clustered_observations_df : pd.DataFrame, optional
        Pre-loaded clustered observations (e.g., from
        ``load_observations_csv``). Will be processed with spatial
        clustering.
    independent_observations_df : pd.DataFrame, optional
        Pre-loaded independent observations. Will be processed without
        clustering.
    dbscan_nproc : int, optional
        Number of processes for DBSCAN clustering of clustered observations.
        Default -1 (all cores).

    Returns
    -------
    pd.DataFrame
        Updated categorical model with posterior mean and stdv columns.

    Raises
    ------
    ValueError
        If neither observations DataFrame is provided, if neither
        ``categorical_model_csv`` nor ``categorical_model_df`` is provided,
        if ``model_type`` is invalid, or if required CSV columns are missing.
    """
    if clustered_observations_df is None and independent_observations_df is None:
        raise ValueError(
            "At least one of clustered_observations_df or "
            "independent_observations_df must be provided"
        )

    if model_type not in constants.ModelType:
        raise ValueError(f"model_type must be a valid ModelType, got '{model_type}'")

    logger.info(f"Model type: {model_type}")

    if categorical_model_df is None:
        if categorical_model_csv is None:
            raise ValueError(
                "Either categorical_model_csv or categorical_model_df must be provided"
            )
        logger.info(f"Loading categorical model from: {categorical_model_csv}")
        categorical_model_df = pd.read_csv(
            categorical_model_csv, comment="#", skipinitialspace=True
        ).rename(columns=str.strip)

    # Drop rows with placeholder values for excluded categories (e.g., water)
    categorical_model_df = categorical_model_df[
        categorical_model_df[constants.COL_MEAN] != constants.NODATA_VALUE
    ]

    utils.validate_csv_columns(
        categorical_model_df,
        [constants.COL_MEAN, constants.COL_STDV],
        "Categorical model CSV",
    )

    current_prior_df = categorical_model_df.copy()

    if clustered_observations_df is not None:
        clustered_observations_df = assign_observations_to_category(
            clustered_observations_df, model_type
        )
        logger.info("Performing spatial clustering...")
        clustered_observations_df = category.perform_clustering(
            clustered_observations_df, dbscan_nproc
        )

    if independent_observations_df is not None:
        independent_observations_df = assign_observations_to_category(
            independent_observations_df, model_type
        )

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



def compute_component_grid(
    model_type: constants.ModelType,
    grid_config: config.GridConfig,
    apply_alluvium_slope_mod: bool,
    categorical_model_csv: Path | None = None,
    posterior_df: pd.DataFrame | None = None,
    clustered_observations_df: pd.DataFrame | None = None,
    independent_observations_df: pd.DataFrame | None = None,
    do_bayesian_update: bool = True,
    mvn: bool = True,
    noisy: bool = True,
    dbscan_nproc: int = -1,
    max_spatial_boolean_array_memory_gb: float = constants.MAX_SPATIAL_BOOLEAN_ARRAY_MEMORY_GB,
    output_dir: Path | None = None,
    include_intermediate: bool = False,
    corr_fn: Callable | None = None,
    apply_coastal_distance_mod: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Compute the Vs30 grid for one component (geology or terrain).

    Parameters
    ----------
    model_type : ModelType
        Model type: either GEOLOGY or TERRAIN.
    grid_config : GridConfig
        Grid domain and resolution parameters.
    apply_alluvium_slope_mod : bool
        Whether to apply slope-based interpolation for GID 4 (alluvium).
    categorical_model_csv : Path, optional
        Path to CSV file with categorical Vs30 values. Required unless
        ``posterior_df`` is provided.
    posterior_df : pd.DataFrame, optional
        Pre-computed posterior categorical model. When provided, the
        Bayesian update is skipped; ``do_bayesian_update`` and
        ``categorical_model_csv`` are ignored.
    clustered_observations_df : pd.DataFrame, optional
        Pre-loaded clustered observations (e.g., CPT data).
    independent_observations_df : pd.DataFrame, optional
        Pre-loaded independent observations (e.g., measured filtered).
    do_bayesian_update : bool, optional
        Whether to perform Bayesian update of categorical model values.
        Ignored when ``posterior_df`` is provided.
    mvn : bool, optional
        Whether to perform MVN spatial adjustment.
    noisy : bool, optional
        Whether to apply noise weighting in spatial adjustment.
    dbscan_nproc : int, optional
        Number of processes for DBSCAN clustering. Default -1 (all cores).
        No effect when ``do_bayesian_update`` is False.
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

    Raises
    ------
    ValueError
        If neither ``categorical_model_csv`` nor ``posterior_df`` is provided;
        if ``mvn`` is True and ``corr_fn`` is not provided; or if ``mvn`` is
        True and no observations are provided.
    """
    if posterior_df is None and categorical_model_csv is None:
        raise ValueError(
            f"Either categorical_model_csv or posterior_df is required for "
            f"{model_type} pipeline."
        )

    if output_dir is not None:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting {model_type} grid computation")

    if posterior_df is not None:
        logger.info("\n=== STEP 1: SKIPPED - Using pre-computed posterior_df ===")
    elif do_bayesian_update:
        logger.info("\n=== STEP 1: Updating Categorical Models ===")
        posterior_df = compute_categorical_vs30_updates(
            model_type=model_type,
            categorical_model_csv=categorical_model_csv,
            clustered_observations_df=clustered_observations_df,
            independent_observations_df=independent_observations_df,
            dbscan_nproc=dbscan_nproc,
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
        posterior_df = pd.read_csv(
            categorical_model_csv, comment="#", skipinitialspace=True
        ).rename(columns=str.strip)

    logger.info("\n=== STEP 2: Creating Initial VS30 Arrays ===")
    logger.info(f"Using grid parameters: {grid_config}")
    id_array, profile = raster.create_category_id_array(model_type, grid_config)
    vs30_array, stdv_array = raster.create_vs30_arrays_from_ids(
        id_array, posterior_df
    )

    if output_dir is not None and include_intermediate:
        id_filename = (
            constants.GEOLOGY_ID_FILENAME
            if model_type == constants.ModelType.GEOLOGY
            else constants.TERRAIN_ID_FILENAME
        )
        grid.write_raster(
            output_dir / id_filename,
            profile,
            [id_array],
            (constants.BAND_DESCRIPTION_ID_INDEX,),
            dtype="uint8",
            nodata=constants.RASTER_ID_NODATA_VALUE,
        )

        initial_filename = (
            constants.GEOLOGY_INITIAL_VS30_FILENAME
            if model_type == constants.ModelType.GEOLOGY
            else constants.TERRAIN_INITIAL_VS30_FILENAME
        )
        grid.write_raster(
            output_dir / initial_filename,
            profile,
            [vs30_array, stdv_array],
            (constants.BAND_DESCRIPTION_VS30, constants.BAND_DESCRIPTION_STDV),
        )

    slope_array = None
    coast_dist_array = None

    if model_type == constants.ModelType.GEOLOGY:
        logger.info("\n=== STEP 3: Slope and Coastal Distance Adjusted Geology ===")

        slope_array = raster.compute_slope_array(profile)

        if apply_coastal_distance_mod:
            coast_dist_array = raster.compute_coast_distance_raster(profile)
        else:
            logger.info("Skipping coast distance computation (disabled in config)")
            coast_dist_array = np.zeros_like(vs30_array)

        vs30_array, stdv_array = raster.apply_hybrid_geology_modifications(
            vs30_array,
            stdv_array,
            id_array,
            slope_array,
            coast_dist_array,
            apply_alluvium_slope_mod=apply_alluvium_slope_mod,
            apply_coastal_distance_mod=apply_coastal_distance_mod,
        )

        if output_dir is not None and include_intermediate:
            grid.write_raster(
                output_dir / constants.SLOPE_RASTER_FILENAME,
                profile,
                [slope_array],
                (constants.BAND_DESCRIPTION_SLOPE,),
            )
            grid.write_raster(
                output_dir / constants.COAST_DISTANCE_RASTER_FILENAME,
                profile,
                [coast_dist_array],
                (constants.BAND_DESCRIPTION_COAST_DISTANCE,),
                nodata=None,
            )
            grid.write_raster(
                output_dir
                / constants.GEOLOGY_VS30_SLOPE_AND_COASTAL_DISTANCE_ADJUSTED_FILENAME,
                profile,
                [vs30_array, stdv_array],
                (
                    constants.BAND_DESCRIPTION_VS30_HYBRID,
                    constants.BAND_DESCRIPTION_STDV_HYBRID,
                ),
            )

    if mvn:
        if corr_fn is None:
            raise ValueError("corr_fn must be provided for spatial adjustment.")

        logger.info("\n=== STEP 4: Spatial Adjustment ===")

        observations_df = concat_observation_dfs(
            clustered_observations_df, independent_observations_df
        )
        if len(observations_df) == 0:
            raise ValueError(
                "No observations provided for spatial fit. "
                "At least one of clustered or independent observations must be specified."
            )

        vs30_array, stdv_array = grid.compute_spatial_adjustment_on_grid(
            vs30_array=vs30_array,
            stdv_array=stdv_array,
            profile=profile,
            observations_df=observations_df,
            model_values_df=posterior_df,
            model_type=model_type,
            corr_fn=corr_fn,
            apply_alluvium_slope_mod=apply_alluvium_slope_mod,
            apply_coastal_distance_mod=apply_coastal_distance_mod,
            noisy=noisy,
            max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
            slope_array=slope_array,
            coast_dist_array=coast_dist_array,
        )
    else:
        logger.info("\n=== STEP 4: SKIPPED - MVN spatial adjustment disabled ===")

    if output_dir is not None:
        output_filename = constants.OUTPUT_FILENAMES[model_type]
        grid.write_raster(
            output_dir / output_filename,
            profile,
            [vs30_array, stdv_array],
            (constants.BAND_DESCRIPTION_VS30, constants.BAND_DESCRIPTION_STDV),
        )

    logger.info(f"\n{model_type} grid computation completed")

    return vs30_array, stdv_array, id_array, profile


def grid_pipeline(
    grid_config: config.GridConfig,
    apply_alluvium_slope_mod: bool,
    geology_corr_fn: Callable,
    terrain_corr_fn: Callable,
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
    dbscan_nproc: int = -1,
    max_spatial_boolean_array_memory_gb: float = constants.MAX_SPATIAL_BOOLEAN_ARRAY_MEMORY_GB,
    apply_coastal_distance_mod: bool = True,
    fill_gaps: bool = False,
    clustered_observations_df: pd.DataFrame | None = None,
    independent_observations_df: pd.DataFrame | None = None,
    geology_posterior_df: pd.DataFrame | None = None,
    terrain_posterior_df: pd.DataFrame | None = None,
) -> dict[str, np.ndarray | dict | None]:
    """
    Compute the Vs30 grid.

    Parameters
    ----------
    grid_config : GridConfig
        Grid domain and resolution parameters.
    apply_alluvium_slope_mod : bool
        Whether to apply slope-based interpolation for GID 4 (alluvium).
    geology_corr_fn : Callable
        Correlation function for geology spatial adjustment.
    terrain_corr_fn : Callable
        Correlation function for terrain spatial adjustment.
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
        Whether to perform MVN spatial adjustment.
    do_bayesian_update : bool, optional
        Whether to perform Bayesian update of categorical model values.
    include_intermediate : bool, optional
        Whether to write intermediate files (ID rasters, initial VS30, slope,
        coast distance, hybrid geology). Default False.
    dbscan_nproc : int, optional
        Number of processes for DBSCAN clustering. Default -1 (all cores).
        No effect when ``do_bayesian_update`` is False.
    max_spatial_boolean_array_memory_gb : float, optional
        Maximum memory for spatial boolean arrays.
    apply_coastal_distance_mod : bool
        Whether to apply coastal distance modification for GID 4 and GID 10.
    fill_gaps : bool
        Whether to fill on-land nodata gaps in the combined output using
        nearest-neighbor interpolation.
    clustered_observations_df : pd.DataFrame, optional
        Pre-loaded clustered observations. When provided, the corresponding
        CSV path is ignored and this DataFrame is reused directly.
    independent_observations_df : pd.DataFrame, optional
        Pre-loaded independent observations. Same semantics as
        ``clustered_observations_df``.
    geology_posterior_df : pd.DataFrame, optional
        Pre-computed posterior categorical model for geology. When provided,
        the geology Bayesian update step is skipped.
    terrain_posterior_df : pd.DataFrame, optional
        Pre-computed posterior categorical model for terrain. When provided,
        the terrain Bayesian update step is skipped.

    Returns
    -------
    dict
        Dictionary containing the computed raster data with keys:

        - ``"geology_vs30"``, ``"geology_stdv"`` : 2D arrays (when geology is computed)
        - ``"geology_ids"`` : 2D uint8 array of geology category IDs (always
          populated when geology is computed; required by the gap-fill
          classifier in ``points_pipeline``).
        - ``"terrain_vs30"``, ``"terrain_stdv"`` : 2D arrays (when terrain is computed)
        - ``"combined_vs30"``, ``"combined_stdv"`` : 2D arrays (when both models are computed)
        - ``"profile"`` : rasterio profile dict with CRS, transform, dimensions, etc.

    Raises
    ------
    ValueError
        If a single-model run (``model_type=GEOLOGY`` or ``TERRAIN``) is
        requested without ``include_intermediate=True``.
    """
    if model_type != constants.ModelType.COMBINED and not include_intermediate:
        raise ValueError(
            "Single-model output (model_type=geology or terrain) requires "
            "include_intermediate=True, as per-model results are intermediate "
            "data products. The only final product is the combined model."
        )

    start_time = time.time()

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

    # Load observation CSVs once and reuse across geology + terrain (skipped
    # when caller has already supplied the corresponding DataFrame).
    if clustered_observations_df is None and clustered_observations_csv is not None:
        clustered_observations_df = load_observations_csv(clustered_observations_csv)
    if independent_observations_df is None and independent_observations_csv is not None:
        independent_observations_df = load_observations_csv(independent_observations_csv)

    result: dict[str, np.ndarray | dict | None] = {}
    profile: dict | None = None

    if run_geology:
        logger.info("\n" + "=" * 80 + "\nRUNNING GEOLOGY PIPELINE\n" + "=" * 80)
        geol_vs30, geol_stdv, geol_ids, profile = compute_component_grid(
            model_type=constants.ModelType.GEOLOGY,
            grid_config=grid_config,
            apply_alluvium_slope_mod=apply_alluvium_slope_mod,
            categorical_model_csv=geology_categorical_csv,
            posterior_df=geology_posterior_df,
            clustered_observations_df=clustered_observations_df,
            independent_observations_df=independent_observations_df,
            do_bayesian_update=do_bayesian_update,
            mvn=mvn,
            noisy=noisy,
            dbscan_nproc=dbscan_nproc,
            max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
            output_dir=output_dir,
            include_intermediate=include_intermediate,
            corr_fn=geology_corr_fn,
            apply_coastal_distance_mod=apply_coastal_distance_mod,
        )
        result["geology_vs30"] = geol_vs30
        result["geology_stdv"] = geol_stdv
        result["geology_ids"] = geol_ids

    if run_terrain:
        logger.info("\n" + "=" * 80 + "\nRUNNING TERRAIN PIPELINE\n" + "=" * 80)
        terr_vs30, terr_stdv, _, profile = compute_component_grid(
            model_type=constants.ModelType.TERRAIN,
            grid_config=grid_config,
            apply_alluvium_slope_mod=apply_alluvium_slope_mod,
            categorical_model_csv=terrain_categorical_csv,
            posterior_df=terrain_posterior_df,
            clustered_observations_df=clustered_observations_df,
            independent_observations_df=independent_observations_df,
            do_bayesian_update=do_bayesian_update,
            mvn=mvn,
            noisy=noisy,
            dbscan_nproc=dbscan_nproc,
            max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
            output_dir=output_dir,
            include_intermediate=include_intermediate,
            corr_fn=terrain_corr_fn,
            apply_coastal_distance_mod=apply_coastal_distance_mod,
        )
        result["terrain_vs30"] = terr_vs30
        result["terrain_stdv"] = terr_stdv

    if run_geology and run_terrain:
        logger.info(
            "\n" + "=" * 80 + "\nCOMBINING GEOLOGY AND TERRAIN RESULTS\n" + "=" * 80
        )

        combined_vs30, combined_stdv = grid.combine_model_arrays(
            geol_vs30=geol_vs30,
            geol_stdv=geol_stdv,
            terr_vs30=terr_vs30,
            terr_stdv=terr_stdv,
            combination_method=combination_method,
            combine_ratio=combine_ratio,
        )

        assert profile is not None  # invariant: set by compute_component_grid above

        if fill_gaps:
            # Gap-fill on-land nodata pixels in combined output
            logger.info(
                "\n" + "=" * 80 + "\nGAP-FILLING COMBINED OUTPUT\n" + "=" * 80
            )

            if output_dir is not None and include_intermediate:
                grid.write_raster(
                    output_dir / constants.COMBINED_VS30_BEFORE_GAPFILL_FILENAME,
                    profile,
                    [utils.nan_to_nodata(combined_vs30), utils.nan_to_nodata(combined_stdv)],
                    (
                        constants.BAND_DESCRIPTION_VS30_COMBINED,
                        constants.BAND_DESCRIPTION_STDV_COMBINED,
                    ),
                )

            combined_vs30, combined_stdv = gapfill.fill_nodata_grid(
                combined_vs30, combined_stdv, geol_ids, profile
            )
        result["combined_vs30"] = combined_vs30
        result["combined_stdv"] = combined_stdv

        if output_dir is not None:
            grid.write_raster(
                output_dir / constants.COMBINED_VS30_FILENAME,
                profile,
                [utils.nan_to_nodata(combined_vs30), utils.nan_to_nodata(combined_stdv)],
                (
                    constants.BAND_DESCRIPTION_VS30_COMBINED,
                    constants.BAND_DESCRIPTION_STDV_COMBINED,
                ),
            )

    result["profile"] = profile

    elapsed_time = time.time() - start_time
    logger.info(f"  Total execution time: {elapsed_time:.1f} seconds")
    if output_dir is not None:
        logger.info(f"  Output available in: {output_dir}")

    return result


def fill_one_point_via_local_grid(
    easting: float,
    northing: float,
    gapfill_grid_config: config.GridConfig,
    grid_pipeline_kwargs: dict,
) -> tuple[float, float]:
    """
    Fill a single nodata point by running ``grid_pipeline`` on a local grid.

    Parameters
    ----------
    easting : float
        Query point easting (NZTM).
    northing : float
        Query point northing (NZTM).
    gapfill_grid_config : config.GridConfig
        Reference grid config defining the pixel alignment for local grids.
    grid_pipeline_kwargs : dict
        Keyword arguments forwarded to ``grid_pipeline`` (everything except
        ``grid_config`` and ``output_dir``).

    Returns
    -------
    tuple[float, float]
        ``(fill_vs30, fill_stdv)``. Both are NaN if no donor was found.
    """
    half_width = constants.GAPFILL_INITIAL_HALF_WIDTH_M
    while half_width <= constants.GAPFILL_MAX_HALF_WIDTH_M:
        local_config = gapfill.create_local_grid_config(
            easting, northing, gapfill_grid_config, half_width
        )
        local_result = grid_pipeline(
            grid_config=local_config,
            output_dir=None,
            **grid_pipeline_kwargs,
        )

        local_vs30, local_stdv = gapfill.fill_nodata_grid(
            local_result["combined_vs30"],
            local_result["combined_stdv"],
            local_result["geology_ids"],
            local_result["profile"],
        )
        row, col = rasterio.transform.rowcol(
            local_result["profile"]["transform"], easting, northing
        )
        fill_vs30 = local_vs30[row, col]
        fill_stdv = local_stdv[row, col]

        if not np.isnan(fill_vs30):
            return float(fill_vs30), float(fill_stdv)

        half_width += constants.GAPFILL_HALF_WIDTH_EXPANSION_M
        logger.info(
            f"  Gap-fill: expanding local grid to "
            f"{half_width * 2}m wide for point ({easting:.0f}, {northing:.0f})"
        )

    logger.warning(
        f"  Gap-fill: no valid donor found for point "
        f"({easting:.0f}, {northing:.0f}), leaving as nodata"
    )
    return float("nan"), float("nan")


def points_pipeline(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    apply_alluvium_slope_mod: bool,
    geology_corr_fn: Callable,
    terrain_corr_fn: Callable,
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
    dbscan_nproc: int = -1,
    apply_coastal_distance_mod: bool = True,
    fill_gaps: bool = False,
    gapfill_grid_config: config.GridConfig = config.FULL_NZ_GRID_CONFIG,
) -> pd.DataFrame:
    """
    Compute Vs30 values at specific latitude/longitude locations.

    Parameters
    ----------
    longitudes : ndarray
        Array of longitude values (WGS84).
    latitudes : ndarray
        Array of latitude values (WGS84).
    apply_alluvium_slope_mod : bool
        Whether to apply slope-based interpolation for GID 4 (alluvium).
    geology_corr_fn : Callable
        Correlation function for geology spatial adjustment.
    terrain_corr_fn : Callable
        Correlation function for terrain spatial adjustment.
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
        Whether to perform MVN spatial adjustment.
    do_bayesian_update : bool, optional
        Whether to perform Bayesian update of categorical Vs30 values
        using observations before computing Vs30. Default False.
    include_intermediate : bool, optional
        Include intermediate values (geology/terrain separately) in output.
    dbscan_nproc : int, optional
        Number of processes for DBSCAN clustering. Default -1 (all cores).
        No effect when ``do_bayesian_update`` is False.
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

    Raises
    ------
    ValueError
        If a single-model run (``model_type=GEOLOGY`` or ``TERRAIN``) is
        requested without ``include_intermediate=True``; if running geology
        without ``geology_categorical_csv``; or if running terrain without
        ``terrain_categorical_csv``.
    """
    if model_type != constants.ModelType.COMBINED and not include_intermediate:
        raise ValueError(
            "Single-model output (model_type=geology or terrain) requires "
            "include_intermediate=True, as per-model results are intermediate "
            "data products. The only final product is the combined model."
        )

    nztm_coords = coordinates.wgs_depth_to_nztm(
        np.column_stack([latitudes, longitudes])
    )
    locations = nztm_coords[:, ::-1]  # (easting, northing)

    logger.info(f"Processing {len(locations)} locations")

    # Load observation CSVs once and reuse across geology + terrain stages.
    clustered_observations_df = (
        load_observations_csv(clustered_observations_csv)
        if clustered_observations_csv is not None
        else None
    )
    independent_observations_df = (
        load_observations_csv(independent_observations_csv)
        if independent_observations_csv is not None
        else None
    )

    if mvn:
        observations_df = concat_observation_dfs(
            clustered_observations_df, independent_observations_df
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
                model_type=constants.ModelType.GEOLOGY,
                categorical_model_csv=geology_categorical_csv,
                clustered_observations_df=clustered_observations_df,
                independent_observations_df=independent_observations_df,
                dbscan_nproc=dbscan_nproc,
            )
        else:
            geol_model_df = pd.read_csv(
                geology_categorical_csv, comment="#", skipinitialspace=True
            ).rename(columns=str.strip)

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
                model_type=constants.ModelType.TERRAIN,
                categorical_model_csv=terrain_categorical_csv,
                clustered_observations_df=clustered_observations_df,
                independent_observations_df=independent_observations_df,
                dbscan_nproc=dbscan_nproc,
            )
        else:
            terr_model_df = pd.read_csv(
                terrain_categorical_csv, comment="#", skipinitialspace=True
            ).rename(columns=str.strip)

    geology_obs_data = (
        points.prepare_geology_obs_data(
            observations_df,
            geol_model_df,
            apply_alluvium_slope_mod=apply_alluvium_slope_mod,
            apply_coastal_distance_mod=apply_coastal_distance_mod,
            noisy=noisy,
        )
        if run_geology
        else None
    )
    terrain_obs_data = (
        points.prepare_terrain_obs_data(observations_df, terr_model_df, noisy=noisy)
        if run_terrain
        else None
    )

    result = {}
    result[constants.ObservationColumn.EASTING] = locations[:, 0]
    result[constants.ObservationColumn.NORTHING] = locations[:, 1]

    if run_geology:
        assert geol_model_df is not None  # invariant: required when run_geology
        with tqdm(
            total=len(locations), desc="Geology: spatial adjustment", unit="point"
        ) as pbar:
            (
                geol_ids,
                geol_vs30,
                geol_stdv,
                geol_vs30_hybrid,
                geol_stdv_hybrid,
                geol_mvn_vs30,
                geol_mvn_stdv,
            ) = points.process_geology_at_points(
                locations,
                geol_model_df,
                geology_obs_data,
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

    if run_terrain:
        assert terr_model_df is not None  # invariant: required when run_terrain
        with tqdm(
            total=len(locations), desc="Terrain: spatial adjustment", unit="point"
        ) as pbar:
            (
                terr_ids,
                terr_vs30,
                terr_stdv,
                terr_mvn_vs30,
                terr_mvn_stdv,
            ) = points.process_terrain_at_points(
                locations,
                terr_model_df,
                terrain_obs_data,
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

    logger.info(f"  Total locations: {len(locations)}")
    result_df = pd.DataFrame(result)

    if fill_gaps and model_type == constants.ModelType.COMBINED:
        combined_vs30 = result_df[constants.ObservationColumn.VS30].to_numpy()
        combined_stdv = result_df[constants.COL_COMBINED_STDV].to_numpy()

        # COMBINED implies run_geology, so geol_ids was already computed above.
        fillable_mask = gapfill.classify_nodata(combined_vs30, geol_ids, locations)

        if np.any(fillable_mask):
            if include_intermediate:
                result_df[constants.COL_VS30_BEFORE_GAPFILL] = combined_vs30.copy()
                result_df[constants.COL_STDV_BEFORE_GAPFILL] = combined_stdv.copy()

            fillable_indices = np.where(fillable_mask)[0]
            logger.info(
                f"  Gap-fill: filling {len(fillable_indices)} point(s) "
                f"via local grid pipeline"
            )

            # Local grid_pipeline calls reuse already-computed observations
            # and posteriors, and disable Bayesian update + internal gap-fill
            # (grid_pipeline's fill_gaps only searches within the local grid
            # bounds; fill_one_point_via_local_grid runs gapfill.fill_nodata_grid
            # itself so it can also grow the local grid on miss).
            grid_pipeline_kwargs = {
                "apply_alluvium_slope_mod": apply_alluvium_slope_mod,
                "geology_corr_fn": geology_corr_fn,
                "terrain_corr_fn": terrain_corr_fn,
                "model_type": constants.ModelType.COMBINED,
                "combination_method": combination_method,
                "combine_ratio": combine_ratio,
                "noisy": noisy,
                "mvn": mvn,
                "apply_coastal_distance_mod": apply_coastal_distance_mod,
                "clustered_observations_df": clustered_observations_df,
                "independent_observations_df": independent_observations_df,
                "geology_posterior_df": geol_model_df,
                "terrain_posterior_df": terr_model_df,
                # Overrides:
                "do_bayesian_update": False,
                "include_intermediate": False,
                "fill_gaps": False,
            }

            for idx in fillable_indices:
                easting, northing = locations[idx]
                fill_vs30, fill_stdv = fill_one_point_via_local_grid(
                    easting, northing, gapfill_grid_config, grid_pipeline_kwargs
                )
                if not np.isnan(fill_vs30):
                    result_df.at[idx, constants.ObservationColumn.VS30] = fill_vs30
                    result_df.at[idx, constants.COL_COMBINED_STDV] = fill_stdv

    return result_df

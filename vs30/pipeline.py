"""
Pipeline functions for generating Vs30 models.
"""

import logging
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from qcore import coordinates
from tqdm import tqdm

from vs30 import category, constants, parallel, raster, spatial, utils
from vs30 import config as config_module

logger = logging.getLogger(__name__)


def update_categorical_vs30_models(
    categorical_model_csv: Path,
    output_dir: Path,
    model_type: constants.ModelType,
    clustered_observations_csv: Path | None = None,
    independent_observations_csv: Path | None = None,
    n_proc: int = 1,
) -> None:
    """
    Update categorical model values using Bayesian updates and save to CSV files.

    Loads observations and categorical model values, applies Bayesian updates to the
    categorical model values (mean and standard deviation per category), and writes
    the updated values back to CSV files.

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
    output_dir : Path
        Path to output directory. Will be created if it does not exist.
    model_type : str
        Model type: either 'geology' or 'terrain'.
    clustered_observations_csv : Path, optional
        Path to CSV file with clustered observations (e.g., measured_vs30_cpt.csv).
        These will be processed with spatial clustering.
    independent_observations_csv : Path, optional
        Path to CSV file with independent observations
        (e.g., measured_vs30_independent_observations.csv).
        These will be processed without clustering.
    n_proc : int, optional
        Number of processes for DBSCAN clustering. Use -1 for all available cores.

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
            clustered_observations_csv, skipinitialspace=True
        )

        utils.validate_csv_columns(
            clustered_observations_df,
            constants.REQUIRED_OBSERVATION_COLUMNS_BASIC,
            "Clustered observations CSV",
        )

        logger.info(f"Loaded {len(clustered_observations_df)} clustered observations")

        # Assign category IDs
        obs_locs = clustered_observations_df[
            [constants.COL_EASTING, constants.COL_NORTHING]
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
            independent_observations_csv, skipinitialspace=True
        )

        utils.validate_csv_columns(
            independent_observations_df,
            constants.REQUIRED_OBSERVATION_COLUMNS,
            "Independent observations CSV",
        )

        logger.info(
            f"Loaded {len(independent_observations_df)} independent observations"
        )

        # Assign category IDs
        obs_locs = independent_observations_df[
            [constants.COL_EASTING, constants.COL_NORTHING]
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

    # Create output directory if it doesn't exist
    output_dir.mkdir(parents=True, exist_ok=True)

    output_filename = constants.POSTERIOR_PREFIX + categorical_model_csv.name
    output_path = output_dir / output_filename

    current_prior_df.to_csv(output_path, index=False)


def make_initial_vs30_raster(
    output_dir: Path,
    grid_config: config_module.GridConfig,
    terrain: bool = False,
    geology: bool = False,
    geology_csv: Path | None = None,
    terrain_csv: Path | None = None,
) -> None:
    """
    Create initial VS30 mean and standard deviation rasters from category IDs.

    Generates initial VS30 rasters by:

    1. Creating category ID rasters (from terrain raster or geology shapefile)
    2. Mapping category IDs to VS30 mean and standard deviation values from CSV files
    3. Writing 2-band GeoTIFFs with VS30 mean (band 1) and standard deviation (band 2)

    Output files are saved as terrain_initial_vs30_with_uncertainty.tif and/or
    geology_initial_vs30_with_uncertainty.tif in output_dir.

    Parameters
    ----------
    output_dir : Path
        Output directory. Created if it does not exist.
    grid_config : GridConfig
        Grid domain and resolution parameters.
    terrain : bool, optional
        Create terrain VS30 raster.
    geology : bool, optional
        Create geology VS30 raster.
    geology_csv : Path, optional
        Custom geology model CSV file. Defaults to bundled resource.
    terrain_csv : Path, optional
        Custom terrain model CSV file. Defaults to bundled resource.

    Raises
    ------
    ValueError
        If neither terrain nor geology is True.
    """
    if not terrain and not geology:
        raise ValueError("At least one of terrain or geology must be True")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    grid_params = {
        "xmin": grid_config.grid_xmin,
        "xmax": grid_config.grid_xmax,
        "ymin": grid_config.grid_ymin,
        "ymax": grid_config.grid_ymax,
        "dx": grid_config.grid_dx,
        "dy": grid_config.grid_dy,
    }
    logger.info(f"Using grid parameters: {grid_params}")

    if terrain:
        logger.info("Processing terrain model...")
        terrain_model_csv = (
            terrain_csv
            if terrain_csv
            else constants.TERRAIN_MEAN_AND_STANDARD_DEVIATION_PER_CATEGORY_FILE
        )
        logger.info(f"Using terrain model values from {terrain_model_csv}")

        logger.info("Creating terrain category ID raster...")
        id_raster = raster.create_category_id_raster(
            constants.ModelType.TERRAIN, output_dir, **grid_params
        )

        logger.info("Creating terrain VS30 raster...")
        vs30_raster = output_dir / constants.TERRAIN_INITIAL_VS30_FILENAME
        raster.create_vs30_raster_from_ids(
            id_raster,
            terrain_model_csv,
            vs30_raster,
            model_type=constants.ModelType.TERRAIN,
        )

    if geology:
        logger.info("Processing geology model...")
        geology_model_csv = (
            geology_csv
            if geology_csv
            else constants.GEOLOGY_MEAN_AND_STANDARD_DEVIATION_PER_CATEGORY_FILE
        )
        logger.info(f"Using geology model values from {geology_model_csv}")

        logger.info("Creating geology category ID raster...")
        id_raster = raster.create_category_id_raster(
            constants.ModelType.GEOLOGY, output_dir, **grid_params
        )

        logger.info("Creating geology VS30 raster...")
        vs30_raster = output_dir / constants.GEOLOGY_INITIAL_VS30_FILENAME
        raster.create_vs30_raster_from_ids(
            id_raster,
            geology_model_csv,
            vs30_raster,
            model_type=constants.ModelType.GEOLOGY,
        )


def adjust_geology_vs30_by_slope_and_coastal_distance(
    input_raster: Path,
    id_raster: Path,
    output_dir: Path,
) -> None:
    """
    Apply hybrid geology modifications to an initial VS30 raster.

    Adds slope-based and coast-distance-based modifications to the geology model.
    Generates intermediate slope and coast distance rasters in the output directory.

    Parameters
    ----------
    input_raster : Path
        Path to initial geology VS30 raster (created by make_initial_vs30_raster).
        Must be a 2-band raster with Vs30 and standard deviation.
    id_raster : Path
        Path to category ID raster (e.g., gid.tif used to create the input raster).
    output_dir : Path
        Directory to save output hybrid raster and intermediate files.

    Raises
    ------
    ValueError
        If raster dimensions do not match.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(
        f"Processing slope and coastal distance adjusted model for: {input_raster}"
    )

    with rasterio.open(input_raster) as src:
        vs30_array = src.read(1)
        stdv_array = src.read(2)
        profile = src.profile.copy()

        with rasterio.open(id_raster) as id_src:
            if id_src.width != src.width or id_src.height != src.height:
                raise ValueError(
                    f"Dimension mismatch! Input raster: {src.width}x{src.height}, "
                    f"ID raster: {id_src.width}x{id_src.height}"
                )
            id_array = id_src.read(1)

    slope_path = output_dir / constants.SLOPE_RASTER_FILENAME
    logger.info(f"Generating slope raster: {slope_path}")
    slope_array, _ = raster.create_slope_raster(slope_path, profile)

    coast_path = output_dir / constants.COAST_DISTANCE_RASTER_FILENAME
    logger.info(f"Generating coast distance raster: {coast_path}")
    coast_dist_array, _ = raster.create_coast_distance_raster(coast_path, profile)

    logger.info("Applying slope and coastal distance based geology modifications...")
    mod_vs30, mod_stdv = raster.apply_hybrid_geology_modifications(
        vs30_array,
        stdv_array,
        id_array,
        slope_array,
        coast_dist_array,
    )

    output_path = (
        output_dir / constants.GEOLOGY_VS30_SLOPE_AND_COASTAL_DISTANCE_ADJUSTED_FILENAME
    )

    profile.update({"dtype": "float32", "compress": "deflate"})

    logger.info(f"Saving hybrid raster to: {output_path}")
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(mod_vs30, 1)
        dst.write(mod_stdv, 2)
        dst.descriptions = (
            constants.BAND_DESCRIPTION_VS30_HYBRID,
            constants.BAND_DESCRIPTION_STDV_HYBRID,
        )


def spatial_fit(
    input_raster: Path,
    observations_csv: Path,
    model_values_csv: Path,
    output_dir: Path,
    model_type: constants.ModelType,
    noisy: bool = True,
    n_proc: int = 1,
    is_clustered_observations: bool = False,
    obs_subsample_step_for_clustered: int = 100,
    max_spatial_boolean_array_memory_gb: float = 1.0,
) -> None:
    """
    Adjust a VS30 raster based on measurements using spatial conditioning.

    Performs a spatial adjustment of an input raster by:

    1. Loading the 2-band input raster (VS30 mean and stdv)
    2. Loading measurements and mapping them to categories
    3. Computing spatial fits to update pixels affected by measurements
    4. Applying updates and saving the resulting 2-band GeoTIFF

    Parameters
    ----------
    input_raster : Path
        Path to input 2-band VS30 raster (Vs30 mean and standard deviation).
    observations_csv : Path
        Path to CSV file with measured VS30 values. Must contain columns:
        easting, northing, vs30, uncertainty.
    model_values_csv : Path
        Path to CSV file with updated categorical Vs30 values
        (e.g., updated_geology_model.csv).
    output_dir : Path
        Directory to save the adjusted raster.
    model_type : ModelType
        Model type: either GEOLOGY or TERRAIN.
    noisy : bool, optional
        Whether to apply noise weighting in spatial adjustment.
    n_proc : int, optional
        Number of parallel processes. Use -1 for all cores.
    is_clustered_observations : bool, optional
        Whether the observations are clustered (for subsampling optimization).
    obs_subsample_step_for_clustered : int, optional
        Subsampling step for clustered observations.
    max_spatial_boolean_array_memory_gb : float, optional
        Maximum memory for spatial boolean arrays.
    """
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    n_proc_resolved = parallel.resolve_n_proc(n_proc)

    logger.info(f"Starting spatial fit for {model_type} model")
    logger.info(f"Input raster: {input_raster}")
    logger.info(f"Observations: {observations_csv}")
    logger.info(f"Model values: {model_values_csv}")

    # 1. Load Raster Data
    logger.info("Loading raster data...")
    raster_data = spatial.RasterData.from_file(input_raster)
    spatial.validate_raster_data(raster_data)

    # 2. Load Observations
    logger.info("Loading observations...")
    observations = pd.read_csv(observations_csv, skipinitialspace=True)
    spatial.validate_observations(observations)

    # 3. Load Model Values (updated categorical table)
    logger.info("Loading updated model table...")
    model_df = pd.read_csv(model_values_csv, skipinitialspace=True)
    mean_col, std_col = raster.select_vs30_columns_by_priority(list(model_df.columns))

    # Build table indexed by category ID
    max_id = model_df[constants.STANDARD_ID_COLUMN].max()
    updated_model_table = np.full((max_id, 2), np.nan)
    for _, row in model_df.iterrows():
        idx = int(row[constants.STANDARD_ID_COLUMN]) - 1
        if 0 <= idx < max_id:
            updated_model_table[idx, 0] = row[mean_col]
            updated_model_table[idx, 1] = row[std_col]

    # 4. Prepare Observation Data for Spatial Adjustment
    logger.info("Preparing observation data for spatial adjustment...")
    obs_data = spatial.prepare_observation_data(
        observations,
        raster_data,
        updated_model_table,
        model_type,
        output_dir,
        noisy=noisy,
    )
    logger.info(f"Prepared {len(obs_data.locations)} valid observations")

    if len(obs_data.locations) == 0:
        logger.warning(
            "No valid observations found within model bounds. Copying input raster to output."
        )
        output_filename = constants.OUTPUT_FILENAMES[model_type]
        output_path = output_dir / output_filename
        shutil.copyfile(input_raster, output_path)
        return

    # 5. Find Affected Pixels (with optional clustered subsampling)
    obs_data_for_bbox = spatial.apply_clustered_subsampling(
        obs_data,
        is_clustered_observations,
        n_proc_resolved,
        obs_subsample_step_for_clustered,
    )

    logger.info("Finding pixels affected by observations...")
    bbox_result = spatial.find_affected_pixels(
        raster_data,
        obs_data_for_bbox,
        max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
        model_type=model_type,
        max_dist_m=constants.MAX_DIST_M,
        n_proc=n_proc_resolved,
    )
    logger.info(f"Found {bbox_result.n_affected_pixels:,} affected pixels")

    # 6. Compute Spatial Adjustments
    logger.info("Computing spatial updates...")
    if n_proc_resolved > 1:
        logger.info(f"Using {n_proc_resolved} parallel workers")
        affected_flat_indices = np.where(bbox_result.mask)[0]
        updates = parallel.run_parallel_spatial_fit(
            affected_flat_indices=affected_flat_indices,
            raster_data=raster_data,
            obs_data=obs_data,
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
            model_type,
            max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
            max_dist_m=constants.MAX_DIST_M,
            max_points=constants.MAX_POINTS,
            noisy=noisy,
            cov_reduc=constants.COV_REDUC,
        )

    # 7. Apply Updates and Write Output
    logger.info("Applying updates and writing output...")
    spatial.apply_and_write_updates(raster_data, updates, model_type, output_dir)


def combine(
    geology_tif: Path,
    terrain_tif: Path,
    output_path: Path,
    combination_method: constants.CombinationMethod,
    combine_ratio: float | None = None,
) -> None:
    """
    Combine geology and terrain VS30 rasters using a weighted average.

    Combines the two model outputs in log-space using the specified weighting
    method. The output is a 2-band GeoTIFF with combined Vs30 mean and standard
    deviation.

    Parameters
    ----------
    geology_tif : Path
        Path to geology VS30 raster.
    terrain_tif : Path
        Path to terrain VS30 raster.
    output_path : Path
        Path to combined output raster.
    combination_method : CombinationMethod
        Method for combining models: STANDARD_DEVIATION_WEIGHTING for
        variance-based weighting, or RATIO for fixed-ratio weighting.
    combine_ratio : float, optional
        Geology-to-terrain weight ratio. Required when combination_method is RATIO.
    """
    if (
        combination_method is constants.CombinationMethod.RATIO
        and combine_ratio is None
    ):
        raise ValueError(
            "combination_method is set to 'ratio' but combine_ratio is not provided"
        )

    logger.info(f"Averaging {geology_tif} and {terrain_tif}")

    with rasterio.open(geology_tif) as src_g, rasterio.open(terrain_tif) as src_t:
        profile = src_g.profile.copy()
        geol_data = src_g.read()
        terr_data = src_t.read()

        # Use nodata from geology (should be same for terrain)
        nodata = src_g.nodata

        # Set nodata values to NaN for easier calculation
        geol_data[geol_data == nodata] = np.nan
        terr_data[terr_data == nodata] = np.nan

        # Combine models using shared function (log-space mixture)
        combined_vs30, combined_stdv = utils.combine_vs30_models(
            geol_vs30=geol_data[0],
            geol_stdv=geol_data[1],
            terr_vs30=terr_data[0],
            terr_stdv=terr_data[1],
            combination_method=combination_method,
            combine_ratio=combine_ratio,
        )

        # Create output array correctly restoring nodata where things are NaN
        combined_data = np.stack([combined_vs30, combined_stdv])
        combined_data[np.isnan(combined_data)] = nodata

        profile.update(
            {
                "dtype": "float32",
                "count": 2,
                "nodata": nodata,
                "compress": "deflate",
            }
        )

        logger.info(f"Saving combined raster to: {output_path}")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(combined_data)
            dst.descriptions = (
                constants.BAND_DESCRIPTION_VS30_COMBINED,
                constants.BAND_DESCRIPTION_STDV_COMBINED,
            )


def run_pipeline_for_model_type(
    model_type: constants.ModelType,
    grid_config: config_module.GridConfig,
    output_dir: Path,
    categorical_model_csv: Path | None = None,
    clustered_observations_csv: Path | None = None,
    independent_observations_csv: Path | None = None,
    do_bayesian_update: bool = True,
    mvn: bool = True,
    noisy: bool = True,
    n_proc: int = 1,
    max_spatial_boolean_array_memory_gb: float = 1.0,
    obs_subsample_step_for_clustered: int = 100,
) -> None:
    """
    Run the full VS30 generation pipeline for a single model type.

    Executes the complete pipeline in sequence:

    1. Bayesian update of categorical model values using observations (conditional).
    2. Create initial 2-band VS30 raster from categorical model.
    3. Apply hybrid modifications for slope and coastal distance (geology only).
    4. Spatial adjustment using MVN conditioning with observations (conditional).

    Parameters
    ----------
    model_type : ModelType
        Model type: either GEOLOGY or TERRAIN.
    grid_config : GridConfig
        Grid domain and resolution parameters.
    output_dir : Path
        Directory to save all pipeline outputs.
    categorical_model_csv : Path, optional
        Path to CSV file with categorical Vs30 values. Default from resources.
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
    obs_subsample_step_for_clustered : int, optional
        Subsampling step for clustered observations.

    Raises
    ------
    FileNotFoundError
        If a pipeline step fails to produce its expected output file.
    """
    if categorical_model_csv is None:
        if model_type == constants.ModelType.GEOLOGY:
            categorical_model_csv = (
                constants.RESOURCE_PATH
                / constants.GEOLOGY_MEAN_AND_STANDARD_DEVIATION_PER_CATEGORY_FILE
            )
        elif model_type == constants.ModelType.TERRAIN:
            categorical_model_csv = (
                constants.RESOURCE_PATH
                / constants.TERRAIN_MEAN_AND_STANDARD_DEVIATION_PER_CATEGORY_FILE
            )
        else:
            raise ValueError(f"Invalid model_type: {model_type}")

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Starting full pipeline for {model_type}")
    logger.info(f"Output directory: {output_dir}")

    # --- Step 1: Bayesian update of categorical model values (conditional) ---
    if do_bayesian_update:
        logger.info("\n=== STEP 1: Updating Categorical Models ===")
        update_categorical_vs30_models(
            categorical_model_csv=categorical_model_csv,
            clustered_observations_csv=clustered_observations_csv,
            independent_observations_csv=independent_observations_csv,
            output_dir=output_dir,
            model_type=model_type,
            n_proc=n_proc,
        )

        posterior_csv = (
            output_dir / f"{constants.POSTERIOR_PREFIX}{categorical_model_csv.name}"
        )
        if not posterior_csv.exists():
            raise FileNotFoundError(f"Step 1 failed to produce {posterior_csv}")
    else:
        logger.info(
            "\n=== STEP 1: SKIPPED - Using prior categorical models directly ==="
        )
        posterior_csv = categorical_model_csv

    # --- Step 2: Create initial VS30 raster from categorical model ---
    logger.info("\n=== STEP 2: Creating Initial Raster ===")
    make_initial_vs30_raster(
        output_dir=output_dir,
        grid_config=grid_config,
        terrain=(model_type == constants.ModelType.TERRAIN),
        geology=(model_type == constants.ModelType.GEOLOGY),
        geology_csv=(
            posterior_csv if model_type == constants.ModelType.GEOLOGY else None
        ),
        terrain_csv=(
            posterior_csv if model_type == constants.ModelType.TERRAIN else None
        ),
    )

    initial_raster = output_dir / (
        constants.GEOLOGY_INITIAL_VS30_FILENAME
        if model_type == constants.ModelType.GEOLOGY
        else constants.TERRAIN_INITIAL_VS30_FILENAME
    )
    id_raster_name = (
        constants.GEOLOGY_ID_FILENAME
        if model_type == constants.ModelType.GEOLOGY
        else constants.TERRAIN_ID_FILENAME
    )
    id_raster = output_dir / id_raster_name

    if not initial_raster.exists():
        raise FileNotFoundError(f"Step 2 failed to produce {initial_raster}")

    # --- Step 3: Apply hybrid modifications for slope and coastal distance (geology only) ---
    current_raster = initial_raster

    if model_type == constants.ModelType.GEOLOGY:
        logger.info(
            "\n=== STEP 3: Creating Slope and Coastal Distance Adjusted Geology Raster ==="
        )

        adjust_geology_vs30_by_slope_and_coastal_distance(
            input_raster=initial_raster,
            id_raster=id_raster,
            output_dir=output_dir,
        )
        current_raster = (
            output_dir
            / constants.GEOLOGY_VS30_SLOPE_AND_COASTAL_DISTANCE_ADJUSTED_FILENAME
        )
        if not current_raster.exists():
            raise FileNotFoundError(f"Step 3 failed to produce {current_raster}")

    # --- Step 4: MVN spatial adjustment using observations (conditional) ---
    if mvn:
        logger.info("\n=== STEP 4: Spatial Adjustment ===")

        # Prefer independent observations for spatial fit; fall back to clustered
        spatial_obs_csv = independent_observations_csv or clustered_observations_csv

        if spatial_obs_csv is None:
            raise ValueError(
                "No observation CSVs provided for spatial fit. "
                "At least one of clustered or independent observations must be specified."
            )

        spatial_fit(
            input_raster=current_raster,
            observations_csv=spatial_obs_csv,
            model_values_csv=posterior_csv,
            output_dir=output_dir,
            model_type=model_type,
            noisy=noisy,
            n_proc=n_proc,
            is_clustered_observations=(independent_observations_csv is None),
            obs_subsample_step_for_clustered=obs_subsample_step_for_clustered,
            max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
        )
    else:
        logger.info("\n=== STEP 4: SKIPPED - MVN spatial adjustment disabled ===")
        # Copy the current raster as the final output
        output_filename = constants.OUTPUT_FILENAMES[model_type]
        output_path = output_dir / output_filename
        shutil.copyfile(current_raster, output_path)

    logger.info(f"\nFull pipeline for {model_type} completed successfully")
    logger.info(f"  Final output available in: {output_dir}")


def compute_grid(
    grid_config: config_module.GridConfig,
    output_dir: Path | None = None,
    model_type: constants.ModelType = constants.ModelType.COMBINED,
    combination_method: constants.CombinationMethod = constants.CombinationMethod.STANDARD_DEVIATION_WEIGHTING,
    combine_ratio: float | None = None,
    geology_categorical_csv: Path | None = None,
    terrain_categorical_csv: Path | None = None,
    clustered_observations_csv: Path | None = None,
    independent_observations_csv: Path | None = None,
    do_bayesian_update: bool = True,
    mvn: bool = True,
    noisy: bool = True,
    n_proc: int = 1,
    max_spatial_boolean_array_memory_gb: float = 1.0,
    obs_subsample_step_for_clustered: int = 100,
) -> dict[str, np.ndarray | dict]:
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
    output_dir : Path, optional
        Directory to save all pipeline outputs. If None, a temporary directory
        is used and cleaned up after results are read.
    model_type : ModelType, optional
        Which model(s) to run: GEOLOGY, TERRAIN, or COMBINED (default).
    combination_method : CombinationMethod, optional
        Method for combining models: STANDARD_DEVIATION_WEIGHTING (default)
        or RATIO.
    combine_ratio : float, optional
        Geology-to-terrain weight ratio. Required when combination_method is RATIO.
    geology_categorical_csv : Path, optional
        Path to geology categorical CSV. Default from resources.
    terrain_categorical_csv : Path, optional
        Path to terrain categorical CSV. Default from resources.
    clustered_observations_csv : Path, optional
        Path to CSV file with clustered observations (e.g., CPT data).
    independent_observations_csv : Path, optional
        Path to CSV file with independent observations.
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
    obs_subsample_step_for_clustered : int, optional
        Subsampling step for clustered observations.

    Returns
    -------
    dict
        Dictionary containing the computed raster data with keys:

        - ``"geology_vs30"``, ``"geology_stdv"`` : 2D arrays (when geology is computed)
        - ``"terrain_vs30"``, ``"terrain_stdv"`` : 2D arrays (when terrain is computed)
        - ``"combined_vs30"``, ``"combined_stdv"`` : 2D arrays (when both models are computed)
        - ``"profile"`` : rasterio profile dict with CRS, transform, dimensions, etc.
    """
    # Use a temporary directory if output_dir is not provided
    temp_dir = None
    if output_dir is None:
        temp_dir = tempfile.TemporaryDirectory()
        output_dir = Path(temp_dir.name)

    try:
        result = run_grid_pipeline(
            grid_config=grid_config,
            output_dir=output_dir,
            model_type=model_type,
            combination_method=combination_method,
            combine_ratio=combine_ratio,
            geology_categorical_csv=geology_categorical_csv,
            terrain_categorical_csv=terrain_categorical_csv,
            clustered_observations_csv=clustered_observations_csv,
            independent_observations_csv=independent_observations_csv,
            do_bayesian_update=do_bayesian_update,
            mvn=mvn,
            noisy=noisy,
            n_proc=n_proc,
            max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
            obs_subsample_step_for_clustered=obs_subsample_step_for_clustered,
        )
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()

    return result


def read_raster_bands(path: Path) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read VS30 and standard deviation bands from a 2-band raster file.

    Returns (vs30_array, stdv_array, profile).
    """
    with rasterio.open(path) as src:
        vs30 = src.read(1)
        stdv = src.read(2)
        profile = src.profile.copy()
    return vs30, stdv, profile


def run_grid_pipeline(
    grid_config: config_module.GridConfig,
    output_dir: Path,
    model_type: constants.ModelType,
    combination_method: constants.CombinationMethod,
    combine_ratio: float | None,
    geology_categorical_csv: Path | None,
    terrain_categorical_csv: Path | None,
    clustered_observations_csv: Path | None,
    independent_observations_csv: Path | None,
    do_bayesian_update: bool,
    mvn: bool,
    noisy: bool,
    n_proc: int,
    max_spatial_boolean_array_memory_gb: float,
    obs_subsample_step_for_clustered: int,
) -> dict[str, np.ndarray | dict]:
    """Run the grid pipeline and return computed raster data."""
    start_time = time.time()

    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    run_geology = model_type in (constants.ModelType.GEOLOGY, constants.ModelType.COMBINED)
    run_terrain = model_type in (constants.ModelType.TERRAIN, constants.ModelType.COMBINED)

    # 1. Run Geology Pipeline
    if run_geology:
        logger.info("\n" + "=" * 80 + "\nRUNNING GEOLOGY PIPELINE\n" + "=" * 80)
        run_pipeline_for_model_type(
            model_type=constants.ModelType.GEOLOGY,
            grid_config=grid_config,
            categorical_model_csv=geology_categorical_csv,
            clustered_observations_csv=clustered_observations_csv,
            independent_observations_csv=independent_observations_csv,
            output_dir=output_dir,
            do_bayesian_update=do_bayesian_update,
            mvn=mvn,
            noisy=noisy,
            n_proc=n_proc,
            max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
            obs_subsample_step_for_clustered=obs_subsample_step_for_clustered,
        )

    # 2. Run Terrain Pipeline
    if run_terrain:
        logger.info("\n" + "=" * 80 + "\nRUNNING TERRAIN PIPELINE\n" + "=" * 80)
        run_pipeline_for_model_type(
            model_type=constants.ModelType.TERRAIN,
            grid_config=grid_config,
            categorical_model_csv=terrain_categorical_csv,
            clustered_observations_csv=clustered_observations_csv,
            independent_observations_csv=independent_observations_csv,
            output_dir=output_dir,
            do_bayesian_update=do_bayesian_update,
            mvn=mvn,
            noisy=noisy,
            n_proc=n_proc,
            max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
            obs_subsample_step_for_clustered=obs_subsample_step_for_clustered,
        )

    # 3. Combine geology and terrain models using weighted average
    if run_geology and run_terrain:
        logger.info(
            "\n" + "=" * 80 + "\nCOMBINING GEOLOGY AND TERRAIN RESULTS\n" + "=" * 80
        )

        combined_tif = output_dir / constants.COMBINED_VS30_FILENAME
        combine(
            geology_tif=output_dir
            / constants.OUTPUT_FILENAMES[constants.ModelType.GEOLOGY],
            terrain_tif=output_dir
            / constants.OUTPUT_FILENAMES[constants.ModelType.TERRAIN],
            output_path=combined_tif,
            combination_method=combination_method,
            combine_ratio=combine_ratio,
        )

    elapsed_time = time.time() - start_time
    logger.info(f"  Total execution time: {elapsed_time:.1f} seconds")
    logger.info(f"  Output available in: {output_dir}")

    # Read output rasters into the result dict
    result: dict[str, np.ndarray | dict] = {}
    profile = None

    if run_geology:
        geol_path = output_dir / constants.OUTPUT_FILENAMES[constants.ModelType.GEOLOGY]
        geol_vs30, geol_stdv, profile = read_raster_bands(geol_path)
        result["geology_vs30"] = geol_vs30
        result["geology_stdv"] = geol_stdv

    if run_terrain:
        terr_path = output_dir / constants.OUTPUT_FILENAMES[constants.ModelType.TERRAIN]
        terr_vs30, terr_stdv, profile = read_raster_bands(terr_path)
        result["terrain_vs30"] = terr_vs30
        result["terrain_stdv"] = terr_stdv

    if run_geology and run_terrain:
        combined_path = output_dir / constants.COMBINED_VS30_FILENAME
        comb_vs30, comb_stdv, profile = read_raster_bands(combined_path)
        result["combined_vs30"] = comb_vs30
        result["combined_stdv"] = comb_stdv

    result["profile"] = profile

    return result


def compute_at_locations(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    combination_method: constants.CombinationMethod = constants.CombinationMethod.STANDARD_DEVIATION_WEIGHTING,
    combine_ratio: float | None = None,
    geology_categorical_csv: Path | None = None,
    terrain_categorical_csv: Path | None = None,
    clustered_observations_csv: Path | None = None,
    independent_observations_csv: Path | None = None,
    include_intermediate: bool = True,
    mvn: bool = True,
    noisy: bool = True,
    n_proc: int = 1,
) -> pd.DataFrame:
    """
    Compute Vs30 values at specific latitude/longitude locations.

    Runs the Vs30 pipeline at the specified query points without generating
    raster grids. This is efficient for querying Vs30 at a small number of
    locations.

    The pipeline stages mirror those in compute_grid:
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
    combination_method : CombinationMethod, optional
        Method for combining models: STANDARD_DEVIATION_WEIGHTING (default)
        or RATIO.
    combine_ratio : float, optional
        Geology-to-terrain weight ratio. Required when combination_method is RATIO.
    geology_categorical_csv : Path, optional
        Path to geology categorical CSV. Default from resources.
    terrain_categorical_csv : Path, optional
        Path to terrain categorical CSV. Default from resources.
    clustered_observations_csv : Path, optional
        Path to CSV file with clustered observations (e.g., CPT).
    independent_observations_csv : Path, optional
        Path to CSV file with independent observations.
    include_intermediate : bool, optional
        Include intermediate values (geology/terrain separately) in output.
    mvn : bool, optional
        Whether to perform MVN spatial adjustment. If False, spatial fit is skipped.
    noisy : bool, optional
        Whether to apply noise weighting in spatial adjustment.
    n_proc : int, optional
        Number of parallel processes. Use -1 for all cores.

    Returns
    -------
    DataFrame
        Results with columns: easting, northing, geology_id, geology_mvn_vs30,
        geology_mvn_stdv, terrain_id, terrain_mvn_vs30, terrain_mvn_stdv,
        vs30, stdv. If include_intermediate is True, also includes
        geology_vs30, geology_stdv, geology_vs30_hybrid, geology_stdv_hybrid,
        terrain_vs30, terrain_stdv.
    """
    # Resolve default categorical model CSVs from resources
    if geology_categorical_csv is None:
        geology_categorical_csv = (
            constants.RESOURCE_PATH
            / constants.GEOLOGY_MEAN_AND_STANDARD_DEVIATION_PER_CATEGORY_FILE
        )
    if terrain_categorical_csv is None:
        terrain_categorical_csv = (
            constants.RESOURCE_PATH
            / constants.TERRAIN_MEAN_AND_STANDARD_DEVIATION_PER_CATEGORY_FILE
        )

    # Convert WGS84 to NZTM
    nztm_coords = coordinates.wgs_depth_to_nztm(
        np.column_stack([latitudes, longitudes])
    )
    points = nztm_coords[:, ::-1]  # (easting, northing)

    logger.info(f"Processing {len(points)} locations")

    # Load and combine all available observation files for spatial adjustment
    observation_csvs = [
        csv
        for csv in [clustered_observations_csv, independent_observations_csv]
        if csv is not None and csv.exists()
    ]

    if mvn and observation_csvs:
        observations_df = pd.concat(
            [pd.read_csv(csv) for csv in observation_csvs],
            ignore_index=True,
        )
    else:
        observations_df = pd.DataFrame(columns=constants.REQUIRED_OBSERVATION_COLUMNS)

    logger.info(f"Loaded {len(observations_df)} observations for spatial adjustment")

    # Load categorical models (skipinitialspace handles spaces after commas)
    geol_model_df = pd.read_csv(geology_categorical_csv, skipinitialspace=True)
    terr_model_df = pd.read_csv(terrain_categorical_csv, skipinitialspace=True)

    n_proc_resolved = parallel.resolve_n_proc(n_proc)

    # ================================================================
    # Parallel Processing Path
    # ================================================================
    if n_proc_resolved > 1:
        logger.info(f"\nProcessing with {n_proc_resolved} parallel workers...")

        loc_config = parallel.LocationsChunkConfig(
            include_intermediate=include_intermediate,
            combination_method=combination_method,
            combine_ratio=combine_ratio,
            noisy=noisy,
        )

        result_df = parallel.run_parallel_locations(
            points=points,
            observations_df=observations_df,
            geol_model_df=geol_model_df,
            terr_model_df=terr_model_df,
            config=loc_config,
            n_proc=n_proc_resolved,
        )

        # Add coordinate columns at the front
        result_df.insert(0, constants.COL_EASTING, points[:, 0])
        result_df.insert(1, constants.COL_NORTHING, points[:, 1])

        logger.info(f"  Total locations: {len(result_df)}")
        return result_df

    # ================================================================
    # Sequential Processing Path
    # ================================================================
    result = {}
    result[constants.COL_EASTING] = points[:, 0]
    result[constants.COL_NORTHING] = points[:, 1]

    # --- Stage 1-3: Geology model (categorical lookup, hybrid mods, spatial adjustment) ---
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
            noisy=noisy,
            progress_bar=pbar,
        )

    result[constants.COL_GEOLOGY_ID] = geol_ids
    if include_intermediate:
        result[constants.COL_GEOLOGY_VS30] = geol_vs30
        result[constants.COL_GEOLOGY_STDV] = geol_stdv
        result[constants.COL_GEOLOGY_VS30_HYBRID] = geol_vs30_hybrid
        result[constants.COL_GEOLOGY_STDV_HYBRID] = geol_stdv_hybrid
    result[constants.COL_GEOLOGY_MVN_VS30] = geol_mvn_vs30
    result[constants.COL_GEOLOGY_MVN_STDV] = geol_mvn_stdv

    # --- Stage 1, 3: Terrain model (categorical lookup, spatial adjustment — no hybrid mods) ---
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
            noisy=noisy,
            progress_bar=pbar,
        )

    result[constants.COL_TERRAIN_ID] = terr_ids
    if include_intermediate:
        result[constants.COL_TERRAIN_VS30] = terr_vs30
        result[constants.COL_TERRAIN_STDV] = terr_stdv
    result[constants.COL_TERRAIN_MVN_VS30] = terr_mvn_vs30
    result[constants.COL_TERRAIN_MVN_STDV] = terr_mvn_stdv

    # --- Stage 4: Combine geology and terrain models using weighted average ---
    logger.info("Combining models...")
    combined_vs30, combined_stdv = utils.combine_vs30_models(
        geol_mvn_vs30,
        geol_mvn_stdv,
        terr_mvn_vs30,
        terr_mvn_stdv,
        combination_method,
        combine_ratio,
    )

    result[constants.COL_VS30] = combined_vs30
    result[constants.COL_COMBINED_STDV] = combined_stdv

    logger.info(f"  Total locations: {len(points)}")
    return pd.DataFrame(result)

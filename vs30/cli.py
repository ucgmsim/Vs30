"""
Command-line interface for the vs30 package.

This module provides CLI commands for the full Vs30 modelling pipeline:

Commands
--------
- update-categorical-vs30-models: Bayesian update of categorical Vs30 values
- make-initial-vs30-raster: Create initial Vs30 rasters from categories
- adjust-geology-vs30-by-slope-and-coastal-distance: Apply hybrid modifications
- spatial-fit: Spatial adjustment using observations
- combine: Combine geology and terrain models
- full-pipeline: Run the complete workflow

Usage
-----
    vs30 full-pipeline  # Run everything with default config
    vs30 --config /path/to/config.yaml full-pipeline  # Use custom config
    vs30 --help         # See all available commands
"""

import logging
import shutil
import typing
from pathlib import Path

import matplotlib.pyplot
import numpy as np
import pandas as pd
import rasterio
import typer

from qcore import cli
from vs30 import config as config_module
from vs30 import constants
from vs30 import raster
from vs30 import spatial
from vs30 import utils

logger = logging.getLogger(__name__)

# Create Typer app for CLI
app = typer.Typer(
    name="vs30",
    help="VS30 map generation and categorical model updates",
    add_completion=False,
)

# =============================================================================
# Helpers
# =============================================================================


def resolve_observation_csv(
    csv_path: Path | None,
    config_filename: str | None,
    res_dir: Path,
) -> Path | None:
    """
    Resolve an observation CSV path from CLI argument or config default.

    Parameters
    ----------
    csv_path : Path or None
        Explicitly provided path (from CLI argument).
    config_filename : str or None
        Filename from config (e.g., cfg.clustered_observations_file).
    res_dir : Path
        Resources directory to look for config-specified files.

    Returns
    -------
    Path or None
        Resolved path, or None if no valid file found.
    """
    if csv_path is not None:
        return csv_path
    if config_filename is not None:
        candidate = res_dir / config_filename
        if candidate.exists():
            return candidate
    return None


def validate_csv_columns(
    df: pd.DataFrame, required_cols: list[str], label: str
) -> None:
    """
    Raise typer.Exit if the DataFrame is missing any required columns.

    Parameters
    ----------
    df : DataFrame
        DataFrame to validate.
    required_cols : list[str]
        Column names that must be present.
    label : str
        Descriptive label used in the error message (e.g. "Clustered observations CSV").
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        typer.echo(f"Error: {label} missing required columns: {missing}", err=True)
        raise typer.Exit(1)


# =============================================================================
# Global Config State
# =============================================================================

_cli_config: config_module.Vs30Config | None = None


def get_config() -> config_module.Vs30Config:
    """
    Get the current CLI configuration.

    Returns the config set by the --config option, or the default
    package config if no custom config was specified.
    """
    global _cli_config
    if _cli_config is None:
        _cli_config = config_module.Vs30Config.default()
    return _cli_config


@app.callback()
def main(
    config: typing.Annotated[
        Path | None,
        typer.Option(
            "--config",
            "-c",
            exists=True,
            dir_okay=False,
            help="Path to config.yaml file (default: package config.yaml)",
        ),
    ] = None,
    verbose: typing.Annotated[
        bool,
        typer.Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
) -> None:
    """
    VS30 map generation and categorical model updates.

    This is the main entry point for the VS30 CLI. Global options like config
    file path and verbosity are specified here and apply to all subcommands.

    Parameters
    ----------
    config : Path, optional
        Path to config.yaml file. If not specified, uses the default
        configuration bundled with the package.
    verbose : bool, optional
        Enable verbose (DEBUG level) logging output.
    """
    global _cli_config

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Load config from specified path or default
    # Note: typer's exists=True already validates the file exists before reaching here
    if config is not None:
        _cli_config = config_module.Vs30Config.from_yaml(config)
        logger.info(f"Loaded config from {config}")
    else:
        _cli_config = config_module.Vs30Config.default()
        logger.debug(f"Using default config from {config_module.Vs30Config.default_config_path()}")


@cli.from_docstring(app)
def update_categorical_vs30_models(
    categorical_model_csv: typing.Annotated[
        Path, typer.Option("--categorical-model-csv", "-m", exists=True, dir_okay=False)
    ],
    output_dir: typing.Annotated[Path, typer.Option("--output-dir", "-d", file_okay=False)],
    model_type: typing.Annotated[str, typer.Option("--model-type", "-t")],
    clustered_observations_csv: typing.Annotated[
        Path | None, typer.Option("--clustered-observations-csv", "-c", exists=True, dir_okay=False)
    ] = None,
    independent_observations_csv: typing.Annotated[
        Path | None, typer.Option("--independent-observations-csv", "-o", exists=True, dir_okay=False)
    ] = None,
    nproc: typing.Annotated[int | None, typer.Option("--nproc")] = None,
) -> None:
    """
    Update categorical model values using Bayesian updates and save to CSV files.

    This command loads observations and categorical model values, applies Bayesian
    updates to the categorical model values (mean and standard deviation per category),
    and writes the updated values back to CSV files.

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
    nproc : int, optional
        Number of processes for DBSCAN clustering. Use -1 for all available cores.
        Default from config.
    """
    try:
        cfg = get_config()
        nproc = nproc if nproc is not None else cfg.n_proc

        if clustered_observations_csv is None and independent_observations_csv is None:
            typer.echo(
                "Error: At least one of --clustered-observations-csv or --independent-observations-csv must be provided",
                err=True,
            )
            raise typer.Exit(1)

        if model_type not in ["geology", "terrain"]:
            typer.echo(
                f"Error: model_type must be 'geology' or 'terrain', got '{model_type}'",
                err=True,
            )
            raise typer.Exit(1)

        logger.info(f"Model type: {model_type}")
        logger.info(f"Loading categorical model from: {categorical_model_csv}")

        categorical_model_df = pd.read_csv(categorical_model_csv, skipinitialspace=True)

        # Drop rows with placeholder values for excluded categories (e.g., water)
        categorical_model_df = categorical_model_df[
            categorical_model_df["mean_vs30_km_per_s"] != constants.NODATA_VALUE
        ]

        validate_csv_columns(
            categorical_model_df,
            ["mean_vs30_km_per_s", "standard_deviation_vs30_km_per_s"],
            "Categorical model CSV",
        )

        # Import category module
        from vs30 import category

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

            validate_csv_columns(
                clustered_observations_df,
                ["easting", "northing", "vs30"],
                "Clustered observations CSV",
            )

            logger.info(
                f"Loaded {len(clustered_observations_df)} clustered observations"
            )

            # Assign category IDs
            obs_locs = clustered_observations_df[["easting", "northing"]].values
            if model_type == "geology":
                model_ids = category._assign_to_category_geology(obs_locs)
            else:  # terrain
                model_ids = category._assign_to_category_terrain(obs_locs)

            clustered_observations_df[constants.STANDARD_ID_COLUMN] = model_ids

            # Log assignment statistics
            unique_assigned_ids = clustered_observations_df[constants.STANDARD_ID_COLUMN].unique()
            n_valid = np.sum(clustered_observations_df[constants.STANDARD_ID_COLUMN] != constants.RASTER_ID_NODATA_VALUE)
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
                clustered_observations_df, model_type, nproc
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

            validate_csv_columns(
                independent_observations_df,
                ["easting", "northing", "vs30", "uncertainty"],
                "Independent observations CSV",
            )

            logger.info(
                f"Loaded {len(independent_observations_df)} independent observations"
            )

            # Assign category IDs
            obs_locs = independent_observations_df[["easting", "northing"]].values
            if model_type == "geology":
                model_ids = category._assign_to_category_geology(obs_locs)
            else:  # terrain
                model_ids = category._assign_to_category_terrain(obs_locs)

            independent_observations_df[constants.STANDARD_ID_COLUMN] = model_ids

        # Perform Bayesian update(s) via dispatcher
        logger.info("Applying Bayesian updates...")
        current_prior_df = category.posterior_from_bayesian_update(
            current_prior_df,
            independent_observations_df=independent_observations_df,
            clustered_observations_df=clustered_observations_df,
            model_type=model_type,
        )

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = constants.POSTERIOR_PREFIX + categorical_model_csv.name
        output_path = output_dir / output_filename

        current_prior_df.to_csv(output_path, index=False)

        typer.echo("✓ Successfully updated categorical model values")
        typer.echo(f"  Output saved to: {output_path}")

    except Exception as e:
        logger.exception("Error updating category values")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@cli.from_docstring(app)
def make_initial_vs30_raster(
    terrain: typing.Annotated[bool, typer.Option("--terrain")] = False,
    geology: typing.Annotated[bool, typer.Option("--geology")] = False,
    output_dir: typing.Annotated[Path | None, typer.Option("--output-dir", "-o", file_okay=False)] = None,
    geology_csv: typing.Annotated[
        Path | None, typer.Option("--geology-csv", exists=True, dir_okay=False)
    ] = None,
    terrain_csv: typing.Annotated[
        Path | None, typer.Option("--terrain-csv", exists=True, dir_okay=False)
    ] = None,
) -> None:
    """
    Create initial VS30 mean and standard deviation rasters from category IDs.

    This command generates initial VS30 rasters by:

    1. Creating category ID rasters (from terrain raster or geology shapefile)
    2. Mapping category IDs to VS30 mean and standard deviation values from CSV files
    3. Writing 2-band GeoTIFFs with VS30 mean (band 1) and standard deviation (band 2)

    Output files are saved as terrain_initial_vs30_with_uncertainty.tif and/or
    geology_initial_vs30_with_uncertainty.tif.

    Parameters
    ----------
    terrain : bool, optional
        Create terrain VS30 raster.
    geology : bool, optional
        Create geology VS30 raster.
    output_dir : Path, optional
        Output directory. Default from config.yaml.
    geology_csv : Path, optional
        Custom geology model CSV file.
    terrain_csv : Path, optional
        Custom terrain model CSV file.
    """
    try:
        if not terrain and not geology:
            typer.echo(
                "Error: At least one of --terrain or --geology must be specified",
                err=True,
            )
            raise typer.Exit(1)

        cfg = get_config()

        if output_dir is None:
            output_dir = Path(cfg.output_dir)

        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        grid_params = {
            "xmin": cfg.grid_xmin,
            "xmax": cfg.grid_xmax,
            "ymin": cfg.grid_ymin,
            "ymax": cfg.grid_ymax,
            "dx": cfg.grid_dx,
            "dy": cfg.grid_dy,
        }
        logger.info(f"Using grid parameters: {grid_params}")

        # Process terrain if requested
        if terrain:
            logger.info("Processing terrain model...")
            terrain_model_csv = terrain_csv if terrain_csv else constants.TERRAIN_MEAN_AND_STANDARD_DEVIATION_PER_CATEGORY_FILE
            logger.info(f"Using terrain model values from {terrain_model_csv}")

            logger.info("Creating terrain category ID raster...")
            id_raster = raster.create_category_id_raster(
                "terrain", output_dir, **grid_params
            )

            logger.info("Creating terrain VS30 raster...")
            vs30_raster = output_dir / constants.TERRAIN_INITIAL_VS30_FILENAME
            raster.create_vs30_raster_from_ids(id_raster, terrain_model_csv, vs30_raster)
            typer.echo(f"✓ Created terrain VS30 raster: {vs30_raster}")

        # Process geology if requested
        if geology:
            logger.info("Processing geology model...")
            geology_model_csv = geology_csv if geology_csv else constants.GEOLOGY_MEAN_AND_STANDARD_DEVIATION_PER_CATEGORY_FILE
            logger.info(f"Using geology model values from {geology_model_csv}")

            logger.info("Creating geology category ID raster...")
            id_raster = raster.create_category_id_raster(
                "geology", output_dir, **grid_params
            )

            logger.info("Creating geology VS30 raster...")
            vs30_raster = output_dir / constants.GEOLOGY_INITIAL_VS30_FILENAME
            raster.create_vs30_raster_from_ids(id_raster, geology_model_csv, vs30_raster)
            typer.echo(f"✓ Created geology VS30 raster: {vs30_raster}")

        typer.echo("✓ Successfully created initial VS30 rasters")

    except Exception as e:
        logger.exception("Error creating initial VS30 rasters")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@cli.from_docstring(app)
def adjust_geology_vs30_by_slope_and_coastal_distance(
    input_raster: typing.Annotated[
        Path, typer.Option("--input-raster", "-i", exists=True, dir_okay=False)
    ],
    id_raster: typing.Annotated[Path, typer.Option("--id-raster", exists=True, dir_okay=False)],
    output_dir: typing.Annotated[Path, typer.Option("--output-dir", "-o", file_okay=False)],
) -> None:
    """
    Apply hybrid geology modifications to an initial VS30 raster.

    This command adds slope-based and coast-distance-based modifications to the
    geology model. It generates intermediate slope and coast distance rasters
    in the output directory.

    Parameters
    ----------
    input_raster : Path
        Path to initial geology VS30 raster (created by make-initial-vs30-raster).
        Must be a 2-band raster with Vs30 and standard deviation.
    id_raster : Path
        Path to category ID raster (e.g., gid.tif used to create the input raster).
    output_dir : Path
        Directory to save output hybrid raster and intermediate files.
    """
    try:
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = get_config()

        logger.info(
            f"Processing slope and coastal distance adjusted model for: {input_raster}"
        )

        # 1. Load Input Raster
        with rasterio.open(input_raster) as src:
            vs30_array = src.read(1)
            stdv_array = src.read(2)
            profile = src.profile.copy()

            # Check dimensions against ID raster
            with rasterio.open(id_raster) as id_src:
                if id_src.width != src.width or id_src.height != src.height:
                    raise ValueError(
                        f"Dimension mismatch! Input raster: {src.width}x{src.height}, "
                        f"ID raster: {id_src.width}x{id_src.height}"
                    )
                id_array = id_src.read(1)

        # 2. Create/Load Intermediate Rasters
        slope_path = output_dir / constants.SLOPE_RASTER_FILENAME
        logger.info(f"Generating slope raster: {slope_path}")
        slope_array, _ = raster.create_slope_raster(slope_path, profile)

        coast_path = output_dir / constants.COAST_DISTANCE_RASTER_FILENAME
        logger.info(f"Generating coast distance raster: {coast_path}")
        coast_dist_array, _ = raster.create_coast_distance_raster(coast_path, profile)

        # 3. Apply Modifications
        logger.info(
            "Applying slope and coastal distance based geology modifications..."
        )
        mod_vs30, mod_stdv = raster.apply_hybrid_geology_modifications(
            vs30_array,
            stdv_array,
            id_array,
            slope_array,
            coast_dist_array,
        )

        # 4. Save Output
        output_path = (
            output_dir
            / constants.GEOLOGY_VS30_SLOPE_AND_COASTAL_DISTANCE_ADJUSTED_FILENAME
        )

        # Update profile for output
        profile.update({"dtype": "float32", "compress": "deflate"})

        logger.info(f"Saving hybrid raster to: {output_path}")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mod_vs30, 1)
            dst.write(mod_stdv, 2)
            dst.descriptions = ("Vs30 (Hybrid)", "Standard Deviation (Hybrid)")

        typer.echo(
            "✓ Successfully created slope and coastal distance adjusted geology raster"
        )
        typer.echo(f"  Output saved to: {output_path}")

    except Exception as e:
        logger.exception("Error creating hybrid raster")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


# =============================================================================
# Helper Functions for spatial_fit
# =============================================================================


def prepare_observations_for_spatial_fit(
    observations: pd.DataFrame,
    raster_data: "spatial.RasterData",
    updated_model_table: np.ndarray,
    model_type: str,
    output_dir: Path,
    cfg: "config_module.Vs30Config",
) -> "spatial.ObservationData | None":
    """
    Prepare observation data for spatial adjustment.

    Parameters
    ----------
    observations : DataFrame
        Raw observation data with easting, northing, vs30, uncertainty columns.
    raster_data : RasterData
        Input raster data object.
    updated_model_table : ndarray
        Updated categorical model values [mean, stdv] indexed by category ID.
    model_type : str
        Either 'geology' or 'terrain'.
    output_dir : Path
        Output directory for any intermediate files.
    cfg : config_module.Vs30Config
        Configuration object.

    Returns
    -------
    ObservationData or None
        Prepared observation data, or None if no valid observations found.
    """
    logger.info("Preparing observation data for spatial adjustment...")

    obs_data = spatial.prepare_observation_data(
        observations,
        raster_data,
        updated_model_table,
        model_type,
        output_dir,
        noisy=cfg.noisy,  # noisy is still user-configurable
    )

    logger.info(f"Prepared {len(obs_data.locations)} valid observations")

    if len(obs_data.locations) == 0:
        return None

    return obs_data


def apply_clustered_subsampling(
    obs_data: "spatial.ObservationData",
    is_clustered_obs: bool,
    cfg: "config_module.Vs30Config",
) -> "spatial.ObservationData":
    """
    Apply DBSCAN clustering and subsampling to observations for optimized pixel search.

    For clustered observations (e.g., CPT data), this function identifies spatial
    clusters and subsamples within each cluster to reduce the number of observations
    used for the bounding box search, while maintaining spatial coverage.

    Parameters
    ----------
    obs_data : ObservationData
        Full observation data.
    is_clustered_obs : bool
        Whether the observations are from a clustered observations file.
    cfg : config_module.Vs30Config
        Configuration object with clustering parameters.

    Returns
    -------
    ObservationData
        Either the original obs_data (if not clustered) or a subsampled version
        for use in the affected pixel search.
    """
    if not is_clustered_obs:
        return obs_data

    import sklearn.cluster

    logger.info("Clustering observations for optimized affected pixel search...")

    # Run DBSCAN directly on the filtered observation locations
    dbscan = sklearn.cluster.DBSCAN(eps=constants.EPS, min_samples=constants.MIN_GROUP, n_jobs=cfg.n_proc)
    cluster_labels = dbscan.fit_predict(obs_data.locations)

    n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
    n_noise = np.sum(cluster_labels == -1)
    logger.info(
        f"Found {n_clusters} clusters and {n_noise} noise points "
        f"in {len(obs_data.locations)} observations"
    )

    # Subsample observations within each cluster
    subsample_indices = spatial.subsample_by_cluster(
        cluster_labels, step=cfg.obs_subsample_step_for_clustered
    )

    # Create subsampled ObservationData for bbox search only
    obs_data_for_bbox = spatial.ObservationData(
        locations=obs_data.locations[subsample_indices],
        vs30=obs_data.vs30[subsample_indices],
        model_vs30=obs_data.model_vs30[subsample_indices],
        model_stdv=obs_data.model_stdv[subsample_indices],
        residuals=obs_data.residuals[subsample_indices],
        omega=obs_data.omega[subsample_indices],
        uncertainty=obs_data.uncertainty[subsample_indices],
    )

    logger.info(
        f"Subsampled to {len(subsample_indices)} observations for affected pixel search "
        f"(step={obs_subsample_step})"
    )

    return obs_data_for_bbox


@cli.from_docstring(app)
def spatial_fit(
    input_raster: typing.Annotated[
        Path, typer.Option("--input-raster", "-i", exists=True, dir_okay=False)
    ],
    observations_csv: typing.Annotated[
        Path, typer.Option("--observations-csv", "-o", exists=True, dir_okay=False)
    ],
    model_values_csv: typing.Annotated[
        Path, typer.Option("--model-values-csv", "-m", exists=True, dir_okay=False)
    ],
    output_dir: typing.Annotated[Path, typer.Option("--output-dir", "-d", file_okay=False)],
    model_type: typing.Annotated[str, typer.Option("--model-type", "-t")],
    n_proc: typing.Annotated[int | None, typer.Option("--n-proc")] = None,
) -> None:
    """
    Adjust a VS30 raster based on measurements using spatial conditioning.

    This command performs a spatial adjustment of an input raster by:

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
    model_type : str
        Model type: either 'geology' or 'terrain'.
    n_proc : int, optional
        Number of parallel processes. Use -1 for all cores. Default from config.
    """
    try:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        cfg = get_config()

        max_dist_m = constants.MAX_DIST_M
        max_points = constants.MAX_POINTS
        phi = constants.PHI[model_type]
        noisy = cfg.noisy  # noisy is still user-configurable
        cov_reduc = constants.COV_REDUC

        from vs30 import parallel

        n_proc_resolved = parallel.resolve_n_proc(
            n_proc if n_proc is not None else cfg.n_proc
        )

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

        # Check if this is clustered observations file (for subsampling optimization)
        clustered_obs_file = cfg.clustered_observations_file
        is_clustered_obs = (
            clustered_obs_file is not None
            and observations_csv.name in clustered_obs_file
        )

        # 3. Load Model Values (updated categorical table)
        logger.info("Loading updated model table...")
        # We need the table as a numpy array [mean, stdv] for each category index (1-indexed in raster)
        # category.py functions usually handle this.
        model_df = pd.read_csv(model_values_csv, skipinitialspace=True)
        # Determine columns
        mean_col, std_col = raster._select_vs30_columns_by_priority(list(model_df.columns))

        # Build table indexed by category ID
        max_id = model_df["id"].max()
        updated_model_table = np.full((max_id, 2), np.nan)
        for _, row in model_df.iterrows():
            idx = int(row["id"]) - 1
            if 0 <= idx < max_id:
                updated_model_table[idx, 0] = row[mean_col]
                updated_model_table[idx, 1] = row[std_col]

        # 4. Prepare Observation Data for Spatial Adjustment
        obs_data = prepare_observations_for_spatial_fit(
            observations, raster_data, updated_model_table, model_type, output_dir, cfg
        )

        if obs_data is None:
            logger.warning(
                "No valid observations found within model bounds. Copying input raster to output."
            )
            output_filename = constants.OUTPUT_FILENAMES[model_type]
            output_path = output_dir / output_filename
            shutil.copyfile(input_raster, output_path)
            typer.echo(f"✓ No updates needed. Copied to {output_path}")
            return

        # 5. Find Affected Pixels (with optional clustered subsampling)
        obs_data_for_bbox = apply_clustered_subsampling(obs_data, is_clustered_obs, cfg)

        logger.info("Finding pixels affected by observations...")
        bbox_result = spatial.find_affected_pixels(
            raster_data,
            obs_data_for_bbox,
            max_dist_m=max_dist_m,
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
                phi=phi,
                max_dist_m=max_dist_m,
                max_points=max_points,
                noisy=noisy,
                cov_reduc=cov_reduc,
                n_proc=n_proc_resolved,
            )
        else:
            updates = spatial.compute_spatial_adjustments(
                raster_data,
                obs_data,
                bbox_result,
                model_type,
                phi=phi,
                max_dist_m=max_dist_m,
                max_points=max_points,
                noisy=noisy,
                cov_reduc=cov_reduc,
            )

        # 7. Apply Updates and Write Output
        logger.info("Applying updates and writing output...")
        spatial.apply_and_write_updates(raster_data, updates, model_type, output_dir)

        typer.echo(f"✓ Successfully completed spatial fit for {model_type}")
        typer.echo(f"  Results saved to: {output_dir}")

    except Exception as e:
        logger.exception("Error in spatial-fit")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@cli.from_docstring(app)
def plot_posterior_values(
    csv_path: typing.Annotated[Path, typer.Option("--csv-path", "-c", exists=True, dir_okay=False)],
    output_dir: typing.Annotated[Path, typer.Option("--output-dir", "-o", file_okay=False)],
) -> None:
    """
    Plot prior and posterior Vs30 mean values with error bars.

    Creates a plot showing Vs30 mean values (y-axis) vs category ID (x-axis)
    for both prior and posterior values, with error bars showing standard
    deviation.

    Parameters
    ----------
    csv_path : Path
        Path to CSV file with prior and posterior values.
    output_dir : Path
        Path to output directory for plots. Will be created if it does not exist.
    """
    try:
        logger.info(f"Loading data from: {csv_path}")

        df = pd.read_csv(csv_path, skipinitialspace=True)

        cfg = get_config()

        # Filter out rows with placeholder values for excluded categories (e.g., water)
        df = df[df["prior_mean_vs30_km_per_s"] != constants.NODATA_VALUE].copy()

        validate_csv_columns(
            df,
            [
                "id",
                "prior_mean_vs30_km_per_s",
                "prior_standard_deviation_vs30_km_per_s",
                "posterior_mean_vs30_km_per_s",
                "posterior_standard_deviation_vs30_km_per_s",
            ],
            "Plot input CSV",
        )

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract data
        category_ids = df["id"].values
        prior_mean = df["prior_mean_vs30_km_per_s"].values
        prior_std = df["prior_standard_deviation_vs30_km_per_s"].values
        posterior_mean = df["posterior_mean_vs30_km_per_s"].values
        posterior_std = df["posterior_standard_deviation_vs30_km_per_s"].values

        # Create figure
        fig, ax = matplotlib.pyplot.subplots(figsize=constants.PLOT_FIGSIZE)

        # Offset for x positions to separate prior and posterior
        prior_x = category_ids - 0.2
        posterior_x = category_ids + 0.2

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
        matplotlib.pyplot.tight_layout()
        matplotlib.pyplot.savefig(output_path, dpi=constants.PLOT_DPI, bbox_inches="tight")
        matplotlib.pyplot.close()

        logger.info(f"Plot saved to: {output_path}")
        typer.echo(f"✓ Plot saved to: {output_path}")

    except Exception as e:
        logger.exception("Error creating plot")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@cli.from_docstring(app)
def full_pipeline_for_geology_or_terrain(
    model_type: typing.Annotated[str, typer.Option("--model-type", "-t")],
    categorical_model_csv: typing.Annotated[
        Path, typer.Option("--categorical-model-csv", "-m", exists=True, dir_okay=False)
    ],
    output_dir: typing.Annotated[Path, typer.Option("--output-dir", "-d", file_okay=False)],
    clustered_observations_csv: typing.Annotated[
        Path | None, typer.Option("--clustered-observations-csv", "-c", exists=True, dir_okay=False)
    ] = None,
    independent_observations_csv: typing.Annotated[
        Path | None, typer.Option("--independent-observations-csv", "-o", exists=True, dir_okay=False)
    ] = None,
    nproc: typing.Annotated[int | None, typer.Option("--nproc")] = None,
    n_proc: typing.Annotated[int | None, typer.Option("--n-proc")] = None,
) -> None:
    """
    Run the full VS30 generation pipeline for a single model type.

    Executes the complete pipeline in sequence:

    1. update-categorical-vs30-models: (Conditional) Updates categorical priors
       with observations if do_bayesian_update is enabled in config.
    2. make-initial-vs30-raster: Creates initial 2-band VS30 raster using
       posteriors (or priors if updates skipped).
    3. create-hybrid-raster: (Geology only) Applies slope/coastal modifications.
    4. spatial-fit: Final spatial adjustment using observations.

    Parameters
    ----------
    model_type : str
        Model type: either 'geology' or 'terrain'.
    categorical_model_csv : Path
        Path to CSV file with categorical Vs30 values.
    output_dir : Path
        Directory to save all pipeline outputs.
    clustered_observations_csv : Path, optional
        Path to CSV file with clustered observations (e.g., CPT data).
    independent_observations_csv : Path, optional
        Path to CSV file with independent observations (e.g., measured filtered).
    nproc : int, optional
        Number of processes for clustering.
    n_proc : int, optional
        Number of parallel processes for spatial adjustment.
    """
    try:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting full pipeline for {model_type}")
        logger.info(f"Output directory: {output_dir}")

        cfg = get_config()

        nproc = nproc if nproc is not None else cfg.n_proc

        do_bayesian_update = cfg.do_bayesian_update_of_geology_and_terrain_categorical_vs30_values

        # Resolve observations from config if not provided
        res_dir = Path(__file__).parent / "resources"
        clustered_observations_csv = resolve_observation_csv(
            clustered_observations_csv, cfg.clustered_observations_file, res_dir
        )
        independent_observations_csv = resolve_observation_csv(
            independent_observations_csv, cfg.independent_observations_file, res_dir
        )

        # --- Step 1: Update Categorical Models (conditional) ---
        if do_bayesian_update:
            logger.info("\n=== STEP 1: Updating Categorical Models ===")
            update_categorical_vs30_models(
                categorical_model_csv=categorical_model_csv,
                clustered_observations_csv=clustered_observations_csv,
                independent_observations_csv=independent_observations_csv,
                output_dir=output_dir,
                model_type=model_type,
                nproc=nproc,
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

        # --- Step 2: Make Initial Raster ---
        logger.info("\n=== STEP 2: Creating Initial Raster ===")
        make_initial_vs30_raster(
            terrain=(model_type == "terrain"),
            geology=(model_type == "geology"),
            output_dir=output_dir,
            geology_csv=posterior_csv if model_type == "geology" else None,
            terrain_csv=posterior_csv if model_type == "terrain" else None,
        )

        initial_raster = output_dir / (
            constants.GEOLOGY_INITIAL_VS30_FILENAME
            if model_type == "geology"
            else constants.TERRAIN_INITIAL_VS30_FILENAME
        )
        id_raster_name = (
            constants.GEOLOGY_ID_FILENAME
            if model_type == "geology"
            else constants.TERRAIN_ID_FILENAME
        )
        id_raster = output_dir / id_raster_name

        if not initial_raster.exists():
            raise FileNotFoundError(f"Step 2 failed to produce {initial_raster}")

        # --- Step 3: Hybrid Modification (Geology Only) ---
        current_raster = initial_raster

        if model_type == "geology":
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

        # --- Step 4: Spatial Fit ---
        logger.info("\n=== STEP 4: Spatial Adjustment ===")

        # Prefer independent observations for spatial fit; fall back to clustered
        spatial_obs_csv = independent_observations_csv or clustered_observations_csv

        spatial_fit(
            input_raster=current_raster,
            observations_csv=spatial_obs_csv,
            model_values_csv=posterior_csv,
            output_dir=output_dir,
            model_type=model_type,
            n_proc=n_proc,
        )

        typer.echo(f"\n✓ Full pipeline for {model_type} completed successfully")
        typer.echo(f"  Final output available in: {output_dir}")

    except Exception as e:
        logger.exception("Error in full-pipeline")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@cli.from_docstring(app)
def combine(
    geology_tif: typing.Annotated[Path, typer.Option("--geology-tif", exists=True, dir_okay=False)],
    terrain_tif: typing.Annotated[Path, typer.Option("--terrain-tif", exists=True, dir_okay=False)],
    output_path: typing.Annotated[Path, typer.Option("--output-path", "-o", dir_okay=False)],
    combination_method: typing.Annotated[str | None, typer.Option("--combination-method")] = None,
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
    combination_method : str, optional
        Method for combining models. Either a ratio (float, e.g., '1.0' for equal
        weighting) or 'standard_deviation_weighting'. Default from config.
    """
    try:
        cfg = get_config()

        if combination_method is None:
            combination_method = cfg.combination_method

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
            )

            # Create output array correctly restoring nodata where things are NaN
            combined_data = np.stack([combined_vs30, combined_stdv])
            combined_data[np.isnan(combined_data)] = nodata

            profile.update({
                "dtype": "float32",
                "count": 2,
                "nodata": nodata,
                "compress": "deflate",
            })

            logger.info(f"Saving combined raster to: {output_path}")
            with rasterio.open(output_path, "w", **profile) as dst:
                dst.write(combined_data)
                dst.descriptions = (
                    "Vs30 (Combined Average)",
                    "Standard Deviation (Combined Average)",
                )

    except Exception as e:
        logger.exception("Error in combine")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@cli.from_docstring(app)
def full_pipeline(
    geology_categorical_csv: typing.Annotated[
        Path | None, typer.Option("--geology-csv", exists=True, dir_okay=False)
    ] = None,
    terrain_categorical_csv: typing.Annotated[
        Path | None, typer.Option("--terrain-csv", exists=True, dir_okay=False)
    ] = None,
    clustered_observations_csv: typing.Annotated[
        Path | None, typer.Option("--clustered-observations-csv", "-c", exists=True, dir_okay=False)
    ] = None,
    independent_observations_csv: typing.Annotated[
        Path | None, typer.Option("--independent-observations-csv", "-o", exists=True, dir_okay=False)
    ] = None,
    output_dir: typing.Annotated[Path | None, typer.Option("--output-dir", "-d", file_okay=False)] = None,
    nproc: typing.Annotated[int | None, typer.Option("--nproc")] = None,
    combination_method: typing.Annotated[str | None, typer.Option("--combination-method")] = None,
    n_proc: typing.Annotated[int | None, typer.Option("--n-proc")] = None,
) -> None:
    """
    Run the full VS30 generation pipeline for both geology and terrain models.

    Executes the complete pipeline for both geology and terrain models, then
    combines the results into a final averaged raster. This is the main entry
    point for generating VS30 maps.

    Parameters
    ----------
    geology_categorical_csv : Path, optional
        Path to geology categorical CSV. Default from config/resources.
    terrain_categorical_csv : Path, optional
        Path to terrain categorical CSV. Default from config/resources.
    clustered_observations_csv : Path, optional
        Path to CSV file with clustered observations (e.g., CPT data).
    independent_observations_csv : Path, optional
        Path to CSV file with independent observations (e.g., measured filtered).
    output_dir : Path, optional
        Directory to save all pipeline outputs. Default from config.
    nproc : int, optional
        Number of processes for clustering.
    combination_method : str, optional
        Method for combining models. Either a ratio (float) or
        'standard_deviation_weighting'.
    n_proc : int, optional
        Number of parallel processes for spatial adjustment. Use -1 for all cores.
        Default from config.
    """
    import time

    start_time = time.time()

    try:
        cfg = get_config()

        if output_dir is None:
            output_dir = Path(cfg.output_dir)
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        nproc = nproc if nproc is not None else cfg.n_proc
        combination_method = (
            combination_method
            if combination_method is not None
            else cfg.combination_method
        )

        from vs30 import parallel as parallel_module

        n_proc_resolved = parallel_module.resolve_n_proc(
            n_proc if n_proc is not None else cfg.n_proc
        )

        # Resolve CSV paths if not provided
        res_dir = Path(__file__).parent / "resources"
        if geology_categorical_csv is None:
            geology_categorical_csv = res_dir / constants.GEOLOGY_MEAN_AND_STANDARD_DEVIATION_PER_CATEGORY_FILE
        if terrain_categorical_csv is None:
            terrain_categorical_csv = res_dir / constants.TERRAIN_MEAN_AND_STANDARD_DEVIATION_PER_CATEGORY_FILE

        # 1. Run Geology Pipeline
        logger.info("\n" + "=" * 80 + "\nRUNNING GEOLOGY PIPELINE\n" + "=" * 80)
        full_pipeline_for_geology_or_terrain(
            model_type="geology",
            categorical_model_csv=geology_categorical_csv,
            clustered_observations_csv=clustered_observations_csv,
            independent_observations_csv=independent_observations_csv,
            output_dir=output_dir,
            nproc=nproc,
            n_proc=n_proc_resolved,
        )

        # 2. Run Terrain Pipeline
        logger.info("\n" + "=" * 80 + "\nRUNNING TERRAIN PIPELINE\n" + "=" * 80)
        full_pipeline_for_geology_or_terrain(
            model_type="terrain",
            categorical_model_csv=terrain_categorical_csv,
            clustered_observations_csv=clustered_observations_csv,
            independent_observations_csv=independent_observations_csv,
            output_dir=output_dir,
            nproc=nproc,
            n_proc=n_proc_resolved,
        )

        # 3. Average Results
        logger.info(
            "\n" + "=" * 80 + "\nCOMBINING GEOLOGY AND TERRAIN RESULTS\n" + "=" * 80
        )

        geol_tif = output_dir / constants.OUTPUT_FILENAMES["geology"]
        terr_tif = output_dir / constants.OUTPUT_FILENAMES["terrain"]
        combined_tif = output_dir / constants.COMBINED_VS30_FILENAME

        combine(
            geology_tif=geol_tif,
            terrain_tif=terr_tif,
            output_path=combined_tif,
            combination_method=combination_method,
        )

        elapsed_time = time.time() - start_time
        typer.echo(f"  Total execution time: {elapsed_time:.1f} seconds")
        typer.echo(f"  Combined output available at: {combined_tif}")

    except Exception as e:
        logger.exception("Error in full-pipeline")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@cli.from_docstring(app)
def compute_at_locations(
    locations_csv: typing.Annotated[
        Path | None, typer.Option("--locations-csv", "-l", exists=True, dir_okay=False)
    ] = None,
    output_csv: typing.Annotated[
        Path | None, typer.Option("--output-csv", "-o", dir_okay=False)
    ] = None,
    lon_column: typing.Annotated[str | None, typer.Option("--lon-column")] = None,
    lat_column: typing.Annotated[str | None, typer.Option("--lat-column")] = None,
    geology_categorical_csv: typing.Annotated[
        Path | None, typer.Option("--geology-csv", exists=True, dir_okay=False)
    ] = None,
    terrain_categorical_csv: typing.Annotated[
        Path | None, typer.Option("--terrain-csv", exists=True, dir_okay=False)
    ] = None,
    clustered_observations_csv: typing.Annotated[
        Path | None, typer.Option("--clustered-observations-csv", "-c", exists=True, dir_okay=False)
    ] = None,
    independent_observations_csv: typing.Annotated[
        Path | None, typer.Option("--independent-observations-csv", "-i", exists=True, dir_okay=False)
    ] = None,
    coast_distance_raster: typing.Annotated[
        Path | None, typer.Option("--coast-distance-raster", exists=True, dir_okay=False)
    ] = None,
    include_intermediate: typing.Annotated[
        bool, typer.Option("--include-intermediate/--final-only")
    ] = True,
    combination_method: typing.Annotated[
        str | None, typer.Option("--combination-method")
    ] = None,
    n_proc: typing.Annotated[int | None, typer.Option("--n-proc")] = None,
) -> None:
    """
    Compute Vs30 values at specific latitude/longitude locations.

    Runs the full Vs30 pipeline but only at the specified query points,
    without generating raster grids. This is efficient for querying
    Vs30 at a small number of locations.

    The input CSV must have columns for longitude and latitude (WGS84).
    Column names can be specified with --lon-column and --lat-column.

    Example:
        vs30 compute-at-locations \\
            --locations-csv sites.csv \\
            --output-csv results.csv

    Parameters
    ----------
    locations_csv : Path
        CSV file with latitude/longitude columns (WGS84).
    output_csv : Path
        Output CSV file path.
    lon_column : str
        Name of longitude column in input CSV.
    lat_column : str
        Name of latitude column in input CSV.
    geology_categorical_csv : Path, optional
        Path to geology categorical CSV (default from config/resources).
    terrain_categorical_csv : Path, optional
        Path to terrain categorical CSV (default from config/resources).
    clustered_observations_csv : Path, optional
        Path to CSV file with clustered observations (e.g., CPT).
    independent_observations_csv : Path, optional
        Path to CSV file with independent observations.
    coast_distance_raster : Path, optional
        Path to coastal distance raster (required for hybrid geology mods).
    include_intermediate : bool
        Include intermediate values (geology/terrain separately) in output.
    combination_method : str, optional
        Method for combining: ratio (float) or 'standard_deviation_weighting'.
    n_proc : int, optional
        Number of parallel processes (default from config, -1 for all cores).
    """
    from qcore import coordinates

    from vs30 import parallel as parallel_locations

    try:
        cfg = get_config()

        # Resolve parameters from CLI arguments or config defaults
        if locations_csv is None:
            if cfg.locations_csv is None:
                typer.echo(
                    "Error: Missing required locations_csv. "
                    "Set in config.yaml or provide --locations-csv.",
                    err=True,
                )
                raise typer.Exit(1)
            locations_csv = Path(cfg.locations_csv)
            if not locations_csv.exists():
                typer.echo(
                    f"Error: locations_csv from config not found: {locations_csv}",
                    err=True,
                )
                raise typer.Exit(1)

        if output_csv is None:
            if cfg.locations_output_csv is None:
                typer.echo(
                    "Error: Missing required locations_output_csv. "
                    "Set in config.yaml or provide --output-csv.",
                    err=True,
                )
                raise typer.Exit(1)
            output_csv = Path(cfg.locations_output_csv)

        if lon_column is None:
            lon_column = constants.LOCATIONS_LON_COLUMN
        if lat_column is None:
            lat_column = constants.LOCATIONS_LAT_COLUMN

        if combination_method is None:
            combination_method = cfg.combination_method

        typer.echo(f"Loading locations from {locations_csv}...")
        df = pd.read_csv(locations_csv)

        if lon_column not in df.columns:
            typer.echo(
                f"Error: Column '{lon_column}' not found in {locations_csv}", err=True
            )
            raise typer.Exit(1)
        if lat_column not in df.columns:
            typer.echo(
                f"Error: Column '{lat_column}' not found in {locations_csv}", err=True
            )
            raise typer.Exit(1)

        # Convert to NZTM
        nztm_coords = coordinates.wgs_depth_to_nztm(
            np.column_stack([df[lat_column].values, df[lon_column].values])
        )
        northing, easting = nztm_coords[:, 0], nztm_coords[:, 1]
        df["easting"] = easting
        df["northing"] = northing
        points = np.column_stack([easting, northing])
        typer.echo(f"Loaded {len(points)} locations")

        # Resolve CSV paths if not provided
        res_dir = Path(__file__).parent / "resources"
        if geology_categorical_csv is None:
            geology_categorical_csv = res_dir / constants.GEOLOGY_MEAN_AND_STANDARD_DEVIATION_PER_CATEGORY_FILE
        if terrain_categorical_csv is None:
            terrain_categorical_csv = res_dir / constants.TERRAIN_MEAN_AND_STANDARD_DEVIATION_PER_CATEGORY_FILE

        # Load observations for spatial adjustment
        clustered_observations_csv = resolve_observation_csv(
            clustered_observations_csv, cfg.clustered_observations_file, res_dir
        )
        independent_observations_csv = resolve_observation_csv(
            independent_observations_csv, cfg.independent_observations_file, res_dir
        )

        # Load and combine all available observation files
        observation_csvs = [
            csv for csv in [clustered_observations_csv, independent_observations_csv]
            if csv is not None and csv.exists()
        ]

        if observation_csvs:
            observations_df = pd.concat(
                [pd.read_csv(csv) for csv in observation_csvs],
                ignore_index=True,
            )
        else:
            observations_df = pd.DataFrame(
                columns=["easting", "northing", "vs30", "uncertainty"]
            )

        typer.echo(
            f"Loaded {len(observations_df)} observations for spatial adjustment"
        )

        # Load categorical models (skipinitialspace handles spaces after commas)
        geol_model_df = pd.read_csv(geology_categorical_csv, skipinitialspace=True)
        terr_model_df = pd.read_csv(terrain_categorical_csv, skipinitialspace=True)

        # Resolve n_proc from CLI or config
        n_proc_resolved = parallel_locations.resolve_n_proc(
            n_proc if n_proc is not None else cfg.n_proc
        )

        # ================================================================
        # Parallel Processing Path
        # ================================================================
        if n_proc_resolved > 1:
            typer.echo(f"\nProcessing with {n_proc_resolved} parallel workers...")

            # Re-read the original CSV (without NZTM conversion - workers will do it)
            locations_df_raw = pd.read_csv(locations_csv)

            loc_config = parallel_locations.LocationsChunkConfig(
                lon_column=lon_column,
                lat_column=lat_column,
                include_intermediate=include_intermediate,
                combination_method=combination_method,
                coast_distance_raster=coast_distance_raster,
            )

            df = parallel_locations.run_parallel_locations(
                locations_df=locations_df_raw,
                observations_df=observations_df,
                geol_model_df=geol_model_df,
                terr_model_df=terr_model_df,
                config=loc_config,
                n_proc=n_proc_resolved,
            )

            # Write output
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)
            typer.echo(f"\nResults written to {output_csv}")
            typer.echo(f"  Total locations: {len(df)}")
            return

        # ================================================================
        # Sequential Processing Path
        # ================================================================
        typer.echo("\nProcessing geology model...")
        if coast_distance_raster is None or not coast_distance_raster.exists():
            typer.echo(
                "  Warning: No coastal distance raster provided, skipping hybrid modifications"
            )

        (
            geol_ids,
            geol_vs30,
            geol_stdv,
            geol_vs30_hybrid,
            geol_stdv_hybrid,
            geol_mvn_vs30,
            geol_mvn_stdv,
        ) = parallel_locations.process_geology_at_points(
            points, geol_model_df, observations_df, coast_distance_raster
        )

        df["geology_id"] = geol_ids
        if include_intermediate:
            df["geology_vs30"] = geol_vs30
            df["geology_stdv"] = geol_stdv
            df["geology_vs30_hybrid"] = geol_vs30_hybrid
            df["geology_stdv_hybrid"] = geol_stdv_hybrid
        df["geology_mvn_vs30"] = geol_mvn_vs30
        df["geology_mvn_stdv"] = geol_mvn_stdv

        typer.echo("Processing terrain model...")
        (
            terr_ids,
            terr_vs30,
            terr_stdv,
            terr_mvn_vs30,
            terr_mvn_stdv,
        ) = parallel_locations.process_terrain_at_points(points, terr_model_df, observations_df)

        df["terrain_id"] = terr_ids
        if include_intermediate:
            df["terrain_vs30"] = terr_vs30
            df["terrain_stdv"] = terr_stdv
        df["terrain_mvn_vs30"] = terr_mvn_vs30
        df["terrain_mvn_stdv"] = terr_mvn_stdv

        typer.echo("Combining models...")
        combined_vs30, combined_stdv = utils.combine_vs30_models(
            geol_mvn_vs30,
            geol_mvn_stdv,
            terr_mvn_vs30,
            terr_mvn_stdv,
            combination_method,
        )

        df["vs30"] = combined_vs30
        df["stdv"] = combined_stdv

        # Write output
        output_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        typer.echo(f"\nResults written to {output_csv}")
        typer.echo(f"  Total locations: {len(df)}")

    except Exception as e:
        logger.exception("Error in compute-at-locations")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":  # pragma: no cover
    app()

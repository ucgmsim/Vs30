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

import importlib.resources
import logging
from pathlib import Path
import shutil
from typing import Annotated, Optional

import numpy as np
import pandas as pd
import rasterio
import typer
from matplotlib import pyplot as plt
from typer import Option

from vs30 import raster, spatial, utils
from vs30.config import Vs30Config

logger = logging.getLogger(__name__)

# Create Typer app for CLI
app = typer.Typer(
    name="vs30",
    help="VS30 map generation and categorical model updates",
    add_completion=False,
)

# =============================================================================
# Global Config State
# =============================================================================

_cli_config: Vs30Config | None = None


def get_config() -> Vs30Config:
    """
    Get the current CLI configuration.

    Returns the config set by the --config option, or the default
    package config if no custom config was specified.
    """
    global _cli_config
    if _cli_config is None:
        _cli_config = Vs30Config.default()
    return _cli_config


@app.callback()
def main(
    config: Annotated[
        Optional[Path],
        Option(
            "--config",
            "-c",
            help="Path to config.yaml file (default: package config.yaml)",
        ),
    ] = None,
    verbose: Annotated[
        bool,
        Option("--verbose", "-v", help="Enable verbose logging"),
    ] = False,
):
    """VS30 map generation and categorical model updates."""
    global _cli_config

    if verbose:
        logging.basicConfig(level=logging.DEBUG)

    # Load config from specified path or default
    if config is not None:
        if not config.exists():
            typer.echo(f"Error: Config file not found: {config}", err=True)
            raise typer.Exit(1)
        _cli_config = Vs30Config.from_yaml(config)
        logger.info(f"Loaded config from {config}")
    else:
        _cli_config = Vs30Config.default()
        logger.debug(f"Using default config from {Vs30Config.default_config_path()}")


@app.command()
def update_categorical_vs30_models(
    categorical_model_csv: Path = Option(
        ...,
        "--categorical-model-csv",
        "-m",
        help="Path to CSV file with categorical vs30 mean and standard deviation values (e.g., geology_model_prior_mean_and_standard_deviation.csv)",
    ),
    clustered_observations_csv: Path = Option(
        None,
        "--clustered-observations-csv",
        "-c",
        help="Path to CSV file with clustered observations (e.g., measured_vs30_cpt.csv). These will be processed with spatial clustering.",
    ),
    independent_observations_csv: Path = Option(
        None,
        "--independent-observations-csv",
        "-o",
        help="Path to CSV file with independent observations (e.g., measured_vs30_independent_observations.csv). These will be processed without clustering.",
    ),
    output_dir: Path = Option(
        ...,
        "--output-dir",
        "-d",
        help="Path to output directory (will be created if it does not exist)",
    ),
    model_type: str = Option(
        ...,
        "--model-type",
        "-t",
        help="Model type: either 'geology' or 'terrain'",
    ),
    n_prior: int = Option(
        None,
        "--n-prior",
        help="Effective number of prior observations (default from config)",
    ),
    min_sigma: float = Option(
        None,
        "--min-sigma",
        help="Minimum standard deviation allowed (default from config)",
    ),
    min_group: int = Option(
        None,
        "--min-group",
        help="Minimum group size for DBSCAN clustering (default from config)",
    ),
    eps: float = Option(
        None,
        "--eps",
        help="Maximum distance (metres) for points to be considered in same cluster (default from config)",
    ),
    nproc: int = Option(
        None,
        "--nproc",
        help="Number of processes for DBSCAN clustering. -1 to use all available cores (default from config)",
    ),
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
    - Clustered observations (e.g., CPT data) may have spatial sampling biases from
      geotechnical investigations; clustering corrects for over-weighting dense samples
    - Independent observations (e.g., direct Vs30 measurements) are typically higher-quality
      and more representative; they refine the bias-corrected model from clustered data
    """
    try:
        # Get config and resolve defaults
        cfg = get_config()
        n_prior = n_prior if n_prior is not None else cfg.n_prior
        min_sigma = min_sigma if min_sigma is not None else cfg.min_sigma
        min_group = min_group if min_group is not None else cfg.min_group
        eps = eps if eps is not None else cfg.eps
        nproc = nproc if nproc is not None else cfg.n_proc

        # Validate that at least one observations CSV is provided
        if clustered_observations_csv is None and independent_observations_csv is None:
            typer.echo(
                "Error: At least one of --clustered-observations-csv or --independent-observations-csv must be provided",
                err=True,
            )
            raise typer.Exit(1)

        # Validate input files exist
        if not categorical_model_csv.exists():
            typer.echo(
                f"Error: Categorical model CSV file not found: {categorical_model_csv}",
                err=True,
            )
            raise typer.Exit(1)

        if (
            clustered_observations_csv is not None
            and not clustered_observations_csv.exists()
        ):
            typer.echo(
                f"Error: Clustered observations CSV file not found: {clustered_observations_csv}",
                err=True,
            )
            raise typer.Exit(1)

        if (
            independent_observations_csv is not None
            and not independent_observations_csv.exists()
        ):
            typer.echo(
                f"Error: Independent observations CSV file not found: {independent_observations_csv}",
                err=True,
            )
            raise typer.Exit(1)

        # Validate model_type parameter
        if model_type not in ["geology", "terrain"]:
            typer.echo(
                f"Error: model_type must be 'geology' or 'terrain', got '{model_type}'",
                err=True,
            )
            raise typer.Exit(1)

        logger.info(f"Model type: {model_type}")
        logger.info(f"Loading categorical model from: {categorical_model_csv}")

        # Load categorical model CSV (absolute path)
        categorical_model_df = pd.read_csv(categorical_model_csv, skipinitialspace=True)

        # Drop rows with placeholder values for excluded categories (e.g., water)
        categorical_model_df = categorical_model_df[
            categorical_model_df["mean_vs30_km_per_s"] != cfg.nodata_value
        ]

        # Validate required columns
        required_cols = ["mean_vs30_km_per_s", "standard_deviation_vs30_km_per_s"]
        missing_cols = [
            col for col in required_cols if col not in categorical_model_df.columns
        ]
        if missing_cols:
            typer.echo(
                f"Error: CSV file missing required columns: {missing_cols}",
                err=True,
            )
            raise typer.Exit(1)

        # Import assignment functions and constants from category
        from vs30.category import (
            RASTER_ID_NODATA_VALUE,
            STANDARD_ID_COLUMN,
            _assign_to_category_geology,
            _assign_to_category_terrain,
            perform_clustering,
            posterior_from_bayesian_update,
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

            # Validate clustered observations have required columns
            obs_required_cols = ["easting", "northing", "vs30"]
            missing_obs_cols = [
                col
                for col in obs_required_cols
                if col not in clustered_observations_df.columns
            ]
            if missing_obs_cols:
                typer.echo(
                    f"Error: Clustered observations CSV missing required columns: {missing_obs_cols}",
                    err=True,
                )
                raise typer.Exit(1)

            logger.info(
                f"Loaded {len(clustered_observations_df)} clustered observations"
            )

            # Assign category IDs
            obs_locs = clustered_observations_df[["easting", "northing"]].values
            if model_type == "geology":
                model_ids = _assign_to_category_geology(obs_locs)
            else:  # terrain
                model_ids = _assign_to_category_terrain(obs_locs)

            clustered_observations_df[STANDARD_ID_COLUMN] = model_ids

            # Log assignment statistics
            unique_assigned_ids = clustered_observations_df[STANDARD_ID_COLUMN].unique()
            valid_assigned = clustered_observations_df[
                clustered_observations_df[STANDARD_ID_COLUMN] != RASTER_ID_NODATA_VALUE
            ]
            logger.info(
                f"Assigned category IDs: {len(valid_assigned)} valid observations "
                f"(out of {len(clustered_observations_df)} total)"
            )
            logger.info(
                f"Unique category IDs in observations: {sorted(unique_assigned_ids[unique_assigned_ids != RASTER_ID_NODATA_VALUE])[:20]}"
            )
            logger.info(
                f"Category IDs in prior model: {sorted(current_prior_df[STANDARD_ID_COLUMN].unique())}"
            )

            # Perform clustering
            logger.info("Performing spatial clustering...")
            clustered_observations_df = perform_clustering(
                clustered_observations_df, model_type, min_group, eps, nproc
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

            # Validate independent observations have required columns
            obs_required_cols = ["easting", "northing", "vs30", "uncertainty"]
            missing_obs_cols = [
                col
                for col in obs_required_cols
                if col not in independent_observations_df.columns
            ]
            if missing_obs_cols:
                typer.echo(
                    f"Error: Independent observations CSV missing required columns: {missing_obs_cols}",
                    err=True,
                )
                raise typer.Exit(1)

            logger.info(
                f"Loaded {len(independent_observations_df)} independent observations"
            )

            # Assign category IDs
            obs_locs = independent_observations_df[["easting", "northing"]].values
            if model_type == "geology":
                model_ids = _assign_to_category_geology(obs_locs)
            else:  # terrain
                model_ids = _assign_to_category_terrain(obs_locs)

            independent_observations_df[STANDARD_ID_COLUMN] = model_ids

        # Perform Bayesian update(s) via dispatcher
        logger.info("Applying Bayesian updates...")
        current_prior_df = posterior_from_bayesian_update(
            current_prior_df,
            independent_observations_df=independent_observations_df,
            clustered_observations_df=clustered_observations_df,
            n_prior=n_prior,
            min_sigma=min_sigma,
            model_type=model_type,
        )

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        output_filename = cfg.posterior_prefix + categorical_model_csv.name
        output_path = output_dir / output_filename

        current_prior_df.to_csv(output_path, index=False)

        typer.echo("✓ Successfully updated categorical model values")
        typer.echo(f"  Output saved to: {output_path}")

    except Exception as e:
        logger.exception("Error updating category values")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def make_initial_vs30_raster(
    terrain: bool = Option(False, "--terrain", help="Create terrain VS30 raster"),
    geology: bool = Option(False, "--geology", help="Create geology VS30 raster"),
    output_dir: Path = Option(
        None,
        "--output-dir",
        "-o",
        help="Output directory (default: output_dir from config.yaml)",
    ),
    geology_csv: Path = Option(
        None, "--geology-csv", help="Custom geology model CSV (optional)"
    ),
    terrain_csv: Path = Option(
        None, "--terrain-csv", help="Custom terrain model CSV (optional)"
    ),
) -> None:
    """
    Create initial VS30 mean and standard deviation rasters from category IDs.

    This command generates initial VS30 rasters by:
    1. Creating category ID rasters (from terrain raster or geology shapefile)
    2. Mapping category IDs to VS30 mean and standard deviation values from CSV files
    3. Writing 2-band GeoTIFFs with VS30 mean (band 1) and standard deviation (band 2)

    Output files are saved as terrain_initial_vs30_with_uncertainty.tif and/or geology_initial_vs30_with_uncertainty.tif.
    """
    try:
        # Validate that at least one model type is specified
        if not terrain and not geology:
            typer.echo(
                "Error: At least one of --terrain or --geology must be specified",
                err=True,
            )
            raise typer.Exit(1)

        # Get config
        cfg = get_config()

        # Determine output directory
        if output_dir is None:
            output_dir = Path(cfg.output_dir)

        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Load grid parameters from config
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
            csv_path = terrain_csv if terrain_csv else cfg.TERRAIN_MEAN_STDDEV_CSV
            logger.info(f"Using terrain model values from {csv_path}")

            logger.info("Creating terrain category ID raster...")
            id_raster = raster.create_category_id_raster(
                "terrain", output_dir, **grid_params
            )

            logger.info("Creating terrain VS30 raster...")
            vs30_raster = output_dir / cfg.terrain_initial_vs30_filename
            raster.create_vs30_raster_from_ids(id_raster, csv_path, vs30_raster)
            typer.echo(f"✓ Created terrain VS30 raster: {vs30_raster}")

        # Process geology if requested
        if geology:
            logger.info("Processing geology model...")
            csv_path = geology_csv if geology_csv else cfg.GEOLOGY_MEAN_STDDEV_CSV
            logger.info(f"Using geology model values from {csv_path}")

            logger.info("Creating geology category ID raster...")
            id_raster = raster.create_category_id_raster(
                "geology", output_dir, **grid_params
            )

            logger.info("Creating geology VS30 raster...")
            vs30_raster = output_dir / cfg.geology_initial_vs30_filename
            raster.create_vs30_raster_from_ids(id_raster, csv_path, vs30_raster)
            typer.echo(f"✓ Created geology VS30 raster: {vs30_raster}")

        typer.echo("✓ Successfully created initial VS30 rasters")

    except Exception as e:
        logger.exception("Error creating initial VS30 rasters")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def adjust_geology_vs30_by_slope_and_coastal_distance(
    input_raster: Path = Option(
        ...,
        "--input-raster",
        "-i",
        help="Path to initial geology VS30 raster (created by make-initial-vs30-raster)",
    ),
    id_raster: Path = Option(
        ...,
        "--id-raster",
        help="Path to category ID raster (e.g. gid.tif used to create the input raster)",
    ),
    output_dir: Path = Option(
        ...,
        "--output-dir",
        "-o",
        help="Directory to save output hybrid raster and intermediate files",
    ),
) -> None:
    """
    Apply hybrid geology modifications to an initial VS30 raster.

    This command adds slope-based and coast-distance-based modifications to the geology model.
    It generates intermediate slope and coast distance rasters in the output directory.

    Requires:
    - Input geology VS30 raster (2 bands: Vs30, Stdv)
    - Corresponding ID raster (gid.tif)
    """
    try:
        if not input_raster.exists():
            raise FileNotFoundError(f"Input raster not found: {input_raster}")
        if not id_raster.exists():
            raise FileNotFoundError(f"ID raster not found: {id_raster}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Get config
        cfg = get_config()

        logger.info(
            f"Processing slope and coastal distance adjusted model for: {input_raster}"
        )

        # 1. Load Input Raster
        with rasterio.open(input_raster) as src:
            vs30_array = src.read(1)
            stdv_array = src.read(2)
            profile = src.profile.copy()
            # Ensure profile uses float64 for output calculations
            # (though input might be float32 or float64)

            # Check dimensions against ID raster
            with rasterio.open(id_raster) as id_src:
                if id_src.width != src.width or id_src.height != src.height:
                    raise ValueError(
                        f"Dimension mismatch! Input raster: {src.width}x{src.height}, "
                        f"ID raster: {id_src.width}x{id_src.height}"
                    )
                id_array = id_src.read(1)

        # 2. Create/Load Intermediate Rasters
        slope_path = output_dir / cfg.slope_raster_filename
        logger.info(f"Generating slope raster: {slope_path}")
        slope_array, _ = raster.create_slope_raster(slope_path, profile)

        coast_path = output_dir / cfg.coast_distance_raster_filename
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
            mod6=True,  # Enable all mods by default for hybrid model
            mod13=True,
            hybrid=True,
            # Pass parameters from config
            hybrid_mod6_dist_min=cfg.hybrid_mod6_dist_min,
            hybrid_mod6_dist_max=cfg.hybrid_mod6_dist_max,
            hybrid_mod6_vs30_min=cfg.hybrid_mod6_vs30_min,
            hybrid_mod6_vs30_max=cfg.hybrid_mod6_vs30_max,
            hybrid_mod13_dist_min=cfg.hybrid_mod13_dist_min,
            hybrid_mod13_dist_max=cfg.hybrid_mod13_dist_max,
            hybrid_mod13_vs30_min=cfg.hybrid_mod13_vs30_min,
            hybrid_mod13_vs30_max=cfg.hybrid_mod13_vs30_max,
        )

        # 4. Save Output
        output_path = (
            output_dir
            / cfg.geology_vs30_slope_and_coastal_distance_adjusted_filename
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


@app.command()
def spatial_fit(
    input_raster: Path = Option(
        ...,
        "--input-raster",
        "-i",
        help="Path to input 2-band VS30 raster (Vs30 mean and stdv)",
    ),
    observations_csv: Path = Option(
        ...,
        "--observations-csv",
        "-o",
        help="Path to CSV file with measured VS30 values (easting, northing, vs30, uncertainty)",
    ),
    model_values_csv: Path = Option(
        ...,
        "--model-values-csv",
        "-m",
        help="Path to CSV file with updated categorical vs30 values (e.g., updated_geology_model.csv)",
    ),
    output_dir: Path = Option(
        ...,
        "--output-dir",
        "-d",
        help="Directory to save the adjusted raster",
    ),
    model_type: str = Option(
        ...,
        "--model-type",
        "-t",
        help="Model type: either 'geology' or 'terrain'",
    ),
    n_proc: int = Option(
        None,
        "--n-proc",
        help="Number of parallel processes (default from config, -1 for all cores)",
    ),
) -> None:
    """
    Adjust a VS30 raster based on measurements using spatial conditioning.

    This command performs a spatial adjustment of an input raster by:
    1. Loading the 2-band input raster (VS30 mean and stdv)
    2. Loading measurements and mapping them to categories
    3. Computing spatial fits to update pixels affected by measurements
    4. Applying updates and saving the resulting 2-band GeoTIFF
    """
    try:
        if not input_raster.exists():
            raise FileNotFoundError(f"Input raster not found: {input_raster}")
        if not observations_csv.exists():
            raise FileNotFoundError(f"Observations CSV not found: {observations_csv}")
        if not model_values_csv.exists():
            raise FileNotFoundError(f"Model values CSV not found: {model_values_csv}")

        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Get config
        cfg = get_config()

        # Get parameters from config
        max_dist_m = cfg.max_dist_m
        max_points = cfg.max_points
        phi = cfg.phi[model_type]
        noisy = cfg.noisy
        cov_reduc = cfg.cov_reduc

        # Resolve n_proc from CLI or config
        from vs30.parallel import resolve_n_proc, run_parallel_spatial_fit

        n_proc_resolved = resolve_n_proc(
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
            clustered_obs_file
            and clustered_obs_file.lower() != "none"
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
        logger.info("Preparing observation data for spatial adjustment...")

        if model_type == "geology":
            # Geology models use hybrid parameters from config
            obs_data = spatial.prepare_observation_data(
                observations,
                raster_data,
                updated_model_table,
                model_type,
                output_dir,
                noisy=noisy,
                hybrid_mod6_dist_min=cfg.hybrid_mod6_dist_min,
                hybrid_mod6_dist_max=cfg.hybrid_mod6_dist_max,
                hybrid_mod6_vs30_min=cfg.hybrid_mod6_vs30_min,
                hybrid_mod6_vs30_max=cfg.hybrid_mod6_vs30_max,
                hybrid_mod13_dist_min=cfg.hybrid_mod13_dist_min,
                hybrid_mod13_dist_max=cfg.hybrid_mod13_dist_max,
                hybrid_mod13_vs30_min=cfg.hybrid_mod13_vs30_min,
                hybrid_mod13_vs30_max=cfg.hybrid_mod13_vs30_max,
            )
        else:
            # Terrain models don't use hybrid parameters
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
            output_filename = cfg.output_filenames[model_type]
            output_path = output_dir / output_filename
            shutil.copyfile(input_raster, output_path)
            typer.echo(f"✓ No updates needed. Copied to {output_path}")
            return

        # 5. Find Affected Pixels
        # For clustered observations, subsample within each cluster before finding affected pixels
        obs_data_for_bbox = obs_data
        if is_clustered_obs:
            from sklearn.cluster import DBSCAN

            logger.info(
                "Clustering observations for optimized affected pixel search..."
            )
            min_group = cfg.min_group
            eps = cfg.eps
            nproc_clustering = cfg.n_proc

            # Run DBSCAN directly on the filtered observation locations
            dbscan = DBSCAN(eps=eps, min_samples=min_group, n_jobs=nproc_clustering)
            cluster_labels = dbscan.fit_predict(obs_data.locations)

            n_clusters = len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)
            n_noise = np.sum(cluster_labels == -1)
            logger.info(
                f"Found {n_clusters} clusters and {n_noise} noise points "
                f"in {len(obs_data.locations)} observations"
            )

            # Subsample observations within each cluster
            obs_subsample_step = cfg.obs_subsample_step_for_clustered
            subsample_indices = spatial.subsample_by_cluster(
                cluster_labels, step=obs_subsample_step
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
            updates = run_parallel_spatial_fit(
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


@app.command()
def plot_posterior_values(
    csv_path: Path = Option(
        ...,
        "--csv-path",
        "-c",
        help="Path to CSV file with prior and posterior values",
    ),
    output_dir: Path = Option(
        ...,
        "--output-dir",
        "-o",
        help="Path to output directory for plots (will be created if it does not exist)",
    ),
) -> None:
    """
    Plot prior and posterior vs30 mean values with error bars.

    Creates a plot showing vs30 mean values (y-axis) vs category ID (x-axis)
    for both prior and posterior values, with error bars showing standard deviation.
    """
    try:
        # Validate input file exists
        if not csv_path.exists():
            typer.echo(f"Error: CSV file not found: {csv_path}", err=True)
            raise typer.Exit(1)

        logger.info(f"Loading data from: {csv_path}")

        # Load CSV
        df = pd.read_csv(csv_path, skipinitialspace=True)

        # Get config
        cfg = get_config()

        # Filter out rows with placeholder values for excluded categories (e.g., water)
        df = df[df["prior_mean_vs30_km_per_s"] != cfg.nodata_value].copy()

        # Validate required columns
        required_cols = [
            "id",
            "prior_mean_vs30_km_per_s",
            "prior_standard_deviation_vs30_km_per_s",
            "posterior_mean_vs30_km_per_s",
            "posterior_standard_deviation_vs30_km_per_s",
        ]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            typer.echo(
                f"Error: CSV file missing required columns: {missing_cols}",
                err=True,
            )
            raise typer.Exit(1)

        # Create output directory if it doesn't exist
        output_dir.mkdir(parents=True, exist_ok=True)

        # Extract data
        category_ids = df["id"].values
        prior_mean = df["prior_mean_vs30_km_per_s"].values
        prior_std = df["prior_standard_deviation_vs30_km_per_s"].values
        posterior_mean = df["posterior_mean_vs30_km_per_s"].values
        posterior_std = df["posterior_standard_deviation_vs30_km_per_s"].values

        # Create figure
        fig, ax = plt.subplots(figsize=cfg.plot_figsize)

        # Offset for x positions to separate prior and posterior
        offset = 0.2
        prior_x = category_ids - offset
        posterior_x = category_ids + offset

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
        plt.tight_layout()
        plt.savefig(output_path, dpi=cfg.plot_dpi, bbox_inches="tight")
        plt.close()

        logger.info(f"Plot saved to: {output_path}")
        typer.echo(f"✓ Plot saved to: {output_path}")

    except Exception as e:
        logger.exception("Error creating plot")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def full_pipeline_for_geology_or_terrain(
    model_type: str = Option(
        ..., "--model-type", "-t", help="Model type: either 'geology' or 'terrain'"
    ),
    categorical_model_csv: Path = Option(
        ...,
        "--categorical-model-csv",
        "-m",
        help="Path to CSV file with categorical vs30 values",
    ),
    clustered_observations_csv: Path = Option(
        None,
        "--clustered-observations-csv",
        "-c",
        help="Path to CSV file with clustered observations (e.g., CPT)",
    ),
    independent_observations_csv: Path = Option(
        None,
        "--independent-observations-csv",
        "-o",
        help="Path to CSV file with independent observations (e.g., measured filtered)",
    ),
    output_dir: Path = Option(
        ..., "--output-dir", "-d", help="Directory to save all pipeline outputs"
    ),
    # Bayesian Update Parameters (None defaults)
    n_prior: int | None = Option(
        None, "--n-prior", help="Effective number of prior observations"
    ),
    min_sigma: float | None = Option(
        None, "--min-sigma", help="Minimum standard deviation allowed"
    ),
    min_group: int | None = Option(
        None, "--min-group", help="Minimum group size for DBSCAN"
    ),
    eps: float | None = Option(
        None, "--eps", help="Max distance for DBSCAN clustering"
    ),
    nproc: int | None = Option(
        None, "--nproc", help="Number of processes for clustering"
    ),
    n_proc: int | None = Option(
        None, "--n-proc", help="Number of parallel processes for spatial adjustment"
    ),
) -> None:
    """
    Run the full VS30 generation pipeline in sequence.

    1. update-categorical-vs30-models: (Conditional) Updates categorical priors with observations if do_bayesian_update is enabled.
    2. make-initial-vs30-raster: Creates initial 2-band VS30 raster using posteriors (or priors if updates skipped).
    3. create-hybrid-raster: (Geology only) Applies slope/coastal modifications.
    4. spatial-fit: Final spatial adjustment using observations.
    """
    try:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting full pipeline for {model_type}")
        logger.info(f"Output directory: {output_dir}")

        # Get config
        cfg = get_config()

        # Resolve parameters
        n_prior = n_prior if n_prior is not None else cfg.n_prior
        min_sigma = min_sigma if min_sigma is not None else cfg.min_sigma
        min_group = min_group if min_group is not None else cfg.min_group
        eps = eps if eps is not None else cfg.eps
        nproc = nproc if nproc is not None else cfg.n_proc

        # Resolve Bayesian update flag
        do_bayesian_update = cfg.do_bayesian_update_of_geology_and_terrain_categorical_vs30_values

        # Resolve observations from config if not provided
        if clustered_observations_csv is None:
            clustered_obs_file = cfg.clustered_observations_file
            if clustered_obs_file and clustered_obs_file.lower() != "none":
                # Find observation file relative to package resources
                resource_path = importlib.resources.files("vs30") / "resources"
                with importlib.resources.as_file(resource_path) as res_dir:
                    candidate = res_dir / clustered_obs_file
                    if candidate.exists():
                        clustered_observations_csv = candidate

        if independent_observations_csv is None:
            indep_obs_file = cfg.independent_observations_file
            if indep_obs_file and indep_obs_file.lower() != "none":
                # Find observation file relative to package resources
                resource_path = importlib.resources.files("vs30") / "resources"
                with importlib.resources.as_file(resource_path) as res_dir:
                    candidate = res_dir / indep_obs_file
                    if candidate.exists():
                        independent_observations_csv = candidate

        # --- Step 1: Update Categorical Models (conditional) ---
        if do_bayesian_update:
            logger.info("\n=== STEP 1: Updating Categorical Models ===")
            update_categorical_vs30_models(
                categorical_model_csv=categorical_model_csv,
                clustered_observations_csv=clustered_observations_csv,
                independent_observations_csv=independent_observations_csv,
                output_dir=output_dir,
                model_type=model_type,
                n_prior=n_prior,
                min_sigma=min_sigma,
                min_group=min_group,
                eps=eps,
                nproc=nproc,
            )

            posterior_csv = (
                output_dir / f"{cfg.posterior_prefix}{categorical_model_csv.name}"
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
            cfg.geology_initial_vs30_filename
            if model_type == "geology"
            else cfg.terrain_initial_vs30_filename
        )
        id_raster_name = (
            cfg.geology_id_filename
            if model_type == "geology"
            else cfg.terrain_id_filename
        )
        id_raster = output_dir / id_raster_name

        if not initial_raster.exists():
            raise FileNotFoundError(f"Step 2 failed to produce {initial_raster}")

        # --- Step 3: Hybrid Modification (Geology Only) ---
        current_raster = initial_raster
        # id_raster was already set above using the correct filename config

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
                / cfg.geology_vs30_slope_and_coastal_distance_adjusted_filename
            )
            if not current_raster.exists():
                raise FileNotFoundError(f"Step 3 failed to produce {current_raster}")

        # --- Step 4: Spatial Fit ---
        logger.info("\n=== STEP 4: Spatial Adjustment ===")

        spatial_fit(
            input_raster=current_raster,
            observations_csv=independent_observations_csv
            if independent_observations_csv
            else clustered_observations_csv,
            model_values_csv=posterior_csv,
            output_dir=output_dir,
            model_type=model_type,
            n_proc=n_proc,
        )

        typer.echo("\n✓ Full pipeline for {model_type} completed successfully")
        typer.echo(f"  Final output available in: {output_dir}")

    except Exception as e:
        logger.exception("Error in full-pipeline")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def combine(
    geology_tif: Path = Option(
        ..., "--geology-tif", help="Path to geology VS30 raster"
    ),
    terrain_tif: Path = Option(
        ..., "--terrain-tif", help="Path to terrain VS30 raster"
    ),
    output_path: Path = Option(
        ..., "--output-path", "-o", help="Path to combined raster"
    ),
    combination_method: str | None = Option(
        None,
        "--combination-method",
        help="Method for combining models: Ratio (float) or 'standard_deviation_weighting'",
    ),
) -> None:
    """
    Combine geology and terrain VS30 rasters using a weighted average.
    """
    try:
        if not geology_tif.exists():
            raise FileNotFoundError(f"Geology raster not found: {geology_tif}")
        if not terrain_tif.exists():
            raise FileNotFoundError(f"Terrain raster not found: {terrain_tif}")

        # Get config
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

            vs30_g, stdv_g = geol_data[0], geol_data[1]
            vs30_t, stdv_t = terr_data[0], terr_data[1]

            # Calculate weights based on combination method
            use_stdv_weight = False

            # Try parsing as float (ratio) first
            try:
                ratio = float(combination_method)
                # It's a ratio R = geology_weight / terrain_weight
                # w_g = R / (R + 1), w_t = 1 / (R + 1)
                total_w = ratio + 1.0
                w_g = ratio / total_w
                w_t = 1.0 / total_w
            except ValueError:
                # Not a float, check for specific string
                if str(combination_method).strip() == "standard_deviation_weighting":
                    use_stdv_weight = True
                    k_val = cfg.k_value
                else:
                    raise ValueError(
                        f"Unknown combination method: {combination_method}"
                    )

            if use_stdv_weight:
                # Weighting based on variance: m = (sigma^2)^-k
                m_g = (stdv_g**2) ** -k_val
                m_t = (stdv_t**2) ** -k_val
                w_g = m_g / (m_g + m_t)
                w_t = m_t / (m_g + m_t)

            # Combine models using log-space mixture (matches legacy logic)
            log_g = np.log(vs30_g)
            log_t = np.log(vs30_t)

            log_comb = log_g * w_g + log_t * w_t
            combined_vs30 = np.exp(log_comb)

            # Combined standard deviation formula for mixture of normals in log-space
            combined_stdv = np.sqrt(
                w_g * ((log_g - log_comb) ** 2 + stdv_g**2)
                + w_t * ((log_t - log_comb) ** 2 + stdv_t**2)
            )

            # Create output array correctly restoring nodata where things are NaN
            combined_data = np.stack([combined_vs30, combined_stdv])
            combined_data[np.isnan(combined_data)] = nodata

            # Update profile for output
            profile.update(
                {
                    "dtype": "float32",
                    "count": geol_data.shape[0],
                    "nodata": nodata,
                    "compress": "deflate",
                }
            )

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


@app.command()
def full_pipeline(
    geology_categorical_csv: Path = Option(
        None,
        "--geology-csv",
        help="Path to geology categorical CSV (default from config/resources)",
    ),
    terrain_categorical_csv: Path = Option(
        None,
        "--terrain-csv",
        help="Path to terrain categorical CSV (default from config/resources)",
    ),
    clustered_observations_csv: Path = Option(
        None,
        "--clustered-observations-csv",
        "-c",
        help="Path to CSV file with clustered observations (e.g., CPT)",
    ),
    independent_observations_csv: Path = Option(
        None,
        "--independent-observations-csv",
        "-o",
        help="Path to CSV file with independent observations (e.g., measured filtered)",
    ),
    output_dir: Path | None = Option(
        None, "--output-dir", "-d", help="Directory to save all pipeline outputs"
    ),
    # Bayesian Update Parameters
    n_prior: int | None = Option(
        None, "--n-prior", help="Effective number of prior observations"
    ),
    min_sigma: float | None = Option(
        None, "--min-sigma", help="Minimum standard deviation allowed"
    ),
    min_group: int | None = Option(
        None, "--min-group", help="Minimum group size for DBSCAN"
    ),
    eps: float | None = Option(
        None, "--eps", help="Max distance for DBSCAN clustering"
    ),
    nproc: int | None = Option(
        None, "--nproc", help="Number of processes for clustering"
    ),
    combination_method: str | None = Option(
        None,
        "--combination-method",
        help="Method for combining models: Ratio (float) or 'standard_deviation_weighting'",
    ),
    n_proc: int = Option(
        None,
        "--n-proc",
        help="Number of parallel processes for spatial adjustment (default from config, -1 for all cores)",
    ),
) -> None:
    """
    Run the full VS30 generation pipeline for both geology and terrain models,
    then average the results into a final combined raster.
    """
    import time

    start_time = time.time()

    try:
        # Get config
        cfg = get_config()

        # Resolve output_dir
        if output_dir is None:
            output_dir = Path(cfg.output_dir)

        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve parameters
        n_prior = n_prior if n_prior is not None else cfg.n_prior
        min_sigma = min_sigma if min_sigma is not None else cfg.min_sigma
        min_group = min_group if min_group is not None else cfg.min_group
        eps = eps if eps is not None else cfg.eps
        nproc = nproc if nproc is not None else cfg.n_proc
        combination_method = (
            combination_method
            if combination_method is not None
            else cfg.combination_method
        )
        # Resolve n_proc for MVN spatial adjustment
        from vs30.parallel import resolve_n_proc

        n_proc_resolved = resolve_n_proc(
            n_proc if n_proc is not None else cfg.n_proc
        )

        # Resolve CSV paths if not provided
        resource_path = importlib.resources.files("vs30") / "resources"

        with importlib.resources.as_file(resource_path) as res_dir:
            if geology_categorical_csv is None:
                geology_categorical_csv = res_dir / cfg.GEOLOGY_MEAN_STDDEV_CSV
            if terrain_categorical_csv is None:
                terrain_categorical_csv = res_dir / cfg.TERRAIN_MEAN_STDDEV_CSV

            # 1. Run Geology Pipeline
            logger.info("\n" + "=" * 80 + "\nRUNNING GEOLOGY PIPELINE\n" + "=" * 80)
            full_pipeline_for_geology_or_terrain(
                model_type="geology",
                categorical_model_csv=geology_categorical_csv,
                clustered_observations_csv=clustered_observations_csv,
                independent_observations_csv=independent_observations_csv,
                output_dir=output_dir,
                n_prior=n_prior,
                min_sigma=min_sigma,
                min_group=min_group,
                eps=eps,
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
                n_prior=n_prior,
                min_sigma=min_sigma,
                min_group=min_group,
                eps=eps,
                nproc=nproc,
                n_proc=n_proc_resolved,
            )

        # 3. Average Results
        logger.info(
            "\n" + "=" * 80 + "\nCOMBINING GEOLOGY AND TERRAIN RESULTS\n" + "=" * 80
        )

        geol_tif = output_dir / cfg.output_filenames["geology"]
        terr_tif = output_dir / cfg.output_filenames["terrain"]
        combined_tif = output_dir / cfg.combined_vs30_filename

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


@app.command()
def compute_at_locations(
    locations_csv: Path = Option(
        ...,
        "--locations-csv",
        "-l",
        help="CSV file with latitude/longitude columns (WGS84)",
    ),
    output_csv: Path = Option(
        ...,
        "--output-csv",
        "-o",
        help="Output CSV file path",
    ),
    lon_column: str = Option(
        "longitude",
        "--lon-column",
        help="Name of longitude column in input CSV",
    ),
    lat_column: str = Option(
        "latitude",
        "--lat-column",
        help="Name of latitude column in input CSV",
    ),
    geology_categorical_csv: Path = Option(
        None,
        "--geology-csv",
        help="Path to geology categorical CSV (default from config/resources)",
    ),
    terrain_categorical_csv: Path = Option(
        None,
        "--terrain-csv",
        help="Path to terrain categorical CSV (default from config/resources)",
    ),
    clustered_observations_csv: Path = Option(
        None,
        "--clustered-observations-csv",
        "-c",
        help="Path to CSV file with clustered observations (e.g., CPT)",
    ),
    independent_observations_csv: Path = Option(
        None,
        "--independent-observations-csv",
        "-i",
        help="Path to CSV file with independent observations",
    ),
    coast_distance_raster: Path = Option(
        None,
        "--coast-distance-raster",
        help="Path to coastal distance raster (required for hybrid geology mods)",
    ),
    include_intermediate: bool = Option(
        True,
        "--include-intermediate/--final-only",
        help="Include intermediate values (geology/terrain separately) in output",
    ),
    combination_method: str = Option(
        None,
        "--combination-method",
        help="Method for combining: ratio (float) or 'standard_deviation_weighting'",
    ),
    n_proc: int = Option(
        None,
        "--n-proc",
        help="Number of parallel processes (default from config, -1 for all cores)",
    ),
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
    """
    from qcore import coordinates

    from vs30.parallel import (
        LocationsChunkConfig,
        process_geology_at_points,
        process_terrain_at_points,
        resolve_n_proc,
        run_parallel_locations,
    )
    from vs30.utils import combine_models_at_points

    try:
        # Get config
        cfg = get_config()

        # Resolve combination method
        if combination_method is None:
            combination_method = cfg.combination_method

        # Load input locations
        typer.echo(f"Loading locations from {locations_csv}...")
        if not locations_csv.exists():
            typer.echo(f"Error: Locations file not found: {locations_csv}", err=True)
            raise typer.Exit(1)

        df = pd.read_csv(locations_csv)

        # Validate columns
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
        resource_path = importlib.resources.files("vs30") / "resources"

        with importlib.resources.as_file(resource_path) as res_dir:
            if geology_categorical_csv is None:
                geology_categorical_csv = res_dir / cfg.GEOLOGY_MEAN_STDDEV_CSV
            if terrain_categorical_csv is None:
                terrain_categorical_csv = res_dir / cfg.TERRAIN_MEAN_STDDEV_CSV

            # Load observations for spatial adjustment
            # Note: config paths already include the 'observations/' prefix
            if clustered_observations_csv is None:
                clustered_obs_path = res_dir / cfg.clustered_observations_file
                if clustered_obs_path.exists():
                    clustered_observations_csv = clustered_obs_path
            if independent_observations_csv is None:
                if cfg.independent_observations_file != "none":
                    independent_obs_path = (
                        res_dir / cfg.independent_observations_file
                    )
                    if independent_obs_path.exists():
                        independent_observations_csv = independent_obs_path

            # Combine observations
            obs_dfs = []
            if (
                clustered_observations_csv is not None
                and clustered_observations_csv.exists()
            ):
                obs_dfs.append(pd.read_csv(clustered_observations_csv))
            if (
                independent_observations_csv is not None
                and independent_observations_csv.exists()
            ):
                obs_dfs.append(pd.read_csv(independent_observations_csv))

            if obs_dfs:
                observations_df = pd.concat(obs_dfs, ignore_index=True)
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
            n_proc_resolved = resolve_n_proc(
                n_proc if n_proc is not None else cfg.n_proc
            )

            # ================================================================
            # Parallel Processing Path
            # ================================================================
            if n_proc_resolved > 1:
                typer.echo(f"\nProcessing with {n_proc_resolved} parallel workers...")

                # Re-read the original CSV (without NZTM conversion - workers will do it)
                locations_df_raw = pd.read_csv(locations_csv)

                config = LocationsChunkConfig(
                    lon_column=lon_column,
                    lat_column=lat_column,
                    include_intermediate=include_intermediate,
                    combination_method=combination_method,
                    coast_distance_raster=coast_distance_raster,
                    weight_epsilon_div_by_zero=cfg.weight_epsilon_div_by_zero,
                )

                df = run_parallel_locations(
                    locations_df=locations_df_raw,
                    observations_df=observations_df,
                    geol_model_df=geol_model_df,
                    terr_model_df=terr_model_df,
                    config=config,
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
            ) = process_geology_at_points(
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
            ) = process_terrain_at_points(points, terr_model_df, observations_df)

            df["terrain_id"] = terr_ids
            if include_intermediate:
                df["terrain_vs30"] = terr_vs30
                df["terrain_stdv"] = terr_stdv
            df["terrain_mvn_vs30"] = terr_mvn_vs30
            df["terrain_mvn_stdv"] = terr_mvn_stdv

            typer.echo("Combining models...")
            combined_vs30, combined_stdv = combine_models_at_points(
                geol_mvn_vs30,
                geol_mvn_stdv,
                terr_mvn_vs30,
                terr_mvn_stdv,
                combination_method,
                cfg.weight_epsilon_div_by_zero,
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

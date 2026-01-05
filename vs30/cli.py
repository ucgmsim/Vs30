"""
MVN updates for geology and terrain models.

This module implements multivariate normal (MVN) distribution updates for pixels
in geology.tif and terrain.tif that are within range of observations.
"""

import importlib.resources
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import typer
from matplotlib import pyplot as plt
from typer import Option

from vs30 import constants, raster, spatial, utils
from vs30.raster import (
    apply_hybrid_geology_modifications,
    create_coast_distance_raster,
    create_slope_raster,
)

logger = logging.getLogger(__name__)

# Create Typer app for CLI
app = typer.Typer(
    name="vs30",
    help="VS30 map generation and categorical model updates",
    add_completion=False,
)


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
        help="Path to CSV file with independent observations (e.g., measured_vs30_original_filtered.csv). These will be processed without clustering.",
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
        constants.N_PRIOR,
        "--n-prior",
        help=f"Effective number of prior observations (default: {constants.N_PRIOR})",
    ),
    min_sigma: float = Option(
        constants.MIN_SIGMA,
        "--min-sigma",
        help=f"Minimum standard deviation allowed (default: {constants.MIN_SIGMA})",
    ),
    min_group: int = Option(
        constants.MIN_GROUP,
        "--min-group",
        help=f"Minimum group size for DBSCAN clustering (default: {constants.MIN_GROUP})",
    ),
    eps: float = Option(
        constants.EPS,
        "--eps",
        help=f"Maximum distance (metres) for points to be considered in same cluster (default: {constants.EPS})",
    ),
    nproc: int = Option(
        constants.NPROC,
        "--nproc",
        help=f"Number of processes for DBSCAN clustering. -1 to use all available cores (default: {constants.NPROC})",
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
    """
    try:
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

        # Drop rows with invalid placeholder values (e.g., -9999 for water category)
        categorical_model_df = categorical_model_df[
            categorical_model_df["mean_vs30_km_per_s"] != -9999
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
            ID_NODATA,
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
                clustered_observations_df[STANDARD_ID_COLUMN] != ID_NODATA
            ]
            logger.info(
                f"Assigned category IDs: {len(valid_assigned)} valid observations "
                f"(out of {len(clustered_observations_df)} total)"
            )
            logger.info(
                f"Unique category IDs in observations: {sorted(unique_assigned_ids[unique_assigned_ids != ID_NODATA])[:20]}"
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

        output_filename = constants.POSTERIOR_PREFIX + categorical_model_csv.name
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
    config: Path = Option(
        None,
        "--config",
        "-c",
        help="Path to config.yaml file (default: vs30/config.yaml relative to workspace root)",
    ),
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

        # Load config
        config_path = utils._find_config_file(config)
        if not config_path.exists():
            typer.echo(f"Error: Config file not found: {config_path}", err=True)
            raise typer.Exit(1)

        # Use constants for default values
        base_path = utils._resolve_base_path(config_path)
        logger.info(f"Using base path: {base_path}")

        # Determine output directory
        if output_dir is None:
            output_dir = base_path / constants.OUTPUT_DIR_NAME
        else:
            output_dir = Path(output_dir)

        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Output directory: {output_dir}")

        # Load grid parameters from config if provided
        config_data = utils.load_config(config_path) if config_path else {}
        grid_params = {
            "xmin": config_data.get("grid_xmin", constants.GRID_XMIN),
            "xmax": config_data.get("grid_xmax", constants.GRID_XMAX),
            "ymin": config_data.get("grid_ymin", constants.GRID_YMIN),
            "ymax": config_data.get("grid_ymax", constants.GRID_YMAX),
            "dx": config_data.get("grid_dx", constants.GRID_DX),
            "dy": config_data.get("grid_dy", constants.GRID_DY),
        }
        logger.info(f"Using grid parameters: {grid_params}")

        # Process terrain if requested
        if terrain:
            logger.info("Processing terrain model...")
            csv_path = terrain_csv if terrain_csv else constants.TERRAIN_MEAN_STDDEV_CSV
            logger.info(f"Using terrain model values from {csv_path}")

            logger.info("Creating terrain category ID raster...")
            id_raster = raster.create_category_id_raster(
                "terrain", output_dir, **grid_params
            )

            logger.info("Creating terrain VS30 raster...")
            vs30_raster = output_dir / constants.TERRAIN_INITIAL_VS30_FILENAME
            raster.create_vs30_raster_from_ids(id_raster, csv_path, vs30_raster)
            typer.echo(f"✓ Created terrain VS30 raster: {vs30_raster}")

        # Process geology if requested
        if geology:
            logger.info("Processing geology model...")
            csv_path = geology_csv if geology_csv else constants.GEOLOGY_MEAN_STDDEV_CSV
            logger.info(f"Using geology model values from {csv_path}")

            logger.info("Creating geology category ID raster...")
            id_raster = raster.create_category_id_raster(
                "geology", output_dir, **grid_params
            )

            logger.info("Creating geology VS30 raster...")
            vs30_raster = output_dir / constants.GEOLOGY_INITIAL_VS30_FILENAME
            raster.create_vs30_raster_from_ids(id_raster, csv_path, vs30_raster)
            typer.echo(f"✓ Created geology VS30 raster: {vs30_raster}")

        typer.echo("✓ Successfully created initial VS30 rasters")

    except Exception as e:
        logger.exception("Error creating initial VS30 rasters")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def create_hybrid_raster(
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

        logger.info(f"Processing hybrid model for: {input_raster}")

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
        slope_path = output_dir / constants.SLOPE_RASTER_FILENAME
        logger.info(f"Generating slope raster: {slope_path}")
        slope_array, _ = create_slope_raster(slope_path, profile)

        coast_path = output_dir / constants.COAST_DISTANCE_RASTER_FILENAME
        logger.info(f"Generating coast distance raster: {coast_path}")
        coast_dist_array, _ = create_coast_distance_raster(coast_path, profile)

        # 3. Apply Modifications
        logger.info("Applying hybrid geology modifications...")
        # Note: Arrays are modified in-place or returned new.
        # The function signature was ported to return them.
        mod_vs30, mod_stdv = apply_hybrid_geology_modifications(
            vs30_array.astype(np.float64),  # Ensure standard precision
            stdv_array.astype(np.float64),
            id_array,
            slope_array,
            coast_dist_array,
            mod6=True,  # Enable all mods by default for hybrid model
            mod13=True,
            hybrid=True,
        )

        # 4. Save Output
        output_path = output_dir / constants.HYBRID_VS30_FILENAME

        # Update profile for output
        profile.update(
            {
                "dtype": "float64",
                "compress": "deflate",
                # Ensure nodata matches expected model nodata if needed,
                # though usually inherited from src is fine.
            }
        )

        logger.info(f"Saving hybrid raster to: {output_path}")
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(mod_vs30, 1)
            dst.write(mod_stdv, 2)
            dst.descriptions = ("Vs30 (Hybrid)", "Standard Deviation (Hybrid)")

        typer.echo("✓ Successfully created hybrid geology raster")
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
    max_dist_m: float = Option(
        constants.MAX_DIST_M, "--max-dist", help="Maximum distance for spatial fit"
    ),
    max_points: int = Option(
        constants.MAX_POINTS, "--max-points", help="Maximum number of points for MVN"
    ),
    phi: float = Option(
        None, "--phi", help="Correlation length (phi). Defaults based on model type."
    ),
    noisy: bool = Option(
        constants.NOISY, "--noisy/--not-noisy", help="Apply noise weighting"
    ),
    cov_reduc: float = Option(
        constants.COV_REDUC, "--cov-reduc", help="Covariance reduction factor"
    ),
) -> None:
    """
    Adjust a VS30 raster based on measurements using MVN spatial fitting.

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
        # We need the table as a numpy array [mean, stdv] for each category index (1-indexed in raster)
        # category.py functions usually handle this.
        model_df = pd.read_csv(model_values_csv, skipinitialspace=True)
        # Determine columns
        mean_col, std_col = raster._determine_vs30_columns(list(model_df.columns))

        # Build table indexed by category ID
        max_id = model_df["id"].max()
        updated_model_table = np.full((max_id, 2), np.nan)
        for _, row in model_df.iterrows():
            idx = int(row["id"]) - 1
            if 0 <= idx < max_id:
                updated_model_table[idx, 0] = row[mean_col]
                updated_model_table[idx, 1] = row[std_col]

        # 4. Prepare Observation Data for MVN
        logger.info("Preparing observation data for MVN processing...")
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
            import shutil

            shutil.copyfile(input_raster, output_path)
            typer.echo(f"✓ No updates needed. Copied to {output_path}")
            return

        # 5. Find Affected Pixels
        logger.info("Finding pixels affected by observations...")
        bbox_result = spatial.find_affected_pixels(
            raster_data, obs_data, max_dist_m=max_dist_m
        )
        logger.info(f"Found {bbox_result.n_affected_pixels:,} affected pixels")

        # 6. Compute MVN Updates
        logger.info("Computing spatial updates...")
        updates = spatial.compute_mvn_updates(
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

        # Filter out invalid rows (e.g., -9999 placeholder values)
        df = df[df["prior_mean_vs30_km_per_s"] != -9999].copy()

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
        fig, ax = plt.subplots(figsize=constants.PLOT_FIGSIZE)

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
        plt.savefig(output_path, dpi=constants.PLOT_DPI, bbox_inches="tight")
        plt.close()

        logger.info(f"Plot saved to: {output_path}")
        typer.echo(f"✓ Plot saved to: {output_path}")

    except Exception as e:
        logger.exception("Error creating plot")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def full_pipeline_given_model(
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
    config: Path = Option(None, "--config", help="Path to config.yaml file (optional)"),
    # Bayesian Update Parameters
    n_prior: int = Option(
        constants.N_PRIOR, "--n-prior", help="Effective number of prior observations"
    ),
    min_sigma: float = Option(
        constants.MIN_SIGMA, "--min-sigma", help="Minimum standard deviation allowed"
    ),
    max_dist_m: float = Option(
        constants.MAX_DIST_M, "--max-dist", help="Maximum distance for spatial fit"
    ),
    max_points: int = Option(
        constants.MAX_POINTS, "--max-points", help="Maximum number of points for MVN"
    ),
    phi: float = Option(
        None, "--phi", help="Correlation length (phi). Defaults based on model type."
    ),
    noisy: bool = Option(
        constants.NOISY, "--noisy/--not-noisy", help="Apply noise weighting"
    ),
    cov_reduc: float = Option(
        constants.COV_REDUC, "--cov-reduc", help="Covariance reduction factor"
    ),
    min_group: int = Option(
        constants.MIN_GROUP, "--min-group", help="Minimum group size for DBSCAN"
    ),
    eps: float = Option(
        constants.EPS, "--eps", help="Max distance for DBSCAN clustering"
    ),
    nproc: int = Option(
        constants.NPROC, "--nproc", help="Number of processes for clustering"
    ),
) -> None:
    """
    Run the full VS30 generation pipeline in sequence.

    1. update-categorical-vs30-models: Updates categorical pries with observations.
    2. make-initial-vs30-raster: Creates initial 2-band VS30 raster using posteriors.
    3. create-hybrid-raster: (Geology only) Applies slope/coastal modifications.
    4. spatial-fit: Final spatial adjustment using MVN and observations.
    """
    try:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Starting FULL PIPELINE for {model_type}")
        logger.info(f"Output directory: {output_dir}")

        # Load config to get potential overrides
        config_path = utils._find_config_file(config)
        config_data = utils.load_config(config_path) if config_path else {}

        # Resolve observations from config if not provided
        if clustered_observations_csv is None and independent_observations_csv is None:
            obs_file = config_data.get("observations_file", constants.OBSERVATIONS_FILE)
            if obs_file:
                # Find observation file relative to package resources
                resource_path = importlib.resources.files("vs30") / "resources"
                with importlib.resources.as_file(resource_path) as res_dir:
                    candidate = res_dir / obs_file
                    if candidate.exists():
                        independent_observations_csv = candidate
                    else:
                        logger.warning(
                            f"Observations file not found in resources: {obs_file}"
                        )

        # --- Step 1: Update Categorical Models ---
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
            output_dir / f"{constants.POSTERIOR_PREFIX}{categorical_model_csv.name}"
        )
        if not posterior_csv.exists():
            raise FileNotFoundError(f"Step 1 failed to produce {posterior_csv}")

        # --- Step 2: Make Initial Raster ---
        logger.info("\n=== STEP 2: Creating Initial Raster ===")
        make_initial_vs30_raster(
            terrain=(model_type == "terrain"),
            geology=(model_type == "geology"),
            config=config,
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

        # --- Step 3: Create Hybrid Raster (Geology Only) ---
        current_raster = initial_raster
        if model_type == "geology":
            logger.info("\n=== STEP 3: Creating Hybrid Geology Raster ===")
            create_hybrid_raster(
                input_raster=initial_raster,
                id_raster=id_raster,
                output_dir=output_dir,
            )
            current_raster = output_dir / constants.HYBRID_VS30_FILENAME
            if not current_raster.exists():
                raise FileNotFoundError(f"Step 3 failed to produce {current_raster}")

        # --- Step 4: Spatial Fit ---
        logger.info("\n=== STEP 4: Spatial Adjustment (MVN) ===")
        # Load config to get potential overrides if calling as part of pipeline
        config_path = utils._find_config_file(config)
        config_data = utils.load_config(config_path) if config_path else {}

        # Merge priorities: explicit arguments > config file > defaults
        f_max_dist = (
            max_dist_m
            if max_dist_m != constants.MAX_DIST_M
            else config_data.get("max_dist_m", constants.MAX_DIST_M)
        )
        f_max_points = (
            max_points
            if max_points != constants.MAX_POINTS
            else config_data.get("max_points", constants.MAX_POINTS)
        )
        f_noisy = (
            noisy
            if noisy != constants.NOISY
            else config_data.get("noisy", constants.NOISY)
        )
        f_cov_reduc = (
            cov_reduc
            if cov_reduc != constants.COV_REDUC
            else config_data.get("cov_reduc", constants.COV_REDUC)
        )
        f_phi = phi if phi is not None else config_data.get(f"phi_{model_type}")

        spatial_fit(
            input_raster=current_raster,
            observations_csv=independent_observations_csv
            if independent_observations_csv
            else clustered_observations_csv,
            model_values_csv=posterior_csv,
            output_dir=output_dir,
            model_type=model_type,
            max_dist_m=f_max_dist,
            max_points=f_max_points,
            phi=f_phi,
            noisy=f_noisy,
            cov_reduc=f_cov_reduc,
        )

        typer.echo("\n✓ FULL PIPELINE COMPLETED SUCCESSFULLY")
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
    geology_weight: float = Option(
        constants.GEOLOGY_WEIGHT,
        "--geology-weight",
        help="Weight for geology model in final average (ignored if --stdv-weight is used)",
    ),
    terrain_weight: float = Option(
        constants.TERRAIN_WEIGHT,
        "--terrain-weight",
        help="Weight for terrain model in final average (ignored if --stdv-weight is used)",
    ),
    use_stdv_weight: bool = Option(
        False, "--stdv-weight", help="Use standard deviation for model combination"
    ),
    k: float = Option(
        constants.K, "--k", help="k factor for stdv based weight combination"
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

        logger.info(f"Averaging {geology_tif} and {terrain_tif}")

        with rasterio.open(geology_tif) as src_g, rasterio.open(terrain_tif) as src_t:
            profile = src_g.profile.copy()
            geol_data = src_g.read().astype(np.float64)
            terr_data = src_t.read().astype(np.float64)

            # Use nodata from geology (should be same for terrain)
            nodata = src_g.nodata

            # Set nodata values to NaN for easier calculation
            geol_data[geol_data == nodata] = np.nan
            terr_data[terr_data == nodata] = np.nan

            vs30_g, stdv_g = geol_data[0], geol_data[1]
            vs30_t, stdv_t = terr_data[0], terr_data[1]

            # Calculate weights
            if use_stdv_weight:
                # Weighting based on variance: m = (sigma^2)^-k
                m_g = (stdv_g**2) ** -k
                m_t = (stdv_t**2) ** -k
                w_g = m_g / (m_g + m_t)
                w_t = m_t / (m_g + m_t)
            else:
                total_w = geology_weight + terrain_weight
                w_g = geology_weight / total_w
                w_t = terrain_weight / total_w

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
                    "dtype": "float64",
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
    output_dir: Path = Option(
        ..., "--output-dir", "-d", help="Directory to save all pipeline outputs"
    ),
    config: Path = Option(None, "--config", help="Path to config.yaml file (optional)"),
    # Bayesian Update Parameters
    n_prior: int = Option(
        constants.N_PRIOR, "--n-prior", help="Effective number of prior observations"
    ),
    min_sigma: float = Option(
        constants.MIN_SIGMA, "--min-sigma", help="Minimum standard deviation allowed"
    ),
    max_dist_m: float = Option(
        constants.MAX_DIST_M, "--max-dist", help="Maximum distance for spatial fit"
    ),
    max_points: int = Option(
        constants.MAX_POINTS, "--max-points", help="Maximum number of points for MVN"
    ),
    phi: float = Option(
        None, "--phi", help="Correlation length (phi). Defaults based on model type."
    ),
    noisy: bool = Option(
        constants.NOISY, "--noisy/--not-noisy", help="Apply noise weighting"
    ),
    cov_reduc: float = Option(
        constants.COV_REDUC, "--cov-reduc", help="Covariance reduction factor"
    ),
    min_group: int = Option(
        constants.MIN_GROUP, "--min-group", help="Minimum group size for DBSCAN"
    ),
    eps: float = Option(
        constants.EPS, "--eps", help="Max distance for DBSCAN clustering"
    ),
    nproc: int = Option(
        constants.NPROC, "--nproc", help="Number of processes for clustering"
    ),
    geology_weight: float = Option(
        constants.GEOLOGY_WEIGHT,
        "--geology-weight",
        help="Weight for geology model in final average",
    ),
    terrain_weight: float = Option(
        constants.TERRAIN_WEIGHT,
        "--terrain-weight",
        help="Weight for terrain model in final average",
    ),
    use_stdv_weight: bool = Option(
        False, "--stdv-weight", help="Use standard deviation for model combination"
    ),
    k: float = Option(
        constants.K, "--k", help="k factor for stdv based weight combination"
    ),
) -> None:
    """
    Run the full VS30 generation pipeline for both geology and terrain models,
    then average the results into a final combined raster.
    """
    try:
        output_dir = output_dir.resolve()
        output_dir.mkdir(parents=True, exist_ok=True)

        # Resolve CSV paths if not provided
        resource_path = importlib.resources.files("vs30") / "resources"

        with importlib.resources.as_file(resource_path) as res_dir:
            if geology_categorical_csv is None:
                geology_categorical_csv = res_dir / constants.GEOLOGY_MEAN_STDDEV_CSV
            if terrain_categorical_csv is None:
                terrain_categorical_csv = res_dir / constants.TERRAIN_MEAN_STDDEV_CSV

            # 1. Run Geology Pipeline
            logger.info("\n" + "=" * 80 + "\nRUNNING GEOLOGY PIPELINE\n" + "=" * 80)
            full_pipeline_given_model(
                model_type="geology",
                categorical_model_csv=geology_categorical_csv,
                clustered_observations_csv=clustered_observations_csv,
                independent_observations_csv=independent_observations_csv,
                output_dir=output_dir,
                config=config,
                n_prior=n_prior,
                min_sigma=min_sigma,
                max_dist_m=max_dist_m,
                max_points=max_points,
                phi=phi,
                noisy=noisy,
                cov_reduc=cov_reduc,
                min_group=min_group,
                eps=eps,
                nproc=nproc,
            )

            # 2. Run Terrain Pipeline
            logger.info("\n" + "=" * 80 + "\nRUNNING TERRAIN PIPELINE\n" + "=" * 80)
            full_pipeline_given_model(
                model_type="terrain",
                categorical_model_csv=terrain_categorical_csv,
                clustered_observations_csv=clustered_observations_csv,
                independent_observations_csv=independent_observations_csv,
                output_dir=output_dir,
                config=config,
                n_prior=n_prior,
                min_sigma=min_sigma,
                max_dist_m=max_dist_m,
                max_points=max_points,
                phi=phi,
                noisy=noisy,
                cov_reduc=cov_reduc,
                min_group=min_group,
                eps=eps,
                nproc=nproc,
            )

        # 3. Average Results
        logger.info(
            "\n" + "=" * 80 + "\nAVERAGING GEOLOGY AND TERRAIN RESULTS\n" + "=" * 80
        )

        geol_tif = output_dir / constants.OUTPUT_FILENAMES["geology"]
        terr_tif = output_dir / constants.OUTPUT_FILENAMES["terrain"]
        combined_tif = output_dir / constants.COMBINED_VS30_FILENAME

        combine(
            geology_tif=geol_tif,
            terrain_tif=terr_tif,
            output_path=combined_tif,
            geology_weight=geology_weight,
            terrain_weight=terrain_weight,
            use_stdv_weight=use_stdv_weight,
            k=k,
        )

        typer.echo("\n✓ FULL MULTI-MODEL PIPELINE COMPLETED SUCCESSFULLY")
        typer.echo(f"  Combined output available at: {combined_tif}")

    except Exception as e:
        logger.exception("Error in full-pipeline")
        typer.echo(f"Error: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()

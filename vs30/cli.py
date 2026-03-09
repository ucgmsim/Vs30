"""Command-line interface for the vs30 package."""

import logging
import typing
from pathlib import Path

import typer
import yaml
from qcore import cli

from vs30 import constants, pipeline
from vs30 import config as config_module

logger = logging.getLogger(__name__)

# Create Typer app for CLI
app = typer.Typer(name="vs30", help="VS30 map generation and categorical model updates")


@cli.from_docstring(app)
def update_priors(
    categorical_model_csv: typing.Annotated[
        Path, typer.Argument(exists=True, dir_okay=False)
    ],
    output_dir: typing.Annotated[Path, typer.Argument(file_okay=False)],
    model_type: typing.Annotated[constants.ModelType, typer.Option()],
    clustered_observations_csv: typing.Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False),
    ] = None,
    independent_observations_csv: typing.Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False),
    ] = None,
    n_proc: typing.Annotated[int, typer.Option()] = 1,
) -> None:
    """
    Update categorical model values using Bayesian updates and save to CSV files.

    Parameters
    ----------
    categorical_model_csv : Path
        Path to CSV file with categorical Vs30 mean and standard deviation values.
    output_dir : Path
        Path to output directory. Will be created if it does not exist.
    model_type : ModelType
        Model type: either 'geology' or 'terrain'.
    clustered_observations_csv : Path, optional
        Path to CSV file with clustered observations (e.g., measured_vs30_cpt.csv).
    independent_observations_csv : Path, optional
        Path to CSV file with independent observations.
    n_proc : int, optional
        Number of processes for DBSCAN clustering. Use -1 for all available cores.
    """
    pipeline.update_categorical_vs30_models(
        categorical_model_csv=categorical_model_csv,
        output_dir=output_dir,
        model_type=model_type,
        clustered_observations_csv=clustered_observations_csv,
        independent_observations_csv=independent_observations_csv,
        n_proc=n_proc,
    )


@cli.from_docstring(app)
def grid(
    output_dir: typing.Annotated[Path, typer.Argument(file_okay=False)],
    config: typing.Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = constants.MODEL_VERSION_TO_CONFIG[constants.FixedModelVersion.FOSTER_2019],
    geology_categorical_csv: typing.Annotated[
        Path | None, typer.Option("--geology-csv", exists=True, dir_okay=False)
    ] = None,
    terrain_categorical_csv: typing.Annotated[
        Path | None, typer.Option("--terrain-csv", exists=True, dir_okay=False)
    ] = None,
    clustered_observations_csv: typing.Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False),
    ] = None,
    independent_observations_csv: typing.Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False),
    ] = None,
    model_type: typing.Annotated[
        constants.ModelType, typer.Option()
    ] = constants.ModelType.COMBINED,
    mvn: typing.Annotated[bool, typer.Option()] = True,
    combination_method: typing.Annotated[
        constants.CombinationMethod, typer.Option()
    ] = constants.CombinationMethod.STANDARD_DEVIATION_WEIGHTING,
    combine_ratio: typing.Annotated[float | None, typer.Option()] = None,
    n_proc: typing.Annotated[int, typer.Option()] = 1,
    noisy: typing.Annotated[bool, typer.Option()] = True,
    max_spatial_boolean_array_memory_gb: typing.Annotated[
        float, typer.Option()
    ] = 1.0,
) -> None:
    """
    Run the full VS30 generation pipeline on a raster grid.

    Parameters
    ----------
    output_dir : Path
        Directory to save all pipeline outputs.
    config : Path, optional
        Path to YAML config file with grid parameters and observation file paths.
    geology_categorical_csv : Path, optional
        Path to geology categorical CSV. Default from config/resources.
    terrain_categorical_csv : Path, optional
        Path to terrain categorical CSV. Default from config/resources.
    clustered_observations_csv : Path, optional
        Path to CSV file with clustered observations (e.g., CPT data).
    independent_observations_csv : Path, optional
        Path to CSV file with independent observations.
    model_type : ModelType, optional
        Which model(s) to run: geology, terrain, or combined (default).
    mvn : bool, optional
        Whether to perform MVN spatial adjustment.
    combination_method : CombinationMethod, optional
        Method for combining models.
    combine_ratio : float, optional
        Geology-to-terrain weight ratio. Required when combination_method is ratio.
    n_proc : int, optional
        Number of parallel processes. Use -1 for all cores.
    noisy : bool, optional
        Whether to apply noise weighting in spatial adjustment.
    max_spatial_boolean_array_memory_gb : float, optional
        Maximum memory for spatial boolean arrays.
    """
    with open(config, encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    grid_config = config_module.GridConfig.from_dict(config_data)

    # Resolve observation CSVs from config if not provided on CLI
    if clustered_observations_csv is None and config_data.get("clustered_observations_file") not in (None, "none"):
        candidate = constants.RESOURCE_PATH / config_data["clustered_observations_file"]
        if candidate.exists():
            clustered_observations_csv = candidate

    if independent_observations_csv is None and config_data.get("independent_observations_file") not in (None, "none"):
        candidate = constants.RESOURCE_PATH / config_data["independent_observations_file"]
        if candidate.exists():
            independent_observations_csv = candidate

    pipeline.compute_grid(
        grid_config=grid_config,
        output_dir=output_dir,
        model_type=model_type,
        combination_method=combination_method,
        combine_ratio=combine_ratio,
        geology_categorical_csv=geology_categorical_csv,
        terrain_categorical_csv=terrain_categorical_csv,
        clustered_observations_csv=clustered_observations_csv,
        independent_observations_csv=independent_observations_csv,
        do_bayesian_update=config_data.get(
            "do_bayesian_update_of_geology_and_terrain_categorical_vs30_values", True
        ),
        mvn=mvn,
        noisy=noisy,
        n_proc=n_proc,
        max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
        obs_subsample_step_for_clustered=config_data.get(
            "obs_subsample_step_for_clustered", 100
        ),
    )


@cli.from_docstring(app)
def grid_with_version(
    output_dir: typing.Annotated[Path, typer.Argument(file_okay=False)],
    version: typing.Annotated[
        constants.FixedModelVersion, typer.Argument()
    ],
    n_proc: typing.Annotated[int, typer.Option()] = 1,
    noisy: typing.Annotated[bool, typer.Option()] = True,
    max_spatial_boolean_array_memory_gb: typing.Annotated[
        float, typer.Option()
    ] = 1.0,
) -> None:
    """
    Run the VS30 grid pipeline using a fixed model version's config.

    Parameters
    ----------
    output_dir : Path
        Directory to save all pipeline outputs.
    version : FixedModelVersion
        Model version to use (e.g., foster_2019).
    n_proc : int, optional
        Number of parallel processes. Use -1 for all cores.
    noisy : bool, optional
        Whether to apply noise weighting in spatial adjustment.
    max_spatial_boolean_array_memory_gb : float, optional
        Maximum memory for spatial boolean arrays.
    """
    config_path = constants.MODEL_VERSION_TO_CONFIG[version]
    with open(config_path, encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    grid_config = config_module.GridConfig.from_dict(config_data)

    # Resolve observation CSVs from config
    clustered_observations_csv = None
    if config_data.get("clustered_observations_file") not in (None, "none"):
        candidate = constants.RESOURCE_PATH / config_data["clustered_observations_file"]
        if candidate.exists():
            clustered_observations_csv = candidate

    independent_observations_csv = None
    if config_data.get("independent_observations_file") not in (None, "none"):
        candidate = constants.RESOURCE_PATH / config_data["independent_observations_file"]
        if candidate.exists():
            independent_observations_csv = candidate

    combination_method = constants.CombinationMethod(
        config_data.get("combination_method", "standard_deviation_weighting")
    )

    pipeline.compute_grid(
        grid_config=grid_config,
        output_dir=output_dir,
        combination_method=combination_method,
        clustered_observations_csv=clustered_observations_csv,
        independent_observations_csv=independent_observations_csv,
        do_bayesian_update=config_data.get(
            "do_bayesian_update_of_geology_and_terrain_categorical_vs30_values", True
        ),
        noisy=noisy,
        n_proc=n_proc,
        max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
        obs_subsample_step_for_clustered=config_data.get(
            "obs_subsample_step_for_clustered", 100
        ),
    )


@cli.from_docstring(app)
def points(
    locations_csv: typing.Annotated[
        Path, typer.Argument(exists=True, dir_okay=False)
    ],
    output_csv: typing.Annotated[
        Path, typer.Argument(dir_okay=False)
    ],
    config: typing.Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False)
    ] = None,
    lon_column: typing.Annotated[
        str, typer.Option()
    ] = constants.LOCATIONS_LON_COLUMN,
    lat_column: typing.Annotated[
        str, typer.Option()
    ] = constants.LOCATIONS_LAT_COLUMN,
    geology_categorical_csv: typing.Annotated[
        Path | None, typer.Option("--geology-csv", exists=True, dir_okay=False)
    ] = None,
    terrain_categorical_csv: typing.Annotated[
        Path | None, typer.Option("--terrain-csv", exists=True, dir_okay=False)
    ] = None,
    clustered_observations_csv: typing.Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False),
    ] = None,
    independent_observations_csv: typing.Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False),
    ] = None,
    include_intermediate: typing.Annotated[
        bool, typer.Option("--include-intermediate/--final-only")
    ] = True,
    combination_method: typing.Annotated[
        constants.CombinationMethod, typer.Option()
    ] = constants.CombinationMethod.STANDARD_DEVIATION_WEIGHTING,
    combine_ratio: typing.Annotated[float | None, typer.Option()] = None,
    mvn: typing.Annotated[bool, typer.Option()] = True,
    noisy: typing.Annotated[bool, typer.Option()] = True,
    n_proc: typing.Annotated[int, typer.Option()] = 1,
) -> None:
    """
    Compute Vs30 values at specific latitude/longitude locations.

    Parameters
    ----------
    locations_csv : Path
        CSV file with latitude/longitude columns (WGS84).
    output_csv : Path
        Output CSV file path.
    config : Path, optional
        Path to YAML config file with observation file paths.
    lon_column : str, optional
        Name of longitude column in input CSV.
    lat_column : str, optional
        Name of latitude column in input CSV.
    geology_categorical_csv : Path, optional
        Path to geology categorical CSV (default from resources).
    terrain_categorical_csv : Path, optional
        Path to terrain categorical CSV (default from resources).
    clustered_observations_csv : Path, optional
        Path to CSV file with clustered observations (e.g., CPT).
    independent_observations_csv : Path, optional
        Path to CSV file with independent observations.
    include_intermediate : bool
        Include intermediate values (geology/terrain separately) in output.
    combination_method : CombinationMethod, optional
        Method for combining models.
    combine_ratio : float, optional
        Geology-to-terrain weight ratio. Required when combination_method is ratio.
    mvn : bool, optional
        Whether to perform MVN spatial adjustment.
    noisy : bool, optional
        Whether to apply noise weighting in spatial adjustment.
    n_proc : int, optional
        Number of parallel processes. Use -1 for all cores.
    """
    # Resolve observation CSVs from config if provided
    if config is not None:
        with open(config, encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        if clustered_observations_csv is None and config_data.get("clustered_observations_file") not in (None, "none"):
            candidate = constants.RESOURCE_PATH / config_data["clustered_observations_file"]
            if candidate.exists():
                clustered_observations_csv = candidate

        if independent_observations_csv is None and config_data.get("independent_observations_file") not in (None, "none"):
            candidate = constants.RESOURCE_PATH / config_data["independent_observations_file"]
            if candidate.exists():
                independent_observations_csv = candidate

    # Read CSV and extract coordinate columns
    import pandas as pd

    df = pd.read_csv(locations_csv)
    if lon_column not in df.columns:
        raise typer.BadParameter(f"Column '{lon_column}' not found in {locations_csv}")
    if lat_column not in df.columns:
        raise typer.BadParameter(f"Column '{lat_column}' not found in {locations_csv}")

    result_df = pipeline.compute_at_locations(
        longitudes=df[lon_column].values,
        latitudes=df[lat_column].values,
        combination_method=combination_method,
        combine_ratio=combine_ratio,
        geology_categorical_csv=geology_categorical_csv,
        terrain_categorical_csv=terrain_categorical_csv,
        clustered_observations_csv=clustered_observations_csv,
        independent_observations_csv=independent_observations_csv,
        include_intermediate=include_intermediate,
        mvn=mvn,
        noisy=noisy,
        n_proc=n_proc,
    )

    # Prepend original CSV columns (excluding easting/northing which are in result_df)
    original_cols = [c for c in df.columns if c not in result_df.columns]
    output_df = pd.concat([df[original_cols].reset_index(drop=True), result_df], axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    logger.info(f"Results written to {output_csv}")


@cli.from_docstring(app)
def point_with_version(
    locations_csv: typing.Annotated[
        Path, typer.Argument(exists=True, dir_okay=False)
    ],
    output_csv: typing.Annotated[
        Path, typer.Argument(dir_okay=False)
    ],
    version: typing.Annotated[
        constants.FixedModelVersion, typer.Argument()
    ],
    lon_column: typing.Annotated[
        str, typer.Option()
    ] = constants.LOCATIONS_LON_COLUMN,
    lat_column: typing.Annotated[
        str, typer.Option()
    ] = constants.LOCATIONS_LAT_COLUMN,
    include_intermediate: typing.Annotated[
        bool, typer.Option("--include-intermediate/--final-only")
    ] = True,
    noisy: typing.Annotated[bool, typer.Option()] = True,
    n_proc: typing.Annotated[int, typer.Option()] = 1,
) -> None:
    """
    Compute Vs30 at locations using a fixed model version's config.

    Parameters
    ----------
    locations_csv : Path
        CSV file with latitude/longitude columns (WGS84).
    output_csv : Path
        Output CSV file path.
    version : FixedModelVersion
        Model version to use (e.g., foster_2019).
    lon_column : str, optional
        Name of longitude column in input CSV.
    lat_column : str, optional
        Name of latitude column in input CSV.
    include_intermediate : bool
        Include intermediate values (geology/terrain separately) in output.
    noisy : bool, optional
        Whether to apply noise weighting in spatial adjustment.
    n_proc : int, optional
        Number of parallel processes. Use -1 for all cores.
    """
    config_path = constants.MODEL_VERSION_TO_CONFIG[version]
    with open(config_path, encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # Resolve observation CSVs from config
    clustered_observations_csv = None
    if config_data.get("clustered_observations_file") not in (None, "none"):
        candidate = constants.RESOURCE_PATH / config_data["clustered_observations_file"]
        if candidate.exists():
            clustered_observations_csv = candidate

    independent_observations_csv = None
    if config_data.get("independent_observations_file") not in (None, "none"):
        candidate = constants.RESOURCE_PATH / config_data["independent_observations_file"]
        if candidate.exists():
            independent_observations_csv = candidate

    combination_method = constants.CombinationMethod(
        config_data.get("combination_method", "standard_deviation_weighting")
    )

    # Read CSV and extract coordinate columns
    import pandas as pd

    df = pd.read_csv(locations_csv)
    if lon_column not in df.columns:
        raise typer.BadParameter(f"Column '{lon_column}' not found in {locations_csv}")
    if lat_column not in df.columns:
        raise typer.BadParameter(f"Column '{lat_column}' not found in {locations_csv}")

    result_df = pipeline.compute_at_locations(
        longitudes=df[lon_column].values,
        latitudes=df[lat_column].values,
        combination_method=combination_method,
        clustered_observations_csv=clustered_observations_csv,
        independent_observations_csv=independent_observations_csv,
        include_intermediate=include_intermediate,
        noisy=noisy,
        n_proc=n_proc,
    )

    # Prepend original CSV columns (excluding easting/northing which are in result_df)
    original_cols = [c for c in df.columns if c not in result_df.columns]
    output_df = pd.concat([df[original_cols].reset_index(drop=True), result_df], axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    logger.info(f"Results written to {output_csv}")


if __name__ == "__main__":  # pragma: no cover
    app()

"""Command-line interface for the vs30 package."""

import logging
import typing
from pathlib import Path

import pandas as pd
import typer
import yaml
from qcore import cli

from vs30 import constants, pipeline
from vs30 import config as config_module

logger = logging.getLogger(__name__)

# Create Typer app for CLI
app = typer.Typer(name="vs30", help="VS30 map generation and categorical model updates")

# CLI helper shared by `points` and `points_custom` to handle CSV I/O and column merging.
def run_points_pipeline(
    locations_csv: Path,
    output_csv: Path,
    lon_column: str,
    lat_column: str,
    include_intermediate: bool,
    noisy: bool,
    n_proc: int,
    combination_method: constants.CombinationMethod,
    combine_ratio: float | None = None,
    geology_categorical_csv: Path | None = None,
    terrain_categorical_csv: Path | None = None,
    clustered_observations_csv: Path | None = None,
    independent_observations_csv: Path | None = None,
    mvn: bool = True,
    do_bayesian_update: bool = False,
) -> None:
    """
    Shared implementation for points and points_custom commands.

    Reads the input CSV, validates columns, delegates computation to
    ``pipeline.compute_at_locations``, merges original columns with results,
    and writes the output CSV.

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
    include_intermediate : bool
        Include intermediate values (geology/terrain separately) in output.
    noisy : bool
        Whether to apply noise weighting in spatial adjustment.
    n_proc : int
        Number of parallel processes. Use -1 for all cores.
    combination_method : CombinationMethod
        Method for combining geology and terrain models.
    combine_ratio : float or None, optional
        Geology-to-terrain weight ratio (used when combination_method is ratio).
    geology_categorical_csv : Path or None, optional
        Path to geology categorical CSV.
    terrain_categorical_csv : Path or None, optional
        Path to terrain categorical CSV.
    clustered_observations_csv : Path or None, optional
        Path to CSV file with clustered observations (e.g., CPT).
    independent_observations_csv : Path or None, optional
        Path to CSV file with independent observations.
    mvn : bool, optional
        Whether to perform MVN spatial adjustment.
    do_bayesian_update : bool, optional
        Whether to perform Bayesian update of categorical Vs30 values.

    Raises
    ------
    typer.BadParameter
        If the specified longitude or latitude column is not found in the input CSV.
    """
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
        do_bayesian_update=do_bayesian_update,
    )

    original_cols = [c for c in df.columns if c not in result_df.columns]
    output_df = pd.concat([df[original_cols].reset_index(drop=True), result_df], axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    logger.info(f"Results written to {output_csv}")


@cli.from_docstring(app)
def points(
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
    n_proc: typing.Annotated[int, typer.Option()] = -1,
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
    n_proc : int, optional
        Number of parallel processes. Use -1 for all cores.
    """
    with open(constants.MODEL_VERSION_TO_CONFIG[version], encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # Resolve CSV paths relative to resources directory
    for key in constants.CSV_PATH_KEYS:
        if config_data[key]:
            config_data[key] = constants.RESOURCE_PATH / constants.RESOURCE_SUBDIRS[key] / config_data[key]

    run_points_pipeline(
        locations_csv=locations_csv,
        output_csv=output_csv,
        lon_column=lon_column,
        lat_column=lat_column,
        include_intermediate=include_intermediate,
        noisy=config_data["noisy"],
        n_proc=n_proc,
        combination_method=constants.CombinationMethod(config_data["combination_method"]),
        combine_ratio=config_data["combine_ratio"],
        geology_categorical_csv=config_data["geology_categorical_csv"],
        terrain_categorical_csv=config_data["terrain_categorical_csv"],
        clustered_observations_csv=config_data["clustered_observations_csv"],
        independent_observations_csv=config_data["independent_observations_csv"],
        do_bayesian_update=config_data["do_bayesian_update"],
    )


@cli.from_docstring(app)
def points_custom(
    locations_csv: typing.Annotated[
        Path, typer.Argument(exists=True, dir_okay=False)
    ],
    output_csv: typing.Annotated[
        Path, typer.Argument(dir_okay=False)
    ],
    geology_categorical_csv: typing.Annotated[
        Path, typer.Option("--geology-csv", exists=True, dir_okay=False)
    ] = ...,
    terrain_categorical_csv: typing.Annotated[
        Path, typer.Option("--terrain-csv", exists=True, dir_okay=False)
    ] = ...,
    combination_method: typing.Annotated[
        constants.CombinationMethod, typer.Option()
    ] = ...,
    combine_ratio: typing.Annotated[float, typer.Option()] = ...,
    noisy: typing.Annotated[bool, typer.Option("--noisy/--no-noisy")] = ...,
    mvn: typing.Annotated[bool, typer.Option("--mvn/--no-mvn")] = ...,
    do_bayesian_update: typing.Annotated[bool, typer.Option("--do-bayesian-update/--no-bayesian-update")] = ...,
    clustered_observations_csv: typing.Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False),
    ] = None,
    independent_observations_csv: typing.Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False),
    ] = None,
    lon_column: typing.Annotated[
        str, typer.Option()
    ] = constants.LOCATIONS_LON_COLUMN,
    lat_column: typing.Annotated[
        str, typer.Option()
    ] = constants.LOCATIONS_LAT_COLUMN,
    include_intermediate: typing.Annotated[
        bool, typer.Option("--include-intermediate/--final-only")
    ] = True,
    n_proc: typing.Annotated[int, typer.Option()] = -1,
) -> None:
    """
    Compute Vs30 values at specific latitude/longitude locations with explicit parameters.

    All scientific parameters must be explicitly provided. Use the simpler
    'points' command to run with a predefined model version instead.

    Parameters
    ----------
    locations_csv : Path
        CSV file with latitude/longitude columns (WGS84).
    output_csv : Path
        Output CSV file path.
    geology_categorical_csv : Path
        Path to geology categorical CSV.
    terrain_categorical_csv : Path
        Path to terrain categorical CSV.
    combination_method : CombinationMethod
        Method for combining models.
    combine_ratio : float
        Geology-to-terrain weight ratio (used when combination_method is ratio).
    noisy : bool
        Whether to apply noise weighting in spatial adjustment.
    mvn : bool
        Whether to perform MVN spatial adjustment.
    do_bayesian_update : bool
        Whether to perform Bayesian update of categorical Vs30 values.
    clustered_observations_csv : Path, optional
        Path to CSV file with clustered observations (e.g., CPT).
    independent_observations_csv : Path, optional
        Path to CSV file with independent observations.
    lon_column : str, optional
        Name of longitude column in input CSV.
    lat_column : str, optional
        Name of latitude column in input CSV.
    include_intermediate : bool, optional
        Include intermediate values (geology/terrain separately) in output.
    n_proc : int, optional
        Number of parallel processes. Use -1 for all cores.
    """
    run_points_pipeline(
        locations_csv=locations_csv,
        output_csv=output_csv,
        lon_column=lon_column,
        lat_column=lat_column,
        include_intermediate=include_intermediate,
        noisy=noisy,
        n_proc=n_proc,
        combination_method=combination_method,
        combine_ratio=combine_ratio,
        geology_categorical_csv=geology_categorical_csv,
        terrain_categorical_csv=terrain_categorical_csv,
        clustered_observations_csv=clustered_observations_csv,
        independent_observations_csv=independent_observations_csv,
        mvn=mvn,
        do_bayesian_update=do_bayesian_update,
    )

def grid(
    output_dir: typing.Annotated[Path, typer.Argument(file_okay=False)],
    version: typing.Annotated[
        constants.FixedModelVersion, typer.Argument()
    ],
    grid_xmin: typing.Annotated[int, typer.Option(help=f"Grid minimum X coordinate (NZTM, meters). Suggested for all of NZ: {constants.FULL_NZ_LAND_XMIN}.")] = ...,
    grid_xmax: typing.Annotated[int, typer.Option(help=f"Grid maximum X coordinate (NZTM, meters). Suggested for all of NZ: {constants.FULL_NZ_LAND_XMAX}.")] = ...,
    grid_ymin: typing.Annotated[int, typer.Option(help=f"Grid minimum Y coordinate (NZTM, meters). Suggested for all of NZ: {constants.FULL_NZ_LAND_YMIN}.")] = ...,
    grid_ymax: typing.Annotated[int, typer.Option(help=f"Grid maximum Y coordinate (NZTM, meters). Suggested for all of NZ: {constants.FULL_NZ_LAND_YMAX}.")] = ...,
    grid_dx: typing.Annotated[int, typer.Option(help=f"Grid X spacing (meters). Suggested: {constants.SUGGESTED_GRID_DX}.")] = ...,
    grid_dy: typing.Annotated[int, typer.Option(help=f"Grid Y spacing (meters). Suggested: {constants.SUGGESTED_GRID_DY}.")] = ...,
    n_proc: typing.Annotated[int, typer.Option()] = -1,
    max_spatial_boolean_array_memory_gb: typing.Annotated[
        float, typer.Option()
    ] = constants.MAX_SPATIAL_BOOLEAN_ARRAY_MEMORY_GB,
) -> None:
    """
    Run the VS30 grid pipeline using a fixed model version's config.

    Parameters
    ----------
    output_dir : Path
        Directory to save all pipeline outputs.
    version : FixedModelVersion
        Model version to use (e.g., foster_2019).
    grid_xmin : int
        Grid minimum X coordinate (NZTM, meters).
    grid_xmax : int
        Grid maximum X coordinate (NZTM, meters).
    grid_ymin : int
        Grid minimum Y coordinate (NZTM, meters).
    grid_ymax : int
        Grid maximum Y coordinate (NZTM, meters).
    grid_dx : int
        Grid X spacing (meters).
    grid_dy : int
        Grid Y spacing (meters).
    n_proc : int, optional
        Number of parallel processes. Use -1 for all cores.
    max_spatial_boolean_array_memory_gb : float, optional
        Maximum memory for spatial boolean arrays.
    """
    with open(constants.MODEL_VERSION_TO_CONFIG[version], encoding="utf-8") as f:
        config_data = yaml.safe_load(f)

    # Resolve CSV paths relative to resources directory
    for key in constants.CSV_PATH_KEYS:
        if config_data[key]:
            config_data[key] = constants.RESOURCE_PATH / constants.RESOURCE_SUBDIRS[key] / config_data[key]

    pipeline.compute_grid(
        grid_config=config_module.GridConfig(
            grid_xmin=grid_xmin, grid_xmax=grid_xmax,
            grid_ymin=grid_ymin, grid_ymax=grid_ymax,
            grid_dx=grid_dx, grid_dy=grid_dy,
        ),
        output_dir=output_dir,
        combination_method=constants.CombinationMethod(config_data["combination_method"]),
        combine_ratio=config_data["combine_ratio"],
        geology_categorical_csv=config_data["geology_categorical_csv"],
        terrain_categorical_csv=config_data["terrain_categorical_csv"],
        clustered_observations_csv=config_data["clustered_observations_csv"],
        independent_observations_csv=config_data["independent_observations_csv"],
        do_bayesian_update=config_data["do_bayesian_update"],
        noisy=config_data["noisy"],
        n_proc=n_proc,
        max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
    )

def grid_custom(
    output_dir: typing.Annotated[Path, typer.Argument(file_okay=False)],
    grid_xmin: typing.Annotated[int, typer.Option(help=f"Grid minimum X coordinate (NZTM, meters). Suggested for all of NZ: {constants.FULL_NZ_LAND_XMIN}.")] = ...,
    grid_xmax: typing.Annotated[int, typer.Option(help=f"Grid maximum X coordinate (NZTM, meters). Suggested for all of NZ: {constants.FULL_NZ_LAND_XMAX}.")] = ...,
    grid_ymin: typing.Annotated[int, typer.Option(help=f"Grid minimum Y coordinate (NZTM, meters). Suggested for all of NZ: {constants.FULL_NZ_LAND_YMIN}.")] = ...,
    grid_ymax: typing.Annotated[int, typer.Option(help=f"Grid maximum Y coordinate (NZTM, meters). Suggested for all of NZ: {constants.FULL_NZ_LAND_YMAX}.")] = ...,
    grid_dx: typing.Annotated[int, typer.Option(help=f"Grid X spacing (meters). Suggested: {constants.SUGGESTED_GRID_DX}.")] = ...,
    grid_dy: typing.Annotated[int, typer.Option(help=f"Grid Y spacing (meters). Suggested: {constants.SUGGESTED_GRID_DY}.")] = ...,
    geology_categorical_csv: typing.Annotated[
        Path, typer.Option("--geology-csv", exists=True, dir_okay=False)
    ] = ...,
    terrain_categorical_csv: typing.Annotated[
        Path, typer.Option("--terrain-csv", exists=True, dir_okay=False)
    ] = ...,
    combination_method: typing.Annotated[
        constants.CombinationMethod, typer.Option()
    ] = ...,
    combine_ratio: typing.Annotated[float, typer.Option(help="Geology-to-terrain weight ratio. Required when combination_method is ratio.")] = ...,
    do_bayesian_update: typing.Annotated[bool, typer.Option("--do-bayesian-update/--no-bayesian-update")] = ...,
    noisy: typing.Annotated[bool, typer.Option("--noisy/--no-noisy")] = ...,
    mvn: typing.Annotated[bool, typer.Option("--mvn/--no-mvn")] = ...,
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
    n_proc: typing.Annotated[int, typer.Option()] = -1,
    max_spatial_boolean_array_memory_gb: typing.Annotated[
        float, typer.Option()
    ] = constants.MAX_SPATIAL_BOOLEAN_ARRAY_MEMORY_GB,
) -> None:
    """
    Run the full VS30 generation pipeline on a raster grid with explicit parameters.

    All scientific and grid parameters must be explicitly provided. Use the
    simpler 'grid' command to run with a predefined model version instead.

    Parameters
    ----------
    output_dir : Path
        Directory to save all pipeline outputs.
    grid_xmin : int
        Grid minimum X coordinate (NZTM, meters).
    grid_xmax : int
        Grid maximum X coordinate (NZTM, meters).
    grid_ymin : int
        Grid minimum Y coordinate (NZTM, meters).
    grid_ymax : int
        Grid maximum Y coordinate (NZTM, meters).
    grid_dx : int
        Grid X spacing (meters).
    grid_dy : int
        Grid Y spacing (meters).
    geology_categorical_csv : Path
        Path to geology categorical CSV.
    terrain_categorical_csv : Path
        Path to terrain categorical CSV.
    combination_method : CombinationMethod
        Method for combining models.
    combine_ratio : float
        Geology-to-terrain weight ratio (used when combination_method is ratio).
    do_bayesian_update : bool
        Whether to perform Bayesian update of categorical Vs30 values.
    noisy : bool
        Whether to apply noise weighting in spatial adjustment.
    mvn : bool
        Whether to perform MVN spatial adjustment.
    clustered_observations_csv : Path, optional
        Path to CSV file with clustered observations (e.g., CPT data).
    independent_observations_csv : Path, optional
        Path to CSV file with independent observations.
    model_type : ModelType, optional
        Which model(s) to run: geology, terrain, or combined (default).
    n_proc : int, optional
        Number of parallel processes. Use -1 for all cores.
    max_spatial_boolean_array_memory_gb : float, optional
        Maximum memory for spatial boolean arrays.
    """
    pipeline.compute_grid(
        grid_config=config_module.GridConfig(
            grid_xmin=grid_xmin, grid_xmax=grid_xmax,
            grid_ymin=grid_ymin, grid_ymax=grid_ymax,
            grid_dx=grid_dx, grid_dy=grid_dy,
        ),
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
    )



if __name__ == "__main__":  # pragma: no cover
    app()

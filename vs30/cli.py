"""Command-line interface for the vs30 package."""

import logging
import typing
from pathlib import Path

import pandas as pd
import typer
from qcore import cli

from vs30 import config, constants, pipeline

logger = logging.getLogger(__name__)

app = typer.Typer(name="vs30", help="VS30 map generation and categorical model updates")

_MODEL_ARG_HELP = (
    "Either a bundled model version name ("
    f"{', '.join(v.value for v in constants.FixedModelVersion)}"
    ") or a path to a YAML config file with the same schema."
)


@cli.from_docstring(app)
def points(
    model: typing.Annotated[str, typer.Argument(help=_MODEL_ARG_HELP)],
    locations_csv: typing.Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_csv: typing.Annotated[Path, typer.Argument(dir_okay=False)],
    lon_column: typing.Annotated[str, typer.Option()] = constants.LOCATIONS_LON_COLUMN,
    lat_column: typing.Annotated[str, typer.Option()] = constants.LOCATIONS_LAT_COLUMN,
    include_intermediate: typing.Annotated[
        bool, typer.Option("--include-intermediate/--final-only")
    ] = False,
    dbscan_nproc: typing.Annotated[int, typer.Option()] = -1,
) -> None:
    """
    Compute Vs30 at locations using a bundled or user-supplied model config.

    Parameters
    ----------
    model : str
        Either a bundled ``FixedModelVersion`` enum value or a path to a
        YAML config file with the same schema.
    locations_csv : Path
        CSV file with latitude/longitude columns (WGS84).
    output_csv : Path
        Output CSV file path.
    lon_column : str, optional
        Name of longitude column in input CSV.
    lat_column : str, optional
        Name of latitude column in input CSV.
    include_intermediate : bool
        Include intermediate values (geology/terrain separately) in output.
    dbscan_nproc : int, optional
        Number of processes for DBSCAN clustering. Use -1 for all cores.
        Has no effect unless --do-bayesian-update is set.
    """
    try:
        config_data = config.resolve_model_config(model)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    df = pd.read_csv(locations_csv)
    if lon_column not in df.columns:
        raise typer.BadParameter(f"Column '{lon_column}' not found in {locations_csv}")
    if lat_column not in df.columns:
        raise typer.BadParameter(f"Column '{lat_column}' not found in {locations_csv}")

    result_df = pipeline.points_pipeline(
        longitudes=df[lon_column].to_numpy(),
        latitudes=df[lat_column].to_numpy(),
        geology_categorical_csv=config_data["geology_categorical_csv"],
        terrain_categorical_csv=config_data["terrain_categorical_csv"],
        clustered_observations_csv=config_data["clustered_observations_csv"],
        independent_observations_csv=config_data["independent_observations_csv"],
        combination_method=constants.CombinationMethod(
            config_data["combination_method"]
        ),
        combine_ratio=config_data["combine_ratio"],
        noisy=config_data["noisy"],
        mvn=config_data["mvn"],
        do_bayesian_update=config_data["do_bayesian_update"],
        include_intermediate=include_intermediate,
        dbscan_nproc=dbscan_nproc,
        geology_corr_fn=config_data["geology_corr_fn"],
        terrain_corr_fn=config_data["terrain_corr_fn"],
        apply_coastal_distance_mod=config_data["apply_coastal_distance_mod"],
        apply_alluvium_slope_mod=config_data["apply_alluvium_slope_mod"],
        fill_gaps=config_data["fill_gaps"],
    )

    original_cols = [c for c in df.columns if c not in result_df.columns]
    output_df = pd.concat([df[original_cols].reset_index(drop=True), result_df], axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    logger.info(f"Results written to {output_csv}")


@cli.from_docstring(app)
def grid(
    model: typing.Annotated[str, typer.Option(help=_MODEL_ARG_HELP)] = ...,
    grid_xmin: typing.Annotated[
        int,
        typer.Option(
            help=f"Grid minimum X coordinate (outer edge of domain, NZTM meters). "
            f"Suggested for all of NZ: {config.FULL_NZ_GRID_CONFIG.grid_xmin}."
        ),
    ] = ...,
    grid_xmax: typing.Annotated[
        int,
        typer.Option(
            help=f"Grid maximum X coordinate (outer edge of domain, NZTM meters). "
            f"Suggested for all of NZ: {config.FULL_NZ_GRID_CONFIG.grid_xmax}."
        ),
    ] = ...,
    grid_ymin: typing.Annotated[
        int,
        typer.Option(
            help=f"Grid minimum Y coordinate (outer edge of domain, NZTM meters). "
            f"Suggested for all of NZ: {config.FULL_NZ_GRID_CONFIG.grid_ymin}."
        ),
    ] = ...,
    grid_ymax: typing.Annotated[
        int,
        typer.Option(
            help=f"Grid maximum Y coordinate (outer edge of domain, NZTM meters). "
            f"Suggested for all of NZ: {config.FULL_NZ_GRID_CONFIG.grid_ymax}."
        ),
    ] = ...,
    grid_dx: typing.Annotated[
        int,
        typer.Option(
            help=f"Grid X spacing (meters). Suggested: {config.FULL_NZ_GRID_CONFIG.grid_dx}."
        ),
    ] = ...,
    grid_dy: typing.Annotated[
        int,
        typer.Option(
            help=f"Grid Y spacing (meters). Suggested: {config.FULL_NZ_GRID_CONFIG.grid_dy}."
        ),
    ] = ...,
    output_dir: typing.Annotated[Path, typer.Option(file_okay=False)] = ...,
    dbscan_nproc: typing.Annotated[int, typer.Option()] = -1,
    include_intermediate: typing.Annotated[
        bool, typer.Option("--include-intermediate/--final-only")
    ] = False,
    max_spatial_boolean_array_memory_gb: typing.Annotated[
        float, typer.Option()
    ] = constants.MAX_SPATIAL_BOOLEAN_ARRAY_MEMORY_GB,
) -> None:
    """
    Run the VS30 grid pipeline using a bundled or user-supplied model config.

    Parameters
    ----------
    model : str
        Either a bundled ``FixedModelVersion`` enum value or a path to a
        YAML config file with the same schema.
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
    output_dir : Path
        Directory to save all pipeline outputs.
    dbscan_nproc : int, optional
        Number of processes for DBSCAN clustering. Use -1 for all cores.
    include_intermediate : bool
        Include intermediate rasters in output.
    max_spatial_boolean_array_memory_gb : float, optional
        Maximum memory for spatial boolean arrays.
    """
    try:
        config_data = config.resolve_model_config(model)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e

    pipeline.grid_pipeline(
        grid_config=config.GridConfig(
            grid_xmin=grid_xmin,
            grid_xmax=grid_xmax,
            grid_ymin=grid_ymin,
            grid_ymax=grid_ymax,
            grid_dx=grid_dx,
            grid_dy=grid_dy,
        ),
        output_dir=output_dir,
        geology_categorical_csv=config_data["geology_categorical_csv"],
        terrain_categorical_csv=config_data["terrain_categorical_csv"],
        clustered_observations_csv=config_data["clustered_observations_csv"],
        independent_observations_csv=config_data["independent_observations_csv"],
        combination_method=constants.CombinationMethod(
            config_data["combination_method"]
        ),
        combine_ratio=config_data["combine_ratio"],
        noisy=config_data["noisy"],
        mvn=config_data["mvn"],
        do_bayesian_update=config_data["do_bayesian_update"],
        include_intermediate=include_intermediate,
        dbscan_nproc=dbscan_nproc,
        max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
        geology_corr_fn=config_data["geology_corr_fn"],
        terrain_corr_fn=config_data["terrain_corr_fn"],
        apply_coastal_distance_mod=config_data["apply_coastal_distance_mod"],
        apply_alluvium_slope_mod=config_data["apply_alluvium_slope_mod"],
        fill_gaps=config_data["fill_gaps"],
    )


if __name__ == "__main__":  # pragma: no cover
    app()

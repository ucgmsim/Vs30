"""Command-line interface for the vs30 package."""

import logging
import typing
from collections.abc import Callable
from pathlib import Path

import pandas as pd
import typer
import yaml
from qcore import cli

from vs30 import config, constants, pipeline, utils

logger = logging.getLogger(__name__)

app = typer.Typer(name="vs30", help="VS30 map generation and categorical model updates")

_GRID_XMIN_HELP = (
    f"Grid minimum X coordinate (NZTM, meters). Suggested for all of NZ: "
    f"{constants.FULL_NZ_GRID_CONFIG.grid_xmin}."
)
_GRID_XMAX_HELP = (
    f"Grid maximum X coordinate (NZTM, meters). Suggested for all of NZ: "
    f"{constants.FULL_NZ_GRID_CONFIG.grid_xmax}."
)
_GRID_YMIN_HELP = (
    f"Grid minimum Y coordinate (NZTM, meters). Suggested for all of NZ: "
    f"{constants.FULL_NZ_GRID_CONFIG.grid_ymin}."
)
_GRID_YMAX_HELP = (
    f"Grid maximum Y coordinate (NZTM, meters). Suggested for all of NZ: "
    f"{constants.FULL_NZ_GRID_CONFIG.grid_ymax}."
)
_GRID_DX_HELP = f"Grid X spacing (meters). Suggested: {constants.FULL_NZ_GRID_CONFIG.grid_dx}."
_GRID_DY_HELP = f"Grid Y spacing (meters). Suggested: {constants.FULL_NZ_GRID_CONFIG.grid_dy}."


def load_model_config(version: constants.FixedModelVersion) -> dict:
    """
    Load and resolve a fixed model version's YAML config.

    Validates required fields, resolves CSV paths relative to the resources
    directory, and builds correlation function callables.

    Parameters
    ----------
    version : FixedModelVersion
        Model version to load.

    Returns
    -------
    dict
        Resolved config with keys including geology_corr_fn, terrain_corr_fn,
        apply_coastal_distance_mod, apply_alluvium_slope_mod, and all CSV paths
        resolved to absolute Paths.

    Raises
    ------
    typer.BadParameter
        If the config is missing required fields.
    """
    with open(
        constants.MODEL_VERSION_TO_CONFIG[version],
        encoding=constants.DEFAULT_TEXT_ENCODING,
    ) as f:
        config_data = yaml.safe_load(f)

    for field in (
        "geology_correlation",
        "terrain_correlation",
        "apply_coastal_distance_mod",
        "apply_alluvium_slope_mod",
        "fill_gaps",
        "mvn",
        "noisy",
        "do_bayesian_update",
        "combination_method",
        "combine_ratio",
    ):
        if field not in config_data:
            raise typer.BadParameter(
                f"Config missing required field '{field}'. "
                "All configs must explicitly specify this parameter."
            )

    for key in constants.RESOURCE_SUBDIRS:
        if config_data[key]:
            config_data[key] = (
                constants.RESOURCE_PATH
                / constants.RESOURCE_SUBDIRS[key]
                / config_data[key]
            )

    config_data["geology_corr_fn"] = utils.resolve_correlation_function(
        config_data["geology_correlation"]
    )
    config_data["terrain_corr_fn"] = utils.resolve_correlation_function(
        config_data["terrain_correlation"]
    )

    return config_data


def run_points_pipeline(
    locations_csv: Path,
    output_csv: Path,
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
    lon_column: str = constants.LOCATIONS_LON_COLUMN,
    lat_column: str = constants.LOCATIONS_LAT_COLUMN,
    geology_corr_fn: Callable | None = None,
    terrain_corr_fn: Callable | None = None,
    apply_coastal_distance_mod: bool = True,
    apply_alluvium_slope_mod: bool = False,
    fill_gaps: bool = False,
) -> None:
    """
    Shared implementation for points and points_custom commands.

    Reads the input CSV, validates columns, delegates computation to
    ``pipeline.points_pipeline``, merges original columns with results,
    and writes the output CSV.

    Parameters
    ----------
    locations_csv : Path
        CSV file with latitude/longitude columns (WGS84).
    output_csv : Path
        Output CSV file path.
    model_type : ModelType, optional
        Which model(s) to run: geology, terrain, or combined (default).
    geology_categorical_csv : Path or None, optional
        Path to geology categorical CSV.
    terrain_categorical_csv : Path or None, optional
        Path to terrain categorical CSV.
    clustered_observations_csv : Path or None, optional
        Path to CSV file with clustered observations (e.g., CPT).
    independent_observations_csv : Path or None, optional
        Path to CSV file with independent observations.
    combination_method : CombinationMethod, optional
        Method for combining geology and terrain models.
    combine_ratio : float or None, optional
        Geology-to-terrain weight ratio (used when combination_method is ratio).
    noisy : bool, optional
        Whether to apply noise weighting in spatial adjustment.
    mvn : bool, optional
        Whether to perform MVN spatial adjustment.
    do_bayesian_update : bool, optional
        Whether to perform Bayesian update of categorical Vs30 values.
    include_intermediate : bool, optional
        Include intermediate values (geology/terrain separately) in output.
    dbscan_nproc : int, optional
        Number of processes for DBSCAN clustering of clustered observations
        when do_bayesian_update is True. Default -1 (all cores). Has no
        effect when do_bayesian_update is False.
    lon_column : str, optional
        Name of longitude column in input CSV.
    lat_column : str, optional
        Name of latitude column in input CSV.
    geology_corr_fn : Callable or None, optional
        Correlation function for the geology model.
    terrain_corr_fn : Callable or None, optional
        Correlation function for the terrain model.
    apply_coastal_distance_mod : bool, optional
        Whether to apply the coastal distance modifier.
    apply_alluvium_slope_mod : bool, optional
        Whether to apply the alluvium slope modifier.
    fill_gaps : bool, optional
        Whether to fill on-land nodata gaps using nearest-neighbor interpolation.

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

    result_df = pipeline.points_pipeline(
        longitudes=df[lon_column].to_numpy(),
        latitudes=df[lat_column].to_numpy(),
        model_type=model_type,
        geology_categorical_csv=geology_categorical_csv,
        terrain_categorical_csv=terrain_categorical_csv,
        clustered_observations_csv=clustered_observations_csv,
        independent_observations_csv=independent_observations_csv,
        combination_method=combination_method,
        combine_ratio=combine_ratio,
        noisy=noisy,
        mvn=mvn,
        do_bayesian_update=do_bayesian_update,
        include_intermediate=include_intermediate,
        dbscan_nproc=dbscan_nproc,
        geology_corr_fn=geology_corr_fn,
        terrain_corr_fn=terrain_corr_fn,
        apply_coastal_distance_mod=apply_coastal_distance_mod,
        apply_alluvium_slope_mod=apply_alluvium_slope_mod,
        fill_gaps=fill_gaps,
    )

    original_cols = [c for c in df.columns if c not in result_df.columns]
    output_df = pd.concat([df[original_cols].reset_index(drop=True), result_df], axis=1)

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_df.to_csv(output_csv, index=False)
    logger.info(f"Results written to {output_csv}")


@cli.from_docstring(app)
def points(
    version: typing.Annotated[constants.FixedModelVersion, typer.Argument()],
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
    Compute Vs30 at locations using a fixed model version's config.

    Parameters
    ----------
    version : FixedModelVersion
        Model version to use (e.g., modified_foster_2019).
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
    config_data = load_model_config(version)

    run_points_pipeline(
        locations_csv=locations_csv,
        output_csv=output_csv,
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
        lon_column=lon_column,
        lat_column=lat_column,
        geology_corr_fn=config_data["geology_corr_fn"],
        terrain_corr_fn=config_data["terrain_corr_fn"],
        apply_coastal_distance_mod=config_data["apply_coastal_distance_mod"],
        apply_alluvium_slope_mod=config_data["apply_alluvium_slope_mod"],
        fill_gaps=config_data["fill_gaps"],
    )


@cli.from_docstring(app)
def points_custom(
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
    do_bayesian_update: typing.Annotated[
        bool, typer.Option("--do-bayesian-update/--no-bayesian-update")
    ] = ...,
    apply_alluvium_slope_mod: typing.Annotated[
        bool, typer.Option("--apply-alluvium-slope-mod/--no-apply-alluvium-slope-mod")
    ] = ...,
    apply_coastal_distance_mod: typing.Annotated[
        bool,
        typer.Option("--apply-coastal-distance-mod/--no-apply-coastal-distance-mod"),
    ] = ...,
    fill_gaps: typing.Annotated[bool, typer.Option("--fill-gaps/--no-fill-gaps")] = ...,
    locations_csv: typing.Annotated[
        Path, typer.Option(exists=True, dir_okay=False)
    ] = ...,
    output_csv: typing.Annotated[Path, typer.Option(dir_okay=False)] = ...,
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
    lon_column: typing.Annotated[str, typer.Option()] = constants.LOCATIONS_LON_COLUMN,
    lat_column: typing.Annotated[str, typer.Option()] = constants.LOCATIONS_LAT_COLUMN,
    include_intermediate: typing.Annotated[
        bool, typer.Option("--include-intermediate/--final-only")
    ] = False,
    dbscan_nproc: typing.Annotated[int, typer.Option()] = -1,
) -> None:
    """
    Compute Vs30 values at specific latitude/longitude locations with explicit parameters.

    All scientific parameters must be explicitly provided. Use the simpler
    'points' command to run with a predefined model version instead.

    Parameters
    ----------
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
    apply_alluvium_slope_mod : bool
        Whether to apply the alluvium slope modifier.
    apply_coastal_distance_mod : bool
        Whether to apply the coastal distance modifier.
    fill_gaps : bool
        Whether to fill on-land nodata gaps using nearest-neighbor interpolation.
    locations_csv : Path
        CSV file with latitude/longitude columns (WGS84).
    output_csv : Path
        Output CSV file path.
    clustered_observations_csv : Path, optional
        Path to CSV file with clustered observations (e.g., CPT).
    independent_observations_csv : Path, optional
        Path to CSV file with independent observations.
    model_type : ModelType, optional
        Which model(s) to run: geology, terrain, or combined (default).
    lon_column : str, optional
        Name of longitude column in input CSV.
    lat_column : str, optional
        Name of latitude column in input CSV.
    include_intermediate : bool, optional
        Include intermediate values (geology/terrain separately) in output.
    dbscan_nproc : int, optional
        Number of processes for DBSCAN clustering. Use -1 for all cores.
        Has no effect unless --do-bayesian-update is set.
    """
    run_points_pipeline(
        locations_csv=locations_csv,
        output_csv=output_csv,
        model_type=model_type,
        geology_categorical_csv=geology_categorical_csv,
        terrain_categorical_csv=terrain_categorical_csv,
        clustered_observations_csv=clustered_observations_csv,
        independent_observations_csv=independent_observations_csv,
        combination_method=combination_method,
        combine_ratio=combine_ratio,
        noisy=noisy,
        mvn=mvn,
        do_bayesian_update=do_bayesian_update,
        include_intermediate=include_intermediate,
        dbscan_nproc=dbscan_nproc,
        lon_column=lon_column,
        lat_column=lat_column,
        apply_alluvium_slope_mod=apply_alluvium_slope_mod,
        apply_coastal_distance_mod=apply_coastal_distance_mod,
        fill_gaps=fill_gaps,
    )


@cli.from_docstring(app)
def grid(
    version: typing.Annotated[constants.FixedModelVersion, typer.Option()] = ...,
    grid_xmin: typing.Annotated[int, typer.Option(help=_GRID_XMIN_HELP)] = ...,
    grid_xmax: typing.Annotated[int, typer.Option(help=_GRID_XMAX_HELP)] = ...,
    grid_ymin: typing.Annotated[int, typer.Option(help=_GRID_YMIN_HELP)] = ...,
    grid_ymax: typing.Annotated[int, typer.Option(help=_GRID_YMAX_HELP)] = ...,
    grid_dx: typing.Annotated[int, typer.Option(help=_GRID_DX_HELP)] = ...,
    grid_dy: typing.Annotated[int, typer.Option(help=_GRID_DY_HELP)] = ...,
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
    Run the VS30 grid pipeline using a fixed model version's config.

    Parameters
    ----------
    version : FixedModelVersion
        Model version to use (e.g., modified_foster_2019).
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
    config_data = load_model_config(version)

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


@cli.from_docstring(app)
def grid_custom(
    geology_categorical_csv: typing.Annotated[
        Path, typer.Option("--geology-csv", exists=True, dir_okay=False)
    ] = ...,
    terrain_categorical_csv: typing.Annotated[
        Path, typer.Option("--terrain-csv", exists=True, dir_okay=False)
    ] = ...,
    combination_method: typing.Annotated[
        constants.CombinationMethod, typer.Option()
    ] = ...,
    combine_ratio: typing.Annotated[
        float,
        typer.Option(
            help="Geology-to-terrain weight ratio. Required when combination_method is ratio."
        ),
    ] = ...,
    noisy: typing.Annotated[bool, typer.Option("--noisy/--no-noisy")] = ...,
    mvn: typing.Annotated[bool, typer.Option("--mvn/--no-mvn")] = ...,
    do_bayesian_update: typing.Annotated[
        bool, typer.Option("--do-bayesian-update/--no-bayesian-update")
    ] = ...,
    apply_alluvium_slope_mod: typing.Annotated[
        bool, typer.Option("--apply-alluvium-slope-mod/--no-apply-alluvium-slope-mod")
    ] = ...,
    apply_coastal_distance_mod: typing.Annotated[
        bool,
        typer.Option("--apply-coastal-distance-mod/--no-apply-coastal-distance-mod"),
    ] = ...,
    fill_gaps: typing.Annotated[bool, typer.Option("--fill-gaps/--no-fill-gaps")] = ...,
    grid_xmin: typing.Annotated[int, typer.Option(help=_GRID_XMIN_HELP)] = ...,
    grid_xmax: typing.Annotated[int, typer.Option(help=_GRID_XMAX_HELP)] = ...,
    grid_ymin: typing.Annotated[int, typer.Option(help=_GRID_YMIN_HELP)] = ...,
    grid_ymax: typing.Annotated[int, typer.Option(help=_GRID_YMAX_HELP)] = ...,
    grid_dx: typing.Annotated[int, typer.Option(help=_GRID_DX_HELP)] = ...,
    grid_dy: typing.Annotated[int, typer.Option(help=_GRID_DY_HELP)] = ...,
    output_dir: typing.Annotated[Path, typer.Option(file_okay=False)] = ...,
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
    include_intermediate: typing.Annotated[
        bool, typer.Option("--include-intermediate/--final-only")
    ] = False,
    dbscan_nproc: typing.Annotated[int, typer.Option()] = -1,
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
    apply_alluvium_slope_mod : bool
        Whether to apply the alluvium slope modifier.
    apply_coastal_distance_mod : bool
        Whether to apply the coastal distance modifier.
    fill_gaps : bool
        Whether to fill on-land nodata gaps using nearest-neighbor interpolation.
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
    clustered_observations_csv : Path, optional
        Path to CSV file with clustered observations (e.g., CPT data).
    independent_observations_csv : Path, optional
        Path to CSV file with independent observations.
    model_type : ModelType, optional
        Which model(s) to run: geology, terrain, or combined (default).
    include_intermediate : bool
        Include intermediate rasters in output.
    dbscan_nproc : int, optional
        Number of processes for DBSCAN clustering. Use -1 for all cores.
    max_spatial_boolean_array_memory_gb : float, optional
        Maximum memory for spatial boolean arrays.
    """
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
        model_type=model_type,
        geology_categorical_csv=geology_categorical_csv,
        terrain_categorical_csv=terrain_categorical_csv,
        clustered_observations_csv=clustered_observations_csv,
        independent_observations_csv=independent_observations_csv,
        combination_method=combination_method,
        combine_ratio=combine_ratio,
        noisy=noisy,
        mvn=mvn,
        do_bayesian_update=do_bayesian_update,
        include_intermediate=include_intermediate,
        dbscan_nproc=dbscan_nproc,
        max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
        apply_alluvium_slope_mod=apply_alluvium_slope_mod,
        apply_coastal_distance_mod=apply_coastal_distance_mod,
        fill_gaps=fill_gaps,
    )


if __name__ == "__main__":  # pragma: no cover
    app()

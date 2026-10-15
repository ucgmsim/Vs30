"""Command-line interface for the vs30 package."""

import logging
import os
import typing
from pathlib import Path

import pandas as pd
import threadpoolctl
import typer
from qcore import cli
from tqdm.contrib.logging import logging_redirect_tqdm

from vs30 import config, constants, pipeline

logger = logging.getLogger(__name__)

app = typer.Typer(
    name="vs30",
    help="VS30 map generation and categorical model updates",
    pretty_exceptions_show_locals=False,
)

MODEL_ARG_HELP = (
    "Either a bundled model version name ("
    f"{', '.join(v.value for v in constants.FixedModelVersion)}"
    ") or a path to a custom YAML config file."
)


def configure_logging(verbose: bool) -> None:
    """
    Print the package's log messages to stderr: progress lines, or everything if verbose.

    Parameters
    ----------
    verbose : bool
        Whether to include the step-by-step (DEBUG) messages.
    """
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logging.getLogger("vs30").handlers = [handler]
    logging.getLogger("vs30").setLevel(logging.DEBUG if verbose else logging.INFO)


def ensure_writable_directory(directory: Path) -> None:
    """
    Create ``directory`` if needed and fail fast if it can't be written to.

    Parameters
    ----------
    directory : Path
        Output directory.

    Raises
    ------
    typer.BadParameter
        If the directory can't be created or written to.
    """
    try:
        directory.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise typer.BadParameter(
            f"Can't write to output directory {directory}: {e.strerror}."
        ) from e
    if not os.access(directory, os.W_OK):
        raise typer.BadParameter(
            f"Can't write to output directory {directory}: permission denied."
        )


@cli.from_docstring(app)
def points(
    model: typing.Annotated[str, typer.Argument(help=MODEL_ARG_HELP)],
    locations_csv: typing.Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output_csv: typing.Annotated[Path, typer.Argument(dir_okay=False)],
    lon_column: typing.Annotated[str, typer.Option()] = constants.LOCATIONS_LON_COLUMN,
    lat_column: typing.Annotated[str, typer.Option()] = constants.LOCATIONS_LAT_COLUMN,
    include_intermediate: typing.Annotated[
        bool, typer.Option("--include-intermediate/--final-only")
    ] = False,
    nproc: typing.Annotated[int, typer.Option()] = -1,
    verbose: typing.Annotated[bool, typer.Option("--verbose", "-v")] = False,
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
    nproc : int, optional
        Number of CPU cores to use, for DBSCAN clustering of observations and
        the linear algebra in the MVN spatial adjustment; -1 uses all cores.
    verbose : bool, optional
        Also print step-by-step progress messages.
    """
    configure_logging(verbose)
    if nproc == 0 or nproc < -1:
        raise typer.BadParameter("--nproc must be -1 (all cores) or a positive number.")
    try:
        config_data = config.resolve_model_config(model)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    ensure_writable_directory(output_csv.parent)

    df = pd.read_csv(locations_csv, skipinitialspace=True).rename(columns=str.strip)
    for column, option in ((lon_column, "--lon-column"), (lat_column, "--lat-column")):
        if column not in df.columns:
            raise typer.BadParameter(
                f"Column '{column}' not found in {locations_csv}. Its columns are: "
                f"{', '.join(df.columns)}. Use {option} to name another column."
            )

    longitudes = pd.to_numeric(df[lon_column], errors="coerce")
    latitudes = pd.to_numeric(df[lat_column], errors="coerce")
    valid_coords = longitudes.between(-180, 180) & latitudes.between(-90, 90)
    if not valid_coords.any():
        raise typer.BadParameter(
            f"No row of {locations_csv} has valid coordinates in columns "
            f"'{lon_column}' and '{lat_column}'."
        )
    if not valid_coords.all():
        # Line numbers as seen in a text editor, where the header is line 1.
        bad_lines = [str(row + 2) for row in df.index[~valid_coords]]
        logger.warning(
            f"{len(bad_lines)} row(s) of {locations_csv.name} have missing or invalid "
            f"coordinates (line(s) {', '.join(bad_lines[:10])}"
            f"{', ...' if len(bad_lines) > 10 else ''}), so their results are left "
            "blank. Coordinates must be WGS84 degrees."
        )

    with (
        logging_redirect_tqdm(loggers=[logging.getLogger("vs30")]),
        threadpoolctl.threadpool_limits(
            limits=None if nproc == -1 else nproc, user_api="blas"
        ),
    ):
        result_df = (
            pipeline.points_pipeline(
                longitudes=longitudes[valid_coords].to_numpy(),
                latitudes=latitudes[valid_coords].to_numpy(),
                apply_alluvium_slope_mod=config_data["apply_alluvium_slope_mod"],
                geology_corr_fn=config_data["geology_corr_fn"],
                terrain_corr_fn=config_data["terrain_corr_fn"],
                geology_categorical_csv=config_data["geology_categorical_csv"],
                terrain_categorical_csv=config_data["terrain_categorical_csv"],
                clustered_observations_csv=config_data["clustered_observations_csv"],
                independent_observations_csv=config_data[
                    "independent_observations_csv"
                ],
                combination_method=constants.CombinationMethod(
                    config_data["combination_method"]
                ),
                combine_ratio=config_data["combine_ratio"],
                noisy=config_data["noisy"],
                mvn=config_data["mvn"],
                do_bayesian_update=config_data["do_bayesian_update"],
                include_intermediate=include_intermediate,
                dbscan_nproc=nproc,
                apply_coastal_distance_mod=config_data["apply_coastal_distance_mod"],
                fill_gaps=config_data["fill_gaps"],
            )
            .set_axis(df.index[valid_coords])
            .reindex(df.index)
        )

    n_without_vs30 = (
        result_df.loc[valid_coords, constants.ObservationColumn.VS30].isna().sum()
    )
    if n_without_vs30:
        logger.warning(
            f"{n_without_vs30} site(s) have no Vs30: they are outside the model's "
            "data coverage (offshore, on water, or where the geology or terrain map "
            "has no data)."
        )

    renamed_columns = {
        column: f"{column}_input"
        for column in df.columns
        if column in result_df.columns
    }
    if renamed_columns:
        logger.warning(
            f"Input column(s) {', '.join(renamed_columns)} share a name with output "
            f"columns, so they are kept as {', '.join(renamed_columns.values())}."
        )
    output_df = pd.concat([df.rename(columns=renamed_columns), result_df], axis=1)

    output_df.to_csv(output_csv, index=False)
    logger.info(f"Results written to {output_csv}")


@cli.from_docstring(app)
def grid(
    model: typing.Annotated[str, typer.Argument(help=MODEL_ARG_HELP)],
    output_dir: typing.Annotated[Path, typer.Argument(file_okay=False)],
    grid_xmin: typing.Annotated[
        int, typer.Option()
    ] = config.FULL_NZ_GRID_CONFIG.grid_xmin,
    grid_xmax: typing.Annotated[
        int, typer.Option()
    ] = config.FULL_NZ_GRID_CONFIG.grid_xmax,
    grid_ymin: typing.Annotated[
        int, typer.Option()
    ] = config.FULL_NZ_GRID_CONFIG.grid_ymin,
    grid_ymax: typing.Annotated[
        int, typer.Option()
    ] = config.FULL_NZ_GRID_CONFIG.grid_ymax,
    grid_dx: typing.Annotated[int, typer.Option()] = config.FULL_NZ_GRID_CONFIG.grid_dx,
    grid_dy: typing.Annotated[int, typer.Option()] = config.FULL_NZ_GRID_CONFIG.grid_dy,
    nproc: typing.Annotated[int, typer.Option()] = -1,
    include_intermediate: typing.Annotated[
        bool, typer.Option("--include-intermediate/--final-only")
    ] = False,
    max_spatial_intermediate_array_memory_gb: typing.Annotated[
        float, typer.Option("--mvn-chunk-memory-gb")
    ] = constants.MAX_SPATIAL_INTERMEDIATE_ARRAY_MEMORY_GB,
    verbose: typing.Annotated[bool, typer.Option("--verbose", "-v")] = False,
) -> None:
    """
    Compute a Vs30 grid using a bundled or user-supplied model config.

    By default the grid covers all of New Zealand at 100 m; the bounds and
    spacing options choose a smaller region or a coarser grid.

    Parameters
    ----------
    model : str
        Either a bundled ``FixedModelVersion`` enum value or a path to a
        YAML config file with the same schema.
    output_dir : Path
        Directory to save all pipeline outputs.
    grid_xmin : int
        Western edge of the grid (NZTM2000 metres; a pixel edge, not centre).
    grid_xmax : int
        Eastern edge of the grid (NZTM2000 metres).
    grid_ymin : int
        Southern edge of the grid (NZTM2000 metres).
    grid_ymax : int
        Northern edge of the grid (NZTM2000 metres).
    grid_dx : int
        Pixel width (metres).
    grid_dy : int
        Pixel height (metres).
    nproc : int, optional
        Number of CPU cores to use, for DBSCAN clustering of observations and
        the linear algebra in the MVN spatial adjustment; -1 uses all cores.
    include_intermediate : bool
        Include intermediate rasters in output.
    max_spatial_intermediate_array_memory_gb : float, optional
        Memory cap (GB) for spatial intermediate arrays produced during MVN chunking.
    verbose : bool, optional
        Also print step-by-step progress messages.
    """
    configure_logging(verbose)
    if nproc == 0 or nproc < -1:
        raise typer.BadParameter("--nproc must be -1 (all cores) or a positive number.")
    try:
        config_data = config.resolve_model_config(model)
    except ValueError as e:
        raise typer.BadParameter(str(e)) from e
    for axis, low, high, spacing in (
        ("x", grid_xmin, grid_xmax, grid_dx),
        ("y", grid_ymin, grid_ymax, grid_dy),
    ):
        if low >= high:
            raise typer.BadParameter(
                f"--grid-{axis}min must be less than --grid-{axis}max "
                f"(got {low} and {high})."
            )
        if spacing <= 0:
            raise typer.BadParameter(
                f"--grid-d{axis} must be positive (got {spacing})."
            )
        if (high - low) % spacing:
            raise typer.BadParameter(
                f"The grid's {axis} extent ({high - low} m) must be a whole number of "
                f"--grid-d{axis} pixels ({spacing} m)."
            )
    if (grid_xmin - config.FULL_NZ_GRID_CONFIG.grid_xmin) % grid_dx or (
        grid_ymin - config.FULL_NZ_GRID_CONFIG.grid_ymin
    ) % grid_dy:
        logger.warning(
            "Grid bounds aren't aligned with the full-NZ grid (edges at "
            f"{config.FULL_NZ_GRID_CONFIG.grid_xmin} m E, "
            f"{config.FULL_NZ_GRID_CONFIG.grid_ymin} m N plus multiples of the "
            "spacing), so its pixels won't line up exactly with a full-NZ map."
        )
    ensure_writable_directory(output_dir)
    logger.info(
        f"Computing a {(grid_xmax - grid_xmin) // grid_dx} x "
        f"{(grid_ymax - grid_ymin) // grid_dy} pixel grid "
        f"({grid_dx} x {grid_dy} m pixels)"
    )

    with (
        logging_redirect_tqdm(loggers=[logging.getLogger("vs30")]),
        threadpoolctl.threadpool_limits(
            limits=None if nproc == -1 else nproc, user_api="blas"
        ),
    ):
        pipeline.grid_pipeline(
            grid_config=config.GridConfig(
                grid_xmin=grid_xmin,
                grid_xmax=grid_xmax,
                grid_ymin=grid_ymin,
                grid_ymax=grid_ymax,
                grid_dx=grid_dx,
                grid_dy=grid_dy,
            ),
            apply_alluvium_slope_mod=config_data["apply_alluvium_slope_mod"],
            geology_corr_fn=config_data["geology_corr_fn"],
            terrain_corr_fn=config_data["terrain_corr_fn"],
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
            dbscan_nproc=nproc,
            max_spatial_intermediate_array_memory_gb=max_spatial_intermediate_array_memory_gb,
            apply_coastal_distance_mod=config_data["apply_coastal_distance_mod"],
            fill_gaps=config_data["fill_gaps"],
        )


if __name__ == "__main__":  # pragma: no cover
    app()

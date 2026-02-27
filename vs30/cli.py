"""Command-line interface for the vs30 package."""

import logging
import typing
from pathlib import Path

import typer
from qcore import cli

from vs30 import constants, pipeline
from vs30 import config as config_module

logger = logging.getLogger(__name__)

# Create Typer app for CLI
app = typer.Typer(name="vs30", help="VS30 map generation and categorical model updates")

_cli_config: config_module.Vs30Config | None = None


def get_config() -> config_module.Vs30Config:
    """
    Get the current CLI configuration.

    Returns the config set by the --config option, or the default
    package config if no custom config was specified.

    Returns
    -------
    Vs30Config
        The current CLI configuration object.
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
            exists=True,
            dir_okay=False,
            help="Path to config.yaml file (default: package config.yaml)",
        ),
    ] = None,
    verbose: typing.Annotated[
        bool,
        typer.Option(help="Enable verbose logging"),
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
        logger.debug(
            f"Using default config from {config_module.Vs30Config.default_config_path()}"
        )


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
    nproc: typing.Annotated[int | None, typer.Option()] = None,
) -> None:
    """
    Update categorical model values using Bayesian updates and save to CSV files.

    Parameters
    ----------
    categorical_model_csv : Path
        Path to CSV file with categorical Vs30 mean and standard deviation values.
    output_dir : Path
        Path to output directory. Will be created if it does not exist.
    model_type : str
        Model type: either 'geology' or 'terrain'.
    clustered_observations_csv : Path, optional
        Path to CSV file with clustered observations (e.g., measured_vs30_cpt.csv).
    independent_observations_csv : Path, optional
        Path to CSV file with independent observations.
    nproc : int, optional
        Number of processes for DBSCAN clustering. Use -1 for all available cores.
        Default from config.
    """
    cfg = get_config()
    pipeline.update_categorical_vs30_models(
        categorical_model_csv=categorical_model_csv,
        output_dir=output_dir,
        model_type=model_type,
        clustered_observations_csv=clustered_observations_csv,
        independent_observations_csv=independent_observations_csv,
        nproc=nproc if nproc is not None else cfg.n_proc,
    )


@cli.from_docstring(app)
def grid(
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
    output_dir: typing.Annotated[
        Path | None, typer.Option(file_okay=False)
    ] = None,
    nproc: typing.Annotated[int | None, typer.Option()] = None,
    combination_method: typing.Annotated[str | None, typer.Option()] = None,
    n_proc: typing.Annotated[int | None, typer.Option()] = None,
) -> None:
    """
    Run the full VS30 generation pipeline for both geology and terrain models.

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
    cfg = get_config()
    pipeline.run_full_pipeline(
        cfg=cfg,
        geology_categorical_csv=geology_categorical_csv,
        terrain_categorical_csv=terrain_categorical_csv,
        clustered_observations_csv=clustered_observations_csv,
        independent_observations_csv=independent_observations_csv,
        output_dir=output_dir,
        nproc=nproc,
        combination_method=combination_method,
        n_proc=n_proc,
    )


@cli.from_docstring(app)
def points(
    locations_csv: typing.Annotated[
        Path | None, typer.Option(exists=True, dir_okay=False)
    ] = None,
    output_csv: typing.Annotated[
        Path | None, typer.Option(dir_okay=False)
    ] = None,
    lon_column: typing.Annotated[str | None, typer.Option()] = None,
    lat_column: typing.Annotated[str | None, typer.Option()] = None,
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
    coast_distance_raster: typing.Annotated[
        Path | None,
        typer.Option(exists=True, dir_okay=False),
    ] = None,
    include_intermediate: typing.Annotated[
        bool, typer.Option("--include-intermediate/--final-only")
    ] = True,
    combination_method: typing.Annotated[str | None, typer.Option()] = None,
    n_proc: typing.Annotated[int | None, typer.Option()] = None,
) -> None:
    """
    Compute Vs30 values at specific latitude/longitude locations.

    Parameters
    ----------
    locations_csv : Path, optional
        CSV file with latitude/longitude columns (WGS84). Default from config.
    output_csv : Path, optional
        Output CSV file path. Default from config.
    lon_column : str, optional
        Name of longitude column in input CSV.
    lat_column : str, optional
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
        Path to coastal distance raster (required for hybrid geology modifications).
    include_intermediate : bool
        Include intermediate values (geology/terrain separately) in output.
    combination_method : str, optional
        Method for combining: ratio (float) or 'standard_deviation_weighting'.
    n_proc : int, optional
        Number of parallel processes (default from config, -1 for all cores).
    """
    cfg = get_config()
    pipeline.compute_at_locations(
        cfg=cfg,
        locations_csv=locations_csv,
        output_csv=output_csv,
        lon_column=lon_column,
        lat_column=lat_column,
        geology_categorical_csv=geology_categorical_csv,
        terrain_categorical_csv=terrain_categorical_csv,
        clustered_observations_csv=clustered_observations_csv,
        independent_observations_csv=independent_observations_csv,
        coast_distance_raster=coast_distance_raster,
        include_intermediate=include_intermediate,
        combination_method=combination_method,
        n_proc=n_proc,
    )


if __name__ == "__main__":  # pragma: no cover
    app()

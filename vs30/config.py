"""Configuration data structures and loaders for Vs30 calculations."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from vs30 import constants, correlations


@dataclass
class GridConfig:
    """
    Grid domain and resolution parameters for raster-based Vs30 calculations.

    Defines the NZTM2000 (EPSG:2193) bounding box and pixel spacing for
    the output raster grid. Only used by the grid pipeline; the points
    pipeline does not need grid parameters.

    Attributes
    ----------
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
    """

    grid_xmin: int
    grid_xmax: int
    grid_ymin: int
    grid_ymax: int
    grid_dx: int
    grid_dy: int

    @classmethod
    def from_dict(cls, data: dict) -> "GridConfig":
        """
        Create a GridConfig from a dictionary.

        Parameters
        ----------
        data : dict
            Dictionary containing grid_xmin, grid_xmax, grid_ymin,
            grid_ymax, grid_dx, grid_dy keys.

        Returns
        -------
        GridConfig
            Grid configuration object.
        """
        return cls(
            grid_xmin=data["grid_xmin"],
            grid_xmax=data["grid_xmax"],
            grid_ymin=data["grid_ymin"],
            grid_ymax=data["grid_ymax"],
            grid_dx=data["grid_dx"],
            grid_dy=data["grid_dy"],
        )


# Full New Zealand land extent at standard 100m resolution.
# Used for coastal distance calculations, gap-fill grid alignment, and CLI defaults.
#
# Convention: xmin/xmax/ymin/ymax are PIXEL EDGES (outer bounds), per
# rasterio.transform.from_bounds() and GDAL outputBounds. The number of
# pixels is (xmax-xmin)/dx, and pixel CENTRES are at xmin + dx/2 + n*dx.
#
# These specific bounds are chosen so that pixel CENTRES (1060150, 1060250,
# ..., 2120050 in x) coincide exactly with the bundled IwahashiPike.tif
# pixel centres (which end in ..50 in both axes). This avoids GDAL's
# nearest-neighbour tie-break at every pixel during terrain resampling.
FULL_NZ_GRID_CONFIG: GridConfig = GridConfig(
    grid_xmin=1060100,
    grid_xmax=2120100,
    grid_ymin=4730100,
    grid_ymax=6250100,
    grid_dx=100,
    grid_dy=100,
)


def load_config_from_yaml(yaml_path: Path) -> dict:
    """
    Load and resolve a Vs30 model config from a YAML file.

    Validates required fields, resolves CSV path values according to their
    own syntax (no fallback), and builds correlation function callables
    from the ``geology_correlation`` / ``terrain_correlation`` blocks.

    CSV path resolution rule for each of the four CSV keys
    (``geology_categorical_csv``, ``terrain_categorical_csv``,
    ``clustered_observations_csv``, ``independent_observations_csv``):

    - **Absolute path** (e.g. ``/data/foo.csv``): used as-is.
    - **Bare filename** (no path separator, e.g. ``foo.csv``): resolved
      against the bundled resources directory:
      ``RESOURCE_PATH / RESOURCE_SUBDIRS[key] / value``.
    - **Relative path with a separator** (e.g. ``./foo.csv``,
      ``data/foo.csv``): resolved against the YAML file's directory.

    The form of the value alone determines where the file is looked up;
    there is no fallback between locations. A typo or wrong-form value
    surfaces as a ``FileNotFoundError`` from the downstream loader,
    pointing at the exact path that was tried.

    Parameters
    ----------
    yaml_path : Path
        Path to the YAML config file.

    Returns
    -------
    dict
        Resolved config with keys including geology_corr_fn, terrain_corr_fn,
        apply_coastal_distance_mod, apply_alluvium_slope_mod, and all CSV paths
        resolved to absolute Paths.

    Raises
    ------
    ValueError
        If the config is missing required fields.
    """
    yaml_path = yaml_path.resolve()
    with open(yaml_path, encoding=constants.DEFAULT_TEXT_ENCODING) as f:
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
            raise ValueError(
                f"Config '{yaml_path}' missing required field '{field}'. "
                "All configs must explicitly specify this parameter."
            )

    yaml_dir = yaml_path.parent
    for key in constants.RESOURCE_SUBDIRS:
        if config_data[key]:
            value = config_data[key]
            path = Path(value)
            if path.is_absolute():
                config_data[key] = path
            elif "/" in value or "\\" in value:
                config_data[key] = (yaml_dir / value).resolve()
            else:
                config_data[key] = (
                    constants.RESOURCE_PATH
                    / constants.RESOURCE_SUBDIRS[key]
                    / value
                )

    config_data["geology_corr_fn"] = correlations.resolve_correlation_function(
        config_data["geology_correlation"]
    )
    config_data["terrain_corr_fn"] = correlations.resolve_correlation_function(
        config_data["terrain_correlation"]
    )

    return config_data


def load_model_config(version: constants.FixedModelVersion) -> dict:
    """
    Load and resolve a bundled fixed model version's YAML config.

    Thin wrapper around ``load_config_from_yaml`` that resolves the YAML path
    from ``constants.MODEL_VERSION_TO_CONFIG``.

    Parameters
    ----------
    version : FixedModelVersion
        Model version to load.

    Returns
    -------
    dict
        Resolved config dict (see ``load_config_from_yaml``).
    """
    return load_config_from_yaml(constants.MODEL_VERSION_TO_CONFIG[version])


def resolve_model_config(model: str) -> dict:
    """
    Load a Vs30 model config by bundled name or YAML file path.

    Parameters
    ----------
    model : str
        Either a bundled ``FixedModelVersion`` value (e.g.
        ``"foster_2019_approx"``) or a path to a YAML config file.

    Returns
    -------
    dict
        Resolved config dict (see ``load_config_from_yaml``).

    Raises
    ------
    ValueError
        If ``model`` is neither a known bundled version nor an existing file.
    """
    bundled_names = {v.value for v in constants.FixedModelVersion}
    if model in bundled_names:
        return load_model_config(constants.FixedModelVersion(model))
    path = Path(model)
    if path.is_file():
        return load_config_from_yaml(path)
    raise ValueError(
        f"'{model}' is not a known bundled version and is not an existing file. "
        f"Bundled versions: {sorted(bundled_names)}. "
        "To use a custom config, pass a path to a YAML file with the same schema."
    )

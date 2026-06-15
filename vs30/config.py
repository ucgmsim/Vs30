"""Configuration data structures and loaders for Vs30 calculations."""

from dataclasses import dataclass
from pathlib import Path

import yaml

from vs30 import constants, correlations


@dataclass
class GridConfig:
    """
    NZTM2000 (EPSG:2193) bounding box and pixel spacing.

    Bounds are pixel edges (outer boundaries), not pixel centres.

    Attributes
    ----------
    grid_xmin : int
        Minimum X bound, in metres.
    grid_xmax : int
        Maximum X bound, in metres.
    grid_ymin : int
        Minimum Y bound, in metres.
    grid_ymax : int
        Maximum Y bound, in metres.
    grid_dx : int
        Pixel spacing in X, in metres.
    grid_dy : int
        Pixel spacing in Y, in metres.
    """

    grid_xmin: int
    grid_xmax: int
    grid_ymin: int
    grid_ymax: int
    grid_dx: int
    grid_dy: int


# Full NZ land extent at 100m. Bounds chosen so pixel centres align with
# the bundled IwahashiPike.tif (centres ending in ..50), avoiding GDAL
# nearest-neighbour tie-breaks during terrain resampling.
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
    Load a Vs30 model config from a YAML file.

    Parameters
    ----------
    yaml_path : Path
        Path to the YAML config file.

    Returns
    -------
    dict
        Resolved config.

    Raises
    ------
    ValueError
        If required fields are missing.
    """
    yaml_path = yaml_path.resolve()
    with open(yaml_path, encoding=constants.DEFAULT_TEXT_ENCODING) as f:
        config_data = yaml.safe_load(f)

    for field in constants.REQUIRED_CONFIG_FIELDS:
        if field not in config_data:
            raise ValueError(
                f"Config '{yaml_path}' missing required field '{field}'. "
                "All configs must explicitly specify this parameter."
            )

    # CSV path resolution: absolute paths are used as-is (custom override);
    # otherwise resolved under constants.RESOURCE_PATH / RESOURCE_SUBDIRS[key]
    # (bundled file). Non-existent paths fail downstream when loaded.
    for key in constants.RESOURCE_SUBDIRS:
        if config_data[key]:
            if Path(config_data[key]).is_absolute():
                config_data[key] = Path(config_data[key])
            else:
                config_data[key] = (
                    constants.RESOURCE_PATH
                    / constants.RESOURCE_SUBDIRS[key]
                    / config_data[key]
                )

    config_data["geology_corr_fn"] = correlations.resolve_correlation_function(
        config_data["geology_correlation"]
    )
    config_data["terrain_corr_fn"] = correlations.resolve_correlation_function(
        config_data["terrain_correlation"]
    )

    return config_data


def resolve_model_config(model: str) -> dict:
    """
    Load a Vs30 model config by bundled name or YAML file path.

    Parameters
    ----------
    model : str
        Bundled ``FixedModelVersion`` value or a path to a YAML config file
        (relative paths resolve against the current working directory).

    Returns
    -------
    dict
        Resolved config.

    Raises
    ------
    ValueError
        If ``model`` is neither a known bundled version nor an existing file.
    """
    if model in {v.value for v in constants.FixedModelVersion}:
        return load_config_from_yaml(
            constants.MODEL_VERSION_TO_CONFIG[constants.FixedModelVersion(model)]
        )
    if Path(model).is_file():
        return load_config_from_yaml(Path(model))
    raise ValueError(
        f"'{model}' is not a known bundled version and is not an existing file. "
        f"Bundled versions: {sorted({v.value for v in constants.FixedModelVersion})}. "
        "To use a custom config, pass a path to a YAML file with the same schema."
    )

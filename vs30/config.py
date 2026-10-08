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
    Load and validate a Vs30 model config from a YAML file.

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
        If the file is empty, a field is missing or unknown, or a value is
        invalid (including CSV paths that don't exist).
    """
    yaml_path = yaml_path.resolve()
    with open(yaml_path, encoding=constants.DEFAULT_TEXT_ENCODING) as f:
        config_data = yaml.safe_load(f)

    if not isinstance(config_data, dict):
        raise ValueError(
            f"Config '{yaml_path}' is empty or not a list of 'field: value' lines."
        )

    for field in constants.REQUIRED_CONFIG_FIELDS:
        if field not in config_data:
            raise ValueError(
                f"Config '{yaml_path}' missing required field '{field}'. "
                "All configs must explicitly specify this parameter."
            )

    unknown_fields = sorted(set(config_data) - set(constants.REQUIRED_CONFIG_FIELDS))
    if unknown_fields:
        raise ValueError(
            f"Config '{yaml_path}' has unknown field(s): {', '.join(unknown_fields)}."
        )

    for field in (
        "apply_coastal_distance_mod",
        "apply_alluvium_slope_mod",
        "fill_gaps",
        "mvn",
        "noisy",
        "do_bayesian_update",
    ):
        if not isinstance(config_data[field], bool):
            raise ValueError(
                f"Config '{yaml_path}': {field} must be true or false, "
                f"got {config_data[field]!r}."
            )

    combination_methods = [method.value for method in constants.CombinationMethod]
    if config_data["combination_method"] not in combination_methods:
        raise ValueError(
            f"Config '{yaml_path}': combination_method must be one of "
            f"{', '.join(combination_methods)}, got {config_data['combination_method']!r}."
        )
    if config_data["combination_method"] == constants.CombinationMethod.RATIO and not (
        isinstance(config_data["combine_ratio"], (int, float))
        and config_data["combine_ratio"] >= 0
    ):
        raise ValueError(
            f"Config '{yaml_path}': combine_ratio must be a number >= 0 when "
            f"combination_method is ratio, got {config_data['combine_ratio']!r}."
        )

    if (config_data["mvn"] or config_data["do_bayesian_update"]) and not (
        config_data["clustered_observations_csv"]
        or config_data["independent_observations_csv"]
    ):
        raise ValueError(
            f"Config '{yaml_path}': mvn and do_bayesian_update need at least one "
            "observations CSV (clustered_observations_csv or "
            "independent_observations_csv)."
        )

    # CSV paths are either bundled file names, resolved under
    # constants.RESOURCE_PATH / RESOURCE_SUBDIRS[key], or absolute paths.
    for key in constants.RESOURCE_SUBDIRS:
        if not config_data[key]:
            continue
        if Path(config_data[key]).is_absolute():
            config_data[key] = Path(config_data[key])
        elif Path(config_data[key]).name == config_data[key]:
            config_data[key] = (
                constants.RESOURCE_PATH
                / constants.RESOURCE_SUBDIRS[key]
                / config_data[key]
            )
        else:
            raise ValueError(
                f"Config '{yaml_path}': {key} {config_data[key]!r} must be a "
                "bundled file name or an absolute path."
            )
        if not config_data[key].is_file():
            raise ValueError(
                f"Config '{yaml_path}': {key} file not found: {config_data[key]}"
            )

    for field, corr_fn_key in (
        ("geology_correlation", "geology_corr_fn"),
        ("terrain_correlation", "terrain_corr_fn"),
    ):
        try:
            config_data[corr_fn_key] = correlations.resolve_correlation_function(
                config_data[field]
            )
        except ValueError as e:
            raise ValueError(f"Config '{yaml_path}': {field}: {e}") from e

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

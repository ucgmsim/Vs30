"""
Pydantic configuration model for the vs30 package.

This module provides a typed configuration class that loads and validates
user-configurable settings from config.yaml.

For scientific/algorithmic constants that should not be modified by users,
see vs30/constants.py.

Usage
-----
    from vs30.config import Vs30Config, get_default_config

    # Load default config (from package's config.yaml)
    config = get_default_config()

    # Load from custom path
    config = Vs30Config.from_yaml(Path("/path/to/custom/config.yaml"))

    # Access values with IDE autocomplete
    max_distance = config.max_dist_m
"""

from __future__ import annotations

from pathlib import Path

import pydantic
import yaml


class Vs30Config(pydantic.BaseModel):
    """
    User-configurable settings for VS30 calculations.

    All fields correspond to entries in config.yaml. See that file for
    detailed descriptions of each parameter's meaning and units.

    For scientific/algorithmic constants, see vs30/constants.py.
    """

    # --- Processor settings ---
    n_proc: int = pydantic.Field(
        default=1,
        description="Number of processors for parallel processing (-1 for all cores)",
    )

    # --- Grid parameters ---
    grid_xmin: int = pydantic.Field(description="Grid minimum X coordinate (NZTM, meters)")
    grid_xmax: int = pydantic.Field(description="Grid maximum X coordinate (NZTM, meters)")
    grid_ymin: int = pydantic.Field(description="Grid minimum Y coordinate (NZTM, meters)")
    grid_ymax: int = pydantic.Field(description="Grid maximum Y coordinate (NZTM, meters)")
    grid_dx: int = pydantic.Field(description="Grid X spacing (meters)")
    grid_dy: int = pydantic.Field(description="Grid Y spacing (meters)")

    # --- Compute-at-locations parameters (only used by compute-at-locations) ---
    locations_csv: str | None = pydantic.Field(
        default=None,
        description="Path to input CSV with locations for compute-at-locations"
    )
    locations_output_csv: str | None = pydantic.Field(
        default=None,
        description="Path to output CSV for compute-at-locations results"
    )

    # --- General configuration ---
    noisy: bool = pydantic.Field(
        description="Whether measurements are noisy (affects uncertainty weighting)"
    )
    max_spatial_boolean_array_memory_gb: float = pydantic.Field(
        description="Maximum memory (GB) for spatial boolean arrays per process"
    )
    obs_subsample_step_for_clustered: int = pydantic.Field(
        description="Subsampling step for clustered observations in affected pixel search"
    )

    # --- File paths (relative to resources directory) ---
    independent_observations_file: str = pydantic.Field(
        description="Path to independent observations CSV (relative to resources)"
    )
    clustered_observations_file: str = pydantic.Field(
        description="Path to clustered observations CSV (relative to resources)"
    )
    output_dir: str = pydantic.Field(description="Output directory path")

    # --- Combination settings ---
    combination_method: str | float = pydantic.Field(
        description="Method for combining models: ratio (float) or 'standard_deviation_weighting'"
    )
    do_bayesian_update_of_geology_and_terrain_categorical_vs30_values: bool = pydantic.Field(
        description="Whether to perform Bayesian update of categorical values"
    )

    # =========================================================================
    # Class methods for loading
    # =========================================================================

    @classmethod
    def from_yaml(cls, path: Path) -> "Vs30Config":
        """
        Load configuration from a YAML file.

        Parameters
        ----------
        path : Path
            Path to the YAML configuration file.

        Returns
        -------
        Vs30Config
            Validated configuration object.

        Raises
        ------
        FileNotFoundError
            If the configuration file does not exist.
        pydantic.ValidationError
            If the configuration file is missing required fields or has invalid values.
        """
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        with open(path, encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls(**data)

    @classmethod
    def default_config_path(cls) -> Path:
        """Return the path to the package's default config.yaml."""
        return Path(__file__).parent / "config.yaml"

    @classmethod
    def default(cls) -> "Vs30Config":
        """
        Load from the package's default config.yaml.

        Returns
        -------
        Vs30Config
            Configuration loaded from the package's bundled config.yaml.
        """
        return cls.from_yaml(cls.default_config_path())


# =============================================================================
# Module-level config management
# =============================================================================

_default_config: Vs30Config | None = None


def get_default_config() -> Vs30Config:
    """
    Get the default configuration, loading it on first access.

    This provides lazy loading of the default config, so it's only
    read from disk when first needed.

    Returns
    -------
    Vs30Config
        The default configuration object.
    """
    global _default_config
    if _default_config is None:
        _default_config = Vs30Config.default()
    return _default_config



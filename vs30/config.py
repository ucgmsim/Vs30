"""
Pydantic configuration model for the vs30 package.

This module provides a typed configuration class that loads and validates
settings from config.yaml. It replaces the previous constants.py approach
with a more maintainable Pydantic-based system.

Usage
-----
    from vs30.config import Vs30Config, get_default_config

    # Load default config (from package's config.yaml)
    config = get_default_config()

    # Load from custom path
    config = Vs30Config.from_yaml(Path("/path/to/custom/config.yaml"))

    # Access values with IDE autocomplete
    max_distance = config.max_dist_m
    phi = config.phi["geology"]
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

import yaml
from pydantic import BaseModel, Field


class HybridVs30Param(BaseModel):
    """Parameters for slope-based Vs30 interpolation per geology group."""

    gid: int = Field(description="Geology group ID")
    slope_limits: list[float] = Field(
        description="Log10 slope limits [min, max]", min_length=2, max_length=2
    )
    vs30_values: list[float] = Field(
        description="Vs30 values [min, max] for interpolation", min_length=2, max_length=2
    )


class Vs30Config(BaseModel):
    """
    Configuration for VS30 calculations.

    All fields correspond to entries in config.yaml. See that file for
    detailed descriptions of each parameter's meaning and units.
    """

    # --- Processor settings ---
    n_proc: int = Field(
        default=1,
        description="Number of processors for parallel processing (-1 for all cores)",
    )

    # --- Grid parameters ---
    grid_xmin: int = Field(description="Grid minimum X coordinate (NZTM, meters)")
    grid_xmax: int = Field(description="Grid maximum X coordinate (NZTM, meters)")
    grid_ymin: int = Field(description="Grid minimum Y coordinate (NZTM, meters)")
    grid_ymax: int = Field(description="Grid maximum Y coordinate (NZTM, meters)")
    grid_dx: int = Field(description="Grid X spacing (meters)")
    grid_dy: int = Field(description="Grid Y spacing (meters)")

    # --- Full NZ land extent (for coastal distance calculations) ---
    full_nz_land_xmin: int = Field(description="Full NZ land extent minimum X")
    full_nz_land_xmax: int = Field(description="Full NZ land extent maximum X")
    full_nz_land_ymin: int = Field(description="Full NZ land extent minimum Y")
    full_nz_land_ymax: int = Field(description="Full NZ land extent maximum Y")

    # --- General configuration ---
    max_dist_m: int = Field(
        description="Maximum distance (m) for MVN spatial adjustment observations"
    )
    max_points: int = Field(
        description="Maximum observations to consider per pixel in MVN update"
    )
    cov_reduc: float = Field(
        description="Covariance reduction factor for dissimilar Vs30 values"
    )
    noisy: bool = Field(
        description="Whether measurements are noisy (affects uncertainty weighting)"
    )
    max_spatial_boolean_array_memory_gb: float = Field(
        description="Maximum memory (GB) for spatial boolean arrays per process"
    )
    obs_subsample_step_for_clustered: int = Field(
        description="Subsampling step for clustered observations in affected pixel search"
    )

    # --- Correlation lengths ---
    phi_geology: int = Field(
        description="Correlation length (m) for geology model spatial adjustment"
    )
    phi_terrain: int = Field(
        description="Correlation length (m) for terrain model spatial adjustment"
    )

    # --- Bayesian update parameters ---
    n_prior: int = Field(description="Effective number of prior observations")
    min_sigma: float = Field(
        description="Minimum standard deviation (log-space) after Bayesian update"
    )
    min_group: int = Field(description="Minimum group size for DBSCAN clustering")
    eps: float = Field(
        description="Maximum distance (m) for DBSCAN clustering epsilon"
    )

    # --- NoData values ---
    raster_id_nodata_value: int = Field(
        description="NoData value in categorical ID rasters"
    )
    nodata_value: int = Field(
        description="NoData value for model outputs and CSV placeholders"
    )

    # --- File paths (relative to resources directory) ---
    independent_observations_file: str = Field(
        description="Path to independent observations CSV (relative to resources)"
    )
    clustered_observations_file: str = Field(
        description="Path to clustered observations CSV (relative to resources)"
    )
    output_dir: str = Field(description="Output directory path")
    geology_mean_and_standard_deviation_per_category_file: str = Field(
        description="Path to geology categorical model CSV"
    )
    terrain_mean_and_standard_deviation_per_category_file: str = Field(
        description="Path to terrain categorical model CSV"
    )

    # --- Output filenames ---
    posterior_prefix: str = Field(description="Prefix for posterior output files")
    terrain_initial_vs30_filename: str = Field(
        description="Filename for initial terrain Vs30 raster"
    )
    geology_initial_vs30_filename: str = Field(
        description="Filename for initial geology Vs30 raster"
    )
    slope_raster_filename: str = Field(description="Filename for slope raster")
    coast_distance_raster_filename: str = Field(
        description="Filename for coastal distance raster"
    )
    geology_vs30_slope_and_coastal_distance_adjusted_filename: str = Field(
        description="Filename for hybrid geology raster"
    )
    geology_id_filename: str = Field(description="Filename for geology ID raster")
    terrain_id_filename: str = Field(description="Filename for terrain ID raster")
    terrain_vs30_mean_stddev_filename: str = Field(
        description="Filename for final terrain Vs30 raster"
    )
    geology_vs30_mean_stddev_filename: str = Field(
        description="Filename for final geology Vs30 raster"
    )
    combined_vs30_filename: str = Field(
        description="Filename for combined geology+terrain Vs30 raster"
    )

    # --- Combination settings ---
    combination_method: Union[str, float] = Field(
        description="Method for combining models: ratio (float) or 'standard_deviation_weighting'"
    )
    k_value: float = Field(
        description="K value for standard deviation based weighting"
    )
    weight_epsilon_div_by_zero: float = Field(
        default=1e-10,
        description="Small epsilon added to variance to prevent division by zero in weight calculation"
    )
    do_bayesian_update_of_geology_and_terrain_categorical_vs30_values: bool = Field(
        description="Whether to perform Bayesian update of categorical values"
    )

    # --- Hybrid model parameters ---
    hybrid_mod6_dist_min: float = Field(description="Mod6 minimum coastal distance (m)")
    hybrid_mod6_dist_max: float = Field(description="Mod6 maximum coastal distance (m)")
    hybrid_mod6_vs30_min: float = Field(description="Mod6 minimum Vs30 (m/s)")
    hybrid_mod6_vs30_max: float = Field(description="Mod6 maximum Vs30 (m/s)")
    hybrid_mod13_dist_min: float = Field(
        description="Mod13 minimum coastal distance (m)"
    )
    hybrid_mod13_dist_max: float = Field(
        description="Mod13 maximum coastal distance (m)"
    )
    hybrid_mod13_vs30_min: float = Field(description="Mod13 minimum Vs30 (m/s)")
    hybrid_mod13_vs30_max: float = Field(description="Mod13 maximum Vs30 (m/s)")
    hybrid_vs30_params: list[HybridVs30Param] = Field(
        description="Slope-based Vs30 interpolation parameters per geology group"
    )
    hybrid_sigma_reduction_factors: dict[int, float] = Field(
        description="Sigma reduction factors per geology group ID"
    )
    min_slope_for_log: float = Field(
        description="Minimum slope value to prevent log10(0)"
    )

    # --- Misc ---
    min_dist_enforced: float = Field(
        description="Minimum distance (m) enforced in correlation calculations"
    )
    nztm_crs: str = Field(description="Coordinate Reference System string")

    # --- Plotting ---
    plot_figsize: list[int] = Field(description="Figure size [width, height] in inches")
    plot_dpi: int = Field(description="Plot resolution in DPI")

    # --- DataFrame Column Names ---
    col_posterior_mean_independent: str = Field(
        default="posterior_mean_vs30_km_per_s_independent_observations",
        description="Column name for posterior mean after independent observations update"
    )
    col_posterior_stdv_independent: str = Field(
        default="posterior_standard_deviation_vs30_km_per_s_independent_observations",
        description="Column name for posterior stdv after independent observations update"
    )
    col_posterior_mean_clustered: str = Field(
        default="posterior_mean_vs30_km_per_s_clustered_observations",
        description="Column name for posterior mean after clustered observations update"
    )
    col_posterior_stdv_clustered: str = Field(
        default="posterior_standard_deviation_vs30_km_per_s_clustered_observations",
        description="Column name for posterior stdv after clustered observations update"
    )
    col_posterior_mean: str = Field(
        default="posterior_mean_vs30_km_per_s",
        description="Column name for generic posterior mean"
    )
    col_posterior_stdv: str = Field(
        default="posterior_standard_deviation_vs30_km_per_s",
        description="Column name for generic posterior stdv"
    )
    col_prior_mean: str = Field(
        default="prior_mean_vs30_km_per_s",
        description="Column name for prior mean"
    )
    col_prior_stdv: str = Field(
        default="prior_standard_deviation_vs30_km_per_s",
        description="Column name for prior stdv"
    )
    col_mean: str = Field(
        default="mean_vs30_km_per_s",
        description="Column name for original/raw mean"
    )
    col_stdv: str = Field(
        default="standard_deviation_vs30_km_per_s",
        description="Column name for original/raw stdv"
    )

    # =========================================================================
    # Computed/derived properties
    # =========================================================================

    @property
    def phi(self) -> dict[str, int]:
        """Phi (correlation length) values keyed by model type."""
        return {"geology": self.phi_geology, "terrain": self.phi_terrain}

    @property
    def output_filenames(self) -> dict[str, str]:
        """Final output filenames keyed by model type."""
        return {
            "geology": self.geology_vs30_mean_stddev_filename,
            "terrain": self.terrain_vs30_mean_stddev_filename,
        }

    @property
    def model_csv_paths(self) -> dict[str, str]:
        """Model CSV paths keyed by model type."""
        return {
            "geology": self.geology_mean_and_standard_deviation_per_category_file,
            "terrain": self.terrain_mean_and_standard_deviation_per_category_file,
        }

    # =========================================================================
    # Convenience aliases matching old constants.py names
    # =========================================================================

    @property
    def MODEL_NODATA(self) -> int:
        """Alias for nodata_value (backwards compatibility)."""
        return self.nodata_value

    @property
    def RASTER_ID_NODATA_VALUE(self) -> int:
        """Alias for raster_id_nodata_value (backwards compatibility)."""
        return self.raster_id_nodata_value

    @property
    def GEOLOGY_MEAN_STDDEV_CSV(self) -> str:
        """Alias for geology CSV path (backwards compatibility)."""
        return self.geology_mean_and_standard_deviation_per_category_file

    @property
    def TERRAIN_MEAN_STDDEV_CSV(self) -> str:
        """Alias for terrain CSV path (backwards compatibility)."""
        return self.terrain_mean_and_standard_deviation_per_category_file

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
        with open(path) as f:
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



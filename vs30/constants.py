"""
Centralized constants and configuration loading for the vs30 package.

This module serves as a single source of truth for all configuration parameters.
Values are loaded from config.yaml at import time and exported as module-level
constants. This pattern allows:

1. Easy access via `from vs30 import constants` then `constants.MAX_DIST_M`
2. Validation that all required config keys exist at startup
3. Centralized configuration management

Usage
-----
    from vs30 import constants

    # Access configuration values as constants
    max_distance = constants.MAX_DIST_M
    phi = constants.PHI["geology"]

Configuration File
-----------------
All values are read from config.yaml in this module's directory. See config.yaml
for parameter descriptions and their scientific meaning.
"""

from pathlib import Path

import yaml

# Path to the configuration file (same directory as this module)
CONFIG_FILE = Path(__file__).parent / "config.yaml"


def _load_config():
    if not CONFIG_FILE.exists():
        raise FileNotFoundError(f"Configuration file not found: {CONFIG_FILE}")

    with open(CONFIG_FILE, "r") as f:
        return yaml.safe_load(f)


_config = _load_config()

# General Configuration
MAX_DIST_M = _config["max_dist_m"]
MAX_POINTS = _config["max_points"]
COV_REDUC = _config["cov_reduc"]
NOISY = _config["noisy"]
MAX_SPATIAL_BOOLEAN_ARRAY_MEMORY_GB = _config["max_spatial_boolean_array_memory_gb"]
MODEL_NODATA = _config["model_nodata"]
N_PROC = _config.get("n_proc", 1)

# Bayesian Update Parameters
N_PRIOR = _config["n_prior"]
MIN_SIGMA = _config["min_sigma"]
MIN_GROUP = _config["min_group"]
EPS = _config["eps"]
NPROC_FOR_DBSCAN_CLUSTERING = _config["nproc_for_dbscan_clustering"]

# Correlation Lengths (Phi)
PHI_GEOLOGY = _config["phi_geology"]
PHI_TERRAIN = _config["phi_terrain"]

# Paths and Filenames
INDEPENDENT_OBSERVATIONS_FILE = _config["independent_observations_file"]
CLUSTERED_OBSERVATIONS_FILE = _config["clustered_observations_file"]
OUTPUT_DIR_NAME = _config["output_dir"]

TERRAIN_VS30_MEAN_STDDEV_FILENAME = _config["terrain_vs30_mean_stddev_filename"]
GEOLOGY_VS30_MEAN_STDDEV_FILENAME = _config["geology_vs30_mean_stddev_filename"]

GEOLOGY_MEAN_STDDEV_CSV = _config[
    "geology_mean_and_standard_deviation_per_category_file"
]
TERRAIN_MEAN_STDDEV_CSV = _config[
    "terrain_mean_and_standard_deviation_per_category_file"
]

# Weights for multi-model combining
# Method for combining geology and terrain models
COMBINATION_METHOD = _config["combination_method"]
K_VALUE = _config["k_value"]

# Grid Parameters
GRID_XMIN = _config["grid_xmin"]
GRID_XMAX = _config["grid_xmax"]
GRID_YMIN = _config["grid_ymin"]
GRID_YMAX = _config["grid_ymax"]
GRID_DX = _config["grid_dx"]
GRID_DY = _config["grid_dy"]

# Full NZ Land Extent Bounds (for coastal distance calculations)
# These are separate from GRID_XMIN/MAX/YMIN/YMAX which define the study domain
FULL_NZ_LAND_XMIN = _config["full_nz_land_xmin"]
FULL_NZ_LAND_XMAX = _config["full_nz_land_xmax"]
FULL_NZ_LAND_YMIN = _config["full_nz_land_ymin"]
FULL_NZ_LAND_YMAX = _config["full_nz_land_ymax"]

# NoData and Placeholder Values
ID_NODATA = _config["id_nodata"]
SLOPE_NODATA = _config["slope_nodata"]
CSV_PLACEHOLDER_NODATA = _config["csv_placeholder_nodata"]

# filenames and Prefixes
POSTERIOR_PREFIX = _config["posterior_prefix"]
TERRAIN_INITIAL_VS30_FILENAME = _config["terrain_initial_vs30_filename"]
GEOLOGY_INITIAL_VS30_FILENAME = _config["geology_initial_vs30_filename"]
SLOPE_RASTER_FILENAME = _config["slope_raster_filename"]
COAST_DISTANCE_RASTER_FILENAME = _config["coast_distance_raster_filename"]
GEOLOGY_VS30_SLOPE_AND_COASTAL_DISTANCE_ADJUSTED_FILENAME = _config[
    "geology_vs30_slope_and_coastal_distance_adjusted_filename"
]
GEOLOGY_ID_FILENAME = _config["geology_id_filename"]
TERRAIN_ID_FILENAME = _config["terrain_id_filename"]
COMBINED_VS30_FILENAME = _config["combined_vs30_filename"]

# Hybrid Model constants
HYBRID_MOD6_DIST_MIN = _config["hybrid_mod6_dist_min"]
HYBRID_MOD6_DIST_MAX = _config["hybrid_mod6_dist_max"]
HYBRID_MOD6_VS30_MIN = _config["hybrid_mod6_vs30_min"]
HYBRID_MOD6_VS30_MAX = _config["hybrid_mod6_vs30_max"]

HYBRID_MOD13_DIST_MIN = _config["hybrid_mod13_dist_min"]
HYBRID_MOD13_DIST_MAX = _config["hybrid_mod13_dist_max"]
HYBRID_MOD13_VS30_MIN = _config["hybrid_mod13_vs30_min"]
HYBRID_MOD13_VS30_MAX = _config["hybrid_mod13_vs30_max"]

# Hybrid slope-based Vs30 interpolation parameters
HYBRID_VS30_PARAMS = _config["hybrid_vs30_params"]

# Hybrid standard deviation reduction factors
HYBRID_SIGMA_REDUCTION_FACTORS = _config["hybrid_sigma_reduction_factors"]

# Minimum slope value for log calculations
MIN_SLOPE_FOR_LOG = _config["min_slope_for_log"]

# Misc
MIN_DIST_ENFORCED = _config["min_dist_enforced"]
NZTM_CRS = _config["nztm_crs"]

# Plotting
PLOT_FIGSIZE = _config["plot_figsize"]
PLOT_DPI = _config["plot_dpi"]

# Helper Mappings for Model-Specific Constants
PHI = {"geology": PHI_GEOLOGY, "terrain": PHI_TERRAIN}

OUTPUT_FILENAMES = {
    "geology": GEOLOGY_VS30_MEAN_STDDEV_FILENAME,
    "terrain": TERRAIN_VS30_MEAN_STDDEV_FILENAME,
}

MODEL_CSV_PATHS = {
    "geology": GEOLOGY_MEAN_STDDEV_CSV,
    "terrain": TERRAIN_MEAN_STDDEV_CSV,
}

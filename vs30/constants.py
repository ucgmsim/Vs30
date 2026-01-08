"""
Centralized constants and configuration loading for the vs30 package.
Loads values from config.yaml and exports them as module-level constants.
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

# Bayesian Update Parameters
N_PRIOR = _config["n_prior"]
MIN_SIGMA = _config["min_sigma"]
MIN_GROUP = _config["min_group"]
EPS = _config["eps"]
NPROC = _config["nproc"]

# Correlation Lengths (Phi)
PHI_GEOLOGY = _config["phi_geology"]
PHI_TERRAIN = _config["phi_terrain"]

# Paths and Filenames
OBSERVATIONS_FILE = _config["observations_file"]
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

MODEL_ID_COLUMNS = _config["model_id_columns"]

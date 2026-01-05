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
MODELS_TO_PROCESS = _config.get("models_to_process", ["geology"])
MAX_DIST_M = _config.get("max_dist_m", 10000)
MAX_POINTS = _config.get("max_points", 500)
COV_REDUC = _config.get("cov_reduc", 1.5)
NOISY = _config.get("noisy", True)
TOTAL_MEMORY_GB = _config.get("total_memory_gb", 3)
COMPUTE_BAYESIAN_UPDATE = _config.get("compute_bayesian_update", True)
MODEL_NODATA = _config.get("model_nodata", -32767)

# Bayesian Update Parameters
N_PRIOR = _config.get("n_prior", 3)
MIN_SIGMA = _config.get("min_sigma", 0.5)
MIN_GROUP = _config.get("min_group", 5)
EPS = _config.get("eps", 15000.0)
NPROC = _config.get("nproc", -1)

# Correlation Lengths (Phi)
PHI_GEOLOGY = _config.get("phi_geology", 1407)
PHI_TERRAIN = _config.get("phi_terrain", 993)

# Paths and Filenames
OBSERVATIONS_FILE = _config.get(
    "observations_file", "observations/measured_vs30_original_filtered.csv"
)
OUTPUT_DIR_NAME = _config.get("output_dir", "new_vs30map")

TERRAIN_VS30_MEAN_STDDEV_FILENAME = _config.get(
    "terrain_vs30_mean_stddev_filename", "terrain_vs30_with_uncertainty.tif"
)
GEOLOGY_VS30_MEAN_STDDEV_FILENAME = _config.get(
    "geology_vs30_mean_stddev_filename", "geology_vs30_with_uncertainty.tif"
)

GEOLOGY_MEAN_STDDEV_CSV = _config.get(
    "geology_mean_and_standard_deviation_per_category_file",
    "categorical_vs30_mean_and_stddev/geology/geology_model_prior_mean_and_standard_deviation.csv",
)
TERRAIN_MEAN_STDDEV_CSV = _config.get(
    "terrain_mean_and_standard_deviation_per_category_file",
    "categorical_vs30_mean_and_stddev/terrain/terrain_model_prior_mean_and_standard_deviation.csv",
)

# Weights for multi-model combining
GEOLOGY_WEIGHT = _config.get("geology_weight", 1.0)
TERRAIN_WEIGHT = _config.get("terrain_weight", 1.0)
K = _config.get("k", -1.0)

# Grid Parameters
GRID_XMIN = _config.get("grid_xmin", 1060050)
GRID_XMAX = _config.get("grid_xmax", 2120050)
GRID_YMIN = _config.get("grid_ymin", 4730050)
GRID_YMAX = _config.get("grid_ymax", 6250050)
GRID_DX = _config.get("grid_dx", 100)
GRID_DY = _config.get("grid_dy", 100)

# NoData and Placeholder Values
ID_NODATA = _config.get("id_nodata", 255)
SLOPE_NODATA = _config.get("slope_nodata", -9999)
INVALID_VS30_PLACEHOLDER = _config.get("invalid_vs30_placeholder", -9999)

# filenames and Prefixes
POSTERIOR_PREFIX = _config.get("posterior_prefix", "posterior_")
TERRAIN_INITIAL_VS30_FILENAME = _config.get(
    "terrain_initial_vs30_filename", "terrain_initial_vs30_with_uncertainty.tif"
)
GEOLOGY_INITIAL_VS30_FILENAME = _config.get(
    "geology_initial_vs30_filename", "geology_initial_vs30_with_uncertainty.tif"
)
SLOPE_RASTER_FILENAME = _config.get("slope_raster_filename", "slope.tif")
COAST_DISTANCE_RASTER_FILENAME = _config.get(
    "coast_distance_raster_filename", "coast_distance.tif"
)
HYBRID_VS30_FILENAME = _config.get(
    "hybrid_vs30_filename",
    "geology_vs30_slope_and_coastal_distance_adjusted_with_uncertainty.tif",
)
GEOLOGY_ID_FILENAME = _config.get("geology_id_filename", "gid.tif")
TERRAIN_ID_FILENAME = _config.get("terrain_id_filename", "tid.tif")
COMBINED_VS30_FILENAME = _config.get(
    "combined_vs30_filename",
    "slope_coastal_distance_adjusted_geology_and_terrain_vs30_weighted_mixture_with_uncertainty.tif",
)

# Hybrid Model constants
HYBRID_MOD6_DIST_MIN = _config.get("hybrid_mod6_dist_min", 8000.0)
HYBRID_MOD6_DIST_MAX = _config.get("hybrid_mod6_dist_max", 20000.0)
HYBRID_MOD6_VS30_MIN = _config.get("hybrid_mod6_vs30_min", 240.0)
HYBRID_MOD6_VS30_MAX = _config.get("hybrid_mod6_vs30_max", 500.0)

HYBRID_MOD13_DIST_MIN = _config.get("hybrid_mod13_dist_min", 8000.0)
HYBRID_MOD13_DIST_MAX = _config.get("hybrid_mod13_dist_max", 20000.0)
HYBRID_MOD13_VS30_MIN = _config.get("hybrid_mod13_vs30_min", 197.0)
HYBRID_MOD13_VS30_MAX = _config.get("hybrid_mod13_vs30_max", 500.0)

# Misc
MIN_DIST_ENFORCED = _config.get("min_dist_enforced", 0.1)
NZTM_CRS = _config.get("nztm_crs", "EPSG:2193")

# Plotting
PLOT_FIGSIZE = _config.get("plot_figsize", [12, 8])
PLOT_DPI = _config.get("plot_dpi", 300)

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

MODEL_ID_COLUMNS = _config.get("model_id_columns", {"geology": "gid", "terrain": "tid"})

# Raster filenames used in initial steps
GEOLOGY_OR_TERRAIN_TO_RASTER_FILENAME = _config.get(
    "geology_or_terrain_to_raster_filename",
    {"geology": "geology.tif", "terrain": "terrain.tif"},
)

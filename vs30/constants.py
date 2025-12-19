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

# Correlation Lengths (Phi)
PHI_GEOLOGY = _config.get("phi_geology", 1407)
PHI_TERRAIN = _config.get("phi_terrain", 993)

# Paths and Filenames
OBSERVATIONS_FILE = _config.get(
    "observations_file", "observations/measured_vs30_original_filtered.csv"
)
OUTPUT_DIR_NAME = _config.get("output_dir", "new_vs30map")

TERRAIN_VS30_MEAN_STDDEV_FILENAME = _config.get(
    "terrain_vs30_mean_stddev_filename", "terrain_vs30_mean_stddev.tif"
)
GEOLOGY_VS30_MEAN_STDDEV_FILENAME = _config.get(
    "geology_vs30_mean_stddev_filename", "geology_vs30_mean_stddev.tif"
)

GEOLOGY_MEAN_STDDEV_CSV = _config.get(
    "geology_mean_and_standard_deviation_per_category_file",
    "categorical_vs30_mean_and_stddev/geology/geology_model_prior_mean_and_standard_deviation.csv",
)
TERRAIN_MEAN_STDDEV_CSV = _config.get(
    "terrain_mean_and_standard_deviation_per_category_file",
    "categorical_vs30_mean_and_stddev/terrain/terrain_model_prior_mean_and_standard_deviation.csv",
)

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

MODEL_ID_COLUMNS = {"geology": "gid", "terrain": "tid"}

# Raster filenames used in initial steps
GEOLOGY_OR_TERRAIN_TO_RASTER_FILENAME = _config.get(
    "geology_or_terrain_to_raster_filename",
    {"geology": "geology.tif", "terrain": "terrain.tif"},
)

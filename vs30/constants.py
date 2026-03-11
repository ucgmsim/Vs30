"""Scientific and algorithmic constants for Vs30 calculations."""

from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path


class CombinationMethod(StrEnum):
    """Valid combination methods for geology and terrain Vs30."""

    STANDARD_DEVIATION_WEIGHTING = "standard_deviation_weighting"
    RATIO = "ratio"


class FixedModelVersion(StrEnum):
    """Identifiers for fixed versions of the geology and terrain models."""

    FOSTER_2019 = "foster_2019"
    JAEHWI_V1P0 = "jaehwi_v1p0"
    VIKTOR_CPT_CLUSTERING = "viktor_cpt_clustering"


CONFIGS_DIR = Path(__file__).parent / "configs"

MODEL_VERSION_TO_CONFIG = {
    FixedModelVersion.FOSTER_2019: CONFIGS_DIR / "foster_2019.yaml",
    FixedModelVersion.JAEHWI_V1P0: CONFIGS_DIR / "jaehwi_v1p0.yaml",
    FixedModelVersion.VIKTOR_CPT_CLUSTERING: CONFIGS_DIR / "viktor_cpt_clustering.yaml",
}

# Path to the data directory containing shapefiles, rasters, and other input data
DATA_DIR = Path(__file__).parent / "data"

# Path to the resources directory containing CSV files with categorical model parameters
RESOURCE_PATH = Path(__file__).parent / "resources"

# Covariance reduction factor for dissimilar Vs30 values (dimensionless).
# Controls how much the correlation between two points is reduced when their
# model Vs30 values differ. Higher values = more reduction for dissimilar values.
COV_REDUC: float = 1.5

# Correlation length parameters (phi) in meters.
# Phi represents the distance at which spatial correlation decays to ~37% (1/e).
# Larger phi = smoother spatial interpolation, smaller phi = more localized updates.
# These values were calibrated for New Zealand geology and terrain data.
PHI_GEOLOGY: int = 1407
PHI_TERRAIN: int = 993

# Dictionary for convenient access by model type
# Note: PHI uses string keys for now; will be updated after ModelType is defined below.

# Minimum distance (meters) enforced in correlation calculations to prevent
# division by zero or correlation=1 when points are exactly co-located.
# The correlation function uses exp(-distance/phi), so distance=0 gives correlation=1.
MIN_DIST_ENFORCED: float = 0.1

# Maximum distance (meters) for considering observations in multivariate normal
# (MVN) spatial adjustment. Observations further than this distance from a pixel
# will not influence its update.
MAX_DIST_M: int = 10000

# Maximum number of observations to consider per pixel during multivariate normal
# (MVN) spatial adjustment. If a pixel is within MAX_DIST_M of more than MAX_POINTS
# observations, only the MAX_POINTS closest observations will be considered.
MAX_POINTS: int = 500

# K value for standard deviation based weighting when combining geology and
# terrain models. Represents the exponent for inverse variance weighting:
# weight ~ (sigma^2)^-k. Only used when combination_method is "standard_deviation_weighting".
K_VALUE: float = 3.0

# Small epsilon value added to variance when computing inverse-variance weights
# for combining geology and terrain models. Prevents division by zero when
# standard deviation is exactly zero.
WEIGHT_EPSILON_DIV_BY_ZERO: float = 1.0e-10

# Assumed initial number of prior observations (n0 in Bayesian formulas)
# for Bayesian update of Vs30 mean and standard deviation values
# for each geology and terrain category.
N_PRIOR: int = 3

# Minimum standard deviation (log-space) allowed after Bayesian update.
# Prevents over-confidence when many observations are available.
MIN_SIGMA: float = 0.5

# DBSCAN clustering parameters for spatially clustered observations
# such as Vs30 inferred from dense CPT measurements.

# Minimum number of observations to form a cluster (DBSCAN min_samples parameter)
MIN_GROUP: int = 5

# Maximum distance (meters) between observations to be in the same cluster
# (DBSCAN epsilon parameter). Points further apart will be in separate clusters.
EPS: float = 15000.0

# INTERNAL DATA FILES
# Filenames for input data files bundled with the package.
# These are relative to the vs30/data directory.

# Terrain classification raster (IwahashiPike terrain categories)
TERRAIN_RASTER_FILENAME: str = "IwahashiPike.tif"

# Geology shapefile (QMAP geology polygons)
GEOLOGY_SHAPEFILE_PATH: str = "qmap/qmap.shp"

# Coastline shapefile for coastal distance calculations
COASTLINE_SHAPEFILE_PATH: str = (
    "coast/nz-coastlines-and-islands-polygons-topo-1500k.shp"
)

# Source slope raster (used if slope.tif needs to be generated)
SLOPE_SOURCE_RASTER_FILENAME: str = "slope.tif"

# Archive containing shapefiles (extracted on first use)
SHAPEFILES_ARCHIVE_FILENAME: str = "shapefiles.tar.xz"

# Prefix used to indicate that Bayesian updates have been performed
POSTERIOR_PREFIX: str = "posterior_"

# Initial Vs30 map constructed from terrain classifications
TERRAIN_INITIAL_VS30_FILENAME: str = "initial_terrain_vs30_with_uncertainty.tif"

# Initial Vs30 map constructed from geology classifications
GEOLOGY_INITIAL_VS30_FILENAME: str = "initial_geology_vs30_with_uncertainty.tif"

# Slopes used for hybrid geology Vs30 model
SLOPE_RASTER_FILENAME: str = "slope.tif"

# Coastal distance used for hybrid geology Vs30 model
COAST_DISTANCE_RASTER_FILENAME: str = "coast_distance.tif"

# Geology Vs30 map adjusted for slope and coastal distance
GEOLOGY_VS30_SLOPE_AND_COASTAL_DISTANCE_ADJUSTED_FILENAME: str = (
    "geology_vs30_slope_and_coastal_distance_adjusted_with_uncertainty.tif"
)

# Map showing the geology category ID for each pixel
GEOLOGY_ID_FILENAME: str = "gid.tif"

# Map showing the terrain category ID for each pixel
TERRAIN_ID_FILENAME: str = "tid.tif"

# Final terrain Vs30 after spatial adjustment
TERRAIN_VS30_MEAN_STDDEV_FILENAME: str = (
    "terrain_vs30_spatially_adjusted_with_uncertainty.tif"
)

# Final geology Vs30 after slope, coastal distance, and spatial adjustment
GEOLOGY_VS30_MEAN_STDDEV_FILENAME: str = "geology_vs30_slope_and_coastal_distance_and_spatially_adjusted_with_uncertainty.tif"

# Combined weighted average of geology and terrain Vs30
COMBINED_VS30_FILENAME: str = "combined_vs30.tif"

# OUTPUT_FILENAMES and PHI dictionaries are defined after ModelType class below

# HYBRID GEOLOGY Vs30 MODEL PARAMETERS
# (Adjusts according to slope and coastal distance)

HYBRID_MOD6_DIST_MIN: float = 8000.0
HYBRID_MOD6_DIST_MAX: float = 20000.0
HYBRID_MOD6_VS30_MIN: float = 240.0
HYBRID_MOD6_VS30_MAX: float = 500.0

HYBRID_MOD13_DIST_MIN: float = 8000.0
HYBRID_MOD13_DIST_MAX: float = 20000.0
HYBRID_MOD13_VS30_MIN: float = 197.0
HYBRID_MOD13_VS30_MAX: float = 500.0


@dataclass
class HybridVs30Param:
    """
    Parameters for slope-based Vs30 interpolation per geology group.

    Attributes
    ----------
    gid : int
        Geology group ID.
    slope_limits : list[float]
        Log10(slope) limits for interpolation [min, max].
    vs30_values : list[float]
        Vs30 values (m/s) at the slope limits [at_min_slope, at_max_slope].
    """

    gid: int
    slope_limits: list[float]
    vs30_values: list[float]


# Hybrid slope-based Vs30 interpolation parameters
# Each entry contains: geology group ID, log10(slope) limits, Vs30 values
HYBRID_VS30_PARAMS: list[HybridVs30Param] = [
    HybridVs30Param(gid=2, slope_limits=[-1.85, -1.22], vs30_values=[242, 418]),
    HybridVs30Param(gid=3, slope_limits=[-2.70, -1.35], vs30_values=[171, 228]),
    HybridVs30Param(gid=4, slope_limits=[-3.44, -0.88], vs30_values=[252, 275]),
    HybridVs30Param(gid=6, slope_limits=[-3.56, -0.93], vs30_values=[183, 239]),
]

# Hybrid standard deviation reduction factors for specific geology groups
# Maps geology group ID to sigma reduction factor
HYBRID_SIGMA_REDUCTION_FACTORS: dict[int, float] = {
    2: 0.4888,
    3: 0.7103,
    4: 0.9988,
    6: 0.9348,
}

# Minimum slope value used to prevent log10(0) when calculating hybrid Vs30
MIN_SLOPE_FOR_LOG: float = 1.0e-9

# No data value in the provided categorical rasters
RASTER_ID_NODATA_VALUE: int = 255

# No-data value used for slope rasters, model outputs, and CSV placeholders
# -32767 is a traditional no-data value in many GIS applications
# (minimum value for signed 16-bit integers)
NODATA_VALUE: int = -32767

# FULL NEW ZEALAND LAND EXTENT BOUNDS
# (For coastal distance calculations)
# IMPORTANT: These values define the full extent of New Zealand land coverage
# and MUST NOT be changed. They are used to ensure coastal distance calculations
# are computed on the full NZ land extent, regardless of the configured study
# domain bounds.
FULL_NZ_LAND_XMIN: int = 1060050
FULL_NZ_LAND_XMAX: int = 2120050
FULL_NZ_LAND_YMIN: int = 4730050
FULL_NZ_LAND_YMAX: int = 6250050

# Default column names for longitude and latitude in location input CSV files.
LOCATIONS_LON_COLUMN: str = "longitude"
LOCATIONS_LAT_COLUMN: str = "latitude"

# Number of chunks to split locations into for parallel processing progress updates.
N_PROGRESS_CHUNKS: int = 1000

# Column names used in categorical model DataFrames for Bayesian updates.
# These are used to identify posterior/prior values at different stages.
COL_POSTERIOR_MEAN_INDEPENDENT: str = (
    "posterior_mean_vs30_km_per_s_independent_observations"
)
COL_POSTERIOR_STDV_INDEPENDENT: str = (
    "posterior_standard_deviation_vs30_km_per_s_independent_observations"
)
COL_POSTERIOR_NOBS_INDEPENDENT: str = (
    "posterior_num_observations_independent_observations"
)
COL_POSTERIOR_MEAN_CLUSTERED: str = (
    "posterior_mean_vs30_km_per_s_clustered_observations"
)
COL_POSTERIOR_STDV_CLUSTERED: str = (
    "posterior_standard_deviation_vs30_km_per_s_clustered_observations"
)
COL_POSTERIOR_MEAN: str = "posterior_mean_vs30_km_per_s"
COL_POSTERIOR_STDV: str = "posterior_standard_deviation_vs30_km_per_s"
COL_PRIOR_MEAN: str = "prior_mean_vs30_km_per_s"
COL_PRIOR_STDV: str = "prior_standard_deviation_vs30_km_per_s"
COL_MEAN: str = "mean_vs30_km_per_s"
COL_STDV: str = "standard_deviation_vs30_km_per_s"
COL_ASSUMED_NUM_PRIOR_OBS: str = "assumed_num_prior_observations"
COL_ENFORCED_MIN_SIGMA: str = "enforced_min_sigma"

# Standard column name for category ID in DataFrames.
STANDARD_ID_COLUMN: str = "id"

# Coordinate Reference System for New Zealand Transverse Mercator 2000
NZTM_CRS: str = "EPSG:2193"

PLOT_FIGSIZE: list[int] = [12, 8]
PLOT_DPI: int = 300

# OBSERVATION DATA COLUMN NAMES
# Standard column names for observation DataFrames used throughout the package.

COL_EASTING: str = "easting"
COL_NORTHING: str = "northing"
COL_VS30: str = "vs30"
COL_UNCERTAINTY: str = "uncertainty"
COL_CLUSTER: str = "cluster"

# Column names for the DataFrame returned by get_vs30_for_ids,
# which maps category IDs to their categorical model Vs30 values.
COL_CATEGORY_VS30_MEAN: str = "category_vs30_mean"
COL_CATEGORY_VS30_STDV: str = "category_vs30_stdv"

# Cluster label for unclustered/noise points in DBSCAN output
CLUSTER_UNCLUSTERED_LABEL: int = -1

# OUTPUT CSV COLUMN NAMES
# Column names for compute-at-locations output CSV files.

COL_GEOLOGY_ID: str = "geology_id"
COL_GEOLOGY_VS30: str = "geology_vs30"
COL_GEOLOGY_STDV: str = "geology_stdv"
COL_GEOLOGY_VS30_HYBRID: str = "geology_vs30_hybrid"
COL_GEOLOGY_STDV_HYBRID: str = "geology_stdv_hybrid"
COL_GEOLOGY_MVN_VS30: str = "geology_mvn_vs30"
COL_GEOLOGY_MVN_STDV: str = "geology_mvn_stdv"
COL_TERRAIN_ID: str = "terrain_id"
COL_TERRAIN_VS30: str = "terrain_vs30"
COL_TERRAIN_STDV: str = "terrain_stdv"
COL_TERRAIN_MVN_VS30: str = "terrain_mvn_vs30"
COL_TERRAIN_MVN_STDV: str = "terrain_mvn_stdv"
COL_COMBINED_STDV: str = "stdv"

# PARALLEL PROCESSING DICTIONARY KEYS
# Keys used in dictionaries for multiprocessing data transfer.

KEY_LOCATIONS: str = "locations"
KEY_MODEL_VS30: str = "model_vs30"
KEY_MODEL_STDV: str = "model_stdv"
KEY_RESIDUALS: str = "residuals"
KEY_OMEGA: str = "omega"
KEY_LOCATION: str = "location"
KEY_STDV: str = "stdv"
KEY_INDEX: str = "index"
KEY_MODEL_TYPE: str = "model_type"
KEY_MAX_DIST_M: str = "max_dist_m"
KEY_MAX_POINTS: str = "max_points"
KEY_NOISY: str = "noisy"
KEY_COV_REDUC: str = "cov_reduc"
KEY_CORR_ZERO: str = "corr_zero"

# Required columns for observation DataFrames
REQUIRED_OBSERVATION_COLUMNS: list[str] = [
    COL_EASTING,
    COL_NORTHING,
    COL_VS30,
    COL_UNCERTAINTY,
]
REQUIRED_OBSERVATION_COLUMNS_BASIC: list[str] = [
    COL_EASTING,
    COL_NORTHING,
    COL_VS30,
]

# MODEL TYPE IDENTIFIERS
# String identifiers for the two model types used in the Vs30 pipeline.


class ModelType(StrEnum):
    """Valid model types for VS30 calculations."""

    GEOLOGY = "geology"
    TERRAIN = "terrain"
    COMBINED = "combined"


# Dictionaries for convenient access by model type
PHI: dict[ModelType, int] = {
    ModelType.GEOLOGY: PHI_GEOLOGY,
    ModelType.TERRAIN: PHI_TERRAIN,
}

OUTPUT_FILENAMES: dict[ModelType, str] = {
    ModelType.GEOLOGY: GEOLOGY_VS30_MEAN_STDDEV_FILENAME,
    ModelType.TERRAIN: TERRAIN_VS30_MEAN_STDDEV_FILENAME,
}

# RASTER BAND INDICES
# Band numbers for multi-band VS30 rasters (1-indexed as per rasterio convention).

RASTER_BAND_VS30: int = 1
RASTER_BAND_STDV: int = 2

# GEOTIFF OPTIONS
# Standard options for writing GeoTIFF raster files.

GEOTIFF_DRIVER: str = "GTiff"
GEOTIFF_COMPRESSION: str = "deflate"
GEOTIFF_TILED: bool = True
GEOTIFF_BIGTIFF: str = "yes"

# RASTER BAND DESCRIPTIONS
# Standard descriptions for raster bands.

BAND_DESCRIPTION_ID_INDEX: str = "Model ID Index"
BAND_DESCRIPTION_VS30: str = "Vs30"
BAND_DESCRIPTION_STDV: str = "Standard Deviation"
BAND_DESCRIPTION_VS30_HYBRID: str = "Vs30 (Hybrid)"
BAND_DESCRIPTION_STDV_HYBRID: str = "Standard Deviation (Hybrid)"
BAND_DESCRIPTION_VS30_COMBINED: str = "Vs30 (Combined Average)"
BAND_DESCRIPTION_STDV_COMBINED: str = "Standard Deviation (Combined Average)"
BAND_DESCRIPTION_COAST_DISTANCE: str = "Distance to Coast (m)"
BAND_DESCRIPTION_SLOPE: str = "Slope"

# SHAPEFILE COLUMN NAMES
# Column names used in input shapefiles.

SHAPEFILE_GEOLOGY_ID_COLUMN: str = "gid"
SHAPEFILE_GEOMETRY_COLUMN: str = "geometry"

# SPATIAL PROCESSING CONSTANTS
# Constants used in spatial coordinate and pixel calculations.

# Offset to convert pixel indices to pixel centers (0.5 = center of pixel)
PIXEL_CENTER_OFFSET: float = 0.5

# PLOT STYLING CONSTANTS
# Standard styling parameters for matplotlib plots.

PLOT_X_OFFSET: float = 0.2
PLOT_ERRORBAR_CAPSIZE: int = 5
PLOT_ERRORBAR_CAPTHICK: float = 1.5
PLOT_MARKER_SIZE: int = 6
PLOT_ALPHA: float = 0.7
PLOT_GRID_ALPHA: float = 0.3
PLOT_LABEL_FONTSIZE: int = 12
PLOT_TITLE_FONTSIZE: int = 14
PLOT_LEGEND_FONTSIZE: int = 11

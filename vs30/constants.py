"""Scientific and algorithmic constants for Vs30 calculations."""

from dataclasses import dataclass, field
from enum import StrEnum
from pathlib import Path

import numpy as np

from vs30 import config


class CombinationMethod(StrEnum):
    """Valid combination methods for geology and terrain Vs30."""

    STANDARD_DEVIATION_WEIGHTING = "standard_deviation_weighting"
    RATIO = "ratio"


class FixedModelVersion(StrEnum):
    """Available fixed versions of the Vs30 model."""

    FOSTER_2019_APPROX = "foster_2019_approx"
    MODIFIED_FOSTER_2019 = "modified_foster_2019"
    VIKTOR_CPT_CLUSTERING = "viktor_cpt_clustering"
    JAEHWI_V1P0 = "jaehwi_v1p0"


CONFIGS_DIR = Path(__file__).parent / "configs"

MODEL_VERSION_TO_CONFIG = {
    FixedModelVersion.FOSTER_2019_APPROX: CONFIGS_DIR / "foster_2019_approx.yaml",
    FixedModelVersion.MODIFIED_FOSTER_2019: CONFIGS_DIR / "modified_foster_2019.yaml",
    FixedModelVersion.JAEHWI_V1P0: CONFIGS_DIR / "jaehwi_v1p0.yaml",
    FixedModelVersion.VIKTOR_CPT_CLUSTERING: CONFIGS_DIR / "viktor_cpt_clustering.yaml",
}

# Path to the geospatial directory containing shapefiles, rasters, and other input data
GEOSPATIAL_DIR = Path(__file__).parent / "resources" / "geospatial"

# Path to the resources directory containing CSV files with categorical model parameters
RESOURCE_PATH = Path(__file__).parent / "resources"

# Config YAML keys that hold CSV file paths, mapped to their subdirectory
# under RESOURCE_PATH.
RESOURCE_SUBDIRS: dict[str, str] = {
    "geology_categorical_csv": "categorical_vs30_mean_and_stddev",
    "terrain_categorical_csv": "categorical_vs30_mean_and_stddev",
    "clustered_observations_csv": "observations",
    "independent_observations_csv": "observations",
}

# Covariance reduction factor for dissimilar Vs30 values (dimensionless).
# Controls how much the correlation between two points is reduced when their
# model Vs30 values differ. Higher values = more reduction for dissimilar values.
COV_REDUC: float = 1.5

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
# These are relative to the vs30/resources/geospatial directory.

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

# Combined VS30 output before gap-fill (intermediate output)
COMBINED_VS30_BEFORE_GAPFILL_FILENAME: str = "combined_vs30_before_gapfill.tif"

# Default correlation length parameters (meters) from the Foster 2019 model.
DEFAULT_GEOLOGY_PHI = 1407
DEFAULT_TERRAIN_PHI = 993

# HYBRID GEOLOGY Vs30 MODEL PARAMETERS
# (Adjusts according to slope and coastal distance)

HYBRID_GID4_DIST_MIN: float = 8000.0
HYBRID_GID4_DIST_MAX: float = 20000.0
HYBRID_GID4_VS30_MIN: float = 240.0
HYBRID_GID4_VS30_MAX: float = 500.0

HYBRID_GID10_DIST_MIN: float = 8000.0
HYBRID_GID10_DIST_MAX: float = 20000.0
HYBRID_GID10_VS30_MIN: float = 197.0
HYBRID_GID10_VS30_MAX: float = 500.0


@dataclass
class HybridGeologyParams:
    """
    Per-geology-group parameters for hybrid Vs30 modifications.

    Attributes
    ----------
    gid : int
        Geology group ID.
    slope_limits : list[float]
        Log10(slope) limits for interpolation [min, max].
    vs30_values : list[float]
        Vs30 values (m/s) at the slope limits [at_min_slope, at_max_slope].
    sigma_reduction : float
        Multiplicative reduction factor applied to the prior standard
        deviation for this group.
    vs30_values_log10 : ndarray
        Precomputed ``np.log10`` of ``vs30_values``, used in the slope
        interpolation hot path.
    """

    gid: int
    slope_limits: list[float]
    vs30_values: list[float]
    sigma_reduction: float
    vs30_values_log10: np.ndarray = field(init=False)

    def __post_init__(self):
        self.vs30_values_log10 = np.log10(np.array(self.vs30_values))


HYBRID_GEOLOGY_PARAMS: list[HybridGeologyParams] = [
    HybridGeologyParams(
        gid=2,
        slope_limits=[-1.85, -1.22],
        vs30_values=[242, 418],
        sigma_reduction=0.4888,
    ),
    HybridGeologyParams(
        gid=3,
        slope_limits=[-2.70, -1.35],
        vs30_values=[171, 228],
        sigma_reduction=0.7103,
    ),
    HybridGeologyParams(
        gid=4,
        slope_limits=[-3.44, -0.88],
        vs30_values=[252, 275],
        sigma_reduction=0.9988,
    ),
    HybridGeologyParams(
        gid=6,
        slope_limits=[-3.56, -0.93],
        vs30_values=[183, 239],
        sigma_reduction=0.9348,
    ),
]

# Minimum slope value used to prevent log10(0) when calculating hybrid Vs30
MIN_SLOPE_FOR_LOG: float = 1.0e-9

# Sentinel slope value used when sampling slope at observation locations at
# NODATA pixels. The legacy model's interpolate_raster converts tif-NODATA to
# ID_NODATA=255, and the downstream hybrid calculation does not catch 255 in
# its NODATA check, so log10(255) ≈ 2.41 is used in the slope interpolation.
# Preserved here so the refactored code reproduces the legacy behaviour at
# observations while still using MIN_SLOPE_FOR_LOG for grid pixels.
LEGACY_OBS_SLOPE_NODATA_SENTINEL: float = 255.0

# No data value in the provided categorical rasters
RASTER_ID_NODATA_VALUE: int = 255

# No-data value used for slope rasters, model outputs, and CSV placeholders
# -32767 is a traditional no-data value in many GIS applications
# (minimum value for signed 16-bit integers)
NODATA_VALUE: int = -32767

# Full New Zealand land extent at standard 100m resolution.
# IMPORTANT: These bounds define the canonical NZ domain and MUST NOT be changed.
# Used for coastal distance calculations, gap-fill grid alignment, and CLI defaults.
FULL_NZ_GRID_CONFIG: config.GridConfig = config.GridConfig(
    grid_xmin=1060050,
    grid_xmax=2120050,
    grid_ymin=4730050,
    grid_ymax=6250050,
    grid_dx=100,
    grid_dy=100,
)

# Gap-fill constants
# Half-width (meters) of the local grid generated around each fillable point
# in the points pipeline. A value of 5000 gives a 10 km x 10 km local grid.
GAPFILL_LOCAL_GRID_SIZE_M: int = 5000

# Amount (meters) to expand the local grid half-width if the initial local
# grid has no valid donor pixels for a fillable point.
GAPFILL_LOCAL_GRID_EXPANSION_M: int = 5000

# Maximum half-width (meters) for local grid expansion. Prevents unbounded
# growth if a fillable point has no valid donors nearby.
GAPFILL_MAX_LOCAL_GRID_HALF_WIDTH_M: int = 50000

# Default memory limit (GB) for spatial boolean arrays used during MVN chunking.
MAX_SPATIAL_BOOLEAN_ARRAY_MEMORY_GB: float = 1.0

# Default column names for longitude and latitude in location input CSV files.
LOCATIONS_LON_COLUMN: str = "longitude"
LOCATIONS_LAT_COLUMN: str = "latitude"

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

# Default encoding for reading text files (e.g. YAML configs).
DEFAULT_TEXT_ENCODING: str = "utf-8"


class ObservationColumn:
    """Standard column names for observation DataFrames used throughout the package."""

    EASTING = "easting"
    NORTHING = "northing"
    VS30 = "vs30"
    UNCERTAINTY = "uncertainty"
    CLUSTER = "cluster"

    REQUIRED = [EASTING, NORTHING, VS30, UNCERTAINTY]


# Cluster label for unclustered/noise points in DBSCAN output
CLUSTER_UNCLUSTERED_LABEL: int = -1

# Column names for `points` command output CSV files.
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
COL_VS30_BEFORE_GAPFILL: str = "vs30_before_gapfill"
COL_STDV_BEFORE_GAPFILL: str = "stdv_before_gapfill"


class ModelType(StrEnum):
    """For specifying whether output should be generated using the geology model only, the
    terrain model only, or combination of both models."""

    GEOLOGY = "geology"
    TERRAIN = "terrain"
    COMBINED = "combined"


OUTPUT_FILENAMES: dict[ModelType, str] = {
    ModelType.GEOLOGY: GEOLOGY_VS30_MEAN_STDDEV_FILENAME,
    ModelType.TERRAIN: TERRAIN_VS30_MEAN_STDDEV_FILENAME,
}

# Options for writing GeoTIFF raster files.
GEOTIFF_DRIVER: str = "GTiff"
GEOTIFF_COMPRESSION: str = "deflate"

# Descriptions of raster bands.
BAND_DESCRIPTION_ID_INDEX: str = "Model ID Index"
BAND_DESCRIPTION_VS30: str = "Vs30"
BAND_DESCRIPTION_STDV: str = "Standard Deviation"
BAND_DESCRIPTION_VS30_HYBRID: str = "Vs30 (Hybrid)"
BAND_DESCRIPTION_STDV_HYBRID: str = "Standard Deviation (Hybrid)"
BAND_DESCRIPTION_VS30_COMBINED: str = "Vs30 (Combined Average)"
BAND_DESCRIPTION_STDV_COMBINED: str = "Standard Deviation (Combined Average)"
BAND_DESCRIPTION_COAST_DISTANCE: str = "Distance to Coast (m)"
BAND_DESCRIPTION_SLOPE: str = "Slope"

# Column names used in input shapefiles.
SHAPEFILE_GEOLOGY_ID_COLUMN: str = "gid"
SHAPEFILE_GEOMETRY_COLUMN: str = "geometry"

# Offset to convert pixel indices to pixel centers (0.5 = center of pixel)
# in spatial coordinate and pixel calculations.
PIXEL_CENTER_OFFSET: float = 0.5

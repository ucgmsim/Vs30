"""Categorical VS30 raster creation and hybrid geology modifications."""

import logging
import os
import tarfile
import tempfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import rasterio.enums
import rasterio.features
import rasterio.transform
import rasterio.warp
import shapely
from osgeo import gdal
from tqdm import tqdm

from vs30 import constants

logger = logging.getLogger(__name__)


def ensure_shapefile_extracted(shapefile_path: Path, directory_prefix: str) -> None:
    """
    Ensure a shapefile is extracted from shapefiles.tar.xz.

    Checks if the shapefile exists. If not, extracts it from shapefiles.tar.xz.
    This is needed because shapefiles are stored compressed in the archive and
    may not be present by default.

    Parameters
    ----------
    shapefile_path : Path
        Full path to the shapefile to check/extract.
    directory_prefix : str
        Directory prefix within the archive (e.g., "qmap" or "coast").

    Raises
    ------
    FileNotFoundError
        If shapefiles.tar.xz is not found or extraction fails.
    ValueError
        If the directory is not found in the archive.
    """
    if shapefile_path.exists():
        return

    archive_path = constants.GEOSPATIAL_DIR / constants.SHAPEFILES_ARCHIVE_FILENAME
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Shapefile archive not found: {archive_path}. "
            f"Cannot extract {shapefile_path.name}. Please ensure shapefiles.tar.xz exists."
        )

    with tarfile.open(archive_path, "r:xz") as tar:
        members = [
            member
            for member in tar.getmembers()
            if member.name.startswith(f"{directory_prefix}/")
        ]
        if not members:
            raise ValueError(
                f"No '{directory_prefix}' directory found in archive {archive_path}"
            )
        tar.extractall(path=constants.GEOSPATIAL_DIR, members=members)

    if not shapefile_path.exists():
        raise FileNotFoundError(
            f"Failed to extract {shapefile_path.name} from {archive_path}. "
            f"Expected file at {shapefile_path} but it was not created."
        )


def create_category_id_array(
    model_type: constants.ModelType,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, dict]:
    """
    Create category ID array for terrain or geology in memory.

    For terrain: Resamples IwahashiPike.tif to target grid using reprojection.
    For geology: Rasterizes qmap.shp shapefile to target grid.

    Parameters
    ----------
    model_type : constants.ModelType
        Either ModelType.TERRAIN or ModelType.GEOLOGY.
    xmin : float
        Grid minimum easting (m, NZTM2000).
    xmax : float
        Grid maximum easting (m, NZTM2000).
    ymin : float
        Grid minimum northing (m, NZTM2000).
    ymax : float
        Grid maximum northing (m, NZTM2000).
    dx : float
        Grid cell width (m).
    dy : float
        Grid cell height (m).

    Returns
    -------
    tuple[np.ndarray, dict]
        A tuple containing:
        - The category ID array (uint8).
        - The rasterio profile dict describing the array's spatial properties.

    Raises
    ------
    ValueError
        If model_type is not a valid ModelType.
    FileNotFoundError
        If input files don't exist.
    """
    if model_type not in constants.ModelType:
        raise ValueError(f"model_type must be a valid ModelType, got '{model_type}'")

    # Common setup: calculate grid dimensions and transform
    nx = round((xmax - xmin) / dx)
    ny = round((ymax - ymin) / dy)
    dst_transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, nx, ny)

    # Common output raster profile
    profile = {
        "driver": constants.GEOTIFF_DRIVER,
        "width": nx,
        "height": ny,
        "count": 1,
        "dtype": "uint8",
        "crs": constants.NZTM_CRS,
        "transform": dst_transform,
        "nodata": constants.RASTER_ID_NODATA_VALUE,
        "compress": constants.GEOTIFF_COMPRESSION,
    }

    if model_type == constants.ModelType.TERRAIN:
        # Resample terrain raster to target grid
        terrain_raster_path = (
            constants.GEOSPATIAL_DIR / constants.TERRAIN_RASTER_FILENAME
        )
        if not terrain_raster_path.exists():
            raise FileNotFoundError(f"Terrain raster not found: {terrain_raster_path}")

        # Reproject into a numpy destination array
        id_array = np.full((ny, nx), constants.RASTER_ID_NODATA_VALUE, dtype=np.uint8)
        with rasterio.open(terrain_raster_path) as src:
            rasterio.warp.reproject(
                source=rasterio.band(src, 1),
                destination=id_array,
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=dst_transform,
                dst_crs=constants.NZTM_CRS,
                resampling=rasterio.enums.Resampling.nearest,
            )

    else:  # geology
        # Ensure qmap.shp is extracted from shapefiles.tar.xz if needed
        ensure_shapefile_extracted(
            constants.GEOSPATIAL_DIR / constants.GEOLOGY_SHAPEFILE_PATH, "qmap"
        )

        # Rasterize geology shapefile to target grid
        geology_shapefile_path = (
            constants.GEOSPATIAL_DIR / constants.GEOLOGY_SHAPEFILE_PATH
        )
        if not geology_shapefile_path.exists():
            raise FileNotFoundError(
                f"Geology shapefile not found: {geology_shapefile_path}"
            )

        # Read shapefile
        gdf = gpd.read_file(geology_shapefile_path)
        if constants.SHAPEFILE_GEOLOGY_ID_COLUMN not in gdf.columns:
            raise ValueError(
                f"Shapefile {geology_shapefile_path} missing "
                f"'{constants.SHAPEFILE_GEOLOGY_ID_COLUMN}' column"
            )

        # Ensure shapefile is in NZTM CRS (EPSG:2193)
        if gdf.crs is None or str(gdf.crs) != constants.NZTM_CRS:
            gdf = gdf.to_crs(constants.NZTM_CRS)

        # Create shapes iterator for rasterization
        shapes = (
            (geom, value)
            for geom, value in zip(
                gdf.geometry, gdf[constants.SHAPEFILE_GEOLOGY_ID_COLUMN]
            )
        )

        # Rasterize to array
        id_array = rasterio.features.rasterize(
            shapes=shapes,
            out_shape=(ny, nx),
            transform=dst_transform,
            fill=constants.RASTER_ID_NODATA_VALUE,
            dtype=np.uint8,
            all_touched=False,
        )

    return id_array, profile


def create_category_id_raster(
    model_type: constants.ModelType,
    output_dir: Path,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
    dx: float,
    dy: float,
) -> Path:
    """
    Create category ID raster for terrain or geology.

    For terrain: Resamples IwahashiPike.tif to target grid.
    For geology: Rasterizes qmap.shp shapefile to target grid.

    Parameters
    ----------
    model_type : constants.ModelType
        Either ModelType.TERRAIN or ModelType.GEOLOGY.
    output_dir : Path
        Directory where output raster will be saved.
    xmin : float
        Grid minimum easting (m, NZTM2000).
    xmax : float
        Grid maximum easting (m, NZTM2000).
    ymin : float
        Grid minimum northing (m, NZTM2000).
    ymax : float
        Grid maximum northing (m, NZTM2000).
    dx : float
        Grid cell width (m).
    dy : float
        Grid cell height (m).

    Returns
    -------
    Path
        Path to created ID raster file.

    Raises
    ------
    ValueError
        If model_type is not a valid ModelType.
    FileNotFoundError
        If input files don't exist.
    """
    id_array, profile = create_category_id_array(
        model_type, xmin, xmax, ymin, ymax, dx, dy
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_filename = (
        constants.TERRAIN_ID_FILENAME
        if model_type == constants.ModelType.TERRAIN
        else constants.GEOLOGY_ID_FILENAME
    )
    output_path = output_dir / output_filename

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(id_array, 1)
        dst.descriptions = (constants.BAND_DESCRIPTION_ID_INDEX,)

    return output_path


def select_vs30_columns_by_priority(columns: list[str]) -> tuple[str, str]:
    """
    Determine which columns to use for VS30 mean and standard deviation.

    Prioritizes columns in the following order:
    1. Independent observations posterior (result of second update step)
    2. Clustered observations posterior (result of first update step)
    3. Generic posterior
    4. Explicit prior
    5. Standard/Original names

    Parameters
    ----------
    columns : list[str]
        List of available column names in the CSV.

    Returns
    -------
    tuple[str, str]
        (mean_column_name, std_column_name)

    Raises
    ------
    ValueError
        If no suitable column pair is found.
    """
    priorities = [
        # 1. Independent observations posterior
        (
            constants.COL_POSTERIOR_MEAN_INDEPENDENT,
            constants.COL_POSTERIOR_STDV_INDEPENDENT,
        ),
        # 2. Clustered observations posterior
        (
            constants.COL_POSTERIOR_MEAN_CLUSTERED,
            constants.COL_POSTERIOR_STDV_CLUSTERED,
        ),
        # 3. Generic posterior
        (constants.COL_POSTERIOR_MEAN, constants.COL_POSTERIOR_STDV),
        # 4. Explicit prior
        (constants.COL_PRIOR_MEAN, constants.COL_PRIOR_STDV),
        # 5. Standard/Original names
        (constants.COL_MEAN, constants.COL_STDV),
    ]

    for mean_col, std_col in priorities:
        if mean_col in columns and std_col in columns:
            return mean_col, std_col

    raise ValueError(
        f"Could not find valid VS30 mean and standard deviation columns. "
        f"Available columns: {columns}"
    )


def create_vs30_arrays_from_ids(
    id_array: np.ndarray,
    model_values_df: pd.DataFrame,
    model_type: constants.ModelType | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map category IDs to VS30 mean and standard deviation arrays in memory.

    Uses the priority-based column selection to find the best available VS30
    mean and standard deviation columns in the DataFrame, then maps each
    category ID to its corresponding values.

    Parameters
    ----------
    id_array : np.ndarray
        Category ID array (uint8) from terrain or geology rasterization.
    model_values_df : pd.DataFrame
        DataFrame containing category ID-to-VS30 mapping. Must have an 'id'
        column and at least one pair of mean/stdv columns recognized by
        ``select_vs30_columns_by_priority``.
    model_type : constants.ModelType or None, optional
        Model type label used in progress bar description. Default None.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        A tuple containing:
        - vs30_array (float32): VS30 mean values for each pixel.
        - stdv_array (float32): VS30 standard deviation values for each pixel.

    Raises
    ------
    ValueError
        If the DataFrame is missing required columns or an ID in the array
        is not found in the DataFrame.
    """
    # Strip whitespace from column names without copying the DataFrame
    stripped_columns = {c.strip(): c for c in model_values_df.columns}
    columns_list = list(stripped_columns.keys())

    mean_col, std_col = select_vs30_columns_by_priority(columns_list)

    if constants.STANDARD_ID_COLUMN not in columns_list:
        raise ValueError(
            f"DataFrame is missing required column: {constants.STANDARD_ID_COLUMN}"
        )

    # Map original column names through stripped lookup
    id_col_orig = stripped_columns[constants.STANDARD_ID_COLUMN]
    mean_col_orig = stripped_columns[mean_col]
    std_col_orig = stripped_columns[std_col]

    id_to_vs30_values = dict(
        zip(
            model_values_df[id_col_orig].astype(int),
            zip(
                model_values_df[mean_col_orig].astype(float),
                model_values_df[std_col_orig].astype(float),
            ),
        )
    )

    # Create output arrays
    vs30_array = np.full(id_array.shape, constants.NODATA_VALUE, dtype=np.float32)
    stdv_array = np.full(id_array.shape, constants.NODATA_VALUE, dtype=np.float32)

    # Map pixel IDs to VS30 values
    unique_ids = np.unique(id_array)
    valid_ids = unique_ids[
        (unique_ids != constants.RASTER_ID_NODATA_VALUE) & (unique_ids != 0)
    ]

    label = str(model_type).capitalize() if model_type else "Model"
    for pixel_id in tqdm(
        valid_ids, desc=f"{label}: mapping categories to Vs30", unit="ID"
    ):
        if pixel_id in id_to_vs30_values:
            mean_vs30, stddev_vs30 = id_to_vs30_values[pixel_id]
            mask = id_array == pixel_id
            vs30_array[mask] = mean_vs30
            stdv_array[mask] = stddev_vs30
        else:
            raise ValueError(
                f"ID {pixel_id} found in array but not in DataFrame. "
                f"Available IDs: {sorted(id_to_vs30_values.keys())}"
            )

    return vs30_array, stdv_array


def compute_coast_distance_array(template_profile: dict) -> np.ndarray:
    """
    Compute distance to the nearest coast (in meters) as an in-memory array.

    Uses GDAL to rasterize the coast shapefile and compute proximity distances,
    following the legacy implementation for numerical consistency. A temporary
    file is used internally because GDAL requires a file path for rasterization
    and proximity computation. The temporary file is cleaned up when done.

    Computes on full NZ land extent to ensure accurate distances for all
    observation locations, even those outside the configured study domain.

    Parameters
    ----------
    template_profile : dict
        Rasterio profile of the reference raster (to match resolution and bounds).

    Returns
    -------
    np.ndarray
        The distance array (float32) matching the template grid dimensions.
    """
    ensure_shapefile_extracted(
        constants.GEOSPATIAL_DIR / constants.COASTLINE_SHAPEFILE_PATH, "coast"
    )

    # Get template bounds for final output extent
    dx = template_profile["transform"].a
    dy = abs(template_profile["transform"].e)
    s_xmin = template_profile["transform"].c
    s_ymax = template_profile["transform"].f
    s_xmax = s_xmin + template_profile["width"] * dx
    s_ymin = s_ymax - template_profile["height"] * dy

    # Extend to full NZ land coverage to ensure accurate distances
    # (matching legacy _full_land_grid behavior)
    nz = constants.FULL_NZ_GRID_CONFIG
    g_xmin = min(nz.grid_xmin, s_xmin)
    g_xmax = max(nz.grid_xmax, s_xmax)
    g_ymin = min(nz.grid_ymin, s_ymin)
    g_ymax = max(nz.grid_ymax, s_ymax)

    # Check if grid was extended beyond template bounds (requires cropping later)
    grid_was_extended = (
        g_xmin < s_xmin or g_xmax > s_xmax or g_ymin < s_ymin or g_ymax > s_ymax
    )

    # GDAL requires a file path, so use a temporary file
    fd, tmp_path = tempfile.mkstemp(suffix=".tif")
    os.close(fd)

    try:
        # Rasterize land polygons using GDAL (legacy approach)
        # Use UInt16 data type as in legacy code (sufficient for distance range)
        ds = gdal.Rasterize(
            tmp_path,
            str(constants.GEOSPATIAL_DIR / constants.COASTLINE_SHAPEFILE_PATH),
            creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=YES"],
            outputBounds=[g_xmin, g_ymin, g_xmax, g_ymax],
            xRes=dx,
            yRes=dy,
            noData=0,
            burnValues=1,
            outputType=gdal.GetDataTypeByName("UInt16"),
        )

        # Compute proximity distances using GDAL (legacy approach)
        # DISTUNITS=GEO ensures distances in georeferenced units (meters)
        band = ds.GetRasterBand(1)
        band.SetDescription(constants.BAND_DESCRIPTION_COAST_DISTANCE)
        # Note: ComputeProximity modifies the raster in-place
        ds = gdal.ComputeProximity(band, band, ["VALUES=0", "DISTUNITS=GEO"])
        band = None
        ds = None

        # If grid was extended, crop back to template bounds
        if grid_was_extended:
            with rasterio.open(tmp_path) as src:
                extended_data = src.read(1)

            col_off = round((s_xmin - g_xmin) / dx)
            row_off = round((g_ymax - s_ymax) / dy)

            distance_meters = extended_data[
                row_off : row_off + template_profile["height"],
                col_off : col_off + template_profile["width"],
            ].astype(np.float32)
        else:
            with rasterio.open(tmp_path) as src:
                distance_meters = src.read(1).astype(np.float32)
    finally:
        # Clean up the temporary file
        Path(tmp_path).unlink(missing_ok=True)

    return distance_meters


def create_coast_distance_raster(
    output_path: Path, template_profile: dict
) -> tuple[np.ndarray, dict]:
    """
    Create a raster of distance to the nearest coast (in meters).

    Uses GDAL to rasterize coast shapefile and compute proximity distances,
    following the legacy implementation for numerical consistency.
    Computes on full NZ land extent to ensure accurate distances for all
    observation locations, even those outside the configured study domain.

    Parameters
    ----------
    output_path : Path
        Path where the output coast distance raster will be saved.
    template_profile : dict
        Rasterio profile of the reference raster (to match resolution and bounds).

    Returns
    -------
    tuple[np.ndarray, dict]
        A tuple containing:
        - The distance array (float32).
        - The updated profile used for saving.
    """
    logger.info("Creating coast distance raster...")

    distance_meters = compute_coast_distance_array(template_profile)

    profile = template_profile.copy()
    profile.update(
        {"dtype": "float32", "count": 1, "nodata": None, "compress": "deflate"}
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(distance_meters, 1)
        dst.descriptions = (constants.BAND_DESCRIPTION_COAST_DISTANCE,)

    return distance_meters, profile


def compute_slope_array(template_profile: dict) -> np.ndarray:
    """
    Compute a slope array matching the target grid in memory.

    Resamples the source slope raster to the target grid properties without
    writing any file to disk.

    Parameters
    ----------
    template_profile : dict
        Rasterio profile of the reference raster (to match resolution and bounds).

    Returns
    -------
    np.ndarray
        The slope array matching the template grid dimensions.

    Raises
    ------
    FileNotFoundError
        If the source slope raster is not found.
    """
    slope_raster_path = (
        constants.GEOSPATIAL_DIR / constants.SLOPE_SOURCE_RASTER_FILENAME
    )
    if not slope_raster_path.exists():
        raise FileNotFoundError(f"Slope raster not found: {slope_raster_path}")

    # Reproject the slope raster onto the specified grid
    destination = np.zeros((template_profile["height"], template_profile["width"]))
    with rasterio.open(slope_raster_path) as src:
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_profile["transform"],
            dst_crs=template_profile["crs"],
            resampling=rasterio.enums.Resampling.nearest,
        )

    return destination


def create_slope_raster(
    output_path: Path, template_profile: dict
) -> tuple[np.ndarray, dict]:
    """
    Create a slope raster matching the target grid.

    Resamples the source slope raster to the target properties.

    Parameters
    ----------
    output_path : Path
        Path where the output slope raster will be saved.
    template_profile : dict
        Rasterio profile of the reference raster.

    Returns
    -------
    tuple[np.ndarray, dict]
        A tuple containing:
        - The slope array (float32).
        - The updated profile used for saving.
    """
    logger.info("Creating slope raster...")

    destination = compute_slope_array(template_profile)

    profile = template_profile.copy()
    profile.update(
        {
            "dtype": "float32",
            "count": 1,
            "nodata": constants.NODATA_VALUE,
            "compress": "deflate",
        }
    )
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(destination, 1)
        dst.descriptions = (constants.BAND_DESCRIPTION_SLOPE,)

    return destination, profile


def sample_slope_at_points(points: np.ndarray) -> np.ndarray:
    """
    Sample slope values at specific NZTM points from the bundled slope raster.

    Parameters
    ----------
    points : np.ndarray
        (N, 2) array of [easting, northing] coordinates in NZTM.

    Returns
    -------
    np.ndarray
        Slope values at each point (N,).
    """
    slope_raster_path = (
        constants.GEOSPATIAL_DIR / constants.SLOPE_SOURCE_RASTER_FILENAME
    )
    if not slope_raster_path.exists():
        raise FileNotFoundError(f"Slope raster not found: {slope_raster_path}")

    with rasterio.open(slope_raster_path) as src:
        return np.array([sample[0] for sample in src.sample(points)], dtype=np.float64)


def compute_coastal_distance_at_points(points: np.ndarray) -> np.ndarray:
    """
    Compute distance from each point to the nearest coastline.

    Uses the bundled NZ coastline polygon shapefile and shapely geometry
    operations. For small numbers of points this is much faster than
    generating a full proximity raster with GDAL.

    Parameters
    ----------
    points : np.ndarray
        (N, 2) array of [easting, northing] coordinates in NZTM.

    Returns
    -------
    np.ndarray
        Distance to coast in meters for each point (N,).
    """
    coastline_path = constants.GEOSPATIAL_DIR / constants.COASTLINE_SHAPEFILE_PATH
    ensure_shapefile_extracted(coastline_path, "coast")

    coast_gdf = gpd.read_file(coastline_path)
    # The coastline file contains land polygons. Distance to coast is distance
    # from each point to the nearest polygon boundary.
    coast_boundary = coast_gdf.geometry.boundary.union_all()

    point_geoms = shapely.points(points)
    distances = shapely.distance(point_geoms, coast_boundary)
    return np.asarray(distances, dtype=np.float64)


def apply_coastal_distance_modification(
    vs30_array: np.ndarray,
    id_array: np.ndarray,
    coast_dist_array: np.ndarray,
    gid: int,
    dist_min: float,
    dist_max: float,
    vs30_min: float,
    vs30_max: float,
) -> None:
    """
    Apply linear coastal distance interpolation to a specific geology group.

    Modifies vs30_array in-place for pixels matching the given geology group ID.
    The Vs30 value is linearly interpolated between vs30_min and vs30_max based
    on distance from coast, clamped to [vs30_min, vs30_max].

    Parameters
    ----------
    vs30_array : ndarray
        VS30 array to modify in-place.
    id_array : ndarray
        Category ID array.
    coast_dist_array : ndarray
        Coastal distance array (meters).
    gid : int
        Geology group ID to modify.
    dist_min, dist_max : float
        Distance range (meters) for linear interpolation.
    vs30_min, vs30_max : float
        Vs30 range (m/s) for linear interpolation.
    """
    mask = id_array == gid
    if not np.any(mask):
        return

    dist_vals = coast_dist_array[mask]
    val = vs30_min + (vs30_max - vs30_min) * (dist_vals - dist_min) / (
        dist_max - dist_min
    )
    vs30_array[mask] = np.clip(val, vs30_min, vs30_max)


def apply_hybrid_geology_modifications(
    vs30_array: np.ndarray,
    stdv_array: np.ndarray,
    id_array: np.ndarray,
    slope_array: np.ndarray,
    coast_dist_array: np.ndarray,
    apply_alluvium_slope_mod: bool,
    apply_coastal_distance_mod: bool,
    hybrid: bool = True,
    hybrid_mod6_dist_min: float = constants.HYBRID_MOD6_DIST_MIN,
    hybrid_mod6_dist_max: float = constants.HYBRID_MOD6_DIST_MAX,
    hybrid_mod6_vs30_min: float = constants.HYBRID_MOD6_VS30_MIN,
    hybrid_mod6_vs30_max: float = constants.HYBRID_MOD6_VS30_MAX,
    hybrid_mod13_dist_min: float = constants.HYBRID_MOD13_DIST_MIN,
    hybrid_mod13_dist_max: float = constants.HYBRID_MOD13_DIST_MAX,
    hybrid_mod13_vs30_min: float = constants.HYBRID_MOD13_VS30_MIN,
    hybrid_mod13_vs30_max: float = constants.HYBRID_MOD13_VS30_MAX,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Apply hybrid model modifications to VS30 and standard deviation arrays.

    Implements slope-based Vs30 interpolation and coastal distance adjustments
    for specific geology categories. Modifies arrays in-place but also returns
    them for clarity.

    Parameters
    ----------
    vs30_array : np.ndarray
        Base VS30 array (float).
    stdv_array : np.ndarray
        Base Standard Deviation array (float).
    id_array : np.ndarray
        Category ID array (int).
    slope_array : np.ndarray
        Slope array (float).
    coast_dist_array : np.ndarray
        Distance to coast array (float).
    apply_alluvium_slope_mod : bool
        Whether to apply slope-based interpolation for GID 4 (alluvium).
        When False, GID 4 keeps its categorical Vs30 value.
    apply_coastal_distance_mod : bool
        Whether to apply coastal distance modifications for GID 4 (alluvium)
        and GID 10 (floodplain).
    hybrid : bool, optional
        Whether to apply general hybrid slope-based modifications. Default True.
    hybrid_mod6_dist_min : float
        Min distance threshold for GID 4 coastal distance mod.
    hybrid_mod6_dist_max : float
        Max distance threshold for GID 4 coastal distance mod.
    hybrid_mod6_vs30_min : float
        Min Vs30 for GID 4 coastal distance mod.
    hybrid_mod6_vs30_max : float
        Max Vs30 for GID 4 coastal distance mod.
    hybrid_mod13_dist_min : float
        Min distance threshold for GID 10 coastal distance mod.
    hybrid_mod13_dist_max : float
        Max distance threshold for GID 10 coastal distance mod.
    hybrid_mod13_vs30_min : float
        Min Vs30 for GID 10 coastal distance mod.
    hybrid_mod13_vs30_max : float
        Max Vs30 for GID 10 coastal distance mod.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Modified (vs30_array, stdv_array).
    """
    logger.info("Applying slope and coastal distance based geology modifications...")

    vs30_array = vs30_array.copy()
    stdv_array = stdv_array.copy()

    # 1. Update Standard Deviation for specific groups
    if hybrid:
        # group IDs have reduction factors from constants
        for gid, factor in constants.HYBRID_SIGMA_REDUCTION_FACTORS.items():
            # Find pixels with this ID
            mask = id_array == gid
            stdv_array[mask] *= factor

    # 2. Hybrid slope-based VS30 calculation
    if hybrid:
        # Prevent log10(0) or log10(-NODATA) by capping at constants.MIN_SLOPE_FOR_LOG
        safe_log_slope = np.log10(
            np.where(
                (slope_array <= 0) | (slope_array == constants.NODATA_VALUE),
                constants.MIN_SLOPE_FOR_LOG,
                slope_array,
            )
        )

        for spec in constants.HYBRID_VS30_PARAMS:
            # Skip GID 4 slope interpolation when apply_alluvium_slope_mod is False
            # (GID 4 may instead get coastal distance modification below)
            if spec.gid == 4 and not apply_alluvium_slope_mod:
                continue

            mask = id_array == spec.gid

            if np.any(mask):
                vs30_limits_log10 = np.log10(np.array(spec.vs30_values))
                interpolated_val = np.interp(
                    safe_log_slope[mask], spec.slope_limits, vs30_limits_log10
                )
                vs30_array[mask] = 10**interpolated_val

    # 3. Coastal distance modification for alluvium (GID 4) and floodplain (GID 10)
    if apply_coastal_distance_mod:
        apply_coastal_distance_modification(
            vs30_array,
            id_array,
            coast_dist_array,
            gid=4,
            dist_min=hybrid_mod6_dist_min,
            dist_max=hybrid_mod6_dist_max,
            vs30_min=hybrid_mod6_vs30_min,
            vs30_max=hybrid_mod6_vs30_max,
        )

        apply_coastal_distance_modification(
            vs30_array,
            id_array,
            coast_dist_array,
            gid=10,
            dist_min=hybrid_mod13_dist_min,
            dist_max=hybrid_mod13_dist_max,
            vs30_min=hybrid_mod13_vs30_min,
            vs30_max=hybrid_mod13_vs30_max,
        )

    return vs30_array, stdv_array

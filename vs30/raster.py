"""Categorical VS30 raster creation and hybrid geology modifications."""

import functools
import logging
import math
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

from vs30 import config, constants, utils

logger = logging.getLogger(__name__)


@functools.lru_cache(maxsize=1)
def load_qmap_shapefile() -> gpd.GeoDataFrame:
    """
    Load the bundled QMAP geology shapefile.

    Returns
    -------
    gpd.GeoDataFrame
        QMAP polygons with geology ``gid`` attribute.
    """
    qmap_path = constants.GEOSPATIAL_DIR / constants.GEOLOGY_SHAPEFILE_PATH
    ensure_shapefile_extracted(qmap_path, "qmap")
    return gpd.read_file(qmap_path)


@functools.lru_cache(maxsize=1)
def load_coast_shapefile() -> gpd.GeoDataFrame:
    """
    Load the bundled NZ coastline shapefile.

    Returns
    -------
    gpd.GeoDataFrame
        Coastline land polygons.
    """
    coast_path = constants.GEOSPATIAL_DIR / constants.COASTLINE_SHAPEFILE_PATH
    ensure_shapefile_extracted(coast_path, "coast")
    return gpd.read_file(coast_path)


@functools.lru_cache(maxsize=1)
def load_coast_union():
    """
    Return the unioned NZ coastline geometry.
    """
    return load_coast_shapefile().geometry.union_all()


@functools.lru_cache(maxsize=1)
def load_coast_boundary_union():
    """
    Return the unioned NZ coastline boundary (polygon edges, distinct from
    ``load_coast_union`` which returns the polygon union).
    """
    return load_coast_shapefile().geometry.boundary.union_all()


@functools.lru_cache(maxsize=1)
def load_terrain_raster_array() -> tuple[
    np.ndarray, rasterio.transform.Affine, float | None
]:
    """
    Load the bundled terrain (IwahashiPike) raster fully into memory.

    Returns
    -------
    tuple[ndarray, rasterio.transform.Affine, float or None]
        ``(data, transform, nodata)`` where ``data`` is the full 2D uint8
        category-id raster.
    """
    path = constants.GEOSPATIAL_DIR / constants.TERRAIN_RASTER_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Terrain raster not found: {path}")
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.nodata


@functools.lru_cache(maxsize=1)
def load_slope_raster_array() -> tuple[
    np.ndarray, rasterio.transform.Affine, float | None
]:
    """
    Load the bundled slope raster fully into memory.

    Returns
    -------
    tuple[ndarray, rasterio.transform.Affine, float or None]
        ``(data, transform, nodata)`` where ``data`` is the full 2D float
        slope raster.
    """
    path = constants.GEOSPATIAL_DIR / constants.SLOPE_SOURCE_RASTER_FILENAME
    if not path.exists():
        raise FileNotFoundError(f"Slope raster not found: {path}")
    with rasterio.open(path) as src:
        return src.read(1), src.transform, src.nodata


def ensure_shapefile_extracted(shapefile_path: Path, directory_prefix: str) -> None:
    """
    Ensure a shapefile is extracted from shapefiles.tar.xz.

    Parameters
    ----------
    shapefile_path : Path
        Shapefile to ensure exists.
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
    grid_config: config.GridConfig,
) -> tuple[np.ndarray, dict]:
    """
    Create category ID array for terrain or geology.

    Parameters
    ----------
    model_type : constants.ModelType
        Either ModelType.TERRAIN or ModelType.GEOLOGY.
    grid_config : config.GridConfig
        Grid bounds and spacing.

    Returns
    -------
    id_array : ndarray
        Category ID array (uint8).
    profile : dict
        Rasterio profile describing the array's spatial properties.

    Raises
    ------
    ValueError
        If model_type is invalid or the geology shapefile is missing required
        columns.
    FileNotFoundError
        If the input raster or shapefile is not found.
    """
    if model_type not in (constants.ModelType.GEOLOGY, constants.ModelType.TERRAIN):
        raise ValueError(
            f"model_type must be ModelType.GEOLOGY or ModelType.TERRAIN, got '{model_type}'"
        )

    nx = round((grid_config.grid_xmax - grid_config.grid_xmin) / grid_config.grid_dx)
    ny = round((grid_config.grid_ymax - grid_config.grid_ymin) / grid_config.grid_dy)
    dst_transform = rasterio.transform.from_bounds(
        grid_config.grid_xmin,
        grid_config.grid_ymin,
        grid_config.grid_xmax,
        grid_config.grid_ymax,
        nx,
        ny,
    )

    if model_type == constants.ModelType.TERRAIN:
        terrain_raster_path = (
            constants.GEOSPATIAL_DIR / constants.TERRAIN_RASTER_FILENAME
        )
        if not terrain_raster_path.exists():
            raise FileNotFoundError(f"Terrain raster not found: {terrain_raster_path}")

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

    elif model_type == constants.ModelType.GEOLOGY:
        gdf = load_qmap_shapefile()
        if constants.SHAPEFILE_GEOLOGY_ID_COLUMN not in gdf.columns:
            raise ValueError(
                f"Shapefile {constants.GEOLOGY_SHAPEFILE_PATH} missing "
                f"'{constants.SHAPEFILE_GEOLOGY_ID_COLUMN}' column"
            )

        if gdf.crs is None or str(gdf.crs) != constants.NZTM_CRS:
            gdf = gdf.to_crs(constants.NZTM_CRS)

        id_array = rasterio.features.rasterize(
            shapes=zip(gdf.geometry, gdf[constants.SHAPEFILE_GEOLOGY_ID_COLUMN]),
            out_shape=(ny, nx),
            transform=dst_transform,
            fill=constants.RASTER_ID_NODATA_VALUE,
            dtype=np.uint8,
            all_touched=False,
        )

    return id_array, {
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


def create_vs30_arrays_from_ids(
    id_array: np.ndarray,
    model_values_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Map category IDs to Vs30 mean and standard deviation arrays.

    Parameters
    ----------
    id_array : np.ndarray
        Category ID array (uint8) from terrain or geology rasterization.
    model_values_df : pd.DataFrame
        Category ID → Vs30 mapping. Must have a STANDARD_ID_COLUMN column
        and one of the mean/stdv pairs recognized by
        utils.select_vs30_columns_by_priority.

    Returns
    -------
    vs30_array : ndarray
        Vs30 mean values for each pixel (float32).
    stdv_array : ndarray
        Vs30 standard deviation values for each pixel (float32).

    Raises
    ------
    ValueError
        If model_values_df lacks required columns, or if id_array contains
        an ID not present in model_values_df.
    """
    mean_col, std_col = utils.select_vs30_columns_by_priority(
        list(model_values_df.columns)
    )

    if constants.STANDARD_ID_COLUMN not in model_values_df.columns:
        raise ValueError(
            f"DataFrame is missing required column: {constants.STANDARD_ID_COLUMN}"
        )

    # Build LUTs indexed directly by category ID. RASTER_ID_NODATA_VALUE (255)
    # is the largest id we ever see, so the LUT length is fixed at 256.
    mean_lut = np.full(
        constants.RASTER_ID_NODATA_VALUE + 1, constants.NODATA_VALUE, dtype=np.float32
    )
    stdv_lut = np.full(
        constants.RASTER_ID_NODATA_VALUE + 1, constants.NODATA_VALUE, dtype=np.float32
    )

    df_ids = model_values_df[constants.STANDARD_ID_COLUMN].astype(int).to_numpy()
    mean_lut[df_ids] = model_values_df[mean_col].astype(np.float32).to_numpy()
    stdv_lut[df_ids] = model_values_df[std_col].astype(np.float32).to_numpy()

    unique_ids = np.unique(id_array)
    missing_ids = np.setdiff1d(
        unique_ids[
            (unique_ids != constants.RASTER_ID_NODATA_VALUE) & (unique_ids != 0)
        ],
        df_ids,
    )
    if missing_ids.size:
        raise ValueError(
            f"ID {int(missing_ids[0])} found in array but not in DataFrame. "
            f"Available IDs: {sorted(df_ids.tolist())}"
        )

    return mean_lut[id_array], stdv_lut[id_array]


def compute_coast_distance_array(template_profile: dict) -> np.ndarray:
    """
    Compute distance to the nearest coast (in meters).

    Parameters
    ----------
    template_profile : dict
        Rasterio profile of the reference raster (to match resolution and bounds).

    Returns
    -------
    np.ndarray
        Distance array (float32) matching the template grid dimensions.
    """
    ensure_shapefile_extracted(
        constants.GEOSPATIAL_DIR / constants.COASTLINE_SHAPEFILE_PATH, "coast"
    )

    # Template bounds for output
    transform = template_profile["transform"]
    dx = transform.a
    dy = abs(transform.e)
    s_xmin = transform.c
    s_ymax = transform.f
    s_xmax = s_xmin + template_profile["width"] * dx
    s_ymin = s_ymax - template_profile["height"] * dy

    # Grid extension rationale:
    # - Full-NZ coverage: accurate distances for observations beyond the
    #   template bounds.
    # - Pixel alignment: round up to whole pixels (else GDAL trims to integer
    #   pixel counts and shifts samples by a sub-pixel offset).
    g_xmin = (
        s_xmin
        - math.ceil(max(0, s_xmin - config.FULL_NZ_GRID_CONFIG.grid_xmin) / dx) * dx
    )
    g_xmax = (
        s_xmax
        + math.ceil(max(0, config.FULL_NZ_GRID_CONFIG.grid_xmax - s_xmax) / dx) * dx
    )
    g_ymin = (
        s_ymin
        - math.ceil(max(0, s_ymin - config.FULL_NZ_GRID_CONFIG.grid_ymin) / dy) * dy
    )
    g_ymax = (
        s_ymax
        + math.ceil(max(0, config.FULL_NZ_GRID_CONFIG.grid_ymax - s_ymax) / dy) * dy
    )

    # GDAL requires a file path, so use a temporary file.
    fd, tmp_path = tempfile.mkstemp(suffix=".tif")
    os.close(fd)

    try:
        ds = gdal.Rasterize(
            tmp_path,
            str(constants.GEOSPATIAL_DIR / constants.COASTLINE_SHAPEFILE_PATH),
            creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=YES"],
            outputBounds=[g_xmin, g_ymin, g_xmax, g_ymax],
            xRes=dx,
            yRes=dy,
            noData=0,
            burnValues=1,
            outputType=gdal.GetDataTypeByName("Float32"),
        )

        band = ds.GetRasterBand(1)
        band.SetDescription(constants.BAND_DESCRIPTION_COAST_DISTANCE)
        # band is passed as both source and destination - ComputeProximity
        # overwrites it with distance values.
        ds = gdal.ComputeProximity(band, band, ["VALUES=0", "DISTUNITS=GEO"])

        with rasterio.open(tmp_path) as src:
            data = src.read(1)

        if g_xmin < s_xmin or g_xmax > s_xmax or g_ymin < s_ymin or g_ymax > s_ymax:
            col_offset = round((s_xmin - g_xmin) / dx)
            row_offset = round((g_ymax - s_ymax) / dy)
            distance_meters = data[
                row_offset : row_offset + template_profile["height"],
                col_offset : col_offset + template_profile["width"],
            ].astype(np.float32)
        else:
            distance_meters = data.astype(np.float32)
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return distance_meters


def compute_slope_array(template_profile: dict) -> np.ndarray:
    """
    Compute a slope array matching the target grid.

    Parameters
    ----------
    template_profile : dict
        Rasterio profile of the reference raster (to match resolution and bounds).

    Returns
    -------
    np.ndarray
        Slope array matching the template grid dimensions.

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

    slope_array = np.zeros(
        (template_profile["height"], template_profile["width"]), dtype=np.float32
    )
    with rasterio.open(slope_raster_path) as src:
        rasterio.warp.reproject(
            source=rasterio.band(src, 1),
            destination=slope_array,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_profile["transform"],
            dst_crs=template_profile["crs"],
            resampling=rasterio.enums.Resampling.nearest,
        )

    return slope_array


def sample_slope_at_points(points: np.ndarray) -> np.ndarray:
    """
    Sample slope values at specific NZTM points from the bundled slope raster.

    Out-of-bounds points get the raster's nodata value (matching the legacy
    ``rasterio.sample`` semantics this used to call).

    Parameters
    ----------
    points : np.ndarray
        (N, 2) array of [easting, northing] coordinates in NZTM.

    Returns
    -------
    np.ndarray
        Slope values at each point (N,).
    """
    data, transform, nodata = load_slope_raster_array()
    rows, cols = rasterio.transform.rowcol(transform, points[:, 0], points[:, 1])
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    in_bounds = (
        (rows >= 0) & (rows < data.shape[0]) & (cols >= 0) & (cols < data.shape[1])
    )
    out = np.full(len(points), nodata, dtype=np.float64)
    out[in_bounds] = data[rows[in_bounds], cols[in_bounds]].astype(np.float64)
    return out


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
    # The coastline file contains land polygons. Distance to coast is distance
    # from each point to the nearest polygon boundary; load_coast_boundary_union
    # memoises the expensive union_all().
    coast_boundary = load_coast_boundary_union()
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

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Modified (vs30_array, stdv_array).
    """
    logger.info("Applying slope and coastal distance based geology modifications...")

    vs30_array = vs30_array.copy()
    stdv_array = stdv_array.copy()

    for spec in constants.HYBRID_GEOLOGY_PARAMS:
        mask = id_array == spec.gid
        if not np.any(mask):
            continue
        # sigma_reduction always applies — it tightens the categorical lookup
        # itself (legacy R semantics), not a per-pixel slope refinement.
        stdv_array[mask] *= spec.sigma_reduction

        # GID 4 (alluvium) gets coastal-distance handling below when the
        # slope mod is off, so skip slope interpolation here.
        if spec.gid == 4 and not apply_alluvium_slope_mod:
            continue

        spec_slope = slope_array[mask]
        # Cap slope at MIN_SLOPE_FOR_LOG to avoid log10(0) or log10(-NODATA).
        safe_slope = np.where(
            (spec_slope <= 0) | (spec_slope == constants.NODATA_VALUE),
            constants.MIN_SLOPE_FOR_LOG,
            spec_slope,
        )
        interpolated_val = np.interp(
            np.log10(safe_slope), spec.slope_limits, spec.vs30_values_log10
        )
        vs30_array[mask] = 10**interpolated_val

    if apply_coastal_distance_mod:
        apply_coastal_distance_modification(
            vs30_array,
            id_array,
            coast_dist_array,
            gid=4,
            dist_min=constants.HYBRID_GID4_DIST_MIN,
            dist_max=constants.HYBRID_GID4_DIST_MAX,
            vs30_min=constants.HYBRID_GID4_VS30_MIN,
            vs30_max=constants.HYBRID_GID4_VS30_MAX,
        )

        apply_coastal_distance_modification(
            vs30_array,
            id_array,
            coast_dist_array,
            gid=10,
            dist_min=constants.HYBRID_GID10_DIST_MIN,
            dist_max=constants.HYBRID_GID10_DIST_MAX,
            vs30_min=constants.HYBRID_GID10_VS30_MIN,
            vs30_max=constants.HYBRID_GID10_VS30_MAX,
        )

    return vs30_array, stdv_array

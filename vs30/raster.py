"""
Common functions for creating categorical VS30 rasters from terrain and geology data.

This module provides:
- Category ID raster creation (terrain from IwahashiPike, geology from QMAP shapefile)
- VS30 value mapping from categorical models (CSV files) to raster format
- Slope and coastal distance rasters for hybrid geology modifications
- Hybrid geology modifications based on slope and coastal distance

Hybrid Geology Model
-------------------
The geology-based Vs30 model includes empirical modifications for certain
geology categories where Vs30 is known to vary with:

1. Slope - steeper terrain generally indicates more consolidated material, leading
   to higher Vs30 values. This relationship is captured via log-linear interpolation.
2. Coastal distance - near-coast sediments (especially alluvium and floodplain
   deposits) tend to be unconsolidated with lower Vs30. Values increase inland.

These modifications are based on New Zealand-specific calibration studies and are
applied to geology categories 2, 3, 4, and 6 (slope-based) and categories 4 and 10
(coastal distance-based).
"""

import importlib.resources
import logging
import tarfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from osgeo import gdal
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from tqdm import tqdm

from vs30.config import get_default_config

logger = logging.getLogger(__name__)


# Resources directory path using importlib.resources
RESOURCE_PATH = importlib.resources.files("vs30") / "resources"

# Data directory path
DATA_DIR = Path(__file__).parent / "data"


def _get_terrain_raster_path() -> Path:
    """Get path to terrain classification raster from config."""
    cfg = get_default_config()
    return DATA_DIR / cfg.terrain_raster_filename


def _get_geology_shapefile_path() -> Path:
    """Get path to geology shapefile from config."""
    cfg = get_default_config()
    return DATA_DIR / cfg.geology_shapefile_path


def _get_coastline_shapefile_path() -> Path:
    """Get path to coastline shapefile from config."""
    cfg = get_default_config()
    return DATA_DIR / cfg.coastline_shapefile_path


def _get_slope_raster_path() -> Path:
    """Get path to source slope raster from config."""
    cfg = get_default_config()
    return DATA_DIR / cfg.slope_source_raster_filename


def _get_shapefiles_archive_path() -> Path:
    """Get path to shapefiles archive from config."""
    cfg = get_default_config()
    return DATA_DIR / cfg.shapefiles_archive_filename


def _ensure_shapefile_extracted(shapefile_path: Path, directory_prefix: str) -> None:
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
    # Check if shapefile already exists
    if shapefile_path.exists():
        return

    # Check if archive exists
    archive_path = _get_shapefiles_archive_path()
    if not archive_path.exists():
        raise FileNotFoundError(
            f"Shapefile archive not found: {archive_path}. "
            f"Cannot extract {shapefile_path.name}. Please ensure shapefiles.tar.xz exists."
        )

    # Extract directory from archive
    with tarfile.open(archive_path, "r:xz") as tar:
        # Extract only files in the specified directory
        members = [
            member
            for member in tar.getmembers()
            if member.name.startswith(f"{directory_prefix}/")
        ]
        if not members:
            raise ValueError(
                f"No '{directory_prefix}' directory found in archive {archive_path}"
            )
        tar.extractall(path=DATA_DIR, members=members)

    # Verify extraction was successful
    if not shapefile_path.exists():
        raise FileNotFoundError(
            f"Failed to extract {shapefile_path.name} from {archive_path}. "
            f"Expected file at {shapefile_path} but it was not created."
        )


def _ensure_qmap_shapefile_extracted() -> None:
    """
    Ensure qmap.shp shapefile is extracted from shapefiles.tar.xz.

    This shapefile is required for geology ID raster creation.
    """
    _ensure_shapefile_extracted(_get_geology_shapefile_path(), "qmap")


def _ensure_coast_shapefile_extracted() -> None:
    """
    Ensure coast shapefile is extracted from shapefiles.tar.xz.

    This shapefile is required for creating coastal distance rasters.
    """
    _ensure_shapefile_extracted(_get_coastline_shapefile_path(), "coast")


def load_model_values_from_csv(csv_path: str) -> np.ndarray:
    """
    Load model values (vs30 mean and standard deviation) from CSV file.

    Parameters
    ----------
    csv_path : str
        Path to CSV file relative to resources directory.

    Returns
    -------
    np.ndarray
        Array of shape (n_categories, 2) with dtype np.float64.
        Columns are [mean_vs30_km_per_s, standard_deviation_vs30_km_per_s].

    Raises
    ------
    FileNotFoundError
        If CSV file is not found.
    ValueError
        If CSV file is malformed or missing required columns.
    """
    # Resolve CSV file path using importlib.resources
    csv_file_traversable = RESOURCE_PATH / csv_path

    # Use importlib.resources.as_file to get a context manager for file access
    # This handles both development (files on disk) and installed package scenarios
    with importlib.resources.as_file(csv_file_traversable) as csv_file_path:
        if not csv_file_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {csv_file_path}. "
                f"Expected path relative to resources directory: {csv_path}"
            )

        # Read CSV and check what columns are available
        # Use skipinitialspace=True to handle spaces after commas in CSV
        df = pd.read_csv(csv_file_path, skipinitialspace=True)

        # Check if required columns exist
        required_cols = ["mean_vs30_km_per_s", "standard_deviation_vs30_km_per_s"]
        missing_cols = [col for col in required_cols if col not in df.columns]

        if missing_cols:
            raise ValueError(
                f"CSV file {csv_file_path} is missing required columns. "
                f"Missing columns: {missing_cols}. "
                f"Available columns: {list(df.columns)}"
            )

        # Return only the required columns as numpy array
        return df[required_cols].values


def create_category_id_raster(
    model_type: str,
    output_dir: Path,
    xmin: float | None = None,
    xmax: float | None = None,
    ymin: float | None = None,
    ymax: float | None = None,
    dx: float | None = None,
    dy: float | None = None,
) -> Path:
    """
    Create category ID raster for terrain or geology.

    For terrain: Resamples IwahashiPike.tif to target grid.
    For geology: Rasterizes qmap.shp shapefile to target grid.

    Parameters
    ----------
    model_type : str
        Either "terrain" or "geology".
    output_dir : Path
        Directory where output raster will be saved.

    Returns
    -------
    Path
        Path to created ID raster file.

    Raises
    ------
    ValueError
        If model_type is not "terrain" or "geology".
    FileNotFoundError
        If input files don't exist.
    """
    if model_type not in ["terrain", "geology"]:
        raise ValueError(
            f"model_type must be 'terrain' or 'geology', got '{model_type}'"
        )

    cfg = get_default_config()
    xmin = xmin if xmin is not None else cfg.grid_xmin
    xmax = xmax if xmax is not None else cfg.grid_xmax
    ymin = ymin if ymin is not None else cfg.grid_ymin
    ymax = ymax if ymax is not None else cfg.grid_ymax
    dx = dx if dx is not None else cfg.grid_dx
    dy = dy if dy is not None else cfg.grid_dy

    output_dir.mkdir(parents=True, exist_ok=True)

    # Common setup: calculate grid dimensions and transform
    nx = round((xmax - xmin) / dx)
    ny = round((ymax - ymin) / dy)
    dst_transform = from_bounds(xmin, ymin, xmax, ymax, nx, ny)
    output_filename = "tid.tif" if model_type == "terrain" else "gid.tif"
    output_path = output_dir / output_filename
    band_description = "Model ID Index"

    # Common output raster profile
    profile = {
        "driver": "GTiff",
        "width": nx,
        "height": ny,
        "count": 1,
        "dtype": "uint8",
        "crs": cfg.nztm_crs,
        "transform": dst_transform,
        "nodata": cfg.raster_id_nodata_value,
        "compress": "deflate",
    }

    if model_type == "terrain":
        # Resample terrain raster to target grid
        terrain_raster_path = _get_terrain_raster_path()
        if not terrain_raster_path.exists():
            raise FileNotFoundError(f"Terrain raster not found: {terrain_raster_path}")

        # Read source raster and reproject
        with rasterio.open(terrain_raster_path) as src:
            with rasterio.open(output_path, "w", **profile) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=cfg.nztm_crs,
                    resampling=Resampling.nearest,
                )
                dst.descriptions = (band_description,)

    else:  # geology
        # Ensure qmap.shp is extracted from shapefiles.tar.xz if needed
        _ensure_qmap_shapefile_extracted()

        # Rasterize geology shapefile to target grid
        geology_shapefile_path = _get_geology_shapefile_path()
        if not geology_shapefile_path.exists():
            raise FileNotFoundError(f"Geology shapefile not found: {geology_shapefile_path}")

        # Read shapefile
        gdf = gpd.read_file(geology_shapefile_path)
        if "gid" not in gdf.columns:
            raise ValueError(f"Shapefile {geology_shapefile_path} missing 'gid' column")

        # Ensure shapefile is in NZTM CRS (EPSG:2193)
        if gdf.crs is None or str(gdf.crs) != cfg.nztm_crs:
            gdf = gdf.to_crs(cfg.nztm_crs)

        # Create shapes iterator for rasterization
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf.gid))

        # Rasterize to output file
        with rasterio.open(output_path, "w", **profile) as dst:
            burned = features.rasterize(
                shapes=shapes,
                out_shape=(ny, nx),
                transform=dst_transform,
                fill=cfg.raster_id_nodata_value,
                dtype=np.uint8,
                all_touched=False,
            )
            dst.write(burned, 1)
            dst.descriptions = (band_description,)

    return output_path


def _select_vs30_columns_by_priority(columns: list[str]) -> tuple[str, str]:
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
    cfg = get_default_config()

    priorities = [
        # 1. Independent observations posterior
        (cfg.col_posterior_mean_independent, cfg.col_posterior_stdv_independent),
        # 2. Clustered observations posterior
        (cfg.col_posterior_mean_clustered, cfg.col_posterior_stdv_clustered),
        # 3. Generic posterior
        (cfg.col_posterior_mean, cfg.col_posterior_stdv),
        # 4. Explicit prior
        (cfg.col_prior_mean, cfg.col_prior_stdv),
        # 5. Standard/Original names
        (cfg.col_mean, cfg.col_stdv),
    ]

    for mean_col, std_col in priorities:
        if mean_col in columns and std_col in columns:
            return mean_col, std_col

    raise ValueError(
        f"Could not find valid VS30 mean and standard deviation columns. "
        f"Available columns: {columns}"
    )


def create_vs30_raster_from_ids(
    id_raster_path: Path, csv_path: str, output_path: Path
) -> Path:
    """
    Create VS30 mean and standard deviation raster from category ID raster.

    This function maps category IDs from the spatial raster file (qmap.shp for geology
    or IwahashiPike.tif for terrain) directly to VS30 values from the CSV file by
    matching the ID values between the two files.

    For geology case: Ensures qmap.shp is extracted from shapefiles.tar.xz if needed.
    The shapefile is required to create the ID raster that this function processes.

    Parameters
    ----------
    id_raster_path : Path
        Path to input category ID raster (contains IDs from spatial file).
    csv_path : str
        Path to CSV file (relative to resources directory) containing ID-to-VS30 mapping.
        CSV must have columns: 'id', 'mean_vs30_km_per_s', 'standard_deviation_vs30_km_per_s'.
        The 'id' column values must match the ID values in the spatial raster.
    output_path : Path
        Path where output 2-band raster will be saved.

    Returns
    -------
    Path
        Path to created VS30 raster file.

    Raises
    ------
    FileNotFoundError
        If CSV file is not found or shapefiles.tar.xz is missing.
    ValueError
        If CSV file is missing required columns or IDs don't match.
    """
    # Get config
    cfg = get_default_config()

    logger.info(f"Creating VS30 raster: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _ensure_qmap_shapefile_extracted()

    # Load CSV and create ID-to-values mapping
    csv_file_traversable = RESOURCE_PATH / csv_path
    with importlib.resources.as_file(csv_file_traversable) as csv_file_path:
        if not csv_file_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {csv_file_path}. "
                f"Expected path relative to resources directory: {csv_path}"
            )

        df = pd.read_csv(csv_file_path, skipinitialspace=True)
        df.columns = df.columns.str.strip()

        mean_col, std_col = _select_vs30_columns_by_priority(list(df.columns))

        required_cols = ["id", mean_col, std_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"CSV file {csv_file_path} is missing required columns: {missing_cols}"
            )

        id_to_vs30_values = {
            int(row["id"]): (float(row[mean_col]), float(row[std_col]))
            for _, row in df.iterrows()
        }

    # Read ID raster
    with rasterio.open(id_raster_path) as src:
        id_array = src.read(1)
        profile = src.profile.copy()

    # Create output arrays
    vs30_array = np.full(id_array.shape, cfg.nodata_value, dtype=np.float32)
    stdv_array = np.full(id_array.shape, cfg.nodata_value, dtype=np.float32)

    # Map pixel IDs to VS30 values
    unique_ids = np.unique(id_array)
    valid_ids = unique_ids[(unique_ids != cfg.raster_id_nodata_value) & (unique_ids != 0)]

    for pixel_id in tqdm(valid_ids, desc="Mapping IDs to VS30", unit="ID"):
        if pixel_id in id_to_vs30_values:
            mean_vs30, stddev_vs30 = id_to_vs30_values[pixel_id]
            mask = id_array == pixel_id
            vs30_array[mask] = mean_vs30
            stdv_array[mask] = stddev_vs30
        else:
            raise ValueError(
                f"ID {pixel_id} found in raster {id_raster_path} but not in CSV {csv_path}. "
                f"Available IDs in CSV: {sorted(id_to_vs30_values.keys())}"
            )

    profile.update({
        "count": 2,
        "dtype": "float32",
        "nodata": cfg.nodata_value,
        "compress": "deflate",
    })

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(vs30_array, 1)
        dst.write(stdv_array, 2)
        dst.descriptions = ("Vs30", "Standard Deviation")

    logger.info(f"Completed VS30 raster: {output_path}")
    return output_path


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
    # Get config
    cfg = get_default_config()

    logger.info("Creating coast distance raster...")
    _ensure_coast_shapefile_extracted()

    # Get template bounds for final output extent
    dx = template_profile["transform"].a
    dy = abs(template_profile["transform"].e)
    s_xmin = template_profile["transform"].c
    s_ymax = template_profile["transform"].f
    s_xmax = s_xmin + template_profile["width"] * dx
    s_ymin = s_ymax - template_profile["height"] * dy

    # Extend to full NZ land coverage to ensure accurate distances
    # (matching legacy _full_land_grid behavior)
    g_xmin = min(cfg.full_nz_land_xmin, s_xmin)
    g_xmax = max(cfg.full_nz_land_xmax, s_xmax)
    g_ymin = min(cfg.full_nz_land_ymin, s_ymin)
    g_ymax = max(cfg.full_nz_land_ymax, s_ymax)

    # Check if we need to extend beyond template bounds
    gridmod = g_xmin < s_xmin or g_xmax > s_xmax or g_ymin < s_ymin or g_ymax > s_ymax

    # Rasterize land polygons using GDAL (legacy approach)
    # Use UInt16 data type as in legacy code (sufficient for distance range)
    ds = gdal.Rasterize(
        str(output_path),
        str(_get_coastline_shapefile_path()),
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
    band.SetDescription("Distance to Coast (m)")
    # Note: ComputeProximity modifies the raster in-place
    ds = gdal.ComputeProximity(band, band, ["VALUES=0", "DISTUNITS=GEO"])
    band = None
    ds = None

    # Output profile (shared by both branches)
    profile = template_profile.copy()
    profile.update(
        {"dtype": "float32", "count": 1, "nodata": None, "compress": "deflate"}
    )

    # If grid was extended, crop back to template bounds
    if gridmod:
        with rasterio.open(output_path) as src:
            extended_data = src.read(1)

        col_off = round((s_xmin - g_xmin) / dx)
        row_off = round((g_ymax - s_ymax) / dy)

        distance_meters = extended_data[
            row_off : row_off + template_profile["height"],
            col_off : col_off + template_profile["width"],
        ].astype(np.float32)

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(distance_meters, 1)
            dst.descriptions = ("Distance to Coast (m)",)
    else:
        with rasterio.open(output_path) as src:
            distance_meters = src.read(1).astype(np.float32)

    return distance_meters, profile


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
    cfg = get_default_config()
    logger.info("Creating slope raster...")
    slope_raster_path = _get_slope_raster_path()
    if not slope_raster_path.exists():
        raise FileNotFoundError(f"Slope raster not found: {slope_raster_path}")

    # Initialize destination array
    destination = np.zeros((template_profile["height"], template_profile["width"]))

    with rasterio.open(slope_raster_path) as src:
        reproject(
            source=rasterio.band(src, 1),
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=template_profile["transform"],
            dst_crs=template_profile["crs"],
            resampling=Resampling.nearest,
        )

    # Save to file
    profile = template_profile.copy()
    profile.update(
        {"dtype": "float32", "count": 1, "nodata": cfg.nodata_value, "compress": "deflate"}
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(destination, 1)
        dst.descriptions = ("Slope",)

    return destination, profile


def _apply_coastal_distance_modification(
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
    val = vs30_min + (vs30_max - vs30_min) * (dist_vals - dist_min) / (dist_max - dist_min)
    vs30_array[mask] = np.clip(val, vs30_min, vs30_max)


def apply_hybrid_geology_modifications(
    vs30_array: np.ndarray,
    stdv_array: np.ndarray,
    id_array: np.ndarray,
    slope_array: np.ndarray,
    coast_dist_array: np.ndarray,
    mod6: bool = True,
    mod13: bool = True,
    hybrid: bool = True,
    hybrid_mod6_dist_min: float | None = None,
    hybrid_mod6_dist_max: float | None = None,
    hybrid_mod6_vs30_min: float | None = None,
    hybrid_mod6_vs30_max: float | None = None,
    hybrid_mod13_dist_min: float | None = None,
    hybrid_mod13_dist_max: float | None = None,
    hybrid_mod13_vs30_min: float | None = None,
    hybrid_mod13_vs30_max: float | None = None,
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
    mod6 : bool, optional
        Whether to apply modification for Group 6 (Alluvium). Default True.
    mod13 : bool, optional
        Whether to apply modification for Group 13 (Floodplain). Default True.
    hybrid : bool, optional
        Whether to apply general hybrid slope-based modifications. Default True.
    hybrid_mod6_dist_min : float, optional
        Min distance threshold for mod6. Default from config.
    hybrid_mod6_dist_max : float, optional
        Max distance threshold for mod6. Default from config.
    hybrid_mod6_vs30_min : float, optional
        Min Vs30 for mod6. Default from config.
    hybrid_mod6_vs30_max : float, optional
        Max Vs30 for mod6. Default from config.
    hybrid_mod13_dist_min : float, optional
        Min distance threshold for mod13. Default from config.
    hybrid_mod13_dist_max : float, optional
        Max distance threshold for mod13. Default from config.
    hybrid_mod13_vs30_min : float, optional
        Min Vs30 for mod13. Default from config.
    hybrid_mod13_vs30_max : float, optional
        Max Vs30 for mod13. Default from config.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Modified (vs30_array, stdv_array).
    """
    cfg = get_default_config()
    logger.info("Applying slope and coastal distance based geology modifications...")

    # Fill in defaults from config for any unspecified parameters
    if mod6:
        hybrid_mod6_dist_min = hybrid_mod6_dist_min if hybrid_mod6_dist_min is not None else cfg.hybrid_mod6_dist_min
        hybrid_mod6_dist_max = hybrid_mod6_dist_max if hybrid_mod6_dist_max is not None else cfg.hybrid_mod6_dist_max
        hybrid_mod6_vs30_min = hybrid_mod6_vs30_min if hybrid_mod6_vs30_min is not None else cfg.hybrid_mod6_vs30_min
        hybrid_mod6_vs30_max = hybrid_mod6_vs30_max if hybrid_mod6_vs30_max is not None else cfg.hybrid_mod6_vs30_max
    if mod13:
        hybrid_mod13_dist_min = hybrid_mod13_dist_min if hybrid_mod13_dist_min is not None else cfg.hybrid_mod13_dist_min
        hybrid_mod13_dist_max = hybrid_mod13_dist_max if hybrid_mod13_dist_max is not None else cfg.hybrid_mod13_dist_max
        hybrid_mod13_vs30_min = hybrid_mod13_vs30_min if hybrid_mod13_vs30_min is not None else cfg.hybrid_mod13_vs30_min
        hybrid_mod13_vs30_max = hybrid_mod13_vs30_max if hybrid_mod13_vs30_max is not None else cfg.hybrid_mod13_vs30_max

    # 1. Update Standard Deviation for specific groups
    if hybrid:
        # group IDs have reduction factors loaded from config
        for gid_str, factor in cfg.hybrid_sigma_reduction_factors.items():
            gid = int(gid_str)
            # Find pixels with this ID
            mask = id_array == gid
            stdv_array[mask] *= factor

    # 2. Hybrid slope-based VS30 calculation
    if hybrid:
        # Prevent log10(0) or log10(-NODATA) by capping at min_slope_for_log
        modified_slope = np.copy(slope_array)
        modified_slope[(modified_slope <= 0) | (modified_slope == cfg.nodata_value)] = (
            cfg.min_slope_for_log
        )
        safe_log_slope = np.log10(modified_slope)

        for spec in cfg.hybrid_vs30_params:
            gid = spec.gid
            slope_limits = spec.slope_limits
            # Compute log10 of vs30 values at runtime
            vs30_limits_log10 = np.log10(np.array(spec.vs30_values))

            # Skip ID 4 if mod6 is active (handled separately later)
            if gid == 4 and mod6:
                continue

            # Find mask: ID matches
            mask = id_array == gid

            if np.any(mask):
                # Interpolate for all pixels in this group
                interpolated_val = np.interp(
                    safe_log_slope[mask], slope_limits, vs30_limits_log10
                )
                vs30_array[mask] = 10**interpolated_val

    # 3. Distance-based modification for alluvium (GID 4) and floodplain (GID 10)
    if mod6:
        _apply_coastal_distance_modification(
            vs30_array, id_array, coast_dist_array,
            gid=4,
            dist_min=hybrid_mod6_dist_min, dist_max=hybrid_mod6_dist_max,
            vs30_min=hybrid_mod6_vs30_min, vs30_max=hybrid_mod6_vs30_max,
        )

    if mod13:
        _apply_coastal_distance_modification(
            vs30_array, id_array, coast_dist_array,
            gid=10,
            dist_min=hybrid_mod13_dist_min, dist_max=hybrid_mod13_dist_max,
            vs30_min=hybrid_mod13_vs30_min, vs30_max=hybrid_mod13_vs30_max,
        )

    return vs30_array, stdv_array


def apply_hybrid_modifications_at_points(
    points: np.ndarray,
    vs30: np.ndarray,
    stdv: np.ndarray,
    geology_ids: np.ndarray,
    slope_raster_path: Path | None = None,
    coast_distance_raster_path: Path | None = None,
    mod6: bool = True,
    mod13: bool = True,
    hybrid: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply hybrid geology modifications at specific points.

    This is the point-based equivalent of apply_hybrid_geology_modifications().
    It samples slope and coastal distance values at query points and applies
    the same modification logic.

    Parameters
    ----------
    points : np.ndarray
        (N, 2) array of [easting, northing] coordinates in NZTM.
    vs30 : np.ndarray
        Initial Vs30 values at points (N,).
    stdv : np.ndarray
        Initial standard deviation values at points (N,).
    geology_ids : np.ndarray
        Geology category IDs at points (N,).
    slope_raster_path : Path, optional
        Path to slope raster. If None, uses source slope.tif from data directory.
    coast_distance_raster_path : Path, optional
        Path to coastal distance raster. If None, raises error (must be provided).
    mod6 : bool, optional
        Whether to apply modification for Group 6 (Alluvium). Default True.
    mod13 : bool, optional
        Whether to apply modification for Group 13 (Floodplain). Default True.
    hybrid : bool, optional
        Whether to apply general hybrid slope-based modifications. Default True.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Modified (vs30, stdv) arrays.

    Raises
    ------
    FileNotFoundError
        If required raster files are not found.
    ValueError
        If coast_distance_raster_path is None and mod6 or mod13 is True.
    """
    cfg = get_default_config()
    # Use source slope raster if not specified
    if slope_raster_path is None:
        slope_raster_path = _get_slope_raster_path()

    # Coast distance raster is required for mod6/mod13
    if (mod6 or mod13) and coast_distance_raster_path is None:
        raise ValueError(
            "coast_distance_raster_path is required when mod6 or mod13 is True. "
            "Run the grid pipeline first to create the coastal distance raster, "
            "or provide a path to an existing one."
        )

    # Sample slope at points
    if not slope_raster_path.exists():
        raise FileNotFoundError(f"Slope raster not found: {slope_raster_path}")

    with rasterio.open(slope_raster_path) as src:
        slope_values = np.array(
            [sample[0] for sample in src.sample(points)], dtype=np.float64
        )

    # Sample coastal distance at points (if needed)
    if mod6 or mod13:
        if not coast_distance_raster_path.exists():
            raise FileNotFoundError(
                f"Coastal distance raster not found: {coast_distance_raster_path}"
            )

        with rasterio.open(coast_distance_raster_path) as src:
            coast_dist_values = np.array(
                [sample[0] for sample in src.sample(points)], dtype=np.float64
            )
    else:
        coast_dist_values = None

    # Create copies to modify
    modified_vs30 = vs30.copy()
    modified_stdv = stdv.copy()

    # 1. Update standard deviation for specific groups
    if hybrid:
        for gid_str, factor in cfg.hybrid_sigma_reduction_factors.items():
            gid = int(gid_str)
            mask = geology_ids == gid
            if np.any(mask):
                modified_stdv[mask] *= factor

    # 2. Hybrid slope-based VS30 calculation
    if hybrid:
        # Prevent log10(0) or log10(negative) by capping at min_slope_for_log
        safe_slope = np.maximum(slope_values, cfg.min_slope_for_log)
        # Handle nodata values
        safe_slope[slope_values == cfg.nodata_value] = cfg.min_slope_for_log
        safe_log_slope = np.log10(safe_slope)

        for spec in cfg.hybrid_vs30_params:
            gid = spec.gid
            slope_limits = spec.slope_limits
            vs30_limits_log10 = np.log10(np.array(spec.vs30_values))

            # Skip ID 4 if mod6 is active (handled separately)
            if gid == 4 and mod6:
                continue

            mask = geology_ids == gid
            if np.any(mask):
                interpolated_val = np.interp(
                    safe_log_slope[mask], slope_limits, vs30_limits_log10
                )
                modified_vs30[mask] = 10**interpolated_val

    # 3. Distance-based modification for alluvium (GID 4) and floodplain (GID 10)
    if mod6 and coast_dist_values is not None:
        _apply_coastal_distance_modification(
            modified_vs30, geology_ids, coast_dist_values,
            gid=4,
            dist_min=cfg.hybrid_mod6_dist_min, dist_max=cfg.hybrid_mod6_dist_max,
            vs30_min=cfg.hybrid_mod6_vs30_min, vs30_max=cfg.hybrid_mod6_vs30_max,
        )

    if mod13 and coast_dist_values is not None:
        _apply_coastal_distance_modification(
            modified_vs30, geology_ids, coast_dist_values,
            gid=10,
            dist_min=cfg.hybrid_mod13_dist_min, dist_max=cfg.hybrid_mod13_dist_max,
            vs30_min=cfg.hybrid_mod13_vs30_min, vs30_max=cfg.hybrid_mod13_vs30_max,
        )

    return modified_vs30, modified_stdv

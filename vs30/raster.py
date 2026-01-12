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

from vs30 import constants


# Resources directory path using importlib.resources
RESOURCE_PATH = importlib.resources.files("vs30") / "resources"

# Data directory path
DATA_DIR = Path(__file__).parent / "data"
TERRAIN_RASTER = DATA_DIR / "IwahashiPike.tif"
GEOLOGY_SHAPEFILE = DATA_DIR / "qmap" / "qmap.shp"
COAST_SHAPEFILE = (
    DATA_DIR / "coast" / "nz-coastlines-and-islands-polygons-topo-1500k.shp"
)
SLOPE_RASTER = DATA_DIR / "slope.tif"


# Path to shapefiles archive
SHAPEFILES_ARCHIVE = DATA_DIR / "shapefiles.tar.xz"


def _ensure_qmap_shapefile_extracted() -> None:
    """
    Ensure qmap.shp shapefile is extracted from shapefiles.tar.xz.

    Checks if qmap.shp exists. If not, extracts it from shapefiles.tar.xz.
    This is needed because qmap.shp is required for geology ID raster creation
    but may not be present by default (it must be extracted from the archive).

    Raises
    ------
    FileNotFoundError
        If shapefiles.tar.xz is not found.
    """
    # Check if qmap.shp already exists
    if GEOLOGY_SHAPEFILE.exists():
        return

    # Check if archive exists
    if not SHAPEFILES_ARCHIVE.exists():
        raise FileNotFoundError(
            f"Shapefile archive not found: {SHAPEFILES_ARCHIVE}. "
            f"Cannot extract qmap.shp. Please ensure shapefiles.tar.xz exists."
        )

    # Extract qmap directory from archive
    # The archive contains both 'coast' and 'qmap' directories
    # We only need to extract 'qmap' to get qmap.shp
    with tarfile.open(SHAPEFILES_ARCHIVE, "r:xz") as tar:
        # Extract only files in the qmap directory
        qmap_members = [
            member for member in tar.getmembers() if member.name.startswith("qmap/")
        ]
        if not qmap_members:
            raise ValueError(
                f"No 'qmap' directory found in archive {SHAPEFILES_ARCHIVE}"
            )
        tar.extractall(path=DATA_DIR, members=qmap_members)

    # Verify extraction was successful
    if not GEOLOGY_SHAPEFILE.exists():
        raise FileNotFoundError(
            f"Failed to extract qmap.shp from {SHAPEFILES_ARCHIVE}. "
            f"Expected file at {GEOLOGY_SHAPEFILE} but it was not created."
        )


def load_model_values_from_csv(csv_path: str) -> np.ndarray:
    """
    Load model values (vs30 mean and standard deviation) from CSV file.

    Copied from vs30/model.py and adapted to be self-contained.

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

        # Read CSV - first read all columns to check what's available
        # Use skipinitialspace=True to handle spaces after commas in CSV
        df_all = pd.read_csv(csv_file_path, skipinitialspace=True)

        # Check if required columns exist
        required_cols = ["mean_vs30_km_per_s", "standard_deviation_vs30_km_per_s"]
        missing_cols = [col for col in required_cols if col not in df_all.columns]

        if missing_cols:
            available_cols = list(df_all.columns)
            raise ValueError(
                f"CSV file {csv_file_path} is missing required columns. "
                f"Missing columns: {missing_cols}. "
                f"Available columns: {available_cols}"
            )

        # Select only the columns we need
        df = df_all[required_cols]

        # Convert DataFrame to numpy array
        return df.values


def create_category_id_raster(
    model_type: str,
    output_dir: Path,
    xmin: float = constants.GRID_XMIN,
    xmax: float = constants.GRID_XMAX,
    ymin: float = constants.GRID_YMIN,
    ymax: float = constants.GRID_YMAX,
    dx: float = constants.GRID_DX,
    dy: float = constants.GRID_DY,
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
        "crs": constants.NZTM_CRS,
        "transform": dst_transform,
        "nodata": constants.ID_NODATA,
        "compress": "deflate",
    }

    if model_type == "terrain":
        # Resample terrain raster to target grid
        if not TERRAIN_RASTER.exists():
            raise FileNotFoundError(f"Terrain raster not found: {TERRAIN_RASTER}")

        # Read source raster and reproject
        with rasterio.open(TERRAIN_RASTER) as src:
            with rasterio.open(output_path, "w", **profile) as dst:
                reproject(
                    source=rasterio.band(src, 1),
                    destination=rasterio.band(dst, 1),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=dst_transform,
                    dst_crs=constants.NZTM_CRS,
                    resampling=Resampling.nearest,
                )
                dst.descriptions = (band_description,)

    else:  # geology
        # Ensure qmap.shp is extracted from shapefiles.tar.xz if needed
        _ensure_qmap_shapefile_extracted()

        # Rasterize geology shapefile to target grid
        if not GEOLOGY_SHAPEFILE.exists():
            raise FileNotFoundError(f"Geology shapefile not found: {GEOLOGY_SHAPEFILE}")

        # Read shapefile
        gdf = gpd.read_file(GEOLOGY_SHAPEFILE)
        if "gid" not in gdf.columns:
            raise ValueError(f"Shapefile {GEOLOGY_SHAPEFILE} missing 'gid' column")

        # Ensure shapefile is in NZTM CRS (EPSG:2193)
        if gdf.crs is None or str(gdf.crs) != constants.NZTM_CRS:
            gdf = gdf.to_crs(constants.NZTM_CRS)

        # Create shapes iterator for rasterization
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf.gid))

        # Rasterize to output file
        with rasterio.open(output_path, "w", **profile) as dst:
            burned = features.rasterize(
                shapes=shapes,
                out_shape=(ny, nx),
                transform=dst_transform,
                fill=constants.ID_NODATA,
                dtype=np.uint8,
                all_touched=False,
            )
            dst.write(burned, 1)
            dst.descriptions = (band_description,)

    return output_path


def _determine_vs30_columns(columns: list[str]) -> tuple[str, str]:
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
            "posterior_mean_vs30_km_per_s_independent_observations",
            "posterior_standard_deviation_vs30_km_per_s_independent_observations",
        ),
        # 2. Clustered observations posterior
        (
            "posterior_mean_vs30_km_per_s_clustered_observations",
            "posterior_standard_deviation_vs30_km_per_s_clustered_observations",
        ),
        # 3. Generic posterior
        (
            "posterior_mean_vs30_km_per_s",
            "posterior_standard_deviation_vs30_km_per_s",
        ),
        # 4. Explicit prior
        (
            "prior_mean_vs30_km_per_s",
            "prior_standard_deviation_vs30_km_per_s",
        ),
        # 5. Standard/Original names
        (
            "mean_vs30_km_per_s",
            "standard_deviation_vs30_km_per_s",
        ),
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
    print(f"Creating VS30 raster: {output_path}")

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

        mean_col, std_col = _determine_vs30_columns(list(df.columns))

        required_cols = ["id", mean_col, std_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"CSV file {csv_file_path} is missing required columns: {missing_cols}"
            )

        id_to_vs30_values = {}
        for _, row in df.iterrows():
            category_id = int(row["id"])
            mean_vs30 = float(row[mean_col])
            stddev_vs30 = float(row[std_col])
            id_to_vs30_values[category_id] = (mean_vs30, stddev_vs30)

    # Read ID raster
    with rasterio.open(id_raster_path) as src:
        id_array = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    # Create output arrays
    vs30_array = np.full(id_array.shape, constants.MODEL_NODATA, dtype=np.float32)
    stdv_array = np.full(id_array.shape, constants.MODEL_NODATA, dtype=np.float32)

    # Map pixel IDs to VS30 values
    unique_ids = np.unique(id_array)
    valid_ids = unique_ids[(unique_ids != constants.ID_NODATA) & (unique_ids != 0)]

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

    # Write output raster
    output_profile = {
        "driver": "GTiff",
        "width": profile["width"],
        "height": profile["height"],
        "count": 2,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": constants.MODEL_NODATA,
        "compress": "deflate",
    }

    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(vs30_array, 1)
        dst.write(stdv_array, 2)
        dst.descriptions = ("Vs30", "Standard Deviation")

    print(f"Completed VS30 raster: {output_path}")
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
    print("  Creating coast distance raster...")
    if not COAST_SHAPEFILE.exists():
        raise FileNotFoundError(f"Coast shapefile not found: {COAST_SHAPEFILE}")

    # Get template bounds for final output extent
    dx = template_profile["transform"].a
    dy = abs(template_profile["transform"].e)
    s_xmin = template_profile["transform"].c
    s_ymax = template_profile["transform"].f
    s_xmax = s_xmin + template_profile["width"] * dx
    s_ymin = s_ymax - template_profile["height"] * dy

    # Extend to full NZ land coverage to ensure accurate distances
    # (matching legacy _full_land_grid behavior)
    g_xmin = min(constants.FULL_NZ_LAND_XMIN, s_xmin)
    g_xmax = max(constants.FULL_NZ_LAND_XMAX, s_xmax)
    g_ymin = min(constants.FULL_NZ_LAND_YMIN, s_ymin)
    g_ymax = max(constants.FULL_NZ_LAND_YMAX, s_ymax)

    # Check if we need to extend beyond template bounds
    gridmod = g_xmin < s_xmin or g_xmax > s_xmax or g_ymin < s_ymin or g_ymax > s_ymax

    # Rasterize land polygons using GDAL (legacy approach)
    # Use UInt16 data type as in legacy code (sufficient for distance range)
    ds = gdal.Rasterize(
        str(output_path),
        str(COAST_SHAPEFILE),
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

    # If grid was extended, resample back to template bounds
    if gridmod:
        # Read the extended raster
        with rasterio.open(output_path) as src:
            extended_data = src.read(1)

        # Calculate offsets for template region
        col_off = round((s_xmin - g_xmin) / dx)
        row_off = round((g_ymax - s_ymax) / dy)

        # Extract template region
        distance_meters = extended_data[
            row_off : row_off + template_profile["height"],
            col_off : col_off + template_profile["width"],
        ]

        # Overwrite file with cropped data
        profile = template_profile.copy()
        profile.update(
            {"dtype": "float32", "count": 1, "nodata": None, "compress": "deflate"}
        )

        with rasterio.open(output_path, "w", **profile) as dst:
            dst.write(distance_meters.astype(np.float32), 1)
            dst.descriptions = ("Distance to Coast (m)",)
    else:
        # No resampling needed, just read the data
        with rasterio.open(output_path) as src:
            distance_meters = src.read(1).astype(np.float32)

        # Update profile for consistency
        profile = template_profile.copy()
        profile.update(
            {"dtype": "float32", "count": 1, "nodata": None, "compress": "deflate"}
        )

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
    print("  Creating slope raster...")
    if not SLOPE_RASTER.exists():
        raise FileNotFoundError(f"Slope raster not found: {SLOPE_RASTER}")

    # Initialize destination array
    destination = np.zeros((template_profile["height"], template_profile["width"]))

    with rasterio.open(SLOPE_RASTER) as src:
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
        {"dtype": "float32", "count": 1, "nodata": constants.SLOPE_NODATA, "compress": "deflate"}
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(destination, 1)
        dst.descriptions = ("Slope",)

    return destination, profile


def apply_hybrid_geology_modifications(
    vs30_array: np.ndarray,
    stdv_array: np.ndarray,
    id_array: np.ndarray,
    slope_array: np.ndarray,
    coast_dist_array: np.ndarray,
    mod6: bool = True,
    mod13: bool = True,
    hybrid: bool = True,
    # Hybrid parameters (required)
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

    Implements the logic from `_hyb_calc` in the old `model_geology.py`.
    Modifies arrays in-place where possible but returns them for clarity.

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
        Whether to apply modification for Group 6 (Alluvium), by default True.
    mod13 : bool, optional
        Whether to apply modification for Group 13 (Floodplain), by default True.
    hybrid : bool, optional
        Whether to apply general hybrid slope-based modifications, by default True.
    hybrid_mod6_dist_min : float, optional
        Min distance threshold for mod6.
    ...

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Modified (vs30_array, stdv_array).
    """
    print("  Applying slope and coastal distance based geology modifications...")

    # Validate required params if mods are active
    if mod6 and (
        hybrid_mod6_dist_min is None
        or hybrid_mod6_dist_max is None
        or hybrid_mod6_vs30_min is None
        or hybrid_mod6_vs30_max is None
    ):
        raise ValueError("Missing required parameters for modification 6 (Alluvium)")
    if mod13 and (
        hybrid_mod13_dist_min is None
        or hybrid_mod13_dist_max is None
        or hybrid_mod13_vs30_min is None
        or hybrid_mod13_vs30_max is None
    ):
        raise ValueError("Missing required parameters for modification 13 (Floodplain)")

    # 1. Update Standard Deviation for specific groups
    if hybrid:
        # group IDs have reduction factors loaded from config
        for gid_str, factor in constants.HYBRID_SIGMA_REDUCTION_FACTORS.items():
            gid = int(gid_str)
            # Find pixels with this ID
            mask = id_array == gid
            stdv_array[mask] *= factor

    # 2. Hybrid slope-based VS30 calculation
    if hybrid:
        # Prevent log10(0) or log10(-NODATA) by capping at min_slope_for_log
        modified_slope = np.copy(slope_array)
        modified_slope[(modified_slope <= 0) | (modified_slope == constants.SLOPE_NODATA)] = (
            constants.MIN_SLOPE_FOR_LOG
        )
        safe_log_slope = np.log10(modified_slope)

        for spec in constants.HYBRID_VS30_PARAMS:
            gid = spec["gid"]
            slope_limits = spec["slope_limits"]
            # Compute log10 of vs30 values at runtime
            vs30_limits_log10 = np.log10(np.array(spec["vs30_values"]))

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

    # 3. Distance-based modification for alluvium (GID 4)
    if mod6:
        # GID 4 = "06_alluvium"
        mask = id_array == 4
        if np.any(mask):
            # Formula: 240 + (500-240) * (dist - 8000) / (20000 - 8000)
            # Clamped between 240 and 500
            # dist is in meters.

            dist_vals = coast_dist_array[mask]

            val = hybrid_mod6_vs30_min + (
                hybrid_mod6_vs30_max - hybrid_mod6_vs30_min
            ) * (dist_vals - hybrid_mod6_dist_min) / (
                hybrid_mod6_dist_max - hybrid_mod6_dist_min
            )
            val = np.clip(val, hybrid_mod6_vs30_min, hybrid_mod6_vs30_max)

            vs30_array[mask] = val

    if mod13:
        # GID 10 = "13_floodplain"
        mask = id_array == 10
        if np.any(mask):
            # Formula: 197 + (500-197) * (dist - 8000) / (20000 - 8000)
            # Clamped between 197 and 500

            dist_vals = coast_dist_array[mask]

            val = hybrid_mod13_vs30_min + (
                hybrid_mod13_vs30_max - hybrid_mod13_vs30_min
            ) * (dist_vals - hybrid_mod13_dist_min) / (
                hybrid_mod13_dist_max - hybrid_mod13_dist_min
            )
            val = np.clip(val, hybrid_mod13_vs30_min, hybrid_mod13_vs30_max)

            vs30_array[mask] = val

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
    # Use source slope raster if not specified
    if slope_raster_path is None:
        slope_raster_path = SLOPE_RASTER

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
        for gid_str, factor in constants.HYBRID_SIGMA_REDUCTION_FACTORS.items():
            gid = int(gid_str)
            mask = geology_ids == gid
            if np.any(mask):
                modified_stdv[mask] *= factor

    # 2. Hybrid slope-based VS30 calculation
    if hybrid:
        # Prevent log10(0) or log10(negative) by capping at min_slope_for_log
        safe_slope = np.maximum(slope_values, constants.MIN_SLOPE_FOR_LOG)
        # Handle nodata values
        safe_slope[slope_values == constants.SLOPE_NODATA] = constants.MIN_SLOPE_FOR_LOG
        safe_log_slope = np.log10(safe_slope)

        for spec in constants.HYBRID_VS30_PARAMS:
            gid = spec["gid"]
            slope_limits = spec["slope_limits"]
            vs30_limits_log10 = np.log10(np.array(spec["vs30_values"]))

            # Skip ID 4 if mod6 is active (handled separately)
            if gid == 4 and mod6:
                continue

            mask = geology_ids == gid
            if np.any(mask):
                interpolated_val = np.interp(
                    safe_log_slope[mask], slope_limits, vs30_limits_log10
                )
                modified_vs30[mask] = 10**interpolated_val

    # 3. Distance-based modification for alluvium (GID 4)
    if mod6 and coast_dist_values is not None:
        mask = geology_ids == 4
        if np.any(mask):
            dist_vals = coast_dist_values[mask]
            val = constants.HYBRID_MOD6_VS30_MIN + (
                constants.HYBRID_MOD6_VS30_MAX - constants.HYBRID_MOD6_VS30_MIN
            ) * (dist_vals - constants.HYBRID_MOD6_DIST_MIN) / (
                constants.HYBRID_MOD6_DIST_MAX - constants.HYBRID_MOD6_DIST_MIN
            )
            val = np.clip(val, constants.HYBRID_MOD6_VS30_MIN, constants.HYBRID_MOD6_VS30_MAX)
            modified_vs30[mask] = val

    # 4. Distance-based modification for floodplain (GID 10)
    if mod13 and coast_dist_values is not None:
        mask = geology_ids == 10
        if np.any(mask):
            dist_vals = coast_dist_values[mask]
            val = constants.HYBRID_MOD13_VS30_MIN + (
                constants.HYBRID_MOD13_VS30_MAX - constants.HYBRID_MOD13_VS30_MIN
            ) * (dist_vals - constants.HYBRID_MOD13_DIST_MIN) / (
                constants.HYBRID_MOD13_DIST_MAX - constants.HYBRID_MOD13_DIST_MIN
            )
            val = np.clip(val, constants.HYBRID_MOD13_VS30_MIN, constants.HYBRID_MOD13_VS30_MAX)
            modified_vs30[mask] = val

    return modified_vs30, modified_stdv

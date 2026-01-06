"""
Common functions for creating categorical VS30 rasters from terrain and geology data.

This module provides unified functionality for both terrain and geology models,
using rasterio for all raster operations.
"""

import importlib.resources
import tarfile
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject
from scipy.ndimage import distance_transform_edt
from tqdm import tqdm

from vs30 import constants

# Grid parameters from constants
XMIN = constants.GRID_XMIN
XMAX = constants.GRID_XMAX
YMIN = constants.GRID_YMIN
YMAX = constants.GRID_YMAX
DX = constants.GRID_DX
DY = constants.GRID_DY
NX = round((XMAX - XMIN) / DX)
NY = round((YMAX - YMIN) / DY)

# NoData values from constants
MODEL_NODATA = constants.MODEL_NODATA
ID_NODATA = constants.ID_NODATA
SLOPE_NODATA = constants.SLOPE_NODATA

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

# Hybrid model vs30 based on interpolation of slope (from old model_geology.py)
# group ID, log10(slope) array, vs30 array
HYBRID_VS30_PARAMS = [
    [2, [-1.85, -1.22], np.log10(np.array([242, 418]))],
    [3, [-2.70, -1.35], np.log10(np.array([171, 228]))],
    [4, [-3.44, -0.88], np.log10(np.array([252, 275]))],
    [6, [-3.56, -0.93], np.log10(np.array([183, 239]))],
]
# Hybrid model sigma reduction factors
# IDs, Factors
HYBRID_SRF_IDS = np.array([2, 3, 4, 6])
HYBRID_SRF_FACTORS = np.array([0.4888, 0.7103, 0.9988, 0.9348])

# CRS for NZTM from constants
NZTM_CRS = constants.NZTM_CRS

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

        # Convert DataFrame to numpy array with float64 dtype
        return df.values.astype(np.float64)


def create_category_id_raster(
    model_type: str,
    output_dir: Path,
    xmin: float = XMIN,
    xmax: float = XMAX,
    ymin: float = YMIN,
    ymax: float = YMAX,
    dx: float = DX,
    dy: float = DY,
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
        "crs": NZTM_CRS,
        "transform": dst_transform,
        "nodata": ID_NODATA,
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
                    dst_crs=NZTM_CRS,
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
        if gdf.crs is None or str(gdf.crs) != NZTM_CRS:
            gdf = gdf.to_crs(NZTM_CRS)

        # Create shapes iterator for rasterization
        shapes = ((geom, value) for geom, value in zip(gdf.geometry, gdf.gid))

        # Rasterize to output file
        with rasterio.open(output_path, "w", **profile) as dst:
            burned = features.rasterize(
                shapes=shapes,
                out_shape=(ny, nx),
                transform=dst_transform,
                fill=ID_NODATA,
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
    print("=" * 80)
    print("Creating VS30 raster from category IDs")
    print("This process may take 1-2 minutes depending on raster size...")
    print("=" * 80)
    print("Step 1: Starting VS30 raster creation")
    print(f"  Input ID raster: {id_raster_path}")
    print(f"  CSV file: {csv_path}")
    print(f"  Output raster: {output_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure qmap.shp is extracted from shapefiles.tar.xz if needed
    # This is required for geology ID rasters (qmap.shp may not be present by default)
    print("Step 2: Ensuring qmap.shp shapefile is available")
    _ensure_qmap_shapefile_extracted()
    print("  ✓ qmap.shp shapefile is available")

    # Load CSV file and create ID-to-values mapping
    # This makes it clear that we're matching IDs between spatial file and CSV
    print("Step 3: Loading CSV file with ID-to-VS30 mapping")
    csv_file_traversable = RESOURCE_PATH / csv_path
    with importlib.resources.as_file(csv_file_traversable) as csv_file_path:
        if not csv_file_path.exists():
            raise FileNotFoundError(
                f"CSV file not found: {csv_file_path}. "
                f"Expected path relative to resources directory: {csv_path}"
            )
        print(f"  CSV file found: {csv_file_path}")

        # Read CSV file
        print("Step 4: Reading CSV file")
        df = pd.read_csv(csv_file_path, skipinitialspace=True)
        print(f"  ✓ Loaded {len(df)} rows from CSV")

        # Clean column names (strip whitespace)
        df.columns = df.columns.str.strip()

        # Determine which columns to use for VS30 values
        print("Step 5: Determining VS30 value columns")
        mean_col, std_col = _determine_vs30_columns(list(df.columns))
        print(f"  Using columns: Mean='{mean_col}', StdDev='{std_col}'")

        # Check required columns exist (id + determined value columns)
        required_cols = ["id", mean_col, std_col]
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(
                f"CSV file {csv_file_path} is missing required columns: {missing_cols}"
            )
        print("  ✓ All required columns present")

        # Create dictionary mapping ID -> (mean_vs30, stddev_vs30)
        # This dictionary makes it explicit that we're looking up values by ID
        print("Step 6: Creating ID-to-VS30 values dictionary")
        id_to_vs30_values = {}
        for _, row in df.iterrows():
            category_id = int(row["id"])
            mean_vs30 = float(row[mean_col])
            stddev_vs30 = float(row[std_col])
            id_to_vs30_values[category_id] = (mean_vs30, stddev_vs30)
        print(f"  ✓ Created dictionary with {len(id_to_vs30_values)} ID mappings")
        print(f"  IDs in CSV: {sorted(id_to_vs30_values.keys())}")

    # Read ID raster
    print("Step 7: Reading ID raster")
    with rasterio.open(id_raster_path) as src:
        id_array = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs
    print(f"  ✓ Loaded ID raster: {id_array.shape[0]} x {id_array.shape[1]} pixels")
    print(f"  CRS: {crs}")

    # Create output arrays initialized with NODATA
    print("Step 8: Creating output arrays")
    vs30_array = np.full(id_array.shape, MODEL_NODATA, dtype=np.float64)
    stdv_array = np.full(id_array.shape, MODEL_NODATA, dtype=np.float64)
    print("  ✓ Created VS30 and standard deviation arrays")

    # Map each pixel's ID to VS30 values from CSV
    # For each unique ID in the raster, look up the Vs30 values in the CSV dictionary
    print("Step 9: Mapping pixel IDs to VS30 values")
    unique_ids = np.unique(id_array)
    # Filter out NODATA and ID 0 (background) to get valid IDs for progress tracking
    valid_ids = unique_ids[(unique_ids != ID_NODATA) & (unique_ids != 0)]
    print(
        f"  Found {len(unique_ids)} unique IDs in raster: {sorted([int(x) for x in unique_ids])}"
    )

    for pixel_id in tqdm(valid_ids, desc="Mapping IDs to VS30 values", unit="ID"):
        # Look up this ID in the CSV file's ID-to-values mapping
        if pixel_id in id_to_vs30_values:
            mean_vs30, stddev_vs30 = id_to_vs30_values[pixel_id]
            # Set VS30 values for all pixels with this ID
            mask = id_array == pixel_id
            vs30_array[mask] = mean_vs30
            stdv_array[mask] = stddev_vs30
        else:
            # ID found in raster but not in CSV - this is an error
            raise ValueError(
                f"ID {pixel_id} found in raster {id_raster_path} but not in CSV {csv_path}. "
                f"Available IDs in CSV: {sorted(id_to_vs30_values.keys())}"
            )
    print("  ✓ Completed mapping all IDs to VS30 values")

    # Create output profile for 2-band Float64 raster
    print("Step 10: Preparing output raster profile")
    output_profile = {
        "driver": "GTiff",
        "width": profile["width"],
        "height": profile["height"],
        "count": 2,
        "dtype": "float64",
        "crs": crs,
        "transform": transform,
        "nodata": MODEL_NODATA,
        "compress": "deflate",
    }
    print(
        f"  ✓ Output profile created: {output_profile['width']} x {output_profile['height']}, 2 bands"
    )

    # Write output raster
    print("Step 11: Writing output raster to disk")
    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(vs30_array, 1)
        dst.write(stdv_array, 2)
        dst.descriptions = ("Vs30", "Standard Deviation")
    print(f"  ✓ Successfully wrote VS30 raster: {output_path}")

    print("Step 12: Completed VS30 raster creation")
    return output_path


def create_coast_distance_raster(
    output_path: Path, template_profile: dict
) -> tuple[np.ndarray, dict]:
    """
    Create a raster of distance to the nearest coast (in meters).

    Rasterizes the coast shapefile and calculates the Euclidean distance transform.
    Uses rasterio and scipy.ndimage.

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

    # Read coast shapefile
    gdf = gpd.read_file(COAST_SHAPEFILE)
    if gdf.crs is None or str(gdf.crs) != NZTM_CRS:
        gdf = gdf.to_crs(NZTM_CRS)

    # Legacy logic: Compute on global NZ grid to ensure correct distances for inland patches
    global_xmin = constants.GRID_XMIN
    global_xmax = constants.GRID_XMAX
    global_ymin = constants.GRID_YMIN
    global_ymax = constants.GRID_YMAX
    dx = template_profile["transform"].a
    dy = abs(template_profile["transform"].e)

    # Extend bounds to include template if it's outside default global
    s_xmin = template_profile["transform"].c
    s_ymax = template_profile["transform"].f
    s_xmax = s_xmin + template_profile["width"] * dx
    s_ymin = s_ymax - template_profile["height"] * dy

    g_xmin = min(global_xmin, s_xmin)
    g_xmax = max(global_xmax, s_xmax)
    g_ymin = min(global_ymin, s_ymin)
    g_ymax = max(global_ymax, s_ymax)

    # Calculate global grid dimensions
    gnx = round((g_xmax - g_xmin) / dx)
    gny = round((g_ymax - g_ymin) / dy)
    global_transform = from_bounds(g_xmin, g_ymin, g_xmax, g_ymax, gnx, gny)

    # Rasterize land polygons as 1, sea/background as 0 on the global grid
    shapes = ((geom, 1) for geom in gdf.geometry)
    global_mask = features.rasterize(
        shapes=shapes,
        out_shape=(gny, gnx),
        transform=global_transform,
        fill=0,
        dtype=np.uint8,
        all_touched=False,
    )

    # Distance Transform on global mask
    global_dist_px = distance_transform_edt(global_mask)
    global_dist_m = global_dist_px * dx

    # Extract study area slice
    col_off = round((s_xmin - g_xmin) / dx)
    row_off = round((g_ymax - s_ymax) / dy)

    distance_meters = global_dist_m[
        row_off : row_off + template_profile["height"],
        col_off : col_off + template_profile["width"],
    ]
    # Save to file

    # Save to file
    profile = template_profile.copy()
    profile.update(
        {"dtype": "float64", "count": 1, "nodata": None, "compress": "deflate"}
    )

    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(distance_meters, 1)
        dst.descriptions = ("Distance to Coast (m)",)

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
    destination = np.zeros(
        (template_profile["height"], template_profile["width"])
    )

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
        {"dtype": "float64", "count": 1, "nodata": SLOPE_NODATA, "compress": "deflate"}
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
        # group IDs 2, 3, 4, 6 have reduction factors
        for idx, gid in enumerate(HYBRID_SRF_IDS):
            factor = HYBRID_SRF_FACTORS[idx]
            # Find pixels with this ID
            mask = id_array == gid
            stdv_array[mask] *= factor

    # 2. Hybrid slope-based VS30 calculation
    if hybrid:
        # Prevent log10(0) or log10(-NODATA) by capping at 1e-9 (legacy logic)
        modified_slope = np.copy(slope_array)
        modified_slope[(modified_slope <= 0) | (modified_slope == SLOPE_NODATA)] = 1e-9
        safe_log_slope = np.log10(modified_slope)

        for spec in HYBRID_VS30_PARAMS:
            gid = spec[0]
            slope_limits = spec[1]
            vs30_limits_log10 = spec[2]

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

    # 3. Distance-based modification for Group 6 (Alluvium, ID=4 in old, check ID mapping?)
    # Ensure `id_array` contains these GIDs. The CSV mapping preserves GIDs.

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

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
from tqdm import tqdm

# Hard-coded grid parameters (from old code defaults)
XMIN = 1060050
XMAX = 2120050
YMIN = 4730050
YMAX = 6250050
DX = 100
DY = 100
NX = round((XMAX - XMIN) / DX)
NY = round((YMAX - YMIN) / DY)

# NoData value for model rasters
MODEL_NODATA = -32767
ID_NODATA = 255

# Resources directory path using importlib.resources
RESOURCE_PATH = importlib.resources.files("vs30") / "resources"

# Data directory path
DATA_DIR = Path(__file__).parent / "data"
TERRAIN_RASTER = DATA_DIR / "IwahashiPike.tif"
GEOLOGY_SHAPEFILE = DATA_DIR / "qmap" / "qmap.shp"

# CRS for NZTM
NZTM_CRS = "EPSG:2193"

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


def create_category_id_raster(model_type: str, output_dir: Path) -> Path:
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

    # Common setup: calculate transform and create output path
    dst_transform = from_bounds(XMIN, YMIN, XMAX, YMAX, NX, NY)
    output_filename = "tid.tif" if model_type == "terrain" else "gid.tif"
    output_path = output_dir / output_filename
    band_description = "Model ID Index"

    # Common output raster profile
    profile = {
        "driver": "GTiff",
        "width": NX,
        "height": NY,
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
                out_shape=(NY, NX),
                transform=dst_transform,
                fill=ID_NODATA,
                dtype=np.uint8,
                all_touched=False,
            )
            dst.write(burned, 1)
            dst.descriptions = (band_description,)

    return output_path


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

        # Check required columns exist
        print("Step 5: Validating CSV columns")
        required_cols = ["id", "mean_vs30_km_per_s", "standard_deviation_vs30_km_per_s"]
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
            mean_vs30 = float(row["mean_vs30_km_per_s"])
            stddev_vs30 = float(row["standard_deviation_vs30_km_per_s"])
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
    # Filter out NODATA to get valid IDs for progress tracking
    valid_ids = unique_ids[unique_ids != ID_NODATA]
    print(
        f"  Found {len(valid_ids)} unique IDs in raster: {sorted([int(x) for x in valid_ids])}"
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

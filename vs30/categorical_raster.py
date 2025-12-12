"""
Common functions for creating categorical VS30 rasters from terrain and geology data.

This module provides unified functionality for both terrain and geology models,
using rasterio for all raster operations.
"""

import importlib.resources
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.transform import from_bounds
from rasterio.warp import reproject

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
    id_raster_path: Path, model_values: np.ndarray, output_path: Path
) -> Path:
    """
    Create VS30 mean and standard deviation raster from category ID raster.

    Parameters
    ----------
    id_raster_path : Path
        Path to input category ID raster.
    model_values : np.ndarray
        Array of shape (n_categories, 2) with [mean_vs30, stddev_vs30] per category.
        Index 0 corresponds to category ID 1, index 1 to category ID 2, etc.
    output_path : Path
        Path where output 2-band raster will be saved.

    Returns
    -------
    Path
        Path to created VS30 raster file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read ID raster
    with rasterio.open(id_raster_path) as src:
        id_array = src.read(1)
        profile = src.profile.copy()
        transform = src.transform
        crs = src.crs

    # Create lookup arrays (index 0 for NODATA, then model values)
    # Category IDs in raster are 1-indexed, so we need to map:
    # ID 1 -> model_values[0], ID 2 -> model_values[1], etc.
    vs30_lookup = np.full(len(model_values) + 1, MODEL_NODATA, dtype=np.float64)
    stdv_lookup = np.full(len(model_values) + 1, MODEL_NODATA, dtype=np.float64)

    # Fill lookup arrays (skip index 0 which is for NODATA)
    vs30_lookup[1:] = model_values[:, 0]  # mean VS30
    stdv_lookup[1:] = model_values[:, 1]  # standard deviation

    # Map IDs to VS30 values
    # Handle NODATA: where id_array == ID_NODATA, use index 0 (which has MODEL_NODATA)
    # For valid IDs, use the ID as index (IDs are 1-indexed)
    id_indices = np.where(id_array == ID_NODATA, 0, id_array)
    # Clamp indices to valid range
    id_indices = np.clip(id_indices, 0, len(vs30_lookup) - 1)

    vs30_array = vs30_lookup[id_indices].astype(np.float64)
    stdv_array = stdv_lookup[id_indices].astype(np.float64)

    # Create output profile for 2-band Float64 raster
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

    # Write output raster
    with rasterio.open(output_path, "w", **output_profile) as dst:
        dst.write(vs30_array, 1)
        dst.write(stdv_array, 2)
        dst.descriptions = ("Vs30", "Standard Deviation")

    return output_path

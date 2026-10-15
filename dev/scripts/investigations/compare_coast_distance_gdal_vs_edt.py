"""Compare grid coast distance from GDAL's proximity with rasterio + scipy's exact transform.

Evidence for replacing GDAL in ``raster.compute_coast_distance_raster`` (October
2026 wrap-up). Both versions rasterize the bundled coastline onto the grid padded
by the 20 km cap, then measure from each land pixel centre to the nearest sea
pixel centre. Needs the GDAL Python bindings (``osgeo``) and the vs30 package.

Results on 2026-10-15 (vs30_refactor at 5569310):

- Wellington 30 km @ 100 m: 99.85% identical, max diff 17.2 m (0.17 px)
- Christchurch 30 km @ 100 m: 98.39% identical within 1 mm, 21 px differ by > 1 m, max 17.2 m
- All NZ @ 1 km / 500 m / 200 m: 99.73% / 99.80% / 99.82% identical, max 0.21–0.25 px
- GDAL's distance is never the shorter one: where they differ, its propagation missed
  the true nearest sea pixel (e.g. 300.0 m against the exact diagonal 282.8 m).
- 5 km grid benchmarks with the scipy version: one pixel beyond rtol 1e-3 (2.4% Vs30)
  in each of modified_foster_2019 and viktor_cpt_clustering; jaehwi_v1p0 (no coastal
  adjustment) and points mode are unaffected.
"""

import math
import tempfile
import time
from pathlib import Path

import numpy as np
import rasterio
import rasterio.features
import rasterio.transform
import scipy.ndimage
from osgeo import gdal

from vs30 import constants, raster

gdal.UseExceptions()
MAX_DISTANCE = max(constants.HYBRID_GID4_DIST_MAX, constants.HYBRID_GID10_DIST_MAX)


def padded_grid(template_profile):
    transform = template_profile["transform"]
    dx, dy = transform.a, abs(transform.e)
    pad_cols, pad_rows = math.ceil(MAX_DISTANCE / dx), math.ceil(MAX_DISTANCE / dy)
    return transform, dx, dy, pad_cols, pad_rows


def gdal_coast_distance(template_profile):
    transform, dx, dy, pad_cols, pad_rows = padded_grid(template_profile)
    xmin, ymax = transform.c, transform.f
    xmax = xmin + template_profile["width"] * dx
    ymin = ymax - template_profile["height"] * dy
    with tempfile.TemporaryDirectory() as tmp:
        path = str(Path(tmp) / "coast.tif")
        ds = gdal.Rasterize(
            path,
            str(constants.GEOSPATIAL_DIR / constants.COASTLINE_SHAPEFILE_PATH),
            outputBounds=[
                xmin - pad_cols * dx,
                ymin - pad_rows * dy,
                xmax + pad_cols * dx,
                ymax + pad_rows * dy,
            ],
            xRes=dx,
            yRes=dy,
            noData=0,
            burnValues=1,
            outputType=gdal.GetDataTypeByName("Float32"),
        )
        band = ds.GetRasterBand(1)
        gdal.ComputeProximity(
            band,
            band,
            ["VALUES=0", "DISTUNITS=GEO", f"MAXDIST={MAX_DISTANCE}", "NODATA=-1"],
        )
        ds = None
        with rasterio.open(path) as src:
            data = src.read(1)
    distance = data[
        pad_rows : pad_rows + template_profile["height"],
        pad_cols : pad_cols + template_profile["width"],
    ].astype(np.float32)
    distance[distance < 0] = MAX_DISTANCE
    return distance


def edt_coast_distance(template_profile):
    transform, dx, dy, pad_cols, pad_rows = padded_grid(template_profile)
    land = rasterio.features.rasterize(
        ((geometry, 1) for geometry in raster.load_coast_shapefile().geometry),
        out_shape=(
            template_profile["height"] + 2 * pad_rows,
            template_profile["width"] + 2 * pad_cols,
        ),
        transform=rasterio.transform.from_origin(
            transform.c - pad_cols * dx, transform.f + pad_rows * dy, dx, dy
        ),
        dtype=np.uint8,
    )
    if land.all():
        distance = np.full(land.shape, MAX_DISTANCE)
    else:
        distance = scipy.ndimage.distance_transform_edt(land, sampling=(dy, dx))
    return np.minimum(distance, MAX_DISTANCE)[
        pad_rows : pad_rows + template_profile["height"],
        pad_cols : pad_cols + template_profile["width"],
    ].astype(np.float32)


def compare(name, xmin, xmax, ymin, ymax, d):
    profile = {
        "transform": rasterio.transform.from_origin(xmin, ymax, d, d),
        "width": round((xmax - xmin) / d),
        "height": round((ymax - ymin) / d),
    }
    t0 = time.perf_counter()
    gdal_dist = gdal_coast_distance(profile).astype(np.float64)
    t1 = time.perf_counter()
    edt_dist = edt_coast_distance(profile).astype(np.float64)
    t2 = time.perf_counter()
    diff = edt_dist - gdal_dist
    print(
        f"{name} @ {d} m ({profile['width']}x{profile['height']} px): "
        f"GDAL {t1 - t0:.1f} s, scipy {t2 - t1:.1f} s | "
        f"identical within 1 mm: {np.mean(np.abs(diff) < 1e-3) * 100:.2f}% | "
        f"scipy longer by > 1 m: {np.sum(diff > 1)} px, shorter by > 1 m: {np.sum(diff < -1)} px | "
        f"max |diff| {np.abs(diff).max():.2f} m ({np.abs(diff).max() / d:.3f} px)"
    )


if __name__ == "__main__":
    compare("Wellington 30 km", 1735000, 1765000, 5410000, 5440000, 100)
    compare("Christchurch 30 km", 1555000, 1585000, 5165000, 5195000, 100)
    for spacing in (1000, 500, 200):
        compare("All NZ", 1060050, 2120050, 4730050, 6250050, spacing)

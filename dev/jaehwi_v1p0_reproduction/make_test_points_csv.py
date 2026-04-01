"""Create a lon/lat CSV of 200 sample points for running through Jaehwi's code."""
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer

REFERENCE_SUBGRID = Path("/home/arr65/src/vs30/dev/jaehwi_v1p0_reproduction/reference_subgrid.tif")
OUTPUT = Path("/home/arr65/src/vs30/dev/jaehwi_v1p0_reproduction/test_points.csv")

nztm2wgs = Transformer.from_crs(2193, 4326, always_xy=True)

rng = np.random.default_rng(42)
with rasterio.open(REFERENCE_SUBGRID) as src:
    vs30_band = src.read(1)
    nodata = src.nodata
    transform = src.transform

valid = (vs30_band != nodata) & (~np.isnan(vs30_band)) & (vs30_band > 0)
valid_rows, valid_cols = np.where(valid)
idx = rng.choice(len(valid_rows), size=200, replace=False)
rows, cols = valid_rows[idx], valid_cols[idx]

eastings = transform.c + (cols + 0.5) * transform.a
northings = transform.f + (rows + 0.5) * transform.e
longitudes, latitudes = nztm2wgs.transform(eastings, northings)

with open(OUTPUT, "w") as f:
    f.write("longitude latitude\n")
    for lon, lat in zip(longitudes, latitudes):
        f.write(f"{lon:.10f} {lat:.10f}\n")

print(f"Wrote {len(longitudes)} points to {OUTPUT}")

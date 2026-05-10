"""
Directly compare refactored compute_coast_distance_raster vs legacy coast.tif
for the BENCHMARK_NZ_GRID template.
"""

from pathlib import Path

import numpy as np
import rasterio

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from vs30 import raster, constants, config

BENCHMARK_NZ_GRID = config.GridConfig(
    grid_xmin=1060100,
    grid_xmax=2120100,
    grid_ymin=4730100,
    grid_ymax=6250100,
    grid_dx=400,
    grid_dy=400,
)

LEGACY_COAST = Path("/home/arr65/data/vs30/grid_models/from_original_code/modified_foster_2019/coast.tif")

# Build template profile for refactored code
from rasterio.transform import from_origin
template_profile = {
    "transform": from_origin(BENCHMARK_NZ_GRID.grid_xmin, BENCHMARK_NZ_GRID.grid_ymax,
                             BENCHMARK_NZ_GRID.grid_dx, BENCHMARK_NZ_GRID.grid_dy),
    "width": int((BENCHMARK_NZ_GRID.grid_xmax - BENCHMARK_NZ_GRID.grid_xmin) / BENCHMARK_NZ_GRID.grid_dx),
    "height": int((BENCHMARK_NZ_GRID.grid_ymax - BENCHMARK_NZ_GRID.grid_ymin) / BENCHMARK_NZ_GRID.grid_dy),
    "crs": rasterio.crs.CRS.from_string(constants.NZTM_CRS),
}

print(f"Template: {template_profile['width']}x{template_profile['height']}, transform: {template_profile['transform']}")

print("Computing refactored coast distance...")
refactored_coast = raster.compute_coast_distance_raster(template_profile)
print(f"Refactored coast shape: {refactored_coast.shape}")

with rasterio.open(LEGACY_COAST) as src:
    legacy_coast = src.read(1)
    legacy_transform = src.transform
    print(f"Legacy coast shape: {legacy_coast.shape}, transform: {legacy_transform}")

# Compare at specific worst-offending pixels
pixels = [(543, 1624), (552, 1625), (543, 1623), (542, 1623), (540, 1624),
          (541, 1623), (550, 1626), (553, 1624), (553, 1623), (542, 1621)]

print(f"\n{'(r,c)':>14} {'legacy':>10} {'refactored':>12} {'diff':>8}")
for r, c in pixels:
    leg = legacy_coast[r, c]
    ref = refactored_coast[r, c]
    print(f"({r:>5},{c:>5}) {leg:>10.1f} {ref:>12.1f} {ref-leg:>8.1f}")

# Overall statistics
print("\nOverall difference statistics:")
diff = refactored_coast.astype(np.float64) - legacy_coast.astype(np.float64)
print(f"  Max abs diff: {np.max(np.abs(diff)):.2f}")
print(f"  Mean abs diff: {np.mean(np.abs(diff)):.2f}")
print(f"  Pixels where diff > 10m: {np.sum(np.abs(diff) > 10)}")
print(f"  Pixels where diff > 100m: {np.sum(np.abs(diff) > 100)}")
print(f"  Pixels where diff > 1000m: {np.sum(np.abs(diff) > 1000)}")

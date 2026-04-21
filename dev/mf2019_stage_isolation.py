"""
Isolate which pipeline stage produces the mf2019 divergence.

Compares refactored test output vs legacy intermediate files (geology.tif vs
geology_mvn.tif, terrain.tif vs terrain_mvn.tif, combined.tif vs combined_mvn.tif)
at the worst-offending pixels.
"""

from pathlib import Path

import numpy as np
import rasterio

DUMP_DIR = Path("/tmp/vs30_test_dumps")
LEGACY_DIR = Path("/home/arr65/data/vs30/grid_models/from_original_code/modified_foster_2019")
BENCHMARK = Path("/home/arr65/src/Vs30/tests/benchmarks/modified_foster_2019.tif")

# Load refactored (actual) combined output
act_vs30 = np.load(DUMP_DIR / "modified_foster_2019_actual_vs30.npy").astype(np.float64)

# Load legacy combined_mvn.tif (this is the benchmark — should match exactly if refactor is correct)
with rasterio.open(BENCHMARK) as src:
    bench_vs30 = src.read(1).astype(np.float64)
    nodata = src.nodata
    transform = src.transform
    crs = src.crs

print(f"Benchmark shape: {bench_vs30.shape}, transform: {transform}")

# The legacy intermediate .tif files may have a different grid! Let me check
for name in ["geology.tif", "terrain.tif", "geology_mvn.tif", "terrain_mvn.tif", "combined.tif", "combined_mvn.tif"]:
    path = LEGACY_DIR / name
    with rasterio.open(path) as src:
        print(f"{name:24s} shape={src.shape}  transform={src.transform!r}  band_count={src.count}")

# Find the worst offenders
bench_valid = (~np.isnan(bench_vs30)) & (bench_vs30 != nodata)
act_valid = (~np.isnan(act_vs30)) & (act_vs30 != nodata)
both = bench_valid & act_valid
abs_diff = np.zeros_like(act_vs30)
abs_diff[both] = np.abs(act_vs30[both] - bench_vs30[both])
rel_diff = np.zeros_like(act_vs30)
rel_diff[both] = abs_diff[both] / np.maximum(np.abs(bench_vs30[both]), 1e-30)

# Top 10 worst offenders
flat_rel = rel_diff.flatten()
worst_idx = np.argsort(flat_rel)[-10:][::-1]
rows_flat = worst_idx // act_vs30.shape[1]
cols_flat = worst_idx % act_vs30.shape[1]

# Get transforms from legacy files
with rasterio.open(LEGACY_DIR / "geology.tif") as src:
    gx_transform = src.transform
    gx_shape = src.shape
    geo_premvn = src.read(1).astype(np.float64)
    geo_nodata = src.nodata

with rasterio.open(LEGACY_DIR / "geology_mvn.tif") as src:
    geo_postmvn = src.read(1).astype(np.float64)

with rasterio.open(LEGACY_DIR / "terrain.tif") as src:
    tx_transform = src.transform
    ter_premvn = src.read(1).astype(np.float64)
    ter_nodata = src.nodata

with rasterio.open(LEGACY_DIR / "terrain_mvn.tif") as src:
    ter_postmvn = src.read(1).astype(np.float64)

with rasterio.open(LEGACY_DIR / "combined.tif") as src:
    comb_premvn = src.read(1).astype(np.float64)

# Map benchmark pixel (row, col) to NZTM, then to legacy pixel indices
print(f"\n{'bench(r,c)':>16} {'NZTM':>26} {'act':>10} {'bench':>10} {'diff%':>7} {'g_pre':>10} {'g_post':>10} {'t_pre':>10} {'t_post':>10} {'comb_pre':>10}")
for i in range(10):
    br = int(rows_flat[i])
    bc = int(cols_flat[i])
    x = transform.c + bc * transform.a + transform.a / 2
    y = transform.f + br * transform.e + transform.e / 2  # transform.e is negative

    # Find corresponding legacy pixel
    # legacy transform uses origin=(lx0, ly0), cellsize=lx, ly
    lx0 = gx_transform.c
    ly0 = gx_transform.f
    ldx = gx_transform.a
    ldy = -gx_transform.e
    lc = int(round((x - lx0 - ldx/2) / ldx))
    lr = int(round((ly0 - ldy/2 - y) / ldy))

    if 0 <= lr < geo_premvn.shape[0] and 0 <= lc < geo_premvn.shape[1]:
        g_pre = geo_premvn[lr, lc]
        g_post = geo_postmvn[lr, lc]
        t_pre = ter_premvn[lr, lc]
        t_post = ter_postmvn[lr, lc]
        c_pre = comb_premvn[lr, lc]
    else:
        g_pre = g_post = t_pre = t_post = c_pre = np.nan

    print(f"({br:>5},{bc:>5}) ({x:>11.0f},{y:>11.0f}) {act_vs30[br,bc]:>10.3f} {bench_vs30[br,bc]:>10.3f} {100*rel_diff[br,bc]:>6.3f}% {g_pre:>10.3f} {g_post:>10.3f} {t_pre:>10.3f} {t_post:>10.3f} {c_pre:>10.3f}")

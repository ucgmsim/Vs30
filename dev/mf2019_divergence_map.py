"""
Spatially characterize the mf2019 divergence cluster.

Questions answered:
- Are the 13052 diverging pixels clustered or scattered?
- What's the geographic extent?
- Does the pattern correlate with observation locations?
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio

DUMP_DIR = Path("/tmp/vs30_test_dumps")
BENCHMARK = Path("/home/arr65/src/Vs30/tests/benchmarks/modified_foster_2019.tif")
RTOL = 1e-3

act = np.load(DUMP_DIR / "modified_foster_2019_actual_vs30.npy").astype(np.float64)

with rasterio.open(BENCHMARK) as src:
    exp = src.read(1).astype(np.float64)
    nodata = src.nodata
    transform = src.transform

xmin = transform.c
ymax = transform.f
dx = transform.a
dy = -transform.e
nrows, ncols = act.shape

act_valid = ~np.isnan(act) & (act != nodata)
exp_valid = ~np.isnan(exp) & (exp != nodata)
both = act_valid & exp_valid

abs_diff = np.zeros_like(act)
rel_diff = np.zeros_like(act)
abs_diff[both] = np.abs(act[both] - exp[both])
rel_diff[both] = abs_diff[both] / np.maximum(np.abs(exp[both]), 1e-30)

viol = (rel_diff > RTOL) & both
rows, cols = np.where(viol)

print(f"Total violators: {viol.sum()}")
print(f"Bounding rows: {rows.min()} to {rows.max()}")
print(f"Bounding cols: {cols.min()} to {cols.max()}")

# Cluster by 20x20 superpixel bins to see spatial distribution
bin_r = rows // 40
bin_c = cols // 40
bins, counts = np.unique(list(zip(bin_r.tolist(), bin_c.tolist())), axis=0, return_counts=True)
order = np.argsort(-counts)

print(f"\nTop 20 hotspot bins (40x40 pixel superpixel):")
print(f"{'bin_r':>6} {'bin_c':>6} {'count':>6} {'x_range':>26} {'y_range':>26}")
for i in order[:20]:
    br, bc = bins[i]
    cnt = counts[i]
    x0 = xmin + bc * 40 * dx
    x1 = x0 + 40 * dx
    y1 = ymax - br * 40 * dy
    y0 = y1 - 40 * dy
    print(f"{br:>6} {bc:>6} {cnt:>6} [{x0:>11.0f}, {x1:>11.0f}] [{y0:>11.0f}, {y1:>11.0f}]")

# See if the violators form contiguous regions
from scipy.ndimage import label
viol_bool = viol
labeled, nlabels = label(viol_bool)
print(f"\nNumber of connected components of violators: {nlabels}")
sizes = np.bincount(labeled.ravel())[1:]
top_sizes = sorted(sizes, reverse=True)[:10]
print(f"Top 10 component sizes: {top_sizes}")
print(f"Total pixels in top 10: {sum(top_sizes)}  ({100*sum(top_sizes)/viol.sum():.1f}%)")

# Compute center of divergence for each of top components and convert to NZTM xy
print(f"\nLargest connected components with geographic locations (NZTM):")
for rank, sz in enumerate(top_sizes[:5]):
    comp_id = np.argmax(sizes == sz) + 1  # one-indexed
    mask = labeled == comp_id
    r_idx, c_idx = np.where(mask)
    center_r = int(r_idx.mean())
    center_c = int(c_idx.mean())
    center_x = xmin + center_c * dx + dx/2
    center_y = ymax - center_r * dy - dy/2
    max_diff = rel_diff[mask].max()
    print(f"  #{rank+1}: size={sz}  center=(row={center_r},col={center_c})  NZTM=({center_x:.0f}, {center_y:.0f})  max_rel_diff={max_diff:.4f}")

# Load the observations CSV for mf2019 and see if any observations are nearby
obs_csv = Path("/home/arr65/src/Vs30/vs30/resources/observations/modified_foster_2019_measured_vs30_independent_observations.csv")
if obs_csv.exists():
    df = pd.read_csv(obs_csv)
    print(f"\nObservation CSV has {len(df)} rows, columns: {list(df.columns)}")
    # If it has NZTM columns compute which observations fall in the divergence regions
    if 'easting' in df.columns or 'x' in df.columns or 'nztm_x' in df.columns:
        ecol = 'easting' if 'easting' in df.columns else ('x' if 'x' in df.columns else 'nztm_x')
        ncol = 'northing' if 'northing' in df.columns else ('y' if 'y' in df.columns else 'nztm_y')
        # Check how many observations fall in each top component
        for rank, sz in enumerate(top_sizes[:5]):
            comp_id = np.argmax(sizes == sz) + 1
            mask = labeled == comp_id
            r_idx, c_idx = np.where(mask)
            x_min_c = xmin + c_idx.min() * dx
            x_max_c = xmin + (c_idx.max()+1) * dx
            y_max_c = ymax - r_idx.min() * dy
            y_min_c = ymax - (r_idx.max()+1) * dy
            in_comp = ((df[ecol] >= x_min_c) & (df[ecol] < x_max_c) &
                       (df[ncol] >= y_min_c) & (df[ncol] < y_max_c))
            print(f"  Component #{rank+1}: bbox ({x_min_c:.0f}-{x_max_c:.0f}, {y_min_c:.0f}-{y_max_c:.0f}) has {in_comp.sum()} observations")

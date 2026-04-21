"""
Second-stage diagnostic: compare refactored stdv at worst pixels against legacy
pre-MVN and post-MVN stdv, and compare the combined pre-MVN array.
"""

from pathlib import Path

import numpy as np
import rasterio

DUMP_DIR = Path("/tmp/vs30_test_dumps")
LEGACY_DIR = Path("/home/arr65/data/vs30/grid_models/from_original_code/modified_foster_2019")
BENCHMARK = Path("/home/arr65/src/Vs30/tests/benchmarks/modified_foster_2019.tif")

# Load refactored outputs
act_vs30 = np.load(DUMP_DIR / "modified_foster_2019_actual_vs30.npy").astype(np.float64)
act_stdv = np.load(DUMP_DIR / "modified_foster_2019_actual_stdv.npy").astype(np.float64)

with rasterio.open(BENCHMARK) as src:
    bench_vs30 = src.read(1).astype(np.float64)
    bench_stdv = src.read(2).astype(np.float64)
    nodata = src.nodata
    transform = src.transform

# Load legacy intermediate files (all bands)
def load2band(path):
    with rasterio.open(path) as src:
        return src.read(1).astype(np.float64), src.read(2).astype(np.float64)

g_pre_vs30, g_pre_stdv = load2band(LEGACY_DIR / "geology.tif")
g_post_vs30, g_post_stdv = load2band(LEGACY_DIR / "geology_mvn.tif")
t_pre_vs30, t_pre_stdv = load2band(LEGACY_DIR / "terrain.tif")
t_post_vs30, t_post_stdv = load2band(LEGACY_DIR / "terrain_mvn.tif")
c_pre_vs30, c_pre_stdv = load2band(LEGACY_DIR / "combined.tif")

# Compute bulk stats
bench_valid = (~np.isnan(bench_vs30)) & (bench_vs30 != nodata)
act_valid = (~np.isnan(act_vs30)) & (act_vs30 != nodata)
both = bench_valid & act_valid

# Check: how often is the refactored (actual) closer to PRE-MVN than to POST-MVN?
diff_to_pre = np.abs(act_vs30 - c_pre_vs30)
diff_to_post = np.abs(act_vs30 - bench_vs30)  # benchmark IS post-MVN combined

mask = both
closer_to_pre = (diff_to_pre[mask] < diff_to_post[mask]).sum()
closer_to_post = (diff_to_pre[mask] > diff_to_post[mask]).sum()
same = (diff_to_pre[mask] == diff_to_post[mask]).sum()
print(f"Pixels where refactored is closer to legacy PRE-MVN combined: {closer_to_pre}")
print(f"Pixels where refactored is closer to legacy POST-MVN combined (benchmark): {closer_to_post}")
print(f"Pixels where refactored is equidistant: {same}")

# If refactored is generally closer to pre-MVN, MVN is the difference. Otherwise investigate.
print()

# Legacy diff within its own pipeline (MVN impact in legacy)
mvn_impact = np.abs(c_pre_vs30 - bench_vs30)
print(f"Legacy MVN impact on combined (mean, max): {mvn_impact[mask].mean():.4f}, {mvn_impact[mask].max():.4f}")

# Refactored diff vs legacy pre-MVN combined
ref_to_pre = np.abs(act_vs30 - c_pre_vs30)
print(f"Refactored vs legacy PRE-MVN combined (mean, max): {ref_to_pre[mask].mean():.4f}, {ref_to_pre[mask].max():.4f}")

# Refactored diff vs legacy post-MVN combined (the benchmark)
ref_to_post = np.abs(act_vs30 - bench_vs30)
print(f"Refactored vs legacy POST-MVN combined (mean, max): {ref_to_post[mask].mean():.4f}, {ref_to_post[mask].max():.4f}")
print()

# Show stdv details at vs30 top-10 offenders
rel_diff = np.zeros_like(act_vs30)
rel_diff[both] = np.abs(act_vs30[both] - bench_vs30[both]) / np.maximum(np.abs(bench_vs30[both]), 1e-30)
worst_idx = np.argsort(rel_diff.flatten())[-10:][::-1]
rows_flat = worst_idx // act_vs30.shape[1]
cols_flat = worst_idx % act_vs30.shape[1]
print(f"{'(r,c)':>14} {'actV':>8} {'benchV':>8} {'gPreV':>8} {'actS':>8} {'benchS':>8} {'gPreS':>8} {'gPostS':>8} {'tPreS':>8} {'tPostS':>8}")
for i in range(10):
    r, c = int(rows_flat[i]), int(cols_flat[i])
    print(f"({r:>5},{c:>5}) {act_vs30[r,c]:>8.3f} {bench_vs30[r,c]:>8.3f} {c_pre_vs30[r,c]:>8.3f} {act_stdv[r,c]:>8.4f} {bench_stdv[r,c]:>8.4f} {g_pre_stdv[r,c]:>8.4f} {g_post_stdv[r,c]:>8.4f} {t_pre_stdv[r,c]:>8.4f} {t_post_stdv[r,c]:>8.4f}")

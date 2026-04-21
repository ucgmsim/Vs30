"""
Diagnose why test_modified_foster_2019_multiprocess failed.

Compares the dumped arrays from conftest against the benchmark raster, showing:
- Mask disagreement (valid vs nodata pixel counts)
- Magnitude and location of value divergence beyond TEST_RTOL=1e-3
"""

from pathlib import Path

import numpy as np
import rasterio

DUMP_DIR = Path("/tmp/vs30_test_dumps")
BENCHMARK = Path("/home/arr65/src/Vs30/tests/benchmarks/modified_foster_2019.tif")
RTOL = 1e-3

act_vs30 = np.load(DUMP_DIR / "modified_foster_2019_actual_vs30.npy")
act_stdv = np.load(DUMP_DIR / "modified_foster_2019_actual_stdv.npy")

with rasterio.open(BENCHMARK) as src:
    exp_vs30 = src.read(1)
    exp_stdv = src.read(2)
    nodata = src.nodata

print(f"Benchmark shape: {exp_vs30.shape}, dtype: {exp_vs30.dtype}, nodata: {nodata}")
print(f"Actual shape: {act_vs30.shape}, dtype: {act_vs30.dtype}")
print()

for name, actual, expected in [("vs30", act_vs30, exp_vs30), ("stdv", act_stdv, exp_stdv)]:
    print(f"=== Band: {name} ===")
    act_valid = ~np.isnan(actual)
    exp_valid = ~np.isnan(expected)
    if nodata is not None:
        act_valid &= actual != nodata
        exp_valid &= expected != nodata

    n_act = act_valid.sum()
    n_exp = exp_valid.sum()
    only_act = (act_valid & ~exp_valid).sum()
    only_exp = (exp_valid & ~act_valid).sum()
    both = (act_valid & exp_valid).sum()

    print(f"  Valid pixels: actual={n_act}  expected={n_exp}")
    print(f"  Only in actual (refactored): {only_act}")
    print(f"  Only in expected (benchmark): {only_exp}")
    print(f"  Valid in both: {both}")

    if only_act > 0 or only_exp > 0:
        r, c = np.where(act_valid & ~exp_valid)
        if len(r):
            print(f"  Sample refactored-only pixels (row, col): {list(zip(r[:5].tolist(), c[:5].tolist()))}")
        r, c = np.where(exp_valid & ~act_valid)
        if len(r):
            print(f"  Sample benchmark-only pixels (row, col): {list(zip(r[:5].tolist(), c[:5].tolist()))}")

    if both > 0:
        both_mask = act_valid & exp_valid
        va = actual[both_mask].astype(np.float64)
        ve = expected[both_mask].astype(np.float64)
        abs_diff = np.abs(va - ve)
        rel_diff = abs_diff / np.maximum(np.abs(ve), 1e-30)

        tol_violated = rel_diff > RTOL
        n_bad = tol_violated.sum()
        print(f"  Pixels with rel_diff > {RTOL}: {n_bad} / {both}  ({100 * n_bad / both:.3f}%)")
        if n_bad:
            print(f"    max abs_diff among violators: {abs_diff[tol_violated].max():.6f}")
            print(f"    max rel_diff among violators: {rel_diff[tol_violated].max():.6f}")
            print(f"    median abs_diff among violators: {np.median(abs_diff[tol_violated]):.6f}")
            rows_both, cols_both = np.where(both_mask)
            top = np.argsort(rel_diff)[-5:][::-1]
            print(f"    Top 5 divergences:")
            for i in top:
                r_idx = rows_both[i]
                c_idx = cols_both[i]
                print(
                    f"      (row={r_idx}, col={c_idx}): actual={va[i]:.4f}  expected={ve[i]:.4f}  "
                    f"abs_diff={abs_diff[i]:.4f}  rel_diff={rel_diff[i]:.6f}"
                )
        else:
            print(f"  All matching pixels within tolerance (max abs_diff={abs_diff.max():.6e})")
    print()

"""
Diagnose remaining mask difference between refactored jaehwi_v1p0 pipeline
output and the gap-filled benchmark raster.

Runs the pipeline once, saves vs30/stdv arrays to .npy, then compares masks.
"""

import dataclasses
import sys
from pathlib import Path

import numpy as np
import rasterio

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tests"))

from vs30 import config, constants, pipeline
from conftest import load_fixed_model_config

STANDARD_NZ_GRID = dataclasses.replace(
    constants.FULL_NZ_GRID_CONFIG, grid_dx=400, grid_dy=400
)

BENCH = Path("tests/benchmarks/jaehwi_v1p0.tif")
CACHE_V = Path("dev/_jaehwi_vs30.npy")
CACHE_S = Path("dev/_jaehwi_stdv.npy")


def run_pipeline():
    cfg = load_fixed_model_config(constants.FixedModelVersion.JAEHWI_V1P0)
    result = pipeline.grid_pipeline(
        grid_config=STANDARD_NZ_GRID,
        output_dir=None,
        geology_categorical_csv=cfg["geology_categorical_csv"],
        terrain_categorical_csv=cfg["terrain_categorical_csv"],
        clustered_observations_csv=cfg["clustered_observations_csv"],
        independent_observations_csv=cfg["independent_observations_csv"],
        combination_method=constants.CombinationMethod(cfg["combination_method"]),
        combine_ratio=cfg["combine_ratio"],
        noisy=cfg["noisy"],
        do_bayesian_update=cfg["do_bayesian_update"],
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        apply_coastal_distance_mod=cfg["apply_coastal_distance_mod"],
        fill_gaps=cfg["fill_gaps"],
        geology_corr_fn=cfg["geology_corr_fn"],
        terrain_corr_fn=cfg["terrain_corr_fn"],
        nproc=1,
    )
    np.save(CACHE_V, result["combined_vs30"])
    np.save(CACHE_S, result["combined_stdv"])
    return result["combined_vs30"], result["combined_stdv"]


def compare():
    if CACHE_V.exists() and CACHE_S.exists():
        print(f"Loading cached pipeline output from {CACHE_V}")
        act_v = np.load(CACHE_V)
        act_s = np.load(CACHE_S)
    else:
        print("Running pipeline...")
        act_v, act_s = run_pipeline()

    with rasterio.open(BENCH) as src:
        exp_v = src.read(1)
        exp_s = src.read(2)
        nodata = src.nodata

    act_valid = ~np.isnan(act_v)
    exp_valid = ~np.isnan(exp_v) & (exp_v != nodata)

    print(f"Actual valid: {act_valid.sum()}")
    print(f"Expected valid: {exp_valid.sum()}")

    only_act = act_valid & ~exp_valid  # refactored valid, benchmark NODATA
    only_exp = exp_valid & ~act_valid  # benchmark valid, refactored NODATA

    print(f"Only in refactored (not benchmark): {only_act.sum()}")
    print(f"Only in benchmark (not refactored): {only_exp.sum()}")

    # Sample some pixel coordinates from each discrepancy set
    rows, cols = np.where(only_act)
    print(f"\nSample refactored-only pixels (row, col): {list(zip(rows[:10], cols[:10]))}")
    rows, cols = np.where(only_exp)
    print(f"Sample benchmark-only pixels (row, col): {list(zip(rows[:10], cols[:10]))}")


if __name__ == "__main__":
    compare()

#!/usr/bin/env python3
"""
Compare our refactored pipeline against Jaehwi's code in points mode
at 200 randomly sampled locations across New Zealand.

Generates test points from valid pixels in V1.0_26Mar.tif, runs both
codebases, and prints comparison statistics.
"""

import functools
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

REFERENCE_TIF = Path("/home/arr65/data/vs30/grid_models/jaehwi_v1p0/V1.0_26Mar.tif")
JAEHWI_CODE_DIR = Path("/home/arr65/src/Vs30_2026")
N_POINTS = 200
SEED = 42

NZTM2WGS = Transformer.from_crs(2193, 4326, always_xy=True)


def sample_valid_points(n: int, seed: int) -> np.ndarray:
    """Sample n random valid land points with valid geology and terrain IDs.

    Samples from V1.0_26Mar.tif pixel centres, then shifts by 50 m in both
    axes so that the sample points fall at terrain-raster pixel centres
    instead of on pixel boundaries.  (V1.0_26Mar.tif and IwahashiPike.tif
    share the same 100 m resolution but have a 50 m grid offset, so
    reference-TIF pixel centres coincide with terrain-raster pixel
    boundaries.  Without the shift, micrometer-level coordinate rounding
    from the NZTM -> WGS84 -> NZTM roundtrip flips ~35 % of terrain IDs.)
    """
    from vs30 import category, constants

    rng = np.random.default_rng(seed)
    with rasterio.open(REFERENCE_TIF) as src:
        data = src.read(1)
        valid_rows, valid_cols = np.where(data != src.nodata)

    # Over-sample, then filter to keep only points with valid geology
    oversample = min(n * 10, len(valid_rows))
    indices = rng.choice(len(valid_rows), size=oversample, replace=False)
    rows = valid_rows[indices]
    cols = valid_cols[indices]

    with rasterio.open(REFERENCE_TIF) as src:
        eastings = src.transform.c + (cols + 0.5) * src.transform.a
        northings = src.transform.f + (rows + 0.5) * src.transform.e

    # Shift by 50 m so points land at terrain-raster pixel centres
    eastings += 50
    northings -= 50

    points = np.column_stack([eastings, northings])

    # Filter: keep only points where geology AND terrain rasters have valid IDs
    gids = category.assign_to_category_geology(points)
    tids = category.assign_to_category_terrain(points)
    valid_mask = (gids != constants.RASTER_ID_NODATA_VALUE) & (
        tids != constants.RASTER_ID_NODATA_VALUE
    )

    points = points[valid_mask]
    if len(points) < n:
        raise RuntimeError(
            f"Only {len(points)} valid land points found (need {n}). "
            "Increase oversample factor."
        )

    return points[:n]


def run_jaehwi_points(points_nztm: np.ndarray, output_dir: Path) -> pd.DataFrame:
    """Run Jaehwi's code in points mode on the given NZTM coordinates."""
    # Convert to WGS84 lon/lat for Jaehwi's --ll-path input
    lons, lats = NZTM2WGS.transform(points_nztm[:, 0], points_nztm[:, 1])

    # Write lon/lat CSV (space-separated, no header)
    ll_csv = output_dir / "test_points_lonlat.csv"
    np.savetxt(ll_csv, np.column_stack([lons, lats]), fmt="%.10f", delimiter=" ")

    jaehwi_out = output_dir / "jaehwi_output"
    jaehwi_out.mkdir(exist_ok=True)

    cmd = [
        "python", "run_vs30calc_V1.py",
        "--gupdate", "posterior", "--tupdate", "posterior",
        "--ll-path", str(ll_csv),
        "--nproc", "1",
        "--out", str(jaehwi_out),
        "--overwrite",
    ]

    # Run with oldvs30_venv
    env_cmd = (
        "source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && "
        "source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && "
        "mamba activate oldvs30_venv && "
        f"cd {JAEHWI_CODE_DIR} && " + " ".join(cmd)
    )
    result = subprocess.run(
        ["bash", "-c", env_cmd],
        capture_output=True, text=True, timeout=600,
    )
    if result.returncode != 0:
        print("STDERR:", result.stderr[-2000:] if result.stderr else "")
        raise RuntimeError(f"Jaehwi's code failed with return code {result.returncode}")

    # Read output
    return pd.read_csv(jaehwi_out / "vs30points.csv")


def run_our_pipeline(points_nztm: np.ndarray) -> pd.DataFrame:
    """Run our refactored pipeline in points mode."""
    from vs30 import pipeline, constants, utils

    lons, lats = NZTM2WGS.transform(points_nztm[:, 0], points_nztm[:, 1])

    # Use the reconstructed 671-station observation set that matches Jaehwi's
    # sites_load_NSHM2022 loader (not the 608-station filtered set).
    obs_csv = constants.RESOURCE_PATH / "observations" / "jaehwi_v1p0_independent_observations.csv"

    result_df = pipeline.points_pipeline(
        longitudes=lons,
        latitudes=lats,
        geology_categorical_csv=constants.RESOURCE_PATH / "categorical_vs30_mean_and_stddev" / "geology_model_prior_mean_and_standard_deviation.csv",
        terrain_categorical_csv=constants.RESOURCE_PATH / "categorical_vs30_mean_and_stddev" / "terrain_model_prior_mean_and_standard_deviation.csv",
        independent_observations_csv=obs_csv,
        do_bayesian_update=True,
        combination_method=constants.CombinationMethod.RATIO,
        combine_ratio=1.0,
        apply_alluvium_slope_mod=False,
        apply_coastal_distance_mod=False,
        noisy=True,
        mvn=True,
        include_intermediate=True,
        nproc=-1,
        geology_corr_fn=functools.partial(
            utils.exponential_correlation_function,
            phi=constants.DEFAULT_GEOLOGY_PHI,
        ),
        terrain_corr_fn=functools.partial(
            utils.exponential_correlation_function,
            phi=constants.DEFAULT_TERRAIN_PHI,
        ),
    )
    return result_df


def print_comparison(ours: pd.DataFrame, jaehwi: pd.DataFrame):
    """Print comparison statistics."""
    our_vs30 = ours["vs30"].values
    our_stdv = ours["stdv"].values
    jaehwi_vs30 = jaehwi["mvn_vs30"].values
    jaehwi_stdv = jaehwi["mvn_stdv"].values

    # Filter out NaN values (points outside valid raster areas)
    valid = ~(np.isnan(our_vs30) | np.isnan(jaehwi_vs30)
              | np.isnan(our_stdv) | np.isnan(jaehwi_stdv))
    n_total = len(our_vs30)
    n_nan = n_total - valid.sum()
    our_vs30 = our_vs30[valid]
    our_stdv = our_stdv[valid]
    jaehwi_vs30 = jaehwi_vs30[valid]
    jaehwi_stdv = jaehwi_stdv[valid]

    if n_nan > 0:
        print(f"\n  Filtered out {n_nan}/{n_total} points with NaN values")

    n_valid = len(our_vs30)

    # Vs30 comparison
    vs30_diff = np.abs(our_vs30 - jaehwi_vs30)
    vs30_pct_diff = vs30_diff / jaehwi_vs30 * 100

    print(f"\n{'='*70}")
    print(f"  Combined Vs30 comparison ({n_valid} valid points)")
    print(f"{'='*70}")
    print(f"  Our range:     [{our_vs30.min():.2f}, {our_vs30.max():.2f}], mean={our_vs30.mean():.2f}")
    print(f"  Jaehwi range:  [{jaehwi_vs30.min():.2f}, {jaehwi_vs30.max():.2f}], mean={jaehwi_vs30.mean():.2f}")
    print()
    print(f"  Mean abs diff:   {vs30_diff.mean():.4f} m/s")
    print(f"  Median abs diff: {np.median(vs30_diff):.4f} m/s")
    print(f"  Max abs diff:    {vs30_diff.max():.4f} m/s")
    print(f"  Std abs diff:    {vs30_diff.std():.4f} m/s")
    print()
    print(f"  Mean % diff:     {vs30_pct_diff.mean():.4f}%")
    print(f"  Median % diff:   {np.median(vs30_pct_diff):.4f}%")
    print(f"  Max % diff:      {vs30_pct_diff.max():.4f}%")
    print()
    for t in [0.01, 0.1, 1.0, 5.0]:
        n = np.sum(vs30_pct_diff > t)
        print(f"  Points with > {t:5.2f}% diff: {n}/{n_valid} ({n/n_valid*100:.1f}%)")

    # StdDev comparison
    stdv_diff = np.abs(our_stdv - jaehwi_stdv)
    print(f"\n{'='*70}")
    print(f"  Combined StdDev comparison ({n_valid} valid points)")
    print(f"{'='*70}")
    print(f"  Mean abs diff:   {stdv_diff.mean():.6f}")
    print(f"  Median abs diff: {np.median(stdv_diff):.6f}")
    print(f"  Max abs diff:    {stdv_diff.max():.6f}")

    # Also compare intermediate values if available
    intermediates = [
        ("geology_vs30", "geology_vs30", "Geology categorical Vs30"),
        ("geology_stdv", "geology_stdv", "Geology categorical StdDev"),
        ("terrain_vs30", "terrain_vs30", "Terrain categorical Vs30"),
        ("terrain_stdv", "terrain_stdv", "Terrain categorical StdDev"),
        ("geology_mvn_vs30", "geology_mvn_vs30", "Geology MVN Vs30"),
        ("terrain_mvn_vs30", "terrain_mvn_vs30", "Terrain MVN Vs30"),
    ]

    print(f"\n{'='*70}")
    print("  Intermediate value comparison")
    print(f"{'='*70}")
    print(f"  {'Stage':<30} {'Mean abs diff':>15} {'Max abs diff':>15}")
    print(f"  {'-'*30} {'-'*15} {'-'*15}")

    for our_col, jaehwi_col, label in intermediates:
        if our_col in ours.columns and jaehwi_col in jaehwi.columns:
            o = ours[our_col].values[valid]
            j = jaehwi[jaehwi_col].values[valid]
            mask = ~(np.isnan(o) | np.isnan(j))
            diff = np.abs(o[mask] - j[mask])
            print(f"  {label:<30} {diff.mean():>15.6f} {diff.max():>15.6f}")
        else:
            print(f"  {label:<30} {'(column missing)':>15}")


if __name__ == "__main__":
    print(f"Sampling {N_POINTS} random valid points from {REFERENCE_TIF.name}...")
    points = sample_valid_points(N_POINTS, SEED)
    print(f"  Easting range:  [{points[:, 0].min():.0f}, {points[:, 0].max():.0f}]")
    print(f"  Northing range: [{points[:, 1].min():.0f}, {points[:, 1].max():.0f}]")

    output_dir = Path("/home/arr65/src/vs30/dev/compare_points_mode_output")
    output_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        print("\nRunning Jaehwi's code (points mode, nproc=1)...")
        jaehwi_df = run_jaehwi_points(points, tmpdir)
        print(f"  Got {len(jaehwi_df)} results")

        print("\nRunning our pipeline (points mode)...")
        our_df = run_our_pipeline(points)
        print(f"  Got {len(our_df)} results")

        # Save results for inspection
        jaehwi_df.to_csv(output_dir / "jaehwi_results.csv", index=False)
        our_df.to_csv(output_dir / "our_results.csv", index=False)
        np.savetxt(output_dir / "sample_points_nztm.csv",
                   points, fmt="%.2f", delimiter=",", header="easting,northing",
                   comments="")
        print(f"\n  Results saved to {output_dir}")

        print_comparison(our_df, jaehwi_df)

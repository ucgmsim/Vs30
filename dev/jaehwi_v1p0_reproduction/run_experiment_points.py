"""Fast point-based experiment comparison.

Instead of generating full grids (~hours per experiment due to MVN),
sample ~200 points from the reference raster and run compute_at_locations()
at those points. Each experiment takes seconds instead of hours.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

sys.path.insert(0, str(Path("/home/arr65/src/vs30")))

from vs30 import pipeline, constants

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO = Path("/home/arr65/src/vs30")
VS30_PKG = REPO / "vs30"
EXPERIMENT_DIR = REPO / "dev" / "jaehwi_v1p0_reproduction"
REFERENCE_SUBGRID = EXPERIMENT_DIR / "reference_subgrid.tif"

CATEGORICAL_DIR = VS30_PKG / "resources" / "categorical_vs30_mean_and_stddev"
OBSERVATIONS_DIR = VS30_PKG / "resources" / "observations"
FOSTER_GEOL_POST = CATEGORICAL_DIR / "geology_model_posterior_from_foster_2019_mean_and_standard_deviation.csv"
FOSTER_TERR_POST = CATEGORICAL_DIR / "terrain_model_posterior_from_foster_2019_mean_and_standard_deviation.csv"
GEOL_PRIOR = CATEGORICAL_DIR / "geology_model_prior_mean_and_standard_deviation.csv"
TERR_PRIOR = CATEGORICAL_DIR / "terrain_model_prior_mean_and_standard_deviation.csv"
CURRENT_OBS = OBSERVATIONS_DIR / "jaehwi_v1p0_independent_observations.csv"
RECONSTRUCTED_OBS = EXPERIMENT_DIR / "jaehwi_reconstructed_observations.csv"
JAEHWI_GEOL_POST = EXPERIMENT_DIR / "jaehwi_geology_posterior.csv"
JAEHWI_TERR_POST = EXPERIMENT_DIR / "jaehwi_terrain_posterior.csv"

nztm2wgs = Transformer.from_crs(2193, 4326, always_xy=True)

# ---------------------------------------------------------------------------
# Experiment definitions (same as run_experiment.py)
# ---------------------------------------------------------------------------
EXPERIMENTS = {
    "exp0_baseline_current_config": dict(
        geology_categorical_csv=GEOL_PRIOR,
        terrain_categorical_csv=TERR_PRIOR,
        do_bayesian_update=True,
        combination_method=constants.CombinationMethod.STANDARD_DEVIATION_WEIGHTING,
        combine_ratio=None,
        apply_coastal_distance_mod=True,
        independent_observations_csv=CURRENT_OBS,
    ),
    "exp1_foster_posterior_ratio_no_coast": dict(
        geology_categorical_csv=FOSTER_GEOL_POST,
        terrain_categorical_csv=FOSTER_TERR_POST,
        do_bayesian_update=False,
        combination_method=constants.CombinationMethod.RATIO,
        combine_ratio=1.0,
        apply_coastal_distance_mod=False,
        independent_observations_csv=RECONSTRUCTED_OBS,
    ),
    "exp2_foster_posterior_ratio_coast_on": dict(
        geology_categorical_csv=FOSTER_GEOL_POST,
        terrain_categorical_csv=FOSTER_TERR_POST,
        do_bayesian_update=False,
        combination_method=constants.CombinationMethod.RATIO,
        combine_ratio=1.0,
        apply_coastal_distance_mod=True,
        independent_observations_csv=RECONSTRUCTED_OBS,
    ),
    "exp3_foster_posterior_stdv_no_coast": dict(
        geology_categorical_csv=FOSTER_GEOL_POST,
        terrain_categorical_csv=FOSTER_TERR_POST,
        do_bayesian_update=False,
        combination_method=constants.CombinationMethod.STANDARD_DEVIATION_WEIGHTING,
        combine_ratio=None,
        apply_coastal_distance_mod=False,
        independent_observations_csv=RECONSTRUCTED_OBS,
    ),
    "exp4_prior_bayesian_ratio_no_coast": dict(
        geology_categorical_csv=GEOL_PRIOR,
        terrain_categorical_csv=TERR_PRIOR,
        do_bayesian_update=True,
        combination_method=constants.CombinationMethod.RATIO,
        combine_ratio=1.0,
        apply_coastal_distance_mod=False,
        independent_observations_csv=RECONSTRUCTED_OBS,
    ),
    "exp5_prior_bayesian_stdv_no_coast": dict(
        geology_categorical_csv=GEOL_PRIOR,
        terrain_categorical_csv=TERR_PRIOR,
        do_bayesian_update=True,
        combination_method=constants.CombinationMethod.STANDARD_DEVIATION_WEIGHTING,
        combine_ratio=None,
        apply_coastal_distance_mod=False,
        independent_observations_csv=RECONSTRUCTED_OBS,
    ),
    "exp6_jaehwi_posterior_ratio_no_coast": dict(
        geology_categorical_csv=JAEHWI_GEOL_POST,
        terrain_categorical_csv=JAEHWI_TERR_POST,
        do_bayesian_update=False,
        combination_method=constants.CombinationMethod.RATIO,
        combine_ratio=1.0,
        apply_coastal_distance_mod=False,
        independent_observations_csv=RECONSTRUCTED_OBS,
    ),
    "exp7_jaehwi_posterior_ratio_coast_on": dict(
        # coast_on => mod6=True => skips GID 4 slope interpolation (matching Jaehwi)
        # but also applies coastal distance (Jaehwi had this commented out)
        geology_categorical_csv=JAEHWI_GEOL_POST,
        terrain_categorical_csv=JAEHWI_TERR_POST,
        do_bayesian_update=False,
        combination_method=constants.CombinationMethod.RATIO,
        combine_ratio=1.0,
        apply_coastal_distance_mod=True,
        independent_observations_csv=RECONSTRUCTED_OBS,
    ),
    "exp8_prior_bayesian_ratio_coast_on": dict(
        # Same as exp4 but with coast_on => skip GID 4 slope interpolation
        geology_categorical_csv=GEOL_PRIOR,
        terrain_categorical_csv=TERR_PRIOR,
        do_bayesian_update=True,
        combination_method=constants.CombinationMethod.RATIO,
        combine_ratio=1.0,
        apply_coastal_distance_mod=True,
        independent_observations_csv=RECONSTRUCTED_OBS,
    ),
    "exp9_jaehwi_posterior_ratio_skip_alluvium": dict(
        # Jaehwi posteriors + ratio + no coastal dist + skip GID 4 slope (decoupled)
        geology_categorical_csv=JAEHWI_GEOL_POST,
        terrain_categorical_csv=JAEHWI_TERR_POST,
        do_bayesian_update=False,
        combination_method=constants.CombinationMethod.RATIO,
        combine_ratio=1.0,
        apply_coastal_distance_mod=False,
        skip_alluvium_slope=True,
        independent_observations_csv=RECONSTRUCTED_OBS,
    ),
    "exp10_prior_bayesian_ratio_skip_alluvium": dict(
        # Prior + Bayesian update + ratio + no coastal dist + skip GID 4 slope
        geology_categorical_csv=GEOL_PRIOR,
        terrain_categorical_csv=TERR_PRIOR,
        do_bayesian_update=True,
        combination_method=constants.CombinationMethod.RATIO,
        combine_ratio=1.0,
        apply_coastal_distance_mod=False,
        skip_alluvium_slope=True,
        independent_observations_csv=RECONSTRUCTED_OBS,
    ),
}


def sample_reference_points(n_points=200, seed=42):
    """Sample n_points from the reference subgrid, returning NZTM coords + Vs30 values."""
    rng = np.random.default_rng(seed)

    with rasterio.open(REFERENCE_SUBGRID) as src:
        vs30_band = src.read(1)
        stdv_band = src.read(2)
        nodata = src.nodata
        transform = src.transform

    # Valid pixel mask
    valid = (vs30_band != nodata) & (~np.isnan(vs30_band)) & (vs30_band > 0)
    valid_rows, valid_cols = np.where(valid)

    # Sample indices
    idx = rng.choice(len(valid_rows), size=min(n_points, len(valid_rows)), replace=False)
    rows = valid_rows[idx]
    cols = valid_cols[idx]

    # Pixel center NZTM coordinates
    eastings = transform.c + (cols + 0.5) * transform.a
    northings = transform.f + (rows + 0.5) * transform.e

    ref_vs30 = vs30_band[rows, cols]
    ref_stdv = stdv_band[rows, cols]

    return eastings, northings, ref_vs30, ref_stdv


def run_experiment(name, params, longitudes, latitudes, ref_vs30, ref_stdv):
    """Run one experiment at the sampled points and compare to reference."""
    print(f"\n{'='*70}")
    print(f"Experiment: {name}")
    print(f"{'='*70}")
    for k, v in params.items():
        vstr = Path(v).name if isinstance(v, Path) else str(v)
        print(f"  {k}: {vstr}")

    result_df = pipeline.compute_at_locations(
        longitudes=longitudes,
        latitudes=latitudes,
        noisy=True,
        mvn=True,
        include_intermediate=True,
        n_proc=-1,
        **params,
    )

    our_vs30 = result_df["vs30"].values
    our_stdv = result_df["stdv"].values

    # Vs30 comparison
    diff = our_vs30 - ref_vs30
    abs_diff = np.abs(diff)
    pct_diff = 100.0 * abs_diff / ref_vs30

    print(f"\n  --- Vs30 Comparison ({len(ref_vs30)} points) ---")
    print(f"  Mean abs diff:    {abs_diff.mean():.2f} m/s")
    print(f"  Median abs diff:  {np.median(abs_diff):.2f} m/s")
    print(f"  Max abs diff:     {abs_diff.max():.2f} m/s")
    print(f"  Mean % diff:      {pct_diff.mean():.2f}%")
    print(f"  RMSE:             {np.sqrt((diff**2).mean()):.2f} m/s")
    print(f"  Mean signed diff: {diff.mean():.2f} m/s")

    # Stdv comparison
    stdv_diff = np.abs(our_stdv - ref_stdv)
    print(f"\n  --- StdDev Comparison ---")
    print(f"  Mean abs diff:    {stdv_diff.mean():.4f}")
    print(f"  Max abs diff:     {stdv_diff.max():.4f}")

    # Breakdown by error magnitude
    n_exact = int((abs_diff < 1.0).sum())
    n_close = int((abs_diff < 5.0).sum())
    n_moderate = int(((abs_diff >= 5.0) & (abs_diff < 20.0)).sum())
    n_large = int((abs_diff >= 20.0).sum())
    print(f"\n  --- Error distribution ---")
    print(f"  <1 m/s:    {n_exact:4d} ({100*n_exact/len(ref_vs30):.1f}%)")
    print(f"  <5 m/s:    {n_close:4d} ({100*n_close/len(ref_vs30):.1f}%)")
    print(f"  5-20 m/s:  {n_moderate:4d} ({100*n_moderate/len(ref_vs30):.1f}%)")
    print(f"  >20 m/s:   {n_large:4d} ({100*n_large/len(ref_vs30):.1f}%)")

    # Also check intermediate geology vs terrain contributions
    if "geology_mvn_vs30" in result_df.columns:
        geol_diff = np.abs(result_df["geology_mvn_vs30"].values - ref_vs30)
        terr_diff = np.abs(result_df["terrain_mvn_vs30"].values - ref_vs30)
        print(f"\n  --- Intermediate (mean abs diff to reference) ---")
        print(f"  Geology MVN only:  {geol_diff.mean():.2f} m/s")
        print(f"  Terrain MVN only:  {terr_diff.mean():.2f} m/s")

    return {
        "name": name,
        "mean_abs_diff": abs_diff.mean(),
        "median_abs_diff": np.median(abs_diff),
        "max_abs_diff": abs_diff.max(),
        "rmse": np.sqrt((diff**2).mean()),
        "mean_pct_diff": pct_diff.mean(),
        "mean_stdv_diff": stdv_diff.mean(),
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fast point-based experiment comparison")
    parser.add_argument("experiments", nargs="*", default=list(EXPERIMENTS.keys()),
                        help="Experiments to run (default: all)")
    parser.add_argument("--n-points", type=int, default=200)
    args = parser.parse_args()

    print("Sampling reference points...")
    eastings, northings, ref_vs30, ref_stdv = sample_reference_points(args.n_points)
    longitudes, latitudes = nztm2wgs.transform(eastings, northings)
    print(f"Sampled {len(ref_vs30)} points from reference subgrid")
    print(f"  Vs30 range: [{ref_vs30.min():.1f}, {ref_vs30.max():.1f}] m/s")

    results = []
    for name in args.experiments:
        if name not in EXPERIMENTS:
            print(f"Unknown experiment: {name}")
            sys.exit(1)
        r = run_experiment(name, EXPERIMENTS[name], longitudes, latitudes, ref_vs30, ref_stdv)
        results.append(r)

    # Summary table
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"{'Experiment':<45s} {'MeanAbs':>8s} {'RMSE':>8s} {'Max':>8s} {'%Diff':>7s}")
    print("-" * 78)
    for r in results:
        print(f"{r['name']:<45s} {r['mean_abs_diff']:8.2f} {r['rmse']:8.2f} "
              f"{r['max_abs_diff']:8.2f} {r['mean_pct_diff']:6.2f}%")

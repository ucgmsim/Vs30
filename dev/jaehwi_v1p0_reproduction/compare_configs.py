"""Compare Foster posteriors vs our Bayesian update at sample points."""
import functools
import sys
from pathlib import Path

import numpy as np
import rasterio
from pyproj import Transformer

sys.path.insert(0, str(Path("/home/arr65/src/vs30")))

from vs30 import pipeline, constants, utils

REPO = Path("/home/arr65/src/vs30")
VS30_PKG = REPO / "vs30"
EXPERIMENT_DIR = REPO / "dev" / "jaehwi_v1p0_reproduction"
REFERENCE_SUBGRID = EXPERIMENT_DIR / "reference_subgrid.tif"

CATEGORICAL_DIR = VS30_PKG / "resources" / "categorical_vs30_mean_and_stddev"
FOSTER_GEOL_POST = CATEGORICAL_DIR / "geology_model_posterior_from_foster_2019_mean_and_standard_deviation.csv"
FOSTER_TERR_POST = CATEGORICAL_DIR / "terrain_model_posterior_from_foster_2019_mean_and_standard_deviation.csv"
GEOL_PRIOR = CATEGORICAL_DIR / "geology_model_prior_mean_and_standard_deviation.csv"
TERR_PRIOR = CATEGORICAL_DIR / "terrain_model_prior_mean_and_standard_deviation.csv"
RECONSTRUCTED_OBS = EXPERIMENT_DIR / "jaehwi_reconstructed_observations.csv"
JAEHWI_GEOL_POST = EXPERIMENT_DIR / "jaehwi_geology_posterior.csv"
JAEHWI_TERR_POST = EXPERIMENT_DIR / "jaehwi_terrain_posterior.csv"

nztm2wgs = Transformer.from_crs(2193, 4326, always_xy=True)


def sample_reference_points(n_points=200, seed=42):
    rng = np.random.default_rng(seed)
    with rasterio.open(REFERENCE_SUBGRID) as src:
        vs30_band = src.read(1)
        stdv_band = src.read(2)
        nodata = src.nodata
        transform = src.transform
    valid = (vs30_band != nodata) & (~np.isnan(vs30_band)) & (vs30_band > 0)
    valid_rows, valid_cols = np.where(valid)
    idx = rng.choice(len(valid_rows), size=min(n_points, len(valid_rows)), replace=False)
    rows, cols = valid_rows[idx], valid_cols[idx]
    eastings = transform.c + (cols + 0.5) * transform.a
    northings = transform.f + (rows + 0.5) * transform.e
    return eastings, northings, vs30_band[rows, cols], stdv_band[rows, cols]


def run_config(name, longitudes, latitudes, **params):
    print(f"\n  Running {name}...")
    return pipeline.points_pipeline(
        longitudes=longitudes, latitudes=latitudes,
        noisy=True, mvn=True, include_intermediate=True, nproc=-1,
        geology_corr_fn=functools.partial(
            utils.exponential_correlation_function,
            phi=1407,
        ),
        terrain_corr_fn=functools.partial(
            utils.exponential_correlation_function,
            phi=993,
        ),
        **params,
    )


if __name__ == "__main__":
    eastings, northings, ref_vs30, ref_stdv = sample_reference_points(200)
    longitudes, latitudes = nztm2wgs.transform(eastings, northings)

    configs = {
        "foster_post_ratio": dict(
            geology_categorical_csv=FOSTER_GEOL_POST,
            terrain_categorical_csv=FOSTER_TERR_POST,
            do_bayesian_update=False,
            combination_method=constants.CombinationMethod.RATIO,
            combine_ratio=1.0,
            apply_coastal_distance_mod=False,
            apply_alluvium_slope_mod=False,
            independent_observations_csv=RECONSTRUCTED_OBS,
        ),
        "bayesian_ratio": dict(
            geology_categorical_csv=GEOL_PRIOR,
            terrain_categorical_csv=TERR_PRIOR,
            do_bayesian_update=True,
            combination_method=constants.CombinationMethod.RATIO,
            combine_ratio=1.0,
            apply_coastal_distance_mod=False,
            apply_alluvium_slope_mod=False,
            independent_observations_csv=RECONSTRUCTED_OBS,
        ),
        "jaehwi_post_ratio": dict(
            geology_categorical_csv=JAEHWI_GEOL_POST,
            terrain_categorical_csv=JAEHWI_TERR_POST,
            do_bayesian_update=False,
            combination_method=constants.CombinationMethod.RATIO,
            combine_ratio=1.0,
            apply_coastal_distance_mod=False,
            apply_alluvium_slope_mod=False,
            independent_observations_csv=RECONSTRUCTED_OBS,
        ),
    }

    results = {}
    for name, params in configs.items():
        df = run_config(name, longitudes, latitudes, **params)
        df["ref_vs30"] = ref_vs30
        df["easting"] = eastings
        df["northing"] = northings
        df["abs_diff"] = np.abs(df["vs30"].values - ref_vs30)
        results[name] = df

    # Per-category comparison
    print("\n" + "=" * 90)
    print("Per-category mean absolute difference (Vs30)")
    print("=" * 90)
    print(f"{'GeoID':>6s}  {'nPts':>5s}  {'foster':>10s}  {'bayesian':>10s}  {'jaehwi':>10s}  {'RefMean':>8s}")
    print("-" * 70)

    for gid in sorted(results["foster_post_ratio"]["geology_id"].unique()):
        mask = results["foster_post_ratio"]["geology_id"] == gid
        n = mask.sum()
        ref_mean = ref_vs30[mask].mean()
        row = f"{gid:6.0f}  {n:5d}"
        for name in ["foster_post_ratio", "bayesian_ratio", "jaehwi_post_ratio"]:
            me = results[name].loc[mask, "abs_diff"].mean()
            row += f"  {me:10.2f}"
        row += f"  {ref_mean:8.2f}"
        print(row)

    # Overall
    print("-" * 70)
    row = f"{'ALL':>6s}  {200:5d}"
    for name in ["foster_post_ratio", "bayesian_ratio", "jaehwi_post_ratio"]:
        me = results[name]["abs_diff"].mean()
        row += f"  {me:10.2f}"
    row += f"  {ref_vs30.mean():8.2f}"
    print(row)

    # Also show cat 15 detail: what geology_vs30 each config gives
    mask15 = results["foster_post_ratio"]["geology_id"] == 15
    print(f"\n=== Category 15 detail (n={mask15.sum()}) ===")
    print(f"  Reference mean Vs30: {ref_vs30[mask15].mean():.2f}")
    for name in ["foster_post_ratio", "bayesian_ratio", "jaehwi_post_ratio"]:
        df = results[name]
        g_vs30 = df.loc[mask15, "geology_vs30"].mean()
        g_mvn = df.loc[mask15, "geology_mvn_vs30"].mean()
        t_mvn = df.loc[mask15, "terrain_mvn_vs30"].mean()
        comb = df.loc[mask15, "vs30"].mean()
        print(f"  {name:25s}: geol_cat={g_vs30:.2f}  geol_mvn={g_mvn:.2f}  terr_mvn={t_mvn:.2f}  combined={comb:.2f}")

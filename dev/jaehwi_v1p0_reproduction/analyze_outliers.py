"""Analyze which points have large discrepancies and what categories they fall in."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

sys.path.insert(0, str(Path("/home/arr65/src/vs30")))

from vs30 import pipeline, constants

REPO = Path("/home/arr65/src/vs30")
VS30_PKG = REPO / "vs30"
EXPERIMENT_DIR = REPO / "dev" / "jaehwi_v1p0_reproduction"
REFERENCE_SUBGRID = EXPERIMENT_DIR / "reference_subgrid.tif"

CATEGORICAL_DIR = VS30_PKG / "resources" / "categorical_vs30_mean_and_stddev"
GEOL_PRIOR = CATEGORICAL_DIR / "geology_model_prior_mean_and_standard_deviation.csv"
TERR_PRIOR = CATEGORICAL_DIR / "terrain_model_prior_mean_and_standard_deviation.csv"
OBSERVATIONS_DIR = VS30_PKG / "resources" / "observations"
RECONSTRUCTED_OBS = EXPERIMENT_DIR / "jaehwi_reconstructed_observations.csv"

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
    rows = valid_rows[idx]
    cols = valid_cols[idx]

    eastings = transform.c + (cols + 0.5) * transform.a
    northings = transform.f + (rows + 0.5) * transform.e
    return eastings, northings, vs30_band[rows, cols], stdv_band[rows, cols]


if __name__ == "__main__":
    eastings, northings, ref_vs30, ref_stdv = sample_reference_points(200)
    longitudes, latitudes = nztm2wgs.transform(eastings, northings)

    result_df = pipeline.points_pipeline(
        longitudes=longitudes,
        latitudes=latitudes,
        noisy=True,
        mvn=True,
        include_intermediate=True,
        n_proc=-1,
        geology_categorical_csv=GEOL_PRIOR,
        terrain_categorical_csv=TERR_PRIOR,
        do_bayesian_update=True,
        combination_method=constants.CombinationMethod.RATIO,
        combine_ratio=1.0,
        apply_coastal_distance_mod=False,
        apply_alluvium_slope_mod=False,
        independent_observations_csv=RECONSTRUCTED_OBS,
    )

    result_df["ref_vs30"] = ref_vs30
    result_df["ref_stdv"] = ref_stdv
    result_df["easting"] = eastings
    result_df["northing"] = northings
    result_df["abs_diff"] = np.abs(result_df["vs30"].values - ref_vs30)
    result_df["pct_diff"] = 100.0 * result_df["abs_diff"] / ref_vs30

    # Show error distribution by geology category
    print("\n=== Error by geology category ===")
    print(f"{'GeoID':>6s}  {'Count':>5s}  {'MeanErr':>8s}  {'MaxErr':>8s}  {'<1m/s':>5s}  {'>20m/s':>6s}")
    for gid in sorted(result_df["geology_id"].unique()):
        mask = result_df["geology_id"] == gid
        sub = result_df[mask]
        n = len(sub)
        me = sub["abs_diff"].mean()
        mx = sub["abs_diff"].max()
        n_good = (sub["abs_diff"] < 1).sum()
        n_bad = (sub["abs_diff"] > 20).sum()
        print(f"{gid:6.0f}  {n:5d}  {me:8.2f}  {mx:8.2f}  {n_good:5d}  {n_bad:6d}")

    # Show worst outliers with intermediate values
    worst = result_df.nlargest(20, "abs_diff")
    print("\n=== Top 20 worst outliers ===")
    cols = ["geology_id", "ref_vs30", "vs30", "abs_diff",
            "geology_vs30", "geology_vs30_hybrid", "geology_mvn_vs30",
            "terrain_mvn_vs30", "stdv", "ref_stdv"]
    available = [c for c in cols if c in worst.columns]
    print(worst[available].to_string())

"""Compare Jaehwi's actual code output against reference raster and our pipeline.

Reads:
- jaehwi_output/vs30points.csv  (Jaehwi's code output on 200 test points)
- reference_subgrid.tif          (V1.0_26Mar.tif Wellington subgrid)
- Our pipeline output via pipeline.points_pipeline()

Compares intermediate values (geology_vs30, terrain_vs30, MVN corrections)
to identify where the implementations diverge.
"""
import functools
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer

sys.path.insert(0, str(Path("/home/arr65/src/vs30")))

from vs30 import pipeline, constants, utils

EXPERIMENT_DIR = Path("/home/arr65/src/vs30/dev/jaehwi_v1p0_reproduction")
JAEHWI_OUTPUT = EXPERIMENT_DIR / "jaehwi_output" / "vs30points.csv"
REFERENCE_SUBGRID = EXPERIMENT_DIR / "reference_subgrid.tif"
REPO = Path("/home/arr65/src/vs30")
VS30_PKG = REPO / "vs30"
CATEGORICAL_DIR = VS30_PKG / "resources" / "categorical_vs30_mean_and_stddev"
FOSTER_GEOL_POST = CATEGORICAL_DIR / "geology_model_posterior_from_foster_2019_mean_and_standard_deviation.csv"
FOSTER_TERR_POST = CATEGORICAL_DIR / "terrain_model_posterior_from_foster_2019_mean_and_standard_deviation.csv"
GEOL_PRIOR = CATEGORICAL_DIR / "geology_model_prior_mean_and_standard_deviation.csv"
TERR_PRIOR = CATEGORICAL_DIR / "terrain_model_prior_mean_and_standard_deviation.csv"
RECONSTRUCTED_OBS = EXPERIMENT_DIR / "jaehwi_reconstructed_observations.csv"

nztm2wgs = Transformer.from_crs(2193, 4326, always_xy=True)


def sample_reference_at_points(eastings, northings):
    """Sample reference raster at given NZTM coordinates."""
    with rasterio.open(REFERENCE_SUBGRID) as src:
        vs30_band = src.read(1)
        stdv_band = src.read(2)
        transform = src.transform
        rows, cols = rasterio.transform.rowcol(transform, eastings, northings)
        rows, cols = np.array(rows), np.array(cols)
        valid = (rows >= 0) & (rows < vs30_band.shape[0]) & (cols >= 0) & (cols < vs30_band.shape[1])
        ref_vs30 = np.full(len(eastings), np.nan)
        ref_stdv = np.full(len(eastings), np.nan)
        ref_vs30[valid] = vs30_band[rows[valid], cols[valid]]
        ref_stdv[valid] = stdv_band[rows[valid], cols[valid]]
    return ref_vs30, ref_stdv


def run_our_pipeline(longitudes, latitudes, mode="foster_posterior"):
    """Run our pipeline with Foster posteriors (matching Jaehwi defaults)."""
    if mode == "foster_posterior":
        return pipeline.points_pipeline(
            longitudes=longitudes, latitudes=latitudes,
            geology_categorical_csv=FOSTER_GEOL_POST,
            terrain_categorical_csv=FOSTER_TERR_POST,
            do_bayesian_update=False,
            combination_method=constants.CombinationMethod.RATIO,
            combine_ratio=1.0,
            apply_coastal_distance_mod=False,
            apply_alluvium_slope_mod=False,
            independent_observations_csv=RECONSTRUCTED_OBS,
            noisy=True, mvn=True, include_intermediate=True, nproc=-1,
            geology_corr_fn=functools.partial(
                utils.exponential_correlation_function,
                phi=constants.DEFAULT_GEOLOGY_PHI,
            ),
            terrain_corr_fn=functools.partial(
                utils.exponential_correlation_function,
                phi=constants.DEFAULT_TERRAIN_PHI,
            ),
        )
    elif mode == "bayesian":
        return pipeline.points_pipeline(
            longitudes=longitudes, latitudes=latitudes,
            geology_categorical_csv=GEOL_PRIOR,
            terrain_categorical_csv=TERR_PRIOR,
            do_bayesian_update=True,
            combination_method=constants.CombinationMethod.RATIO,
            combine_ratio=1.0,
            apply_coastal_distance_mod=False,
            apply_alluvium_slope_mod=False,
            independent_observations_csv=RECONSTRUCTED_OBS,
            noisy=True, mvn=True, include_intermediate=True, nproc=-1,
            geology_corr_fn=functools.partial(
                utils.exponential_correlation_function,
                phi=constants.DEFAULT_GEOLOGY_PHI,
            ),
            terrain_corr_fn=functools.partial(
                utils.exponential_correlation_function,
                phi=constants.DEFAULT_TERRAIN_PHI,
            ),
        )


def compare_columns(label, jaehwi_vals, other_vals, mask=None):
    """Print comparison stats for a pair of columns."""
    if mask is not None:
        j, o = jaehwi_vals[mask], other_vals[mask]
    else:
        j, o = jaehwi_vals, other_vals
    valid = np.isfinite(j) & np.isfinite(o)
    j, o = j[valid], o[valid]
    if len(j) == 0:
        print(f"  {label:30s}: no valid comparisons")
        return
    diff = np.abs(j - o)
    exact = np.sum(diff == 0)
    close = np.sum(diff < 0.01)
    print(f"  {label:30s}: n={len(j):4d}  mean_abs={diff.mean():10.4f}  "
          f"median={np.median(diff):10.4f}  max={diff.max():10.4f}  "
          f"exact={exact:4d}  <0.01={close:4d}")


if __name__ == "__main__":
    # Load Jaehwi's output
    print("Loading Jaehwi's output...")
    jdf = pd.read_csv(JAEHWI_OUTPUT)
    print(f"  Columns: {list(jdf.columns)}")
    print(f"  Shape: {jdf.shape}")
    print(f"  First row:\n{jdf.iloc[0]}")

    eastings = jdf["easting"].values
    northings = jdf["northing"].values
    longitudes = jdf["longitude"].values
    latitudes = jdf["latitude"].values

    # Sample reference raster at those points
    print("\nSampling reference raster...")
    ref_vs30, ref_stdv = sample_reference_at_points(eastings, northings)

    # === Part 1: Does Jaehwi's code reproduce V1.0_26Mar.tif? ===
    print("\n" + "=" * 90)
    print("PART 1: Jaehwi's code vs Reference Raster (V1.0_26Mar.tif subgrid)")
    print("=" * 90)

    # The reference raster stores combined MVN values
    # Jaehwi's output should have mvn_vs30 and mvn_stdv columns
    for col_pair in [("mvn_vs30", ref_vs30, "Combined MVN Vs30"),
                     ("mvn_stdv", ref_stdv, "Combined MVN StdDev")]:
        col_name, ref_vals, label = col_pair
        if col_name in jdf.columns:
            compare_columns(f"Jaehwi vs Ref: {label}", jdf[col_name].values, ref_vals)
        else:
            print(f"  Column '{col_name}' not in Jaehwi output")

    # === Part 2: Run our pipeline ===
    print("\n" + "=" * 90)
    print("PART 2: Running our pipeline (Foster posteriors, matching Jaehwi defaults)")
    print("=" * 90)
    our_foster = run_our_pipeline(longitudes, latitudes, mode="foster_posterior")

    print("\nRunning our pipeline (Bayesian update from priors)...")
    our_bayesian = run_our_pipeline(longitudes, latitudes, mode="bayesian")

    # === Part 3: Compare intermediate values ===
    print("\n" + "=" * 90)
    print("PART 3: Intermediate value comparison")
    print("=" * 90)

    # Map Jaehwi column names to our column names
    column_map = {
        # Jaehwi col → (our foster col, our bayesian col, description)
        "geology_vs30": ("geology_vs30", "geology_vs30", "Geology categorical Vs30"),
        "geology_stdv": ("geology_stdv", "geology_stdv", "Geology categorical StdDev"),
        "terrain_vs30": ("terrain_vs30", "terrain_vs30", "Terrain categorical Vs30"),
        "terrain_stdv": ("terrain_stdv", "terrain_stdv", "Terrain categorical StdDev"),
        "geology_mvn_vs30": ("geology_mvn_vs30", "geology_mvn_vs30", "Geology MVN Vs30"),
        "geology_mvn_stdv": ("geology_mvn_stdv", "geology_mvn_stdv", "Geology MVN StdDev"),
        "terrain_mvn_vs30": ("terrain_mvn_vs30", "terrain_mvn_vs30", "Terrain MVN Vs30"),
        "terrain_mvn_stdv": ("terrain_mvn_stdv", "terrain_mvn_stdv", "Terrain MVN StdDev"),
    }

    for j_col, (f_col, b_col, desc) in column_map.items():
        if j_col in jdf.columns:
            print(f"\n  --- {desc} ({j_col}) ---")
            if f_col in our_foster.columns:
                compare_columns("Jaehwi vs Our Foster", jdf[j_col].values, our_foster[f_col].values)
            if b_col in our_bayesian.columns:
                compare_columns("Jaehwi vs Our Bayesian", jdf[j_col].values, our_bayesian[b_col].values)
        else:
            print(f"  Column '{j_col}' not in Jaehwi output — skipping")

    # === Part 4: Combined values ===
    print("\n" + "=" * 90)
    print("PART 4: Final combined values")
    print("=" * 90)

    for j_col, f_col, desc in [("mvn_vs30", "vs30", "Final Vs30"),
                                ("mvn_stdv", "vs30_stdv", "Final StdDev")]:
        if j_col in jdf.columns:
            print(f"\n  --- {desc} ---")
            if f_col in our_foster.columns:
                compare_columns("Jaehwi vs Our Foster", jdf[j_col].values, our_foster[f_col].values)
            if f_col in our_bayesian.columns:
                compare_columns("Jaehwi vs Our Bayesian", jdf[j_col].values, our_bayesian[f_col].values)
            compare_columns("Jaehwi vs Reference", jdf[j_col].values, ref_vs30 if "vs30" in j_col else ref_stdv)

    # === Part 5: Per-category breakdown ===
    print("\n" + "=" * 90)
    print("PART 5: Per-geology-category breakdown (Jaehwi vs Reference)")
    print("=" * 90)

    gid_col = "gid" if "gid" in jdf.columns else "geology_id" if "geology_id" in jdf.columns else None
    vs30_col = "mvn_vs30" if "mvn_vs30" in jdf.columns else "vs30"

    if gid_col and vs30_col in jdf.columns:
        print(f"\n{'GID':>5s}  {'n':>4s}  {'J_mean':>8s}  {'Ref_mean':>8s}  {'MeanDiff':>10s}  {'MaxDiff':>10s}")
        print("-" * 55)
        for gid in sorted(jdf[gid_col].dropna().unique()):
            mask = jdf[gid_col] == gid
            j_vals = jdf.loc[mask, vs30_col].values
            r_vals = ref_vs30[mask.values]
            valid = np.isfinite(j_vals) & np.isfinite(r_vals)
            if valid.sum() == 0:
                continue
            diff = np.abs(j_vals[valid] - r_vals[valid])
            print(f"{gid:5.0f}  {valid.sum():4d}  {j_vals[valid].mean():8.2f}  "
                  f"{r_vals[valid].mean():8.2f}  {diff.mean():10.4f}  {diff.max():10.4f}")
    else:
        print(f"  Cannot find geology ID column. Available: {list(jdf.columns)}")

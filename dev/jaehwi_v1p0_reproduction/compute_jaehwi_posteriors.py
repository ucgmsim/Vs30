"""Compute Bayesian posterior categorical values using Jaehwi's exact formula.

Jaehwi's posterior() in model_new.py differs from our pipeline:
  1. Batch processing (all observations per category at once, not sequential)
  2. No mean_shift residual term in variance
  3. StdDev NOT updated — original prior stdv is retained

Since there are no Q5 observations, all quality weights W=1.

The mean formula simplifies to:
  posterior_mean = exp((n0 * ln(prior) + sum(ln(obs_i))) / (n0 + n))

The variance formula:
  posterior_var = (n0 * sigma^2 + sum(unc_i^2)) / (n0 + n)
  (but this is NOT written back — stdv stays at prior)

Output: CSV files matching the prior format that can be used with
do_bayesian_update=False in the pipeline.
"""
import sys
from math import exp, log
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("/home/arr65/src/vs30")))

from vs30 import category, constants

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CATEGORICAL_DIR = Path("/home/arr65/src/vs30/vs30/resources/categorical_vs30_mean_and_stddev")
GEOL_PRIOR_CSV = CATEGORICAL_DIR / "geology_model_prior_mean_and_standard_deviation.csv"
TERR_PRIOR_CSV = CATEGORICAL_DIR / "terrain_model_prior_mean_and_standard_deviation.csv"
OBS_CSV = Path("/home/arr65/src/vs30/dev/jaehwi_v1p0_reproduction/jaehwi_reconstructed_observations.csv")
OUT_DIR = Path("/home/arr65/src/vs30/dev/jaehwi_v1p0_reproduction")

N_PRIOR = 3


def jaehwi_posterior_mean(prior_mean, obs_vs30, n_prior=N_PRIOR):
    """Batch Bayesian update of mean using Jaehwi's formula.

    All quality weights W=1 (no Q5 observations).
    """
    n = len(obs_vs30)
    if n == 0:
        return prior_mean
    numerator = n_prior * log(prior_mean) + np.sum(np.log(obs_vs30))
    denominator = n_prior + n
    return exp(numerator / denominator)


def compute_posteriors(prior_csv, obs_df, model_type):
    """Compute posteriors for one model type (geology or terrain)."""
    prior_df = pd.read_csv(prior_csv, skipinitialspace=True)

    # Assign category IDs to observations
    obs_locs = obs_df[["easting", "northing"]].values
    if model_type == "geology":
        obs_ids = category.assign_to_category_geology(obs_locs)
    else:
        obs_ids = category.assign_to_category_terrain(obs_locs)
    obs_df = obs_df.copy()
    obs_df["category_id"] = obs_ids

    # Update each category
    posterior_df = prior_df.copy()
    print(f"\n  {'ID':>4s}  {'Prior':>8s}  {'Post':>8s}  {'nObs':>4s}  Description")
    print(f"  {'--':>4s}  {'-----':>8s}  {'----':>8s}  {'----':>4s}  -----------")

    for idx, row in posterior_df.iterrows():
        cat_id = row[constants.STANDARD_ID_COLUMN]
        prior_mean = row[constants.COL_MEAN]

        if prior_mean == constants.NODATA_VALUE:
            continue

        group = obs_df[obs_df["category_id"] == cat_id]
        n = len(group)

        if n > 0:
            new_mean = jaehwi_posterior_mean(prior_mean, group["vs30"].values)
            posterior_df.at[idx, constants.COL_MEAN] = new_mean
            desc = row.get("description", "")
            print(f"  {cat_id:4d}  {prior_mean:8.2f}  {new_mean:8.2f}  {n:4d}  {desc}")
        # stdv is NOT updated (Jaehwi's key difference)

    return posterior_df


def main():
    obs_df = pd.read_csv(OBS_CSV)
    print(f"Loaded {len(obs_df)} observations from {OBS_CSV.name}")

    print("\n=== Geology Posteriors ===")
    geol_post = compute_posteriors(GEOL_PRIOR_CSV, obs_df, "geology")
    geol_out = OUT_DIR / "jaehwi_geology_posterior.csv"
    geol_post.to_csv(geol_out, index=False)
    print(f"\nWrote: {geol_out}")

    print("\n=== Terrain Posteriors ===")
    terr_post = compute_posteriors(TERR_PRIOR_CSV, obs_df, "terrain")
    terr_out = OUT_DIR / "jaehwi_terrain_posterior.csv"
    terr_post.to_csv(terr_out, index=False)
    print(f"\nWrote: {terr_out}")

    # Compare against Foster 2019 posteriors
    print("\n=== Comparison: Jaehwi posterior vs Foster 2019 posterior ===")
    foster_geol = pd.read_csv(
        CATEGORICAL_DIR / "geology_model_posterior_from_foster_2019_mean_and_standard_deviation.csv",
        skipinitialspace=True,
    )
    print("\nGeology (categories with differences > 1 m/s):")
    print(f"  {'ID':>4s}  {'Prior':>8s}  {'Jaehwi':>8s}  {'Foster':>8s}  {'J-F':>8s}")
    for idx, row in geol_post.iterrows():
        cat_id = row[constants.STANDARD_ID_COLUMN]
        if row[constants.COL_MEAN] == constants.NODATA_VALUE:
            continue
        foster_row = foster_geol[foster_geol[constants.STANDARD_ID_COLUMN] == cat_id]
        if foster_row.empty:
            continue
        prior_val = pd.read_csv(GEOL_PRIOR_CSV, skipinitialspace=True).iloc[idx][constants.COL_MEAN]
        j_val = row[constants.COL_MEAN]
        f_val = foster_row[constants.COL_MEAN].values[0]
        if abs(j_val - f_val) > 1:
            print(f"  {cat_id:4d}  {prior_val:8.2f}  {j_val:8.2f}  {f_val:8.2f}  {j_val - f_val:+8.2f}")


if __name__ == "__main__":
    main()

"""Generate the foster_2019_approx points benchmark CSV.

Replaces the 25 MB foster_2019_approx.tif benchmark raster (the paper's
published 100 m map) with a small CSV containing 30 prior-dominated
sample points and their reference Vs30/Stdv values.

Selection logic matches what tests/test_benchmarks.py previously ran at
test time: deterministic from ``SEED``, draws ``N_POINTS`` NZTM pixel
centres that are more than ``constants.MAX_DIST_M`` from any observation
in the foster_2019_approx obs CSV. Prior-dominated pixels are used
because the MVN step has no effect there, so the categorical posterior
and hybrid slope modification reproduce the paper at float precision.

Run once to produce the CSV checked into the repo:

    python dev/scripts/generators/generate_foster_2019_approx_points_benchmark.py

After regeneration the original 25 MB raster can be removed.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
import yaml
from scipy.spatial import cKDTree

from vs30 import constants


REPO_ROOT = Path(__file__).resolve().parents[3]
BENCHMARK_RASTER = REPO_ROOT / "tests/benchmarks/foster_2019_approx.tif"
OUTPUT_CSV = REPO_ROOT / "tests/benchmarks/foster_2019_approx_points.csv"

SEED = 42
N_POINTS = 30


def resolve_obs_csv_path() -> Path:
    """Return the absolute path to the foster_2019_approx independent observations CSV."""
    config_path = constants.MODEL_VERSION_TO_CONFIG[
        constants.FixedModelVersion.FOSTER_2019_APPROX
    ]
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    subdir = constants.RESOURCE_SUBDIRS["independent_observations_csv"]
    return constants.RESOURCE_PATH / subdir / cfg["independent_observations_csv"]


def select_prior_dominated_points(rng: np.random.Generator) -> np.ndarray:
    obs_df = pd.read_csv(resolve_obs_csv_path(), comment="#")
    obs_xy = obs_df[["easting", "northing"]].to_numpy()

    with rasterio.open(BENCHMARK_RASTER) as src:
        band = src.read(1)
        valid = ~np.isnan(band)
        if src.nodata is not None:
            valid &= band != src.nodata
        rows, cols = np.where(valid)
        idx = rng.choice(len(rows), size=min(200_000, len(rows)), replace=False)
        xs, ys = rasterio.transform.xy(
            src.transform, rows[idx], cols[idx], offset="center"
        )
    candidate_xy = np.column_stack([xs, ys])

    dists = cKDTree(obs_xy).query(candidate_xy, k=1)[0]
    prior_xy = candidate_xy[dists > constants.MAX_DIST_M]
    rng.shuffle(prior_xy)
    return prior_xy[:N_POINTS]


def main() -> None:
    rng = np.random.default_rng(SEED)
    pixel_xy = select_prior_dominated_points(rng)

    with rasterio.open(BENCHMARK_RASTER) as src:
        samples = np.array(list(src.sample(pixel_xy, indexes=[1, 2])))

    df = pd.DataFrame(
        {
            "easting": pixel_xy[:, 0],
            "northing": pixel_xy[:, 1],
            "benchmark_vs30": samples[:, 0],
            "benchmark_stdv": samples[:, 1],
        }
    )
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {len(df)} rows to {OUTPUT_CSV}")


if __name__ == "__main__":
    main()

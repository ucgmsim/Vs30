#!/usr/bin/env python3
"""
Can the refactored foster_2019_approx pipeline reproduce the published paper's map?

Approach: sample points from the benchmark raster in two cohorts and compare
the refactored points_pipeline output against the benchmark raster values at
those same coordinates.

Cohort A ("prior-dominated"): pixels whose distance to the nearest
observation is > MAX_DIST_M, so the MVN update has no effect. Tests
categorical priors + hybrid slope mod only.

Cohort B ("observation-influenced"): pixels within NEAR_OBS_M of at least
one observation. Tests MVN conditioning and observation handling.

The refactored code has 412 observations (derived from modified_foster_2019
filtering) while the paper had 393 — so divergence is expected in cohort B
even if the rest of the pipeline is faithful. See
dev/foster_2019_benchmark_status.md and dev/foster_2019_reproduction_comparison.md.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree

from vs30 import constants, pipeline

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from conftest import load_fixed_model_config  # noqa: E402


BENCHMARK_PATH = Path("/home/arr65/src/Vs30/tests/benchmarks/model_versions/foster_2019_approx.tif")
N_PER_COHORT = 30
NEAR_OBS_M = 500.0  # upper bound for cohort B
SEED = 42

NZTM_TO_WGS = Transformer.from_crs(2193, 4326, always_xy=True)


def load_obs_locations(cfg: dict) -> np.ndarray:
    """Return the NZTM easting/northing of every observation used by foster_2019_approx."""
    obs = pd.read_csv(cfg["independent_observations_csv"], comment="#")
    return obs[["easting", "northing"]].to_numpy()


def sample_pixel_centers(raster_path: Path, rng: np.random.Generator, n_candidates: int) -> np.ndarray:
    """Return n_candidates random valid pixel centre coordinates (NZTM) as (N, 2)."""
    with rasterio.open(raster_path) as src:
        band = src.read(1)
        nodata = src.nodata
        valid = ~np.isnan(band)
        if nodata is not None:
            valid &= band != nodata
        rows, cols = np.where(valid)
        # Over-sample the full valid pool; caller will down-select per cohort.
        idx = rng.choice(len(rows), size=min(n_candidates, len(rows)), replace=False)
        xs, ys = rasterio.transform.xy(src.transform, rows[idx], cols[idx], offset="center")
    return np.column_stack([xs, ys])


def split_cohorts(pixel_xy: np.ndarray, obs_xy: np.ndarray, max_dist_m: float, near_obs_m: float):
    """Split pixel coords into (prior_dominated, near_obs) based on distance to nearest obs."""
    tree = cKDTree(obs_xy)
    dists, _ = tree.query(pixel_xy, k=1)
    prior_mask = dists > max_dist_m
    near_mask = dists < near_obs_m
    return pixel_xy[prior_mask], pixel_xy[near_mask], dists


def run_points_pipeline(cfg: dict, pixel_xy: np.ndarray) -> pd.DataFrame:
    """Run points_pipeline at the given NZTM coordinates."""
    lons, lats = NZTM_TO_WGS.transform(pixel_xy[:, 0], pixel_xy[:, 1])
    return pipeline.points_pipeline(
        longitudes=np.asarray(lons),
        latitudes=np.asarray(lats),
        geology_categorical_csv=cfg["geology_categorical_csv"],
        terrain_categorical_csv=cfg["terrain_categorical_csv"],
        clustered_observations_csv=cfg.get("clustered_observations_csv"),
        independent_observations_csv=cfg.get("independent_observations_csv"),
        combination_method=constants.CombinationMethod(cfg["combination_method"]),
        combine_ratio=cfg.get("combine_ratio"),
        noisy=cfg["noisy"],
        do_bayesian_update=cfg["do_bayesian_update"],
        n_proc=1,
        geology_corr_fn=cfg.get("geology_corr_fn"),
        terrain_corr_fn=cfg.get("terrain_corr_fn"),
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        apply_coastal_distance_mod=cfg["apply_coastal_distance_mod"],
        fill_gaps=cfg.get("fill_gaps", False),
    )


def sample_benchmark(raster_path: Path, pixel_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Read benchmark (vs30, stdv) at the given NZTM coordinates."""
    with rasterio.open(raster_path) as src:
        samples = np.array(list(src.sample(pixel_xy, indexes=[1, 2])))
    return samples[:, 0], samples[:, 1]


def summarise(label: str, refactored_vs30, refactored_stdv, bench_vs30, bench_stdv) -> None:
    vs30_diff = refactored_vs30 - bench_vs30
    stdv_diff = refactored_stdv - bench_stdv
    vs30_rel = np.abs(vs30_diff) / np.abs(bench_vs30)
    stdv_rel = np.abs(stdv_diff) / np.abs(bench_stdv)

    def stats(arr):
        return (
            f"min={np.min(arr):.4g}, median={np.median(arr):.4g}, "
            f"mean={np.mean(arr):.4g}, max={np.max(arr):.4g}"
        )

    print(f"\n=== {label} ===")
    print(f"N = {len(bench_vs30)}")
    print(f"Vs30 abs diff (m/s):  {stats(np.abs(vs30_diff))}")
    print(f"Vs30 rel diff:        {stats(vs30_rel)}")
    print(f"  % within 0.1 %:     {100 * np.mean(vs30_rel < 1e-3):.1f}")
    print(f"  % within 1 %:       {100 * np.mean(vs30_rel < 1e-2):.1f}")
    print(f"Stdv abs diff:        {stats(np.abs(stdv_diff))}")
    print(f"Stdv rel diff:        {stats(stdv_rel)}")
    print(f"  % within 1 %:       {100 * np.mean(stdv_rel < 1e-2):.1f}")


def main() -> None:
    cfg = load_fixed_model_config(constants.FixedModelVersion.FOSTER_2019_APPROX)
    obs_xy = load_obs_locations(cfg)
    print(f"Loaded {len(obs_xy)} foster_2019_approx observations")

    rng = np.random.default_rng(SEED)
    candidate_xy = sample_pixel_centers(BENCHMARK_PATH, rng, n_candidates=200_000)
    prior_xy_all, near_xy_all, all_dists = split_cohorts(
        candidate_xy, obs_xy, constants.MAX_DIST_M, NEAR_OBS_M
    )
    print(f"Of {len(candidate_xy)} candidate pixels:")
    print(f"  prior-dominated (dist > {constants.MAX_DIST_M} m): {len(prior_xy_all)}")
    print(f"  near-obs        (dist < {NEAR_OBS_M} m):          {len(near_xy_all)}")

    # Shuffle and take the first N from each cohort for a consistent sample.
    rng.shuffle(prior_xy_all)
    rng.shuffle(near_xy_all)
    prior_xy = prior_xy_all[:N_PER_COHORT]
    near_xy = near_xy_all[:N_PER_COHORT]

    print(f"\nRunning points_pipeline for {len(prior_xy) + len(near_xy)} points…")
    combined_xy = np.concatenate([prior_xy, near_xy])
    result = run_points_pipeline(cfg, combined_xy)

    bench_vs30, bench_stdv = sample_benchmark(BENCHMARK_PATH, combined_xy)

    refactored_vs30 = result[constants.ObservationColumn.VS30].to_numpy()
    refactored_stdv = result[constants.COL_COMBINED_STDV].to_numpy()

    prior_slice = slice(0, N_PER_COHORT)
    near_slice = slice(N_PER_COHORT, 2 * N_PER_COHORT)
    summarise(
        "COHORT A: prior-dominated",
        refactored_vs30[prior_slice], refactored_stdv[prior_slice],
        bench_vs30[prior_slice], bench_stdv[prior_slice],
    )
    summarise(
        "COHORT B: observation-influenced",
        refactored_vs30[near_slice], refactored_stdv[near_slice],
        bench_vs30[near_slice], bench_stdv[near_slice],
    )


if __name__ == "__main__":
    main()

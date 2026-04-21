"""
Benchmark tests that compare pipeline output against reference benchmarks.

Three model versions (modified_foster_2019, jaehwi_v1p0, viktor_cpt_clustering)
run the full grid pipeline at 5000 m resolution and compare the in-memory
result against a stored benchmark raster in under 30 seconds per test.

foster_2019_approx is benchmarked differently: its benchmark raster is the
paper's published 100 m map, and a full-domain refactored run at 100 m takes
~6 hours. Instead, ``test_foster_2019_approx_points_benchmark`` samples the
pipeline at a deterministic set of prior-dominated points (more than
MAX_DIST_M from any observation) and compares against the benchmark raster
at the same coordinates. Prior-dominated points are chosen because the MVN
step has no effect there, so the categorical posterior and hybrid slope
modification reproduce the paper at float precision. A small minority of
pixels have larger discrepancies due to categorical/hybrid edge cases
unrelated to the MVN; the test uses median + percentile assertions to catch
drift while tolerating those known outliers. See
``dev/foster_2019_benchmark_status.md`` for background.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
from conftest import assert_arrays_match_raster_benchmark, load_fixed_model_config
from pyproj import Transformer
from scipy.spatial import cKDTree

from vs30 import config, constants, pipeline

BENCHMARKS_DIR = Path(__file__).parent / "benchmarks" / "model_versions"

# Shared grid for modified_foster_2019, jaehwi_v1p0, and viktor_cpt_clustering.
BENCHMARK_NZ_GRID = config.GridConfig(
    grid_xmin=1060100,
    grid_xmax=2120100,
    grid_ymin=4730100,
    grid_ymax=6250100,
    grid_dx=5000,
    grid_dy=5000,
)

# foster_2019_approx points benchmark — see module docstring.
FOSTER_2019_APPROX_BENCHMARK_PATH = BENCHMARKS_DIR / "foster_2019_approx.tif"
FOSTER_2019_APPROX_POINTS_SEED = 42
FOSTER_2019_APPROX_N_POINTS = 30

_NZTM_TO_WGS = Transformer.from_crs(2193, 4326, always_xy=True)


def _select_foster_2019_approx_prior_points() -> np.ndarray:
    """Deterministically select prior-dominated NZTM pixel centres from the benchmark.

    Returns ``FOSTER_2019_APPROX_N_POINTS`` (easting, northing) pairs
    sampled from foster_2019_approx.tif's valid pixels that are more than
    ``MAX_DIST_M`` from any observation in the foster_2019_approx obs CSV.
    The selection is deterministic from ``FOSTER_2019_APPROX_POINTS_SEED``
    and the benchmark + obs files on disk.
    """
    cfg = load_fixed_model_config(constants.FixedModelVersion.FOSTER_2019_APPROX)
    obs_df = pd.read_csv(cfg["independent_observations_csv"], comment="#")
    obs_xy = obs_df[["easting", "northing"]].to_numpy()

    rng = np.random.default_rng(FOSTER_2019_APPROX_POINTS_SEED)
    with rasterio.open(FOSTER_2019_APPROX_BENCHMARK_PATH) as src:
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
    return prior_xy[:FOSTER_2019_APPROX_N_POINTS]


def run_benchmark(version: constants.FixedModelVersion, grid: config.GridConfig, n_proc: int) -> None:
    """Run the grid pipeline for a fixed model version and compare against the benchmark raster."""
    cfg = load_fixed_model_config(version)

    result = pipeline.grid_pipeline(
        grid_config=grid,
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
        n_proc=n_proc,
    )

    benchmark = BENCHMARKS_DIR / f"{version}.tif"
    assert_arrays_match_raster_benchmark(
        result["combined_vs30"], result["combined_stdv"], benchmark
    )


@pytest.mark.parametrize("n_proc", [1, -1], ids=["single_process", "multiprocess"])
def test_foster_2019_approx_points_benchmark(n_proc):
    """Prior-dominated points from foster_2019_approx.tif match points_pipeline output.

    Tests that the categorical posterior + hybrid slope modification
    reproduce the paper's published Vs30 raster at float precision for
    pixels outside any observation's MVN neighbourhood. Runs in both
    single-process and multiprocess modes to cover both dispatch paths.

    The assertions use median + 80th-percentile thresholds rather than a
    hard rtol because a small minority of prior-dominated pixels have
    larger discrepancies from the paper (up to ~19 % in the worst case)
    due to categorical/hybrid edge-case differences between the legacy
    R pipeline and the refactored Python one — see
    ``dev/foster_2019_reproduction_comparison.md``. Those outliers are
    independent of MVN conditioning and known to exist. The median must
    stay tight because any drift in the common-case codepath would show
    up there immediately.
    """
    cfg = load_fixed_model_config(constants.FixedModelVersion.FOSTER_2019_APPROX)
    pixel_xy = _select_foster_2019_approx_prior_points()
    lons, lats = _NZTM_TO_WGS.transform(pixel_xy[:, 0], pixel_xy[:, 1])

    result = pipeline.points_pipeline(
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
        n_proc=n_proc,
        geology_corr_fn=cfg.get("geology_corr_fn"),
        terrain_corr_fn=cfg.get("terrain_corr_fn"),
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        apply_coastal_distance_mod=cfg["apply_coastal_distance_mod"],
        fill_gaps=cfg.get("fill_gaps", False),
    )

    with rasterio.open(FOSTER_2019_APPROX_BENCHMARK_PATH) as src:
        samples = np.array(list(src.sample(pixel_xy, indexes=[1, 2])))
    bench_vs30 = samples[:, 0]
    bench_stdv = samples[:, 1]

    actual_vs30 = result[constants.ObservationColumn.VS30].to_numpy()
    actual_stdv = result[constants.COL_COMBINED_STDV].to_numpy()

    vs30_rel = np.abs(actual_vs30 - bench_vs30) / np.abs(bench_vs30)
    stdv_rel = np.abs(actual_stdv - bench_stdv) / np.abs(bench_stdv)

    assert np.median(vs30_rel) < 1e-4, (
        f"Vs30 median rel diff {np.median(vs30_rel):.3g} exceeds 1e-4 — "
        "common-case categorical/hybrid pipeline has drifted"
    )
    assert np.median(stdv_rel) < 1e-3, (
        f"Stdv median rel diff {np.median(stdv_rel):.3g} exceeds 1e-3"
    )
    assert np.percentile(vs30_rel, 80) < 1e-2, (
        f"80th-pct Vs30 rel diff {np.percentile(vs30_rel, 80):.3g} exceeds 1 %"
    )
    assert np.percentile(stdv_rel, 80) < 2e-2, (
        f"80th-pct Stdv rel diff {np.percentile(stdv_rel, 80):.3g} exceeds 2 %"
    )


def test_modified_foster_2019_single_process():
    """modified_foster_2019 full-domain pipeline matches benchmark (n_proc=1)."""
    run_benchmark(constants.FixedModelVersion.MODIFIED_FOSTER_2019, BENCHMARK_NZ_GRID, n_proc=1)


def test_modified_foster_2019_multiprocess():
    """modified_foster_2019 full-domain pipeline matches benchmark (all CPUs)."""
    run_benchmark(constants.FixedModelVersion.MODIFIED_FOSTER_2019, BENCHMARK_NZ_GRID, n_proc=os.cpu_count())


def test_jaehwi_v1p0_single_process():
    """jaehwi_v1p0 full-domain pipeline matches benchmark (n_proc=1)."""
    run_benchmark(constants.FixedModelVersion.JAEHWI_V1P0, BENCHMARK_NZ_GRID, n_proc=1)


def test_jaehwi_v1p0_multiprocess():
    """jaehwi_v1p0 full-domain pipeline matches benchmark (all CPUs)."""
    run_benchmark(constants.FixedModelVersion.JAEHWI_V1P0, BENCHMARK_NZ_GRID, n_proc=os.cpu_count())


def test_viktor_cpt_clustering_single_process():
    """viktor_cpt_clustering full-domain pipeline matches benchmark (n_proc=1)."""
    run_benchmark(constants.FixedModelVersion.VIKTOR_CPT_CLUSTERING, BENCHMARK_NZ_GRID, n_proc=1)


def test_viktor_cpt_clustering_multiprocess():
    """viktor_cpt_clustering full-domain pipeline matches benchmark (all CPUs)."""
    run_benchmark(constants.FixedModelVersion.VIKTOR_CPT_CLUSTERING, BENCHMARK_NZ_GRID, n_proc=os.cpu_count())

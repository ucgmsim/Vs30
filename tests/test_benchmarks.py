"""
Benchmark tests that compare pipeline output against reference benchmarks.

Three model versions (modified_foster_2019, jaehwi_v1p0, viktor_cpt_clustering)
run the full grid pipeline at 5000 m resolution and compare the in-memory
result against a stored benchmark raster in under 30 seconds per test.

foster_2019_approx is benchmarked differently: a full-domain refactored run
at the paper's 100 m resolution takes ~6 hours, so
``test_foster_2019_approx_points_benchmark`` instead compares pipeline
output at 30 fixed prior-dominated points (more than MAX_DIST_M from any
observation) against reference values sampled from the paper's published
map. The points and reference values are baked into
``foster_2019_approx_points.csv``; see
``dev/generate_foster_2019_approx_points_benchmark.py`` for regeneration.
Prior-dominated points are used because the MVN step has no effect there,
so the categorical posterior and hybrid slope modification reproduce the
paper at float precision. A small minority of pixels have larger
discrepancies due to categorical/hybrid edge cases unrelated to MVN; the
test uses median + percentile assertions to catch drift while tolerating
those known outliers. See ``dev/docs/foster_2019_benchmark_status.md`` for
background.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from conftest import assert_arrays_match_raster_benchmark, load_fixed_model_config
import pyproj

from vs30 import config, constants, pipeline

BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"

# Shared grid for modified_foster_2019, jaehwi_v1p0, and viktor_cpt_clustering.
#
# At dx=dy=5000, pixel CENTRES sit at xmin + 2500 + n*5000. To land each
# centre exactly on an IwahashiPike pixel centre (which is at coordinates
# ending in ..50 in both axes), xmin/ymin must end in ..50 — different
# from FULL_NZ_GRID_CONFIG's ..100 (which is correct for dx=100). See
# dev/docs/grid_bounds_semantics_investigation.md.
BENCHMARK_NZ_GRID = config.GridConfig(
    grid_xmin=1060050,
    grid_xmax=2120050,
    grid_ymin=4730050,
    grid_ymax=6250050,
    grid_dx=5000,
    grid_dy=5000,
)

FOSTER_2019_APPROX_BENCHMARK_POINTS_CSV = (
    BENCHMARKS_DIR / "foster_2019_approx_points.csv"
)

NZTM_TO_WGS = pyproj.Transformer.from_crs(2193, 4326, always_xy=True)


def run_benchmark(
    version: constants.FixedModelVersion, grid: config.GridConfig
) -> None:
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
        dbscan_nproc=1,
    )

    benchmark = BENCHMARKS_DIR / f"{version}.tif"
    assert_arrays_match_raster_benchmark(
        result["combined_vs30"], result["combined_stdv"], benchmark
    )


def test_foster_2019_approx_points_benchmark():
    """Prior-dominated points from foster_2019_approx.tif match points_pipeline output.

    Tests that the categorical posterior + hybrid slope modification
    reproduce the paper's published Vs30 raster at float precision for
    pixels outside any observation's MVN neighbourhood.

    The assertions use median + 80th-percentile thresholds rather than a
    hard rtol because a small minority of prior-dominated pixels have
    larger discrepancies from the paper (up to ~19 % in the worst case)
    due to categorical/hybrid edge-case differences between the legacy
    R pipeline and the refactored Python one. Those outliers are
    independent of MVN conditioning and known to exist. The median must
    stay tight because any drift in the common-case codepath would show
    up there immediately.
    """
    cfg = load_fixed_model_config(constants.FixedModelVersion.FOSTER_2019_APPROX)
    bench_df = pd.read_csv(FOSTER_2019_APPROX_BENCHMARK_POINTS_CSV)
    lons, lats = NZTM_TO_WGS.transform(
        bench_df["easting"].to_numpy(), bench_df["northing"].to_numpy()
    )

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
        geology_corr_fn=cfg.get("geology_corr_fn"),
        terrain_corr_fn=cfg.get("terrain_corr_fn"),
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        apply_coastal_distance_mod=cfg["apply_coastal_distance_mod"],
        fill_gaps=cfg.get("fill_gaps", False),
    )

    bench_vs30 = bench_df["benchmark_vs30"].to_numpy()
    bench_stdv = bench_df["benchmark_stdv"].to_numpy()

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


def test_modified_foster_2019():
    """modified_foster_2019 full-domain pipeline matches benchmark."""
    run_benchmark(constants.FixedModelVersion.MODIFIED_FOSTER_2019, BENCHMARK_NZ_GRID)


def test_jaehwi_v1p0():
    """jaehwi_v1p0 full-domain pipeline matches benchmark."""
    run_benchmark(constants.FixedModelVersion.JAEHWI_V1P0, BENCHMARK_NZ_GRID)


def test_viktor_cpt_clustering():
    """viktor_cpt_clustering full-domain pipeline matches benchmark."""
    run_benchmark(constants.FixedModelVersion.VIKTOR_CPT_CLUSTERING, BENCHMARK_NZ_GRID)

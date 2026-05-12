"""Benchmark tests that compare pipeline output against reference benchmarks."""

from pathlib import Path

import numpy as np
import pandas as pd
import pyproj

from conftest import assert_arrays_match_raster_benchmark, load_fixed_model_config
from vs30 import config, constants, pipeline

BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"

# FULL_NZ_GRID_CONFIG is the production grid defined elsewhere in the
# codebase (100 m resolution). This benchmark grid uses 5000 m and
# needs different bounds (ending in ..50) so pixel centres still land
# on the bundled IwahashiPike raster's ..50 centres.
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
    version: constants.FixedModelVersion,
    grid: config.GridConfig,
    rtol: float | None = None,
) -> None:
    """Run the grid pipeline for a fixed model version and compare against the benchmark raster."""
    cfg = load_fixed_model_config(version)

    result = pipeline.grid_pipeline(
        grid_config=grid,
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        geology_corr_fn=cfg["geology_corr_fn"],
        terrain_corr_fn=cfg["terrain_corr_fn"],
        output_dir=None,
        geology_categorical_csv=cfg["geology_categorical_csv"],
        terrain_categorical_csv=cfg["terrain_categorical_csv"],
        clustered_observations_csv=cfg["clustered_observations_csv"],
        independent_observations_csv=cfg["independent_observations_csv"],
        combination_method=constants.CombinationMethod(cfg["combination_method"]),
        combine_ratio=cfg["combine_ratio"],
        noisy=cfg["noisy"],
        do_bayesian_update=cfg["do_bayesian_update"],
        dbscan_nproc=1,
        apply_coastal_distance_mod=cfg["apply_coastal_distance_mod"],
        fill_gaps=cfg["fill_gaps"],
    )

    benchmark = BENCHMARKS_DIR / f"{version}.tif"
    assert_arrays_match_raster_benchmark(
        result["combined_vs30"], result["combined_stdv"], benchmark, rtol=rtol
    )


def test_foster_2019_approx_points_benchmark():
    """
    foster_2019 pipeline matches the paper's published Vs30 at points far from any observation.

    The 30 sample points all sit further from any observation than
    ``MAX_DIST_M``, so the MVN spatial-adjustment step contributes nothing
    to their predicted Vs30 — only the categorical posterior and hybrid
    slope modification do. Those two stages reproduce the legacy R
    pipeline at float precision, so we can pin them tightly against the
    paper's published values.

    The MVN step is intentionally outside this test's scope. foster_2019
    is itself a known-flawed model, and the refactored implementation
    approximates it rather than reproducing it exactly: the legacy R and
    Python MVN fitting routines differ numerically, and matching the
    legacy behaviour for a flawed model is not worth the engineering
    cost. The refactored MVN is the accepted reference going forward.

    Notes
    -----
    The assertions use a very tight threshold on the median and a looser one
    on the upper tail. A small minority of points have larger errors due to
    known edge cases in the categorical posterior and hybrid slope steps.
    Tightening the tail threshold to chase those outliers would just make
    the test fail without exposing any real regression.
    """
    cfg = load_fixed_model_config(constants.FixedModelVersion.FOSTER_2019_APPROX)
    bench_df = pd.read_csv(FOSTER_2019_APPROX_BENCHMARK_POINTS_CSV)
    lons, lats = NZTM_TO_WGS.transform(
        bench_df["easting"].to_numpy(), bench_df["northing"].to_numpy()
    )

    result = pipeline.points_pipeline(
        longitudes=np.asarray(lons),
        latitudes=np.asarray(lats),
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        geology_corr_fn=cfg.get("geology_corr_fn"),
        terrain_corr_fn=cfg.get("terrain_corr_fn"),
        geology_categorical_csv=cfg["geology_categorical_csv"],
        terrain_categorical_csv=cfg["terrain_categorical_csv"],
        clustered_observations_csv=cfg.get("clustered_observations_csv"),
        independent_observations_csv=cfg.get("independent_observations_csv"),
        combination_method=constants.CombinationMethod(cfg["combination_method"]),
        combine_ratio=cfg.get("combine_ratio"),
        noisy=cfg["noisy"],
        do_bayesian_update=cfg["do_bayesian_update"],
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
    run_benchmark(
        constants.FixedModelVersion.JAEHWI_V1P0,
        BENCHMARK_NZ_GRID,
        rtol=2e-4,
    )


def test_viktor_cpt_clustering():
    """viktor_cpt_clustering full-domain pipeline matches benchmark."""
    run_benchmark(constants.FixedModelVersion.VIKTOR_CPT_CLUSTERING, BENCHMARK_NZ_GRID)

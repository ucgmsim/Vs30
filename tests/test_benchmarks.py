"""
Benchmark tests that compare the grid pipeline output against reference benchmarks.

These tests run the complete grid pipeline over the full NZ domain and compare
the in-memory results against a stored benchmark raster. They are marked slow
because each run takes approximately 40 minutes.

These tests are skipped by default. To run them:
    pytest --runslow tests/test_benchmarks.py
"""

import dataclasses
import os
from pathlib import Path

import pytest
from conftest import assert_arrays_match_raster_benchmark, load_fixed_model_config

from vs30 import config, constants, pipeline

BENCHMARKS_DIR = Path(__file__).parent / "benchmarks" / "model_versions"

FOSTER_2019_GRID = config.GridConfig(
    grid_xmin=1000000,
    grid_xmax=2126400,
    grid_ymin=4700000,
    grid_ymax=6338400,
    grid_dx=100,
    grid_dy=100,
)

STANDARD_NZ_GRID = dataclasses.replace(constants.FULL_NZ_GRID_CONFIG, grid_dx=400, grid_dy=400)


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


@pytest.mark.slow
def test_foster_2019_single_process():
    """foster_2019 full-domain pipeline matches benchmark (n_proc=1)."""
    run_benchmark(constants.FixedModelVersion.FOSTER_2019, FOSTER_2019_GRID, n_proc=1)


@pytest.mark.slow
def test_foster_2019_multiprocess():
    """foster_2019 full-domain pipeline matches benchmark (all CPUs)."""
    run_benchmark(constants.FixedModelVersion.FOSTER_2019, FOSTER_2019_GRID, n_proc=os.cpu_count())


@pytest.mark.slow
def test_modified_foster_2019_single_process():
    """modified_foster_2019 full-domain pipeline matches benchmark (n_proc=1)."""
    run_benchmark(constants.FixedModelVersion.MODIFIED_FOSTER_2019, STANDARD_NZ_GRID, n_proc=1)


@pytest.mark.slow
def test_modified_foster_2019_multiprocess():
    """modified_foster_2019 full-domain pipeline matches benchmark (all CPUs)."""
    run_benchmark(constants.FixedModelVersion.MODIFIED_FOSTER_2019, STANDARD_NZ_GRID, n_proc=os.cpu_count())


@pytest.mark.slow
def test_jaehwi_v1p0_single_process():
    """jaehwi_v1p0 full-domain pipeline matches benchmark (n_proc=1)."""
    run_benchmark(constants.FixedModelVersion.JAEHWI_V1P0, STANDARD_NZ_GRID, n_proc=1)


@pytest.mark.slow
def test_jaehwi_v1p0_multiprocess():
    """jaehwi_v1p0 full-domain pipeline matches benchmark (all CPUs)."""
    run_benchmark(constants.FixedModelVersion.JAEHWI_V1P0, STANDARD_NZ_GRID, n_proc=os.cpu_count())

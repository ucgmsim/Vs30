"""Tests that grid and points pipelines produce consistent Vs30 values for the same locations."""

import numpy as np
import pandas as pd
import pytest
from qcore import coordinates

from conftest import FIXTURES_DIR, load_fixed_model_config
from vs30 import config, constants, gapfill, pipeline

POINTS_CSV = FIXTURES_DIR / "consistency_test_points.csv"

# Stdv needs a wider tolerance because the points pipeline samples slope/coast distance
# at exact coordinates while the grid pipeline uses resampled pixel values, and these
# small input differences propagate nonlinearly into the MVN posterior variance.
STDV_RTOL = 0.10

# Half-width for the 3x3 local grid (150m each side of center -> 300m / 100m = 3 pixels).
LOCAL_GRID_HALF_WIDTH = 150

# Model versions that do NOT use coastal distance — cheap enough to run
# in the fast tier on a small point subset.
FAST_VERSIONS = [
    constants.FixedModelVersion.FOSTER_2019_APPROX,
    constants.FixedModelVersion.JAEHWI_V1P0,
]

# All model versions, used by the slow tier.
ALL_VERSIONS = [
    constants.FixedModelVersion.FOSTER_2019_APPROX,
    constants.FixedModelVersion.JAEHWI_V1P0,
    constants.FixedModelVersion.MODIFIED_FOSTER_2019,
    constants.FixedModelVersion.VIKTOR_CPT_CLUSTERING,
]

# Fast-tier point subset: 3 geologically distinct cities covering the north
# (volcanic Auckland), central (complex Wellington), and south (alluvial
# Christchurch). Enough to smoke-test the grid/points pipelines.
FAST_POINT_NAMES = ["auckland", "wellington", "christchurch"]


def load_test_points() -> pd.DataFrame:
    """Load the pre-computed test points CSV."""
    df = pd.read_csv(POINTS_CSV)
    assert {"name", "longitude", "latitude", "category"}.issubset(df.columns)
    return df


def run_points_pipeline_for_version(cfg: dict, points_df: pd.DataFrame) -> pd.DataFrame:
    """Run points_pipeline once with all test points for a given model config."""
    return pipeline.points_pipeline(
        longitudes=points_df["longitude"].values,
        latitudes=points_df["latitude"].values,
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


def run_grid_pipeline_at_point(
    cfg: dict, easting: float, northing: float
) -> tuple[float, float]:
    """Run grid_pipeline on a 3x3 grid centered on (easting, northing).

    Returns the center pixel (row=1, col=1) Vs30 and stdv.
    """
    local_config = gapfill.create_local_grid_config(
        easting, northing, config.FULL_NZ_GRID_CONFIG, LOCAL_GRID_HALF_WIDTH
    )

    result = pipeline.grid_pipeline(
        grid_config=local_config,
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        geology_corr_fn=cfg.get("geology_corr_fn"),
        terrain_corr_fn=cfg.get("terrain_corr_fn"),
        output_dir=None,
        geology_categorical_csv=cfg["geology_categorical_csv"],
        terrain_categorical_csv=cfg["terrain_categorical_csv"],
        clustered_observations_csv=cfg.get("clustered_observations_csv"),
        independent_observations_csv=cfg.get("independent_observations_csv"),
        combination_method=constants.CombinationMethod(cfg["combination_method"]),
        combine_ratio=cfg.get("combine_ratio"),
        noisy=cfg["noisy"],
        do_bayesian_update=cfg["do_bayesian_update"],
        dbscan_nproc=1,
        apply_coastal_distance_mod=cfg["apply_coastal_distance_mod"],
        fill_gaps=cfg.get("fill_gaps", False),
    )

    grid_vs30 = result["combined_vs30"]
    grid_stdv = result["combined_stdv"]

    # Center pixel of the 3x3 grid
    return float(grid_vs30[1, 1]), float(grid_stdv[1, 1])


def check_consistency_for_version(
    version: constants.FixedModelVersion,
    point_filter: list[str] | None = None,
):
    """Core comparison logic shared by fast and slow tiers.

    If ``point_filter`` is provided, only points whose ``name`` is in the
    list are tested — used by the fast tier to keep runtime low.
    """
    cfg = load_fixed_model_config(version)
    points_df = load_test_points()
    if point_filter is not None:
        points_df = points_df[points_df["name"].isin(point_filter)].reset_index(
            drop=True
        )

    # Batch points pipeline call
    points_result = run_points_pipeline_for_version(cfg, points_df)

    # Convert lon/lat to NZTM for grid pipeline calls
    lats = points_df["latitude"].values
    lons = points_df["longitude"].values
    nztm = coordinates.wgs_depth_to_nztm(np.column_stack([lats, lons]))
    eastings = nztm[:, 1]
    northings = nztm[:, 0]

    failures = []
    for i in range(len(points_df)):
        name = points_df["name"].iloc[i]
        e, n = eastings[i], northings[i]

        grid_vs30, grid_stdv = run_grid_pipeline_at_point(cfg, e, n)

        pts_vs30 = points_result[constants.ObservationColumn.VS30].iloc[i]
        pts_stdv = points_result[constants.COL_COMBINED_STDV].iloc[i]

        # Skip nodata points (both pipelines should agree on nodata)
        if np.isnan(grid_vs30) and np.isnan(pts_vs30):
            continue

        # Catch asymmetric NaN (one pipeline returns data, the other doesn't)
        if np.isnan(grid_vs30) != np.isnan(pts_vs30):
            failures.append(
                f"  {name}: NaN disagreement — "
                f"grid={'nan' if np.isnan(grid_vs30) else f'{grid_vs30:.2f}'}, "
                f"points={'nan' if np.isnan(pts_vs30) else f'{pts_vs30:.2f}'}"
            )
            continue

        # Check Vs30
        if pts_vs30 != pytest.approx(grid_vs30):
            failures.append(
                f"  {name}: Vs30 mismatch — "
                f"grid={grid_vs30:.2f}, points={pts_vs30:.2f}, "
                f"rdiff={abs(pts_vs30 - grid_vs30) / grid_vs30:.4f}"
            )

        # Check Stdv
        if pts_stdv != pytest.approx(grid_stdv, rel=STDV_RTOL):
            failures.append(
                f"  {name}: Stdv mismatch — "
                f"grid={grid_stdv:.2f}, points={pts_stdv:.2f}, "
                f"rdiff={abs(pts_stdv - grid_stdv) / grid_stdv:.4f}"
            )

    if failures:
        msg = f"\n{version.value}: {len(failures)} failure(s):\n" + "\n".join(failures)
        pytest.fail(msg)


@pytest.mark.parametrize("version", FAST_VERSIONS, ids=lambda v: v.value)
def test_grid_points_consistency_fast(version):
    """Grid/points consistency smoke test: 3 cities, 2 simple models."""
    check_consistency_for_version(version, point_filter=FAST_POINT_NAMES)


@pytest.mark.slow
@pytest.mark.parametrize("version", ALL_VERSIONS, ids=lambda v: v.value)
def test_grid_points_consistency_slow(version):
    """Full grid/points consistency: many test points across all model versions."""
    check_consistency_for_version(version)

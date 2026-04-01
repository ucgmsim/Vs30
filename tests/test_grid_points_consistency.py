"""
Test that grid and points pipelines produce consistent Vs30 values
across the full NZ domain for all fixed model versions.

For each test point, a tiny 3x3 grid (300m x 300m at 100m resolution) is
generated and run through grid_pipeline.  The center pixel is compared
against the batched points_pipeline result at the same coordinates.

The test is split into two tiers:
- Fast tier (foster_2019, jaehwi_v1p0): no coastal distance computation,
  runs in ~4-5 minutes.
- Slow tier (modified_foster_2019, viktor_cpt_clustering): coastal distance
  extends to full NZ domain per point, runs in ~25-30 minutes.
"""

import numpy as np
import pandas as pd
import pytest
from qcore import coordinates

from conftest import FIXTURES_DIR, load_fixed_model_config
from vs30 import constants, gapfill, pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POINTS_CSV = FIXTURES_DIR / "consistency_test_points.csv"

# Tolerances for approximate grid/points agreement.
# The 3x3 grid approach minimises resampling discrepancy (most observations
# fall outside the tiny grid and use direct source sampling in both paths).
VS30_RTOL = 0.03
STDV_RTOL = 0.30

# Half-width for the 3x3 local grid (150m each side of center -> 300m / 100m = 3 pixels).
LOCAL_GRID_HALF_WIDTH = 150

# Model versions that do NOT use coastal distance (fast tier).
FAST_VERSIONS = [
    constants.FixedModelVersion.FOSTER_2019,
    constants.FixedModelVersion.JAEHWI_V1P0,
]

# Model versions that DO use coastal distance (slow tier).
SLOW_VERSIONS = [
    constants.FixedModelVersion.MODIFIED_FOSTER_2019,
    constants.FixedModelVersion.VIKTOR_CPT_CLUSTERING,
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_test_points() -> pd.DataFrame:
    """Load the pre-computed test points CSV."""
    df = pd.read_csv(POINTS_CSV)
    assert {"name", "longitude", "latitude", "category"}.issubset(df.columns)
    return df


def run_points_pipeline_for_version(
    cfg: dict, points_df: pd.DataFrame
) -> pd.DataFrame:
    """Run points_pipeline once with all test points for a given model config."""
    return pipeline.points_pipeline(
        longitudes=points_df["longitude"].values,
        latitudes=points_df["latitude"].values,
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
    )


def run_grid_pipeline_at_point(
    cfg: dict, easting: float, northing: float
) -> tuple[float, float]:
    """Run grid_pipeline on a 3x3 grid centered on (easting, northing).

    Returns the center pixel (row=1, col=1) Vs30 and stdv.
    """
    local_config = gapfill.create_local_grid_config(
        easting, northing, constants.FULL_NZ_GRID_CONFIG, LOCAL_GRID_HALF_WIDTH
    )

    result = pipeline.grid_pipeline(
        grid_config=local_config,
        output_dir=None,
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
    )

    grid_vs30 = result["combined_vs30"]
    grid_stdv = result["combined_stdv"]

    # Center pixel of the 3x3 grid
    return float(grid_vs30[1, 1]), float(grid_stdv[1, 1])


def _check_consistency_for_version(version: constants.FixedModelVersion):
    """Core comparison logic shared by fast and slow tiers."""
    cfg = load_fixed_model_config(version)
    points_df = load_test_points()

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

        # Check Vs30
        if not np.isclose(pts_vs30, grid_vs30, rtol=VS30_RTOL):
            failures.append(
                f"  {name}: Vs30 mismatch — "
                f"grid={grid_vs30:.2f}, points={pts_vs30:.2f}, "
                f"rdiff={abs(pts_vs30 - grid_vs30) / grid_vs30:.4f}"
            )

        # Check Stdv
        if not np.isclose(pts_stdv, grid_stdv, rtol=STDV_RTOL):
            failures.append(
                f"  {name}: Stdv mismatch — "
                f"grid={grid_stdv:.2f}, points={pts_stdv:.2f}, "
                f"rdiff={abs(pts_stdv - grid_stdv) / grid_stdv:.4f}"
            )

    if failures:
        msg = f"\n{version.value}: {len(failures)} failure(s):\n" + "\n".join(failures)
        pytest.fail(msg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", FAST_VERSIONS, ids=lambda v: v.value)
def test_grid_points_consistency_fast(version):
    """Grid/points consistency for models without coastal distance (~2-4 min)."""
    _check_consistency_for_version(version)


@pytest.mark.slow
@pytest.mark.parametrize("version", SLOW_VERSIONS, ids=lambda v: v.value)
def test_grid_points_consistency_slow(version):
    """Grid/points consistency for models with coastal distance (~12-15 min each)."""
    _check_consistency_for_version(version)

"""Tests for pipeline-level input validation."""

import numpy as np
import pandas as pd
import pytest

from conftest import load_fixed_model_config
from vs30 import config, constants, pipeline


def test_points_rejects_non_positive_observation_vs30(tmp_path):
    """Points mode checks observations the way grid mode does: an observed Vs30 of 0 is an error."""
    cfg = load_fixed_model_config(constants.FixedModelVersion.FOSTER_2019_APPROX)
    observations = pipeline.load_observations_csv(cfg["independent_observations_csv"])
    observations.loc[0, constants.ObservationColumn.VS30] = 0.0
    observations_csv = tmp_path / "observations.csv"
    observations.to_csv(observations_csv, index=False)

    with pytest.raises(ValueError, match="Vs30 must be positive"):
        pipeline.points_pipeline(
            longitudes=np.array([174.7762]),
            latitudes=np.array([-41.2865]),
            apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
            geology_corr_fn=cfg["geology_corr_fn"],
            terrain_corr_fn=cfg["terrain_corr_fn"],
            geology_categorical_csv=cfg["geology_categorical_csv"],
            terrain_categorical_csv=cfg["terrain_categorical_csv"],
            independent_observations_csv=observations_csv,
        )


def test_categorical_model_missing_a_column_names_it():
    """A categorical model without a mean column fails with a message naming the column."""
    with pytest.raises(ValueError, match=constants.COL_MEAN):
        pipeline.compute_categorical_vs30_updates(
            constants.ModelType.TERRAIN,
            categorical_model_df=pd.DataFrame(
                {constants.STANDARD_ID_COLUMN: [1], constants.COL_STDV: [0.5]}
            ),
            independent_observations_df=pd.DataFrame(
                columns=constants.ObservationColumn.REQUIRED
            ),
        )


def run_points_with_intermediates(
    version: constants.FixedModelVersion, longitude: float, latitude: float
) -> pd.DataFrame:
    """Run points_pipeline on one site with a bundled model's settings and intermediate columns."""
    cfg = load_fixed_model_config(version)
    return pipeline.points_pipeline(
        longitudes=np.array([longitude]),
        latitudes=np.array([latitude]),
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        geology_corr_fn=cfg["geology_corr_fn"],
        terrain_corr_fn=cfg["terrain_corr_fn"],
        geology_categorical_csv=cfg["geology_categorical_csv"],
        terrain_categorical_csv=cfg["terrain_categorical_csv"],
        independent_observations_csv=cfg["independent_observations_csv"],
        combination_method=constants.CombinationMethod(cfg["combination_method"]),
        combine_ratio=cfg["combine_ratio"],
        do_bayesian_update=cfg["do_bayesian_update"],
        apply_coastal_distance_mod=cfg["apply_coastal_distance_mod"],
        fill_gaps=cfg["fill_gaps"],
        include_intermediate=True,
    )


def test_water_site_gets_nan_not_the_placeholder_without_bayesian_update():
    """A site on water (Lake Taupō) gets NaN geology values, not the -32767 placeholder, when the categories aren't updated."""
    result = run_points_with_intermediates(
        constants.FixedModelVersion.FOSTER_2019_APPROX, 175.90, -38.80
    )

    assert result.loc[0, constants.COL_GEOLOGY_ID] == 0
    assert np.isnan(result.loc[0, constants.COL_GEOLOGY_VS30])
    assert np.isnan(result.loc[0, constants.COL_GEOLOGY_MVN_VS30])


def test_before_gap_fill_columns_are_present_when_no_site_needed_filling():
    """With gap-filling, the before-gap-fill columns are always written, so the columns don't depend on the input."""
    result = run_points_with_intermediates(
        constants.FixedModelVersion.JAEHWI_V1P0, 174.7762, -41.2865
    )

    assert result.loc[0, constants.COL_VS30_BEFORE_GAPFILL] == result.loc[
        0, constants.ObservationColumn.VS30
    ]


def test_no_coast_distance_raster_is_written_when_the_coastal_modification_is_off(
    tmp_path,
):
    """Intermediate output skips coast_distance.tif for models that don't use it, rather than writing zeros."""
    cfg = load_fixed_model_config(constants.FixedModelVersion.FOSTER_2019_APPROX)

    pipeline.grid_pipeline(
        grid_config=config.GridConfig(1748100, 1748400, 5427100, 5427400, 100, 100),
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        geology_corr_fn=cfg["geology_corr_fn"],
        terrain_corr_fn=cfg["terrain_corr_fn"],
        output_dir=tmp_path,
        geology_categorical_csv=cfg["geology_categorical_csv"],
        terrain_categorical_csv=cfg["terrain_categorical_csv"],
        independent_observations_csv=cfg["independent_observations_csv"],
        combination_method=constants.CombinationMethod(cfg["combination_method"]),
        combine_ratio=cfg["combine_ratio"],
        do_bayesian_update=cfg["do_bayesian_update"],
        include_intermediate=True,
        apply_coastal_distance_mod=cfg["apply_coastal_distance_mod"],
        show_progress=False,
    )

    assert (tmp_path / constants.SLOPE_RASTER_FILENAME).exists()
    assert not (tmp_path / constants.COAST_DISTANCE_RASTER_FILENAME).exists()

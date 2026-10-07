"""Tests for pipeline-level input validation."""

import numpy as np
import pandas as pd
import pytest

from conftest import load_fixed_model_config
from vs30 import constants, pipeline


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

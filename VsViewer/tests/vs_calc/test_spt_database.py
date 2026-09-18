"""Check interpretation of sparse soil logs without requiring the NZGD database."""

import numpy as np
import pandas as pd
import pytest

from vs_calc.scripts.validate_spt_database import InvalidRecord, complete_soil_profile


def test_gap_uses_nearest_interval_and_preserves_logged_soils():
    soils = pd.DataFrame(
        {
            "top_depth_m": [0.5, 3.0, 6.0],
            "bottom_depth_m": [1.0, 5.0, 7.0],
            "soil_type": ["SAND", "CLAY", "GRAVEL"],
        }
    )
    profile, assumptions = complete_soil_profile(soils, 8.0)
    np.testing.assert_allclose(profile["top_depth_m"], [0.0, 2.0, 5.5])
    np.testing.assert_allclose(profile["bottom_depth_m"], [2.0, 5.5, 8.0])
    assert profile["soil_type"].tolist() == ["SAND", "CLAY", "GRAVEL"]
    assert assumptions["interval_gaps_filled"] == 2
    assert assumptions["gap_depth_filled_m"] == 3.0
    assert assumptions["surface_extension_m"] == 0.5
    assert assumptions["bottom_extension_m"] == 1.0


def test_top_only_log_infers_bottoms_from_next_top():
    soils = pd.DataFrame(
        {
            "top_depth_m": [1.0, 3.0],
            "bottom_depth_m": [np.nan, np.nan],
            "soil_type": ["SAND", "CLAY"],
        }
    )
    profile, assumptions = complete_soil_profile(soils, 5.0)
    np.testing.assert_allclose(profile["top_depth_m"], [0.0, 3.0])
    np.testing.assert_allclose(profile["bottom_depth_m"], [3.0, 5.0])
    assert assumptions["layer_bottoms_inferred"] == 2


def test_conflicting_overlap_is_reported():
    soils = pd.DataFrame(
        {
            "top_depth_m": [0.0, 2.0],
            "bottom_depth_m": [3.0, 5.0],
            "soil_type": ["SAND", "CLAY"],
        }
    )
    with pytest.raises(InvalidRecord, match="overlapping"):
        complete_soil_profile(soils, 5.0)


def test_overlapping_intervals_of_same_soil_are_merged():
    soils = pd.DataFrame(
        {
            "top_depth_m": [0.0, 2.0, 6.0],
            "bottom_depth_m": [3.0, 5.0, 7.0],
            "soil_type": ["SAND", "SAND", "CLAY"],
        }
    )
    profile, assumptions = complete_soil_profile(soils, 7.0)
    np.testing.assert_allclose(profile["bottom_depth_m"], [5.5, 7.0])
    assert profile["soil_type"].tolist() == ["SAND", "CLAY"]
    assert assumptions["overlapping_same_soil_intervals_merged"] == 1

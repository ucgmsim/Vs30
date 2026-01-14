"""
Tests for the VS30 raster module.

Tests cover:
- _determine_vs30_columns function
- apply_hybrid_geology_modifications function
- Hybrid model calculations
"""

import numpy as np
import pytest

from vs30.raster import (
    _determine_vs30_columns,
    apply_hybrid_geology_modifications,
)


class TestDetermineVs30Columns:
    """Tests for the _determine_vs30_columns function."""

    def test_independent_observations_priority(self):
        """Test that independent observations posterior is preferred."""
        columns = [
            "id",
            "mean_vs30_km_per_s",
            "standard_deviation_vs30_km_per_s",
            "posterior_mean_vs30_km_per_s_independent_observations",
            "posterior_standard_deviation_vs30_km_per_s_independent_observations",
            "posterior_mean_vs30_km_per_s_clustered_observations",
            "posterior_standard_deviation_vs30_km_per_s_clustered_observations",
        ]

        mean_col, std_col = _determine_vs30_columns(columns)

        assert mean_col == "posterior_mean_vs30_km_per_s_independent_observations"
        assert std_col == "posterior_standard_deviation_vs30_km_per_s_independent_observations"

    def test_clustered_observations_second_priority(self):
        """Test that clustered observations posterior is second priority."""
        columns = [
            "id",
            "mean_vs30_km_per_s",
            "standard_deviation_vs30_km_per_s",
            "posterior_mean_vs30_km_per_s_clustered_observations",
            "posterior_standard_deviation_vs30_km_per_s_clustered_observations",
        ]

        mean_col, std_col = _determine_vs30_columns(columns)

        assert mean_col == "posterior_mean_vs30_km_per_s_clustered_observations"
        assert std_col == "posterior_standard_deviation_vs30_km_per_s_clustered_observations"

    def test_generic_posterior_third_priority(self):
        """Test that generic posterior is third priority."""
        columns = [
            "id",
            "mean_vs30_km_per_s",
            "standard_deviation_vs30_km_per_s",
            "posterior_mean_vs30_km_per_s",
            "posterior_standard_deviation_vs30_km_per_s",
        ]

        mean_col, std_col = _determine_vs30_columns(columns)

        assert mean_col == "posterior_mean_vs30_km_per_s"
        assert std_col == "posterior_standard_deviation_vs30_km_per_s"

    def test_prior_fourth_priority(self):
        """Test that prior columns are fourth priority."""
        columns = [
            "id",
            "mean_vs30_km_per_s",
            "standard_deviation_vs30_km_per_s",
            "prior_mean_vs30_km_per_s",
            "prior_standard_deviation_vs30_km_per_s",
        ]

        mean_col, std_col = _determine_vs30_columns(columns)

        assert mean_col == "prior_mean_vs30_km_per_s"
        assert std_col == "prior_standard_deviation_vs30_km_per_s"

    def test_standard_columns_fallback(self):
        """Test that standard columns are used as fallback."""
        columns = [
            "id",
            "mean_vs30_km_per_s",
            "standard_deviation_vs30_km_per_s",
        ]

        mean_col, std_col = _determine_vs30_columns(columns)

        assert mean_col == "mean_vs30_km_per_s"
        assert std_col == "standard_deviation_vs30_km_per_s"

    def test_missing_columns_raises_error(self):
        """Test that missing columns raises ValueError."""
        columns = ["id", "some_other_column"]

        with pytest.raises(ValueError, match="Could not find valid VS30"):
            _determine_vs30_columns(columns)

    def test_partial_pair_not_selected(self):
        """Test that having only mean or only std column doesn't match."""
        columns = [
            "id",
            "posterior_mean_vs30_km_per_s_independent_observations",
            # Missing posterior_standard_deviation_vs30_km_per_s_independent_observations
            "mean_vs30_km_per_s",
            "standard_deviation_vs30_km_per_s",
        ]

        mean_col, std_col = _determine_vs30_columns(columns)

        # Should fall back to standard columns since independent is incomplete
        assert mean_col == "mean_vs30_km_per_s"
        assert std_col == "standard_deviation_vs30_km_per_s"


class TestApplyHybridGeologyModifications:
    """Tests for the apply_hybrid_geology_modifications function."""

    @pytest.fixture
    def sample_arrays(self):
        """Create sample arrays for testing."""
        # 3x3 grid with different geology IDs
        id_array = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 10, 11],
        ], dtype=np.uint8)

        vs30_array = np.array([
            [300.0, 300.0, 300.0],
            [300.0, 300.0, 300.0],
            [300.0, 300.0, 300.0],
        ], dtype=np.float32)

        stdv_array = np.array([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ], dtype=np.float32)

        slope_array = np.array([
            [0.01, 0.05, 0.1],
            [0.2, 0.5, 1.0],
            [2.0, 5.0, 10.0],
        ], dtype=np.float32)

        coast_dist_array = np.array([
            [5000.0, 8000.0, 12000.0],
            [15000.0, 20000.0, 25000.0],
            [30000.0, 10000.0, 5000.0],
        ], dtype=np.float32)

        return id_array, vs30_array, stdv_array, slope_array, coast_dist_array

    def test_mod6_applies_to_gid4(self, sample_arrays):
        """Test that mod6 modification applies to geology ID 4."""
        id_array, vs30_array, stdv_array, slope_array, coast_dist_array = sample_arrays

        # Set specific ID for testing
        id_array[1, 0] = 4  # Alluvium

        result_vs30, result_stdv = apply_hybrid_geology_modifications(
            vs30_array.copy(),
            stdv_array.copy(),
            id_array,
            slope_array,
            coast_dist_array,
            mod6=True,
            mod13=False,
            hybrid=False,
            hybrid_mod6_dist_min=8000.0,
            hybrid_mod6_dist_max=20000.0,
            hybrid_mod6_vs30_min=240.0,
            hybrid_mod6_vs30_max=500.0,
        )

        # GID 4 pixel at (1,0) has coast_dist=15000
        # vs30 = 240 + (500-240) * (15000-8000) / (20000-8000)
        # vs30 = 240 + 260 * 7000/12000 = 240 + 151.67 = 391.67
        expected_vs30 = 240 + (500 - 240) * (15000 - 8000) / (20000 - 8000)
        assert np.isclose(result_vs30[1, 0], expected_vs30, rtol=0.01)

    def test_mod13_applies_to_gid10(self, sample_arrays):
        """Test that mod13 modification applies to geology ID 10."""
        id_array, vs30_array, stdv_array, slope_array, coast_dist_array = sample_arrays

        # GID 10 is at position (2, 1) with coast_dist=10000
        result_vs30, result_stdv = apply_hybrid_geology_modifications(
            vs30_array.copy(),
            stdv_array.copy(),
            id_array,
            slope_array,
            coast_dist_array,
            mod6=False,
            mod13=True,
            hybrid=False,
            hybrid_mod13_dist_min=8000.0,
            hybrid_mod13_dist_max=20000.0,
            hybrid_mod13_vs30_min=197.0,
            hybrid_mod13_vs30_max=500.0,
        )

        # GID 10 pixel at (2,1) has coast_dist=10000
        # vs30 = 197 + (500-197) * (10000-8000) / (20000-8000)
        # vs30 = 197 + 303 * 2000/12000 = 197 + 50.5 = 247.5
        expected_vs30 = 197 + (500 - 197) * (10000 - 8000) / (20000 - 8000)
        assert np.isclose(result_vs30[2, 1], expected_vs30, rtol=0.01)

    def test_mod6_clamps_at_minimum(self, sample_arrays):
        """Test that mod6 clamps vs30 at minimum value."""
        id_array, vs30_array, stdv_array, slope_array, coast_dist_array = sample_arrays

        id_array[0, 0] = 4  # Alluvium
        coast_dist_array[0, 0] = 1000.0  # Very close to coast

        result_vs30, _ = apply_hybrid_geology_modifications(
            vs30_array.copy(),
            stdv_array.copy(),
            id_array,
            slope_array,
            coast_dist_array,
            mod6=True,
            mod13=False,
            hybrid=False,
            hybrid_mod6_dist_min=8000.0,
            hybrid_mod6_dist_max=20000.0,
            hybrid_mod6_vs30_min=240.0,
            hybrid_mod6_vs30_max=500.0,
        )

        # Should clamp at minimum (240)
        assert result_vs30[0, 0] == 240.0

    def test_mod6_clamps_at_maximum(self, sample_arrays):
        """Test that mod6 clamps vs30 at maximum value."""
        id_array, vs30_array, stdv_array, slope_array, coast_dist_array = sample_arrays

        id_array[0, 0] = 4  # Alluvium
        coast_dist_array[0, 0] = 100000.0  # Very far inland

        result_vs30, _ = apply_hybrid_geology_modifications(
            vs30_array.copy(),
            stdv_array.copy(),
            id_array,
            slope_array,
            coast_dist_array,
            mod6=True,
            mod13=False,
            hybrid=False,
            hybrid_mod6_dist_min=8000.0,
            hybrid_mod6_dist_max=20000.0,
            hybrid_mod6_vs30_min=240.0,
            hybrid_mod6_vs30_max=500.0,
        )

        # Should clamp at maximum (500)
        assert result_vs30[0, 0] == 500.0

    def test_missing_params_raises_error(self, sample_arrays):
        """Test that missing required parameters raises ValueError."""
        id_array, vs30_array, stdv_array, slope_array, coast_dist_array = sample_arrays

        with pytest.raises(ValueError, match="Missing required parameters"):
            apply_hybrid_geology_modifications(
                vs30_array.copy(),
                stdv_array.copy(),
                id_array,
                slope_array,
                coast_dist_array,
                mod6=True,
                mod13=False,
                hybrid=False,
                # Missing required params for mod6
            )

    def test_no_modifications_returns_unchanged(self, sample_arrays):
        """Test that disabling all modifications returns unchanged arrays."""
        id_array, vs30_array, stdv_array, slope_array, coast_dist_array = sample_arrays

        original_vs30 = vs30_array.copy()
        original_stdv = stdv_array.copy()

        result_vs30, result_stdv = apply_hybrid_geology_modifications(
            vs30_array.copy(),
            stdv_array.copy(),
            id_array,
            slope_array,
            coast_dist_array,
            mod6=False,
            mod13=False,
            hybrid=False,
        )

        np.testing.assert_array_equal(result_vs30, original_vs30)
        np.testing.assert_array_equal(result_stdv, original_stdv)


class TestHybridSlopeCalculations:
    """Tests for hybrid slope-based Vs30 calculations."""

    def test_slope_interpolation_basic(self):
        """Test basic slope-based Vs30 interpolation logic."""
        # This tests the mathematical relationship without calling the full function

        # For GID 2 in config: slope_limits=[-1.85, -1.22], vs30_values=[242, 418]
        slope_limits = [-1.85, -1.22]
        vs30_values = [242, 418]
        vs30_limits_log10 = np.log10(vs30_values)

        # Test point at middle of slope range
        test_slope = 0.05  # Corresponds to log10(0.05) = -1.3
        log_slope = np.log10(test_slope)

        interpolated_log_vs30 = np.interp(log_slope, slope_limits, vs30_limits_log10)
        interpolated_vs30 = 10 ** interpolated_log_vs30

        # Value should be between the limits
        assert 242 < interpolated_vs30 < 418

    def test_slope_extrapolation_clamped(self):
        """Test that slope values outside range are extrapolated."""
        slope_limits = [-1.85, -1.22]
        vs30_values = [242, 418]
        vs30_limits_log10 = np.log10(vs30_values)

        # Very low slope (below range)
        very_low_slope = 0.001  # log10 = -3
        log_slope_low = np.log10(very_low_slope)
        interpolated_low = 10 ** np.interp(log_slope_low, slope_limits, vs30_limits_log10)

        # Very high slope (above range)
        very_high_slope = 1.0  # log10 = 0
        log_slope_high = np.log10(very_high_slope)
        interpolated_high = 10 ** np.interp(log_slope_high, slope_limits, vs30_limits_log10)

        # numpy interp extrapolates to boundary values
        assert np.isclose(interpolated_low, 242, rtol=0.01)
        assert np.isclose(interpolated_high, 418, rtol=0.01)

    def test_distance_interpolation_formula(self):
        """Test the distance-based interpolation formula."""
        # Formula: vs30_min + (vs30_max - vs30_min) * (dist - dist_min) / (dist_max - dist_min)
        dist_min = 8000.0
        dist_max = 20000.0
        vs30_min = 240.0
        vs30_max = 500.0

        # Test at various distances
        test_cases = [
            (8000.0, 240.0),   # At min distance -> min vs30
            (20000.0, 500.0),  # At max distance -> max vs30
            (14000.0, 370.0),  # Middle distance -> middle vs30
        ]

        for dist, expected_vs30 in test_cases:
            calculated = vs30_min + (vs30_max - vs30_min) * (dist - dist_min) / (dist_max - dist_min)
            calculated = np.clip(calculated, vs30_min, vs30_max)
            assert np.isclose(calculated, expected_vs30, rtol=0.01), \
                f"Failed for dist={dist}: got {calculated}, expected {expected_vs30}"

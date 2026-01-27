"""
Tests for the VS30 utils module.

Tests cover:
- combine_vs30_models: Model combination in log-space
- correlation_function: Exponential correlation calculation
"""

import numpy as np
import pytest

from vs30.utils import (
    combine_vs30_models,
    correlation_function,
)


class TestCorrelationFunction:
    """Tests for the correlation_function."""

    def test_zero_distance_high_correlation(self):
        """Test that near-zero distance gives high correlation."""
        distances = np.array([0.1])  # Very small distance
        phi = 1000

        corr = correlation_function(distances, phi)

        # Should be very close to 1
        assert corr[0] > 0.99

    def test_large_distance_low_correlation(self):
        """Test that large distance gives low correlation."""
        distances = np.array([10000])  # 10km
        phi = 1000  # 1km correlation length

        corr = correlation_function(distances, phi)

        # Should be close to exp(-10) ≈ 0.000045
        assert corr[0] < 0.01

    def test_correlation_at_phi(self):
        """Test correlation at distance = phi."""
        phi = 1000
        distances = np.array([phi])

        corr = correlation_function(distances, phi)

        # At distance = phi, correlation should be 1/e ≈ 0.368
        expected = 1 / np.exp(1)
        assert np.isclose(corr[0], expected, rtol=0.01)

    def test_correlation_decreases_with_distance(self):
        """Test that correlation decreases with distance."""
        distances = np.array([100, 500, 1000, 2000, 5000])
        phi = 1000

        corr = correlation_function(distances, phi)

        # Correlations should be monotonically decreasing
        for i in range(len(corr) - 1):
            assert corr[i] > corr[i + 1]

    def test_correlation_range(self):
        """Test that correlation is always between 0 and 1."""
        distances = np.random.rand(100) * 50000  # 0 to 50km
        phi = 1000

        corr = correlation_function(distances, phi)

        assert (corr >= 0).all()
        assert (corr <= 1).all()

    def test_matrix_input(self):
        """Test with 2D distance matrix input."""
        distances = np.array([
            [0, 1000, 2000],
            [1000, 0, 1000],
            [2000, 1000, 0],
        ])
        phi = 1000

        corr = correlation_function(distances, phi)

        assert corr.shape == (3, 3)
        # Diagonal should have highest correlation
        assert (np.diag(corr) > 0.99).all()

    def test_different_phi_values(self):
        """Test that larger phi gives higher correlation at same distance."""
        distance = np.array([1000])

        corr_small_phi = correlation_function(distance, phi=500)
        corr_large_phi = correlation_function(distance, phi=2000)

        # Larger phi means slower decay, so higher correlation at same distance
        assert corr_large_phi[0] > corr_small_phi[0]


class TestUtilsEdgeCases:
    """Additional edge case tests for utils module."""

    def test_correlation_at_very_large_distance(self):
        """Test correlation function at very large distances approaches zero."""
        distances = np.array([1000000.0])  # 1000 km
        phi = 1000.0  # 1 km correlation length

        corr = correlation_function(distances, phi)

        assert corr[0] < 0.001  # Should be essentially zero


class TestCombineVs30Models:
    """Tests for the combine_vs30_models function.

    This function combines geology and terrain Vs30 models using log-space
    weighted mixture, matching the algorithm used in the raster-based combine
    CLI command.
    """

    def test_equal_ratio_gives_geometric_mean(self):
        """Test that ratio=1.0 gives geometric mean, not arithmetic mean."""
        geol_vs30 = np.array([200.0])
        geol_stdv = np.array([0.3])
        terr_vs30 = np.array([400.0])
        terr_stdv = np.array([0.3])

        combined_vs30, combined_stdv = combine_vs30_models(
            geol_vs30, geol_stdv, terr_vs30, terr_stdv,
            combination_method=1.0,  # Equal weighting
        )

        # Geometric mean of 200 and 400 is sqrt(200*400) ≈ 282.84
        # Arithmetic mean would be 300
        expected_geometric = np.sqrt(200.0 * 400.0)
        assert np.isclose(combined_vs30[0], expected_geometric, rtol=0.01)
        assert combined_vs30[0] < 300.0  # Must be less than arithmetic mean

    def test_ratio_2_gives_more_weight_to_geology(self):
        """Test that ratio=2.0 gives geology twice the weight of terrain."""
        geol_vs30 = np.array([200.0])
        geol_stdv = np.array([0.3])
        terr_vs30 = np.array([400.0])
        terr_stdv = np.array([0.3])

        combined_vs30, combined_stdv = combine_vs30_models(
            geol_vs30, geol_stdv, terr_vs30, terr_stdv,
            combination_method=2.0,  # Geology has 2x weight
        )

        # With ratio=2, w_g = 2/3, w_t = 1/3
        # In log-space: log_comb = (2/3)*log(200) + (1/3)*log(400)
        # exp(log_comb) = 200^(2/3) * 400^(1/3) ≈ 251.98
        # Result should be closer to geology (200) than terrain (400)
        assert combined_vs30[0] < 300.0  # Below midpoint
        assert combined_vs30[0] > 200.0  # Above geology
        # More specifically, check against expected value
        expected = np.exp((2/3) * np.log(200.0) + (1/3) * np.log(400.0))
        assert np.isclose(combined_vs30[0], expected, rtol=0.01)

    def test_high_ratio_favors_geology(self):
        """Test that very high ratio strongly favors geology."""
        geol_vs30 = np.array([200.0])
        geol_stdv = np.array([0.3])
        terr_vs30 = np.array([400.0])
        terr_stdv = np.array([0.3])

        combined_vs30, _ = combine_vs30_models(
            geol_vs30, geol_stdv, terr_vs30, terr_stdv,
            combination_method=100.0,  # Very high geology weight
        )

        # With ratio=100, w_g ≈ 0.99, w_t ≈ 0.01
        # Result should be very close to geology
        assert np.isclose(combined_vs30[0], 200.0, rtol=0.05)

    def test_stdv_weighting_lower_stdv_gets_more_weight(self):
        """Test that stdv weighting gives more weight to model with lower stdv."""
        geol_vs30 = np.array([200.0])
        geol_stdv = np.array([0.1])  # Low uncertainty
        terr_vs30 = np.array([400.0])
        terr_stdv = np.array([0.5])  # High uncertainty

        combined_vs30, _ = combine_vs30_models(
            geol_vs30, geol_stdv, terr_vs30, terr_stdv,
            combination_method="standard_deviation_weighting",
            k_value=3.0,
        )

        # Geology has lower stdv, so it should get more weight
        # Result should be closer to geology (200)
        geometric_mean = np.sqrt(200.0 * 400.0)  # ≈ 282.84
        assert combined_vs30[0] < geometric_mean  # Closer to geology

    def test_stdv_weighting_equal_stdv_gives_equal_weight(self):
        """Test that equal stdv gives equal weight in stdv weighting mode."""
        geol_vs30 = np.array([200.0])
        geol_stdv = np.array([0.3])
        terr_vs30 = np.array([400.0])
        terr_stdv = np.array([0.3])  # Same stdv as geology

        combined_vs30, _ = combine_vs30_models(
            geol_vs30, geol_stdv, terr_vs30, terr_stdv,
            combination_method="standard_deviation_weighting",
            k_value=3.0,
        )

        # Equal stdv means equal weight → geometric mean
        expected_geometric = np.sqrt(200.0 * 400.0)
        assert np.isclose(combined_vs30[0], expected_geometric, rtol=0.01)

    def test_combined_stdv_formula(self):
        """Test that combined stdv uses mixture of log-normals formula."""
        geol_vs30 = np.array([200.0])
        geol_stdv = np.array([0.3])
        terr_vs30 = np.array([400.0])
        terr_stdv = np.array([0.4])

        combined_vs30, combined_stdv = combine_vs30_models(
            geol_vs30, geol_stdv, terr_vs30, terr_stdv,
            combination_method=1.0,  # Equal weighting
        )

        # Manually compute expected stdv
        # With ratio=1.0: w_g = w_t = 0.5
        log_g = np.log(200.0)
        log_t = np.log(400.0)
        log_comb = 0.5 * log_g + 0.5 * log_t
        expected_stdv = np.sqrt(
            0.5 * ((log_g - log_comb)**2 + 0.3**2)
            + 0.5 * ((log_t - log_comb)**2 + 0.4**2)
        )
        assert np.isclose(combined_stdv[0], expected_stdv, rtol=0.01)

    def test_array_input(self):
        """Test with array inputs containing multiple values."""
        geol_vs30 = np.array([200.0, 300.0, 400.0])
        geol_stdv = np.array([0.3, 0.2, 0.4])
        terr_vs30 = np.array([400.0, 300.0, 200.0])
        terr_stdv = np.array([0.3, 0.2, 0.4])

        combined_vs30, combined_stdv = combine_vs30_models(
            geol_vs30, geol_stdv, terr_vs30, terr_stdv,
            combination_method=1.0,
        )

        assert combined_vs30.shape == (3,)
        assert combined_stdv.shape == (3,)

        # Second element: both models equal, so combined = 300
        assert np.isclose(combined_vs30[1], 300.0, rtol=0.01)

    def test_invalid_combination_method_raises(self):
        """Test that invalid combination method raises ValueError."""
        geol_vs30 = np.array([200.0])
        geol_stdv = np.array([0.3])
        terr_vs30 = np.array([400.0])
        terr_stdv = np.array([0.3])

        with pytest.raises(ValueError, match="Unknown combination method"):
            combine_vs30_models(
                geol_vs30, geol_stdv, terr_vs30, terr_stdv,
                combination_method="invalid_method",
            )

    def test_k_value_effect(self):
        """Test that higher k_value gives more weight to lower stdv model."""
        geol_vs30 = np.array([200.0])
        geol_stdv = np.array([0.2])  # Lower stdv
        terr_vs30 = np.array([400.0])
        terr_stdv = np.array([0.4])

        # Low k_value: less sensitivity to stdv differences
        combined_low_k, _ = combine_vs30_models(
            geol_vs30, geol_stdv, terr_vs30, terr_stdv,
            combination_method="standard_deviation_weighting",
            k_value=1.0,
        )

        # High k_value: more sensitivity to stdv differences
        combined_high_k, _ = combine_vs30_models(
            geol_vs30, geol_stdv, terr_vs30, terr_stdv,
            combination_method="standard_deviation_weighting",
            k_value=5.0,
        )

        # With higher k, geology (lower stdv) gets even more weight
        # So combined_high_k should be closer to 200 than combined_low_k
        assert combined_high_k[0] < combined_low_k[0]

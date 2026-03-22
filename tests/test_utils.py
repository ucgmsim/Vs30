"""
Tests for the VS30 utils module.

Tests cover:
- combine_vs30_models: Model combination in log-space
"""

import numpy as np

from vs30 import constants, utils


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

        combined_vs30, combined_stdv = utils.combine_vs30_models(
            geol_vs30,
            geol_stdv,
            terr_vs30,
            terr_stdv,
            combination_method=constants.CombinationMethod.RATIO,
            combine_ratio=1.0,
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

        combined_vs30, combined_stdv = utils.combine_vs30_models(
            geol_vs30,
            geol_stdv,
            terr_vs30,
            terr_stdv,
            combination_method=constants.CombinationMethod.RATIO,
            combine_ratio=2.0,
        )

        # With ratio=2, w_g = 2/3, w_t = 1/3
        # In log-space: log_comb = (2/3)*log(200) + (1/3)*log(400)
        # exp(log_comb) = 200^(2/3) * 400^(1/3) ≈ 251.98
        # Result should be closer to geology (200) than terrain (400)
        assert combined_vs30[0] < 300.0  # Below midpoint
        assert combined_vs30[0] > 200.0  # Above geology
        # More specifically, check against expected value
        expected = np.exp((2 / 3) * np.log(200.0) + (1 / 3) * np.log(400.0))
        assert np.isclose(combined_vs30[0], expected, rtol=0.01)

    def test_stdv_weighting_lower_stdv_gets_more_weight(self):
        """Test that stdv weighting gives more weight to model with lower stdv."""
        geol_vs30 = np.array([200.0])
        geol_stdv = np.array([0.1])  # Low uncertainty
        terr_vs30 = np.array([400.0])
        terr_stdv = np.array([0.5])  # High uncertainty

        combined_vs30, _ = utils.combine_vs30_models(
            geol_vs30,
            geol_stdv,
            terr_vs30,
            terr_stdv,
            combination_method=constants.CombinationMethod.STANDARD_DEVIATION_WEIGHTING,
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

        combined_vs30, _ = utils.combine_vs30_models(
            geol_vs30,
            geol_stdv,
            terr_vs30,
            terr_stdv,
            combination_method=constants.CombinationMethod.STANDARD_DEVIATION_WEIGHTING,
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

        combined_vs30, combined_stdv = utils.combine_vs30_models(
            geol_vs30,
            geol_stdv,
            terr_vs30,
            terr_stdv,
            combination_method=constants.CombinationMethod.RATIO,
            combine_ratio=1.0,
        )

        # Manually compute expected stdv
        # With ratio=1.0: w_g = w_t = 0.5
        log_g = np.log(200.0)
        log_t = np.log(400.0)
        log_comb = 0.5 * log_g + 0.5 * log_t
        expected_stdv = np.sqrt(
            0.5 * ((log_g - log_comb) ** 2 + 0.3**2)
            + 0.5 * ((log_t - log_comb) ** 2 + 0.4**2)
        )
        assert np.isclose(combined_stdv[0], expected_stdv, rtol=0.01)


class TestExponentialCorrelationFunction:
    """Tests for the exponential correlation function."""

    def test_zero_distance_returns_near_one(self):
        """Correlation at zero distance ≈ 1.0 (limited by MIN_DIST_ENFORCED)."""
        distances = np.array([0.0])
        result = utils.exponential_correlation_function(distances, phi=1407)
        assert result[0] > 0.999

    def test_correlation_decays_with_distance(self):
        """Correlation decays as distance increases."""
        distances = np.array([0.0, 100.0, 500.0, 1407.0, 5000.0])
        result = utils.exponential_correlation_function(distances, phi=1407)
        assert np.all(np.diff(result) < 0)  # Monotonically decreasing

    def test_at_phi_correlation_is_1_over_e(self):
        """At distance=phi, correlation ≈ 1/e ≈ 0.368."""
        distances = np.array([1407.0])
        result = utils.exponential_correlation_function(distances, phi=1407)
        assert np.isclose(result[0], np.exp(-1), rtol=0.01)

    def test_practical_range_three_phi(self):
        """At 3*phi, correlation ≈ 0.05 (5% practical range)."""
        phi = 1407
        distances = np.array([3 * phi])
        result = utils.exponential_correlation_function(distances, phi=phi)
        assert np.isclose(result[0], np.exp(-3), rtol=0.01)


class TestMaternCorrelationFunction:
    """Tests for the Matérn correlation function.

    Uses the original Foster (2019) parameters: range=20000, sill=0.15,
    nugget=0.05, kappa=0.9.
    """

    def test_near_zero_distance_returns_sill_over_total(self):
        """At d≈0, correlation ≈ sill/(sill+nugget) = 0.15/0.20 = 0.75."""
        distances = np.array([0.0])
        result = utils.matern_correlation_function(
            distances, range_m=20000, sill=0.15, nugget=0.05, kappa=0.9,
        )
        assert np.isclose(result[0], 0.75, atol=0.02)

    def test_correlation_decays_with_distance(self):
        """Correlation decays as distance increases."""
        distances = np.array([100.0, 1000.0, 5000.0, 20000.0, 50000.0])
        result = utils.matern_correlation_function(
            distances, range_m=20000, sill=0.15, nugget=0.05, kappa=0.9,
        )
        assert np.all(np.diff(result) < 0)

    def test_kappa_half_matches_exponential_shape(self):
        """Matérn with kappa=0.5 and no nugget should match exponential shape.

        This is a mathematical identity: Matérn(κ=0.5) ∝ exp(-d/range).
        With nugget=0, the correlation at d should equal exp(-d/range).
        """
        distances = np.array([100.0, 500.0, 1000.0, 5000.0])
        range_m = 993.0
        result = utils.matern_correlation_function(
            distances, range_m=range_m, sill=1.0, nugget=0.0, kappa=0.5,
        )
        expected = np.exp(-distances / range_m)
        np.testing.assert_allclose(result, expected, rtol=0.05)

    def test_large_distance_approaches_zero(self):
        """At very large distances, correlation → 0."""
        distances = np.array([200000.0])
        result = utils.matern_correlation_function(
            distances, range_m=20000, sill=0.15, nugget=0.05, kappa=0.9,
        )
        assert result[0] < 0.01

    def test_returns_correct_shape(self):
        """Output shape matches input shape."""
        distances = np.array([[100, 200], [300, 400]], dtype=float)
        result = utils.matern_correlation_function(
            distances, range_m=20000, sill=0.15, nugget=0.05, kappa=0.9,
        )
        assert result.shape == (2, 2)

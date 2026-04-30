"""
Tests for the VS30 utils module.

Tests cover:
- combine_vs30_models: Model combination in log-space
"""

import numpy as np
import pytest

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
        assert combined_vs30[0] == pytest.approx(expected_geometric, rel=0.01)
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
        assert combined_vs30[0] == pytest.approx(expected, rel=0.01)

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
        assert combined_vs30[0] == pytest.approx(expected_geometric, rel=0.01)

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
        assert combined_stdv[0] == pytest.approx(expected_stdv, rel=0.01)


class TestMaternCorrelationFunction:
    """Tests for the Matérn correlation function."""

    def test_kappa_half_matches_exponential(self):
        """Matérn with kappa=0.5 and no nugget should match exponential.

        This is a mathematical identity: Matérn(κ=0.5) = exp(-d/range).
        Verifies our implementation against the known analytical result.
        """
        distances = np.array([100.0, 500.0, 1000.0, 5000.0])
        range_m = 993.0
        result = utils.matern_correlation_function(
            distances,
            range_m=range_m,
            sill=1.0,
            nugget=0.0,
            kappa=0.5,
        )
        expected = np.exp(-distances / range_m)
        np.testing.assert_allclose(result, expected, rtol=0.05)

    def test_matches_r_gstat_variogram_line(self):
        """Matérn correlation must match R gstat variogramLine output.

        Reference values generated with R 4.5.2 and gstat using:
            vgm(psill=0.15, model="Mat", range=20e3, nugget=0.05, kappa=0.9)
        Then divided by psill to obtain the correlation values used in the
        Worden et al. MVN formulation (see vs30/utils.py docstring).
        """
        distances = np.array(
            [0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0, 20000.0, 50000.0, 100000.0]
        )
        r_gstat_reference = np.array(
            [
                1.0,
                1.0,
                0.9999974,
                0.9998576,
                0.9933186,
                0.8000460,
                0.5647182,
                0.1637082,
                0.01697042,
            ]
        )
        result = utils.matern_correlation_function(
            distances,
            range_m=20000.0,
            sill=0.15,
            nugget=0.05,
            kappa=0.9,
        )
        np.testing.assert_allclose(result, r_gstat_reference, atol=1e-6)

    def test_zero_distance_gives_unit_correlation(self):
        """At zero distance, correlation must be ≈1 regardless of nugget.

        R gstat treats the nugget as contributing to per-point variance, not
        to the correlation function itself (Worden et al. Eq. 7). The
        correlation-at-zero must therefore be 1, not sill/(sill+nugget).
        """
        result = utils.matern_correlation_function(
            np.array([0.0]),
            range_m=20000.0,
            sill=0.15,
            nugget=0.05,
            kappa=0.9,
        )
        assert result[0] == pytest.approx(1.0, abs=1e-6)



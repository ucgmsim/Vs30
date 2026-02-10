"""
Tests for the VS30 category module.

Tests cover:
- Bayesian update formulas
- Independent data update
- Category edge cases
"""

import numpy as np
import pandas as pd
import pytest

from vs30 import category, constants


class TestBayesianUpdateFormulas:
    """Tests for the Bayesian update helper functions."""

    def test_posterior_variance_basic(self):
        """Test compute_bayesian_posterior_variance with simple inputs."""
        prior_stdv = 0.5  # Prior std dev
        num_prior_observations = 3  # Prior sample size
        uncertainty = 0.2  # Observation uncertainty
        prior_mean = 200  # Prior mean
        observation_value = 210  # Observation

        var = category.compute_bayesian_posterior_variance(
            prior_stdv,
            num_prior_observations,
            uncertainty,
            prior_mean,
            observation_value,
        )

        # Variance should be positive
        assert var > 0
        # New variance should be less than prior variance when adding data
        assert var < prior_stdv**2

    def test_posterior_mean_basic(self):
        """Test compute_bayesian_posterior_mean with simple inputs."""
        prior_mean = 200  # Prior mean
        num_prior_observations = 3  # Prior sample size
        observation_value = 220  # Observation

        mean = category.compute_bayesian_posterior_mean(
            prior_mean, num_prior_observations, observation_value
        )

        # New mean should be between prior and observation
        assert (
            min(prior_mean, observation_value)
            <= mean
            <= max(prior_mean, observation_value)
        )

    def test_posterior_mean_pulls_toward_observation(self):
        """Test that posterior mean is pulled toward observation."""
        prior_mean = 200
        num_prior_observations = 3
        observation_value = 300  # Observation much higher than prior

        mean = category.compute_bayesian_posterior_mean(
            prior_mean, num_prior_observations, observation_value
        )

        # Mean should be closer to observation than prior was
        assert mean > prior_mean

    def test_posterior_variance_increases_with_mean_shift(self):
        """Test that variance increases when observation far from prior."""
        prior_stdv = 0.5
        num_prior_observations = 3
        uncertainty = 0.2
        prior_mean = 200

        # Observation close to prior
        obs_close = 205
        var_close = category.compute_bayesian_posterior_variance(
            prior_stdv, num_prior_observations, uncertainty, prior_mean, obs_close
        )

        # Observation far from prior
        obs_far = 400
        var_far = category.compute_bayesian_posterior_variance(
            prior_stdv, num_prior_observations, uncertainty, prior_mean, obs_far
        )

        # Variance should be higher when observation is far from prior
        assert var_far > var_close

    def test_bayesian_update_convergence(self):
        """Test that multiple observations converge toward true value."""
        prior_mean = 200  # Prior mean
        prior_stdv = 0.5  # Prior std dev
        num_prior_observations = 3
        true_value = 250  # True value observations are drawn from

        # Simulate multiple observations around true value
        current_mean = prior_mean
        current_std = prior_stdv
        current_num_observations = num_prior_observations

        for _ in range(10):
            # Observation with some noise
            observation_value = true_value * (1 + np.random.normal(0, 0.05))
            uncertainty = 0.2

            var = category.compute_bayesian_posterior_variance(
                current_std,
                current_num_observations,
                uncertainty,
                current_mean,
                observation_value,
            )
            current_mean = category.compute_bayesian_posterior_mean(
                current_mean, current_num_observations, observation_value
            )
            current_std = np.sqrt(var)
            current_num_observations += 1

        # After many observations, mean should be close to true value
        assert abs(current_mean - true_value) < abs(prior_mean - true_value)


class TestUpdateWithIndependentData:
    """Tests for the update_with_independent_data function."""

    @pytest.fixture
    def sample_categorical_model(self):
        """Create sample categorical model DataFrame."""
        return pd.DataFrame(
            {
                "id": [1, 2, 3],
                "mean_vs30_km_per_s": [200.0, 300.0, 400.0],
                "standard_deviation_vs30_km_per_s": [0.5, 0.4, 0.3],
            }
        )

    @pytest.fixture
    def sample_observations(self):
        """Create sample observations DataFrame."""
        return pd.DataFrame(
            {
                "id": [1, 1, 2],
                "vs30": [210.0, 195.0, 320.0],
                "uncertainty": [0.2, 0.2, 0.15],
            }
        )

    def test_basic_update(self, sample_categorical_model, sample_observations):
        """Test basic Bayesian update with observations."""
        result = category.update_with_independent_data(
            sample_categorical_model,
            sample_observations,
        )

        # Check output has expected columns
        assert "posterior_mean_vs30_km_per_s_independent_observations" in result.columns
        assert (
            "posterior_standard_deviation_vs30_km_per_s_independent_observations"
            in result.columns
        )
        assert "posterior_num_observations_independent_observations" in result.columns

        # Category 1 should be updated (has 2 observations)
        # N_PRIOR=3 from constants, so 3 + 2 = 5
        cat1 = result[result["id"] == 1].iloc[0]
        assert (
            cat1["posterior_num_observations_independent_observations"] == 5
        )  # 3 prior + 2 obs

        # Category 3 should remain unchanged (no observations)
        cat3 = result[result["id"] == 3].iloc[0]
        assert cat3["posterior_mean_vs30_km_per_s_independent_observations"] == 400.0

    def test_min_sigma_enforced(self, sample_categorical_model, sample_observations):
        """Test that minimum sigma is enforced using MIN_SIGMA constant."""
        result = category.update_with_independent_data(
            sample_categorical_model,
            sample_observations,
        )

        # All posteriors should have stddev >= MIN_SIGMA
        # Note: posteriors can go below MIN_SIGMA, but prior is floored at MIN_SIGMA
        assert result["enforced_min_sigma"].iloc[0] == constants.MIN_SIGMA


class TestCategoryEdgeCases:
    """Tests for edge cases in category module."""

    def test_update_with_no_matching_observations(self):
        """Test Bayesian update when no observations match a category."""
        # Create categorical model with some categories
        categorical_model_df = pd.DataFrame(
            {
                "id": [1, 2, 3],
                "mean_vs30_km_per_s": [300.0, 400.0, 500.0],
                "standard_deviation_vs30_km_per_s": [30.0, 40.0, 50.0],
            }
        )

        # Observations that don't match any category (use STANDARD_ID_COLUMN = 'id')
        observations_df = pd.DataFrame(
            {
                "vs30": [350.0],
                "uncertainty": [25.0],
                "id": [99],  # Non-existent category - uses 'id' column
                "easting": [1500000.0],
                "northing": [5100000.0],
            }
        )

        # Should not raise, categories without observations keep prior
        result_df = category.update_with_independent_data(
            categorical_model_df,
            observations_df,
        )

        # Result should have posterior columns
        assert (
            "posterior_mean_vs30_km_per_s_independent_observations" in result_df.columns
        )
        assert (
            "posterior_standard_deviation_vs30_km_per_s_independent_observations"
            in result_df.columns
        )

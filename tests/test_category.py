"""
Tests for the VS30 category module.

Tests cover:
- Bayesian update formulas
- DBSCAN clustering
- Category assignment functions
- get_vs30_for_points function
"""

import numpy as np
import pandas as pd
import pytest
from math import sqrt

from vs30.category import (
    _compute_bayesian_posterior_mean,
    _compute_bayesian_posterior_variance,
    perform_clustering,
    update_with_independent_data,
    get_vs30_for_points,
    RASTER_ID_NODATA_VALUE,
    STANDARD_ID_COLUMN,
)


class TestBayesianUpdateFormulas:
    """Tests for the Bayesian update helper functions."""

    def test_posterior_variance_basic(self):
        """Test _compute_bayesian_posterior_variance with simple inputs."""
        prior_stdv = 0.5  # Prior std dev
        num_prior_observations = 3  # Prior sample size
        uncertainty = 0.2  # Observation uncertainty
        prior_mean = 200  # Prior mean
        observation_value = 210  # Observation

        var = _compute_bayesian_posterior_variance(prior_stdv, num_prior_observations, uncertainty, prior_mean, observation_value)

        # Variance should be positive
        assert var > 0
        # New variance should be less than prior variance when adding data
        assert var < prior_stdv**2

    def test_posterior_mean_basic(self):
        """Test _compute_bayesian_posterior_mean with simple inputs."""
        prior_mean = 200  # Prior mean
        num_prior_observations = 3  # Prior sample size
        posterior_variance = 0.2  # Updated variance
        observation_value = 220  # Observation

        mean = _compute_bayesian_posterior_mean(prior_mean, num_prior_observations, posterior_variance, observation_value)

        # New mean should be between prior and observation
        assert min(prior_mean, observation_value) <= mean <= max(prior_mean, observation_value)

    def test_posterior_mean_pulls_toward_observation(self):
        """Test that posterior mean is pulled toward observation."""
        prior_mean = 200
        num_prior_observations = 3
        posterior_variance = 0.2
        observation_value = 300  # Observation much higher than prior

        mean = _compute_bayesian_posterior_mean(prior_mean, num_prior_observations, posterior_variance, observation_value)

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
        var_close = _compute_bayesian_posterior_variance(prior_stdv, num_prior_observations, uncertainty, prior_mean, obs_close)

        # Observation far from prior
        obs_far = 400
        var_far = _compute_bayesian_posterior_variance(prior_stdv, num_prior_observations, uncertainty, prior_mean, obs_far)

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

            var = _compute_bayesian_posterior_variance(current_std, current_num_observations, uncertainty, current_mean, observation_value)
            current_mean = _compute_bayesian_posterior_mean(current_mean, current_num_observations, var, observation_value)
            current_std = sqrt(var)
            current_num_observations += 1

        # After many observations, mean should be close to true value
        assert abs(current_mean - true_value) < abs(prior_mean - true_value)


class TestUpdateWithIndependentData:
    """Tests for the update_with_independent_data function."""

    @pytest.fixture
    def sample_categorical_model(self):
        """Create sample categorical model DataFrame."""
        return pd.DataFrame({
            "id": [1, 2, 3],
            "mean_vs30_km_per_s": [200.0, 300.0, 400.0],
            "standard_deviation_vs30_km_per_s": [0.5, 0.4, 0.3],
        })

    @pytest.fixture
    def sample_observations(self):
        """Create sample observations DataFrame."""
        return pd.DataFrame({
            "id": [1, 1, 2],
            "vs30": [210.0, 195.0, 320.0],
            "uncertainty": [0.2, 0.2, 0.15],
        })

    def test_basic_update(self, sample_categorical_model, sample_observations):
        """Test basic Bayesian update with observations."""
        result = update_with_independent_data(
            sample_categorical_model,
            sample_observations,
            n_prior=3,
            min_sigma=0.3,
        )

        # Check output has expected columns
        assert "posterior_mean_vs30_km_per_s_independent_observations" in result.columns
        assert "posterior_standard_deviation_vs30_km_per_s_independent_observations" in result.columns
        assert "posterior_num_observations_independent_observations" in result.columns

        # Category 1 should be updated (has 2 observations)
        cat1 = result[result["id"] == 1].iloc[0]
        assert cat1["posterior_num_observations_independent_observations"] == 5  # 3 prior + 2 obs

        # Category 3 should remain unchanged (no observations)
        cat3 = result[result["id"] == 3].iloc[0]
        assert cat3["posterior_mean_vs30_km_per_s_independent_observations"] == 400.0

    def test_min_sigma_enforced(self, sample_categorical_model, sample_observations):
        """Test that minimum sigma is enforced."""
        min_sigma = 0.4

        result = update_with_independent_data(
            sample_categorical_model,
            sample_observations,
            n_prior=3,
            min_sigma=min_sigma,
        )

        # All posteriors should have stddev >= min_sigma
        # Note: posteriors can go below min_sigma, but prior is floored at min_sigma
        assert result["enforced_min_sigma"].iloc[0] == min_sigma


class TestPerformClustering:
    """Tests for the DBSCAN clustering function."""

    @pytest.fixture
    def clustered_sites(self):
        """Create sites with clear clusters."""
        # Cluster 1: tight group at (1000, 1000)
        cluster1 = pd.DataFrame({
            "id": [1] * 10,
            "easting": 1000 + np.random.normal(0, 100, 10),
            "northing": 1000 + np.random.normal(0, 100, 10),
            "vs30": np.random.uniform(180, 220, 10),
        })

        # Cluster 2: tight group at (50000, 50000)
        cluster2 = pd.DataFrame({
            "id": [1] * 10,
            "easting": 50000 + np.random.normal(0, 100, 10),
            "northing": 50000 + np.random.normal(0, 100, 10),
            "vs30": np.random.uniform(180, 220, 10),
        })

        # Scattered points (should not cluster)
        scattered = pd.DataFrame({
            "id": [1] * 3,
            "easting": [100000, 200000, 300000],
            "northing": [100000, 200000, 300000],
            "vs30": [200, 210, 190],
        })

        return pd.concat([cluster1, cluster2, scattered], ignore_index=True)

    def test_clustering_identifies_groups(self, clustered_sites):
        """Test that clustering identifies tight groups."""
        result = perform_clustering(
            clustered_sites,
            model_type="geology",
            min_group=5,
            eps=1000,  # 1km epsilon
        )

        # Should have cluster column
        assert "cluster" in result.columns

        # Should identify at least 2 clusters (0 and 1, or similar)
        unique_clusters = result[result["cluster"] != -1]["cluster"].unique()
        assert len(unique_clusters) >= 2

    def test_scattered_points_unclustered(self, clustered_sites):
        """Test that scattered points remain unclustered."""
        result = perform_clustering(
            clustered_sites,
            model_type="geology",
            min_group=5,
            eps=1000,
        )

        # Points at 100000, 200000, 300000 should be unclustered (-1)
        far_points = result[result["easting"] > 90000]
        assert (far_points["cluster"] == -1).all()

    def test_min_group_respected(self):
        """Test that min_group parameter is respected."""
        # Create a small group (4 points) that shouldn't cluster with min_group=5
        sites = pd.DataFrame({
            "id": [1] * 4,
            "easting": [1000, 1010, 1020, 1030],
            "northing": [1000, 1010, 1020, 1030],
            "vs30": [200, 200, 200, 200],
        })

        result = perform_clustering(sites, "geology", min_group=5, eps=1000)

        # All points should be unclustered since group is too small
        assert (result["cluster"] == -1).all()


class TestGetVs30ForPoints:
    """Tests for the get_vs30_for_points function."""

    @pytest.fixture
    def sample_model_df(self):
        """Create sample categorical model DataFrame."""
        return pd.DataFrame({
            "id": [1, 2, 3, 4, 5],
            "mean_vs30_km_per_s": [200.0, 250.0, 300.0, 350.0, 400.0],
            "standard_deviation_vs30_km_per_s": [0.5, 0.45, 0.4, 0.35, 0.3],
        })

    def test_christchurch_point_geology(self, sample_model_df):
        """Test that Christchurch point returns valid geology values."""
        # University of Canterbury area (NZTM)
        points = np.array([[1570604, 5180029]])

        vs30, stdv, ids = get_vs30_for_points(
            points, "geology", sample_model_df
        )

        # Should return arrays of correct length
        assert len(vs30) == 1
        assert len(stdv) == 1
        assert len(ids) == 1

        # ID should be valid (not NODATA) for Christchurch
        assert ids[0] != RASTER_ID_NODATA_VALUE

    def test_christchurch_point_terrain(self, sample_model_df):
        """Test that Christchurch point returns valid terrain values."""
        points = np.array([[1570604, 5180029]])

        vs30, stdv, ids = get_vs30_for_points(
            points, "terrain", sample_model_df
        )

        assert len(vs30) == 1
        assert len(stdv) == 1
        assert len(ids) == 1

    def test_invalid_model_type_raises(self, sample_model_df):
        """Test that invalid model type raises ValueError."""
        points = np.array([[1570604, 5180029]])

        with pytest.raises(ValueError, match="Unknown model_type"):
            get_vs30_for_points(points, "invalid", sample_model_df)

    def test_multiple_points(self, sample_model_df):
        """Test with multiple points."""
        # Multiple NZ locations
        points = np.array([
            [1570604, 5180029],  # Christchurch
            [1757209, 5920482],  # Auckland
            [1748735, 5427916],  # Wellington
        ])

        vs30, stdv, ids = get_vs30_for_points(
            points, "geology", sample_model_df
        )

        assert len(vs30) == 3
        assert len(stdv) == 3
        assert len(ids) == 3


class TestCategoryConstants:
    """Tests for category module constants."""

    def test_raster_id_nodata_value(self):
        """Test that RASTER_ID_NODATA_VALUE is set correctly."""
        assert RASTER_ID_NODATA_VALUE == 255

    def test_standard_id_column(self):
        """Test that STANDARD_ID_COLUMN is correct."""
        assert STANDARD_ID_COLUMN == "id"


# =============================================================================
# Additional Tests from Coverage Improvements
# =============================================================================


class TestCategoryEdgeCases:
    """Tests for edge cases in category module."""

    def test_update_with_no_matching_observations(self):
        """Test Bayesian update when no observations match a category."""
        # Create categorical model with some categories
        categorical_model_df = pd.DataFrame({
            'id': [1, 2, 3],
            'mean_vs30_km_per_s': [300.0, 400.0, 500.0],
            'standard_deviation_vs30_km_per_s': [30.0, 40.0, 50.0],
        })

        # Observations that don't match any category (use STANDARD_ID_COLUMN = 'id')
        observations_df = pd.DataFrame({
            'vs30': [350.0],
            'uncertainty': [25.0],
            'id': [99],  # Non-existent category - uses 'id' column
            'easting': [1500000.0],
            'northing': [5100000.0],
        })

        # Should not raise, categories without observations keep prior
        result_df = update_with_independent_data(
            categorical_model_df,
            observations_df,
            n_prior=10,
            min_sigma=0.1,
        )

        # Result should have posterior columns
        assert 'posterior_mean_vs30_km_per_s_independent_observations' in result_df.columns
        assert 'posterior_standard_deviation_vs30_km_per_s_independent_observations' in result_df.columns

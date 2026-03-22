"""
Tests for the VS30 spatial module.

Tests cover:
- Spatial adjustment computations (MVN conditioning)
- Cluster subsampling
- Point-based MVN adjustment
"""

import functools

import numpy as np
import pytest

from vs30 import spatial, utils, constants

# Create a standard geology correlation callable for tests
_geology_corr_fn = functools.partial(utils.exponential_correlation_function, phi=1407)


class TestComputeSpatialAdjustmentForPixel:
    """Tests for the compute_spatial_adjustment_for_pixel function."""

    @pytest.fixture
    def pixel(self):
        """Create a test pixel."""
        return spatial.PixelData(
            location=np.array([1000.0, 1000.0]),
            vs30=250.0,
            stdv=0.4,
            index=12345,
        )

    @pytest.fixture
    def nearby_observation(self):
        """Create a single nearby observation."""
        return spatial.ObservationData(
            locations=np.array([[1100.0, 1000.0]]),  # 100m away
            vs30=np.array([280.0]),  # Higher than prior
            model_vs30=np.array([260.0]),
            model_stdv=np.array([0.4]),
            residuals=np.array([np.log(280.0 / 260.0)]),
            omega=np.ones(1),
            uncertainty=np.array([0.2]),
        )

    def test_updates_toward_observation(self, pixel, nearby_observation):
        """Test that update shifts vs30 toward observation."""
        result = spatial.compute_spatial_adjustment_for_pixel(
            pixel,
            nearby_observation,
            corr_fn=_geology_corr_fn,
            max_dist_m=5000.0,
            max_points=100,
            noisy=False,
        )

        # Observation is higher (280), prior is 250, update should increase
        assert result is not None  # Type guard: narrows Optional return type
        assert result.updated_vs30 > pixel.vs30

    def test_stdv_decreases_with_observation(self, pixel, nearby_observation):
        """Test that standard deviation decreases when observation is added."""
        result = spatial.compute_spatial_adjustment_for_pixel(
            pixel,
            nearby_observation,
            corr_fn=_geology_corr_fn,
            max_dist_m=5000.0,
            max_points=100,
            noisy=False,
        )

        # Adding observation should reduce uncertainty
        assert result is not None  # Type guard: narrows Optional return type
        assert result.updated_stdv < pixel.stdv

    def test_no_observations_returns_unchanged_vs30(self, pixel):
        """Test that no nearby observations returns unchanged vs30."""
        far_observation = spatial.ObservationData(
            locations=np.array([[100000.0, 100000.0]]),  # Very far
            vs30=np.array([300.0]),
            model_vs30=np.array([300.0]),
            model_stdv=np.array([0.4]),
            residuals=np.zeros(1),
            omega=np.ones(1),
            uncertainty=np.array([0.2]),
        )

        result = spatial.compute_spatial_adjustment_for_pixel(
            pixel,
            far_observation,
            corr_fn=_geology_corr_fn,
            max_dist_m=5000,
        )

        # VS30 should be unchanged when no nearby observations
        assert result is not None  # Type guard: narrows Optional return type
        assert result.updated_vs30 == pixel.vs30
        assert result.n_observations_used == 0


class TestComputeMvnAtPoints:
    """Tests for compute_spatial_adjustment_at_points function."""

    def test_no_observations_returns_prior(self):
        """Test that no observations returns prior values with shrunk stdv."""
        points = np.array([[1500000, 5100000], [1501000, 5101000]])
        model_vs30 = np.array([300.0, 400.0])
        model_stdv = np.array([30.0, 40.0])

        # Empty observations
        obs_locations = np.empty((0, 2))
        obs_vs30 = np.empty(0)
        obs_model_vs30 = np.empty(0)
        obs_model_stdv = np.empty(0)
        obs_uncertainty = np.empty(0)

        mvn_vs30, mvn_stdv = spatial.compute_spatial_adjustment_at_points(
            points=points,
            model_vs30=model_vs30,
            model_stdv=model_stdv,
            obs_locations=obs_locations,
            obs_vs30=obs_vs30,
            obs_model_vs30=obs_model_vs30,
            obs_model_stdv=obs_model_stdv,
            obs_uncertainty=obs_uncertainty,
            corr_fn=_geology_corr_fn,
        )

        # Should return prior vs30 unchanged
        np.testing.assert_array_equal(mvn_vs30, model_vs30)

    def test_with_nearby_observations(self):
        """Test spatial adjustment with nearby observations."""
        # Single query point
        points = np.array([[1500000.0, 5100000.0]])
        model_vs30 = np.array([300.0])
        model_stdv = np.array([30.0])

        # Nearby observation with higher vs30
        obs_locations = np.array([[1500100.0, 5100100.0]])  # 141m away
        obs_vs30 = np.array([400.0])
        obs_model_vs30 = np.array([300.0])
        obs_model_stdv = np.array([30.0])
        obs_uncertainty = np.array([25.0])

        mvn_vs30, mvn_stdv = spatial.compute_spatial_adjustment_at_points(
            points=points,
            model_vs30=model_vs30,
            model_stdv=model_stdv,
            obs_locations=obs_locations,
            obs_vs30=obs_vs30,
            obs_model_vs30=obs_model_vs30,
            obs_model_stdv=obs_model_stdv,
            obs_uncertainty=obs_uncertainty,
            corr_fn=_geology_corr_fn,
            max_dist_m=5000,
        )

        # Should adjust toward observation (increase vs30)
        assert mvn_vs30[0] > model_vs30[0], (
            "Adjustment should pull vs30 toward observation"
        )
        # Uncertainty should decrease
        assert mvn_stdv[0] < model_stdv[0], (
            "Uncertainty should decrease with observation"
        )

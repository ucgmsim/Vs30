"""
Tests for the VS30 spatial module.

Tests cover:
- ObservationData dataclass
- PixelData dataclass
- MVNUpdateResult dataclass
- RasterData dataclass
- Covariance matrix building
- MVN update computations
- Observation selection
- Bounding box calculations
- Cluster subsampling
"""

import numpy as np
import pytest
from math import sqrt

from vs30.spatial import (
    ObservationData,
    PixelData,
    MVNUpdateResult,
    build_covariance_matrix,
    select_observations_for_pixel,
    compute_mvn_update_for_pixel,
    grid_points_in_bbox,
    calculate_chunk_size,
    subsample_by_cluster,
    validate_observations,
)


class TestObservationData:
    """Tests for the ObservationData dataclass."""

    def test_empty_creates_zero_length_arrays(self):
        """Test that empty() creates arrays with zero observations."""
        obs = ObservationData.empty()

        assert len(obs.locations) == 0
        assert len(obs.vs30) == 0
        assert len(obs.model_vs30) == 0
        assert len(obs.model_stdv) == 0
        assert len(obs.residuals) == 0
        assert len(obs.omega) == 0
        assert len(obs.uncertainty) == 0

    def test_empty_locations_shape(self):
        """Test that empty locations has correct shape."""
        obs = ObservationData.empty()

        assert obs.locations.shape == (0, 2)

    def test_create_with_data(self):
        """Test creating ObservationData with actual data."""
        n_obs = 5
        locations = np.random.rand(n_obs, 2) * 1000
        vs30 = np.random.uniform(150, 400, n_obs)
        model_vs30 = np.random.uniform(150, 400, n_obs)
        model_stdv = np.random.uniform(0.3, 0.6, n_obs)
        residuals = np.log(vs30 / model_vs30)
        omega = np.ones(n_obs)
        uncertainty = np.random.uniform(0.1, 0.3, n_obs)

        obs = ObservationData(
            locations=locations,
            vs30=vs30,
            model_vs30=model_vs30,
            model_stdv=model_stdv,
            residuals=residuals,
            omega=omega,
            uncertainty=uncertainty,
        )

        assert len(obs.locations) == n_obs
        assert len(obs.vs30) == n_obs
        np.testing.assert_array_almost_equal(obs.vs30, vs30)


class TestPixelData:
    """Tests for the PixelData dataclass."""

    def test_create_pixel(self):
        """Test creating a PixelData object."""
        pixel = PixelData(
            location=np.array([1570604, 5180029]),
            vs30=250.0,
            stdv=0.4,
            index=12345,
        )

        assert pixel.vs30 == 250.0
        assert pixel.stdv == 0.4
        assert pixel.index == 12345
        assert len(pixel.location) == 2


class TestMVNUpdateResult:
    """Tests for the MVNUpdateResult dataclass."""

    def test_create_result(self):
        """Test creating an MVNUpdateResult object."""
        result = MVNUpdateResult(
            updated_vs30=275.0,
            updated_stdv=0.35,
            n_observations_used=5,
            min_distance=500.0,
            pixel_index=12345,
        )

        assert result.updated_vs30 == 275.0
        assert result.updated_stdv == 0.35
        assert result.n_observations_used == 5
        assert result.min_distance == 500.0
        assert result.pixel_index == 12345


class TestBuildCovarianceMatrix:
    """Tests for the build_covariance_matrix function."""

    @pytest.fixture
    def simple_pixel(self):
        """Create a simple pixel for testing."""
        return PixelData(
            location=np.array([1000.0, 1000.0]),
            vs30=250.0,
            stdv=0.4,
            index=0,
        )

    @pytest.fixture
    def nearby_observations(self):
        """Create nearby observations for testing."""
        n_obs = 3
        return ObservationData(
            locations=np.array([
                [1100.0, 1000.0],  # 100m away
                [1000.0, 1200.0],  # 200m away
                [1300.0, 1300.0],  # ~424m away
            ]),
            vs30=np.array([260.0, 240.0, 270.0]),
            model_vs30=np.array([255.0, 245.0, 265.0]),
            model_stdv=np.array([0.4, 0.4, 0.4]),
            residuals=np.log(np.array([260.0, 240.0, 270.0]) / np.array([255.0, 245.0, 265.0])),
            omega=np.ones(n_obs),
            uncertainty=np.array([0.2, 0.2, 0.2]),
        )

    def test_covariance_matrix_shape(self, simple_pixel, nearby_observations):
        """Test that covariance matrix has correct shape."""
        cov = build_covariance_matrix(
            simple_pixel,
            nearby_observations,
            model_name="geology",
            phi=1000.0,
            noisy=False,
            cov_reduc=0.0,
        )

        # Matrix should be (1 pixel + 3 observations) x (1 pixel + 3 observations)
        expected_size = 1 + len(nearby_observations.locations)
        assert cov.shape == (expected_size, expected_size)

    def test_covariance_matrix_symmetric(self, simple_pixel, nearby_observations):
        """Test that covariance matrix is symmetric."""
        cov = build_covariance_matrix(
            simple_pixel,
            nearby_observations,
            model_name="geology",
            phi=1000.0,
            noisy=False,
            cov_reduc=0.0,
        )

        np.testing.assert_array_almost_equal(cov, cov.T)

    def test_covariance_matrix_positive_diagonal(self, simple_pixel, nearby_observations):
        """Test that diagonal elements are positive."""
        cov = build_covariance_matrix(
            simple_pixel,
            nearby_observations,
            model_name="geology",
            phi=1000.0,
            noisy=False,
            cov_reduc=0.0,
        )

        assert (np.diag(cov) > 0).all()

    def test_covariance_decreases_with_distance(self, simple_pixel, nearby_observations):
        """Test that covariance decreases with distance."""
        cov = build_covariance_matrix(
            simple_pixel,
            nearby_observations,
            model_name="geology",
            phi=1000.0,
            noisy=False,
            cov_reduc=0.0,
        )

        # Off-diagonal elements should decrease with distance
        # Observation 1 is closest to pixel, should have highest covariance
        # Observation 3 is farthest, should have lowest covariance
        assert cov[0, 1] > cov[0, 3]

    def test_larger_phi_gives_higher_covariance(self, simple_pixel, nearby_observations):
        """Test that larger phi gives higher off-diagonal covariance."""
        cov_small_phi = build_covariance_matrix(
            simple_pixel,
            nearby_observations,
            model_name="geology",
            phi=500.0,
            noisy=False,
            cov_reduc=0.0,
        )

        cov_large_phi = build_covariance_matrix(
            simple_pixel,
            nearby_observations,
            model_name="geology",
            phi=2000.0,
            noisy=False,
            cov_reduc=0.0,
        )

        # Larger phi means slower decay, so higher covariance at same distance
        assert cov_large_phi[0, 1] > cov_small_phi[0, 1]


class TestSelectObservationsForPixel:
    """Tests for the select_observations_for_pixel function."""

    @pytest.fixture
    def pixel_at_origin(self):
        """Create a pixel at origin for testing."""
        return PixelData(
            location=np.array([0.0, 0.0]),
            vs30=250.0,
            stdv=0.4,
            index=0,
        )

    @pytest.fixture
    def scattered_observations(self):
        """Create observations at various distances."""
        return ObservationData(
            locations=np.array([
                [100.0, 0.0],    # 100m away
                [500.0, 0.0],    # 500m away
                [1000.0, 0.0],   # 1km away
                [5000.0, 0.0],   # 5km away
                [10000.0, 0.0],  # 10km away
                [20000.0, 0.0],  # 20km away
            ]),
            vs30=np.array([260.0, 240.0, 270.0, 230.0, 250.0, 280.0]),
            model_vs30=np.array([255.0, 245.0, 265.0, 235.0, 255.0, 275.0]),
            model_stdv=np.array([0.4, 0.4, 0.4, 0.4, 0.4, 0.4]),
            residuals=np.zeros(6),
            omega=np.ones(6),
            uncertainty=np.array([0.2, 0.2, 0.2, 0.2, 0.2, 0.2]),
        )

    def test_max_dist_filters_observations(self, pixel_at_origin, scattered_observations):
        """Test that max_dist_m filters distant observations."""
        selected = select_observations_for_pixel(
            pixel_at_origin,
            scattered_observations,
            max_dist_m=2000,  # 2km
            max_points=100,
        )

        # Should only include observations within 2km (first 3)
        assert len(selected.locations) == 3

    def test_max_points_limits_count(self, pixel_at_origin, scattered_observations):
        """Test that max_points limits observation count."""
        selected = select_observations_for_pixel(
            pixel_at_origin,
            scattered_observations,
            max_dist_m=50000,  # 50km (include all)
            max_points=2,
        )

        # Should limit to 2 closest observations
        assert len(selected.locations) <= 2

    def test_empty_when_too_far(self, pixel_at_origin):
        """Test that empty result returned when all observations too far."""
        far_observations = ObservationData(
            locations=np.array([[100000.0, 0.0]]),  # 100km away
            vs30=np.array([250.0]),
            model_vs30=np.array([250.0]),
            model_stdv=np.array([0.4]),
            residuals=np.zeros(1),
            omega=np.ones(1),
            uncertainty=np.array([0.2]),
        )

        selected = select_observations_for_pixel(
            pixel_at_origin,
            far_observations,
            max_dist_m=5000,  # 5km
            max_points=100,
        )

        assert len(selected.locations) == 0


class TestComputeMVNUpdateForPixel:
    """Tests for the compute_mvn_update_for_pixel function."""

    @pytest.fixture
    def pixel(self):
        """Create a test pixel."""
        return PixelData(
            location=np.array([1000.0, 1000.0]),
            vs30=250.0,
            stdv=0.4,
            index=12345,
        )

    @pytest.fixture
    def nearby_observation(self):
        """Create a single nearby observation."""
        return ObservationData(
            locations=np.array([[1100.0, 1000.0]]),  # 100m away
            vs30=np.array([280.0]),  # Higher than prior
            model_vs30=np.array([260.0]),
            model_stdv=np.array([0.4]),
            residuals=np.array([np.log(280.0 / 260.0)]),
            omega=np.ones(1),
            uncertainty=np.array([0.2]),
        )

    def test_returns_mvn_update_result(self, pixel, nearby_observation):
        """Test that function returns MVNUpdateResult."""
        result = compute_mvn_update_for_pixel(
            pixel,
            nearby_observation,
            model_name="geology",
            phi=1000.0,
            max_dist_m=5000.0,
            max_points=100,
            noisy=False,
            cov_reduc=0.0,
        )

        assert isinstance(result, MVNUpdateResult)
        assert result.pixel_index == pixel.index

    def test_updates_toward_observation(self, pixel, nearby_observation):
        """Test that update shifts vs30 toward observation."""
        result = compute_mvn_update_for_pixel(
            pixel,
            nearby_observation,
            model_name="geology",
            phi=1000.0,
            max_dist_m=5000.0,
            max_points=100,
            noisy=False,
            cov_reduc=0.0,
        )

        # Observation is higher (280), prior is 250, update should increase
        assert result.updated_vs30 > pixel.vs30

    def test_stdv_decreases_with_observation(self, pixel, nearby_observation):
        """Test that standard deviation decreases when observation is added."""
        result = compute_mvn_update_for_pixel(
            pixel,
            nearby_observation,
            model_name="geology",
            phi=1000.0,
            max_dist_m=5000.0,
            max_points=100,
            noisy=False,
            cov_reduc=0.0,
        )

        # Adding observation should reduce uncertainty
        assert result.updated_stdv < pixel.stdv

    def test_returns_none_for_nan_pixel(self, nearby_observation):
        """Test that None is returned for NaN pixel values."""
        nan_pixel = PixelData(
            location=np.array([1000.0, 1000.0]),
            vs30=np.nan,
            stdv=0.4,
            index=0,
        )

        result = compute_mvn_update_for_pixel(
            nan_pixel,
            nearby_observation,
            model_name="geology",
        )

        assert result is None

    def test_no_observations_returns_unchanged_vs30(self, pixel):
        """Test that no nearby observations returns unchanged vs30."""
        far_observation = ObservationData(
            locations=np.array([[100000.0, 100000.0]]),  # Very far
            vs30=np.array([300.0]),
            model_vs30=np.array([300.0]),
            model_stdv=np.array([0.4]),
            residuals=np.zeros(1),
            omega=np.ones(1),
            uncertainty=np.array([0.2]),
        )

        result = compute_mvn_update_for_pixel(
            pixel,
            far_observation,
            model_name="geology",
            max_dist_m=5000,
        )

        # VS30 should be unchanged when no nearby observations
        assert result.updated_vs30 == pixel.vs30
        assert result.n_observations_used == 0


class TestGridPointsInBbox:
    """Tests for the grid_points_in_bbox function."""

    def test_finds_points_in_bbox(self):
        """Test that function correctly identifies points in bounding boxes."""
        # Create a simple grid
        grid_locs = np.array([
            [0.0, 0.0],
            [100.0, 0.0],
            [200.0, 0.0],
            [300.0, 0.0],
        ])

        # Observation at 150, with max_dist=100
        obs_eastings_min = np.array([[50.0]])
        obs_eastings_max = np.array([[250.0]])
        obs_northings_min = np.array([[-100.0]])
        obs_northings_max = np.array([[100.0]])

        mask, indices = grid_points_in_bbox(
            grid_locs,
            obs_eastings_min,
            obs_eastings_max,
            obs_northings_min,
            obs_northings_max,
        )

        # Points at 100 and 200 should be in bbox
        assert mask[1] == True  # 100
        assert mask[2] == True  # 200
        assert mask[0] == False  # 0 (outside east)
        assert mask[3] == False  # 300 (outside east)

    def test_multiple_observations(self):
        """Test with multiple observations."""
        grid_locs = np.array([
            [0.0, 0.0],
            [500.0, 0.0],
            [1000.0, 0.0],
        ])

        # Two observations
        obs_eastings_min = np.array([[-100.0], [900.0]])
        obs_eastings_max = np.array([[100.0], [1100.0]])
        obs_northings_min = np.array([[-100.0], [-100.0]])
        obs_northings_max = np.array([[100.0], [100.0]])

        mask, indices = grid_points_in_bbox(
            grid_locs,
            obs_eastings_min,
            obs_eastings_max,
            obs_northings_min,
            obs_northings_max,
        )

        # First point in first obs bbox, third point in second obs bbox
        assert mask[0] == True
        assert mask[1] == False
        assert mask[2] == True


class TestCalculateChunkSize:
    """Tests for the calculate_chunk_size function."""

    def test_more_observations_smaller_chunks(self):
        """Test that more observations lead to smaller chunks."""
        chunk_size_few_obs = calculate_chunk_size(100, max_spatial_boolean_array_memory_gb=1.0)
        chunk_size_many_obs = calculate_chunk_size(10000, max_spatial_boolean_array_memory_gb=1.0)

        assert chunk_size_few_obs > chunk_size_many_obs

    def test_more_memory_larger_chunks(self):
        """Test that more memory allows larger chunks."""
        chunk_size_small_mem = calculate_chunk_size(1000, max_spatial_boolean_array_memory_gb=0.5)
        chunk_size_large_mem = calculate_chunk_size(1000, max_spatial_boolean_array_memory_gb=2.0)

        assert chunk_size_large_mem > chunk_size_small_mem

    def test_minimum_chunk_size_is_one(self):
        """Test that chunk size is at least 1."""
        # With extreme number of observations
        chunk_size = calculate_chunk_size(10**15, max_spatial_boolean_array_memory_gb=0.001)

        assert chunk_size >= 1


class TestSubsampleByCluster:
    """Tests for the subsample_by_cluster function."""

    def test_unclustered_all_included(self):
        """Test that unclustered observations (-1) are all included."""
        # All unclustered
        cluster_labels = np.array([-1, -1, -1, -1, -1])

        result = subsample_by_cluster(cluster_labels, step=2)

        # All should be included since they are unclustered
        assert len(result) == 5

    def test_clustered_subsampled(self):
        """Test that clustered observations are subsampled."""
        # All in one cluster
        cluster_labels = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0])  # 10 points

        result = subsample_by_cluster(cluster_labels, step=3)

        # Should take every 3rd: indices 0, 3, 6, 9
        assert len(result) == 4

    def test_mixed_clusters(self):
        """Test with mixed clustered and unclustered."""
        cluster_labels = np.array([
            -1, -1, -1,  # 3 unclustered
            0, 0, 0, 0,  # 4 in cluster 0
            1, 1, 1, 1, 1, 1,  # 6 in cluster 1
        ])

        result = subsample_by_cluster(cluster_labels, step=2)

        # Unclustered: 3 (all included)
        # Cluster 0: 2 (indices 0, 2 within cluster -> original indices 3, 5)
        # Cluster 1: 3 (indices 0, 2, 4 within cluster -> original indices 7, 9, 11)
        # Total: 3 + 2 + 3 = 8
        assert len(result) == 8

    def test_at_least_one_per_cluster(self):
        """Test that at least one observation per cluster is kept."""
        cluster_labels = np.array([0, 1, 2])  # Each cluster has one point

        result = subsample_by_cluster(cluster_labels, step=100)

        # Even with large step, each cluster should keep at least one
        assert len(result) == 3


class TestValidateObservations:
    """Tests for the validate_observations function."""

    def test_valid_observations_pass(self):
        """Test that valid observations pass validation."""
        import pandas as pd

        obs = pd.DataFrame({
            "easting": [1000, 2000],
            "northing": [1000, 2000],
            "vs30": [200, 300],
            "uncertainty": [0.2, 0.3],
        })

        # Should not raise
        validate_observations(obs)

    def test_missing_column_raises(self):
        """Test that missing column raises AssertionError."""
        import pandas as pd

        obs = pd.DataFrame({
            "easting": [1000, 2000],
            "northing": [1000, 2000],
            "vs30": [200, 300],
            # Missing "uncertainty"
        })

        with pytest.raises(AssertionError, match="uncertainty"):
            validate_observations(obs)

    def test_negative_vs30_raises(self):
        """Test that negative vs30 raises AssertionError."""
        import pandas as pd

        obs = pd.DataFrame({
            "easting": [1000, 2000],
            "northing": [1000, 2000],
            "vs30": [200, -100],  # Negative value
            "uncertainty": [0.2, 0.3],
        })

        with pytest.raises(AssertionError, match="positive"):
            validate_observations(obs)

    def test_negative_uncertainty_raises(self):
        """Test that negative uncertainty raises AssertionError."""
        import pandas as pd

        obs = pd.DataFrame({
            "easting": [1000, 2000],
            "northing": [1000, 2000],
            "vs30": [200, 300],
            "uncertainty": [0.2, -0.3],  # Negative value
        })

        with pytest.raises(AssertionError, match="positive"):
            validate_observations(obs)

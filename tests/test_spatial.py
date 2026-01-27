"""
Tests for the VS30 spatial module.

Tests cover:
- ObservationData dataclass
- PixelData dataclass
- SpatialAdjustmentResult dataclass
- RasterData dataclass
- Covariance matrix building
- Spatial adjustment computations
- Observation selection
- Bounding box calculations
- Cluster subsampling
"""

from pathlib import Path

import numpy as np
import pytest

from vs30.spatial import (
    ObservationData,
    PixelData,
    SpatialAdjustmentResult,
    build_covariance_matrix,
    select_observations_for_pixel,
    compute_spatial_adjustment_for_pixel,
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


class TestSpatialAdjustmentResult:
    """Tests for the SpatialAdjustmentResult dataclass."""

    def test_create_result(self):
        """Test creating an SpatialAdjustmentResult object."""
        result = SpatialAdjustmentResult(
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
            model_type="geology",
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
            model_type="geology",
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
            model_type="geology",
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
            model_type="geology",
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
            model_type="geology",
            phi=500.0,
            noisy=False,
            cov_reduc=0.0,
        )

        cov_large_phi = build_covariance_matrix(
            simple_pixel,
            nearby_observations,
            model_type="geology",
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


class TestComputeSpatialAdjustmentForPixel:
    """Tests for the compute_spatial_adjustment_for_pixel function."""

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

    def test_returns_spatial_adjustment_result(self, pixel, nearby_observation):
        """Test that function returns SpatialAdjustmentResult."""
        result = compute_spatial_adjustment_for_pixel(
            pixel,
            nearby_observation,
            model_type="geology",
            phi=1000.0,
            max_dist_m=5000.0,
            max_points=100,
            noisy=False,
            cov_reduc=0.0,
        )

        assert isinstance(result, SpatialAdjustmentResult)
        assert result.pixel_index == pixel.index

    def test_updates_toward_observation(self, pixel, nearby_observation):
        """Test that update shifts vs30 toward observation."""
        result = compute_spatial_adjustment_for_pixel(
            pixel,
            nearby_observation,
            model_type="geology",
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
        result = compute_spatial_adjustment_for_pixel(
            pixel,
            nearby_observation,
            model_type="geology",
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

        result = compute_spatial_adjustment_for_pixel(
            nan_pixel,
            nearby_observation,
            model_type="geology",
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

        result = compute_spatial_adjustment_for_pixel(
            pixel,
            far_observation,
            model_type="geology",
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
        assert mask[1]  # 100
        assert mask[2]  # 200
        assert not mask[0]  # 0 (outside east)
        assert not mask[3]  # 300 (outside east)

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
        assert mask[0]
        assert not mask[1]
        assert mask[2]


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
        """Test that missing column raises ValueError."""
        import pandas as pd

        obs = pd.DataFrame({
            "easting": [1000, 2000],
            "northing": [1000, 2000],
            "vs30": [200, 300],
            # Missing "uncertainty"
        })

        with pytest.raises(ValueError, match="uncertainty"):
            validate_observations(obs)

    def test_negative_vs30_raises(self):
        """Test that negative vs30 raises ValueError."""
        import pandas as pd

        obs = pd.DataFrame({
            "easting": [1000, 2000],
            "northing": [1000, 2000],
            "vs30": [200, -100],  # Negative value
            "uncertainty": [0.2, 0.3],
        })

        with pytest.raises(ValueError, match="positive"):
            validate_observations(obs)

    def test_negative_uncertainty_raises(self):
        """Test that negative uncertainty raises ValueError."""
        import pandas as pd

        obs = pd.DataFrame({
            "easting": [1000, 2000],
            "northing": [1000, 2000],
            "vs30": [200, 300],
            "uncertainty": [0.2, -0.3],  # Negative value
        })

        with pytest.raises(ValueError, match="positive"):
            validate_observations(obs)


# =============================================================================
# Additional Tests from Coverage Improvements
# =============================================================================


class TestRasterDataClass:
    """Tests for RasterData class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        import tempfile
        import shutil
        tmpdir = tempfile.mkdtemp(prefix="vs30_spatial_test_")
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def sample_vs30_raster(self, temp_dir):
        """Create a sample 2-band VS30 raster for testing."""
        import rasterio
        from rasterio.transform import from_bounds

        raster_path = temp_dir / "test_vs30.tif"

        # Create small test raster
        width, height = 10, 10
        transform = from_bounds(1500000, 5100000, 1505000, 5105000, width, height)

        vs30_data = np.random.uniform(200, 600, (height, width)).astype(np.float32)
        stdv_data = np.random.uniform(20, 60, (height, width)).astype(np.float32)

        with rasterio.open(
            raster_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=2,
            dtype='float32',
            crs='EPSG:2193',
            transform=transform,
            nodata=-9999.0,
        ) as dst:
            dst.write(vs30_data, 1)
            dst.write(stdv_data, 2)

        return raster_path

    def test_from_file(self, sample_vs30_raster):
        """Test loading raster from file."""
        from vs30.spatial import RasterData

        raster_data = RasterData.from_file(sample_vs30_raster)

        assert raster_data.vs30.shape == (10, 10)
        assert raster_data.stdv.shape == (10, 10)
        assert raster_data.transform is not None
        assert raster_data.crs is not None

    def test_get_coordinates(self, sample_vs30_raster):
        """Test getting coordinates for valid pixels."""
        from vs30.spatial import RasterData

        raster_data = RasterData.from_file(sample_vs30_raster)
        coords = raster_data.get_coordinates()

        # Should have coordinates for all valid pixels
        assert len(coords) == len(raster_data.valid_flat_indices)
        assert coords.shape[1] == 2  # easting, northing

    def test_write_updated(self, sample_vs30_raster, temp_dir):
        """Test writing updated raster."""
        import rasterio
        from vs30.spatial import RasterData

        raster_data = RasterData.from_file(sample_vs30_raster)

        # Modify arrays
        updated_vs30 = raster_data.vs30 * 1.1
        updated_stdv = raster_data.stdv * 0.9

        output_path = temp_dir / "updated.tif"
        raster_data.write_updated(output_path, updated_vs30, updated_stdv)

        assert output_path.exists()

        # Verify written data
        with rasterio.open(output_path) as src:
            assert src.count == 2
            written_vs30 = src.read(1)
            np.testing.assert_array_almost_equal(written_vs30, updated_vs30)


class TestValidateRasterDataFunc:
    """Tests for validate_raster_data function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        import tempfile
        import shutil
        tmpdir = tempfile.mkdtemp(prefix="vs30_spatial_test_")
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def sample_vs30_raster(self, temp_dir):
        """Create a sample 2-band VS30 raster for testing."""
        import rasterio
        from rasterio.transform import from_bounds

        raster_path = temp_dir / "test_vs30.tif"

        width, height = 10, 10
        transform = from_bounds(1500000, 5100000, 1505000, 5105000, width, height)

        vs30_data = np.random.uniform(200, 600, (height, width)).astype(np.float32)
        stdv_data = np.random.uniform(20, 60, (height, width)).astype(np.float32)

        with rasterio.open(
            raster_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=2,
            dtype='float32',
            crs='EPSG:2193',
            transform=transform,
            nodata=-9999.0,
        ) as dst:
            dst.write(vs30_data, 1)
            dst.write(stdv_data, 2)

        return raster_path

    def test_valid_raster_passes(self, sample_vs30_raster):
        """Test that valid raster data passes validation."""
        from vs30.spatial import RasterData, validate_raster_data

        raster_data = RasterData.from_file(sample_vs30_raster)
        # Should not raise
        validate_raster_data(raster_data)

    def test_shape_mismatch_fails(self, sample_vs30_raster):
        """Test that mismatched band shapes fail validation."""
        from vs30.spatial import RasterData, validate_raster_data

        raster_data = RasterData.from_file(sample_vs30_raster)
        # Artificially corrupt the data
        raster_data.stdv = np.zeros((5, 5))  # Wrong shape

        with pytest.raises(ValueError):
            validate_raster_data(raster_data)


class TestComputeMvnAtPoints:
    """Tests for compute_spatial_adjustment_at_points function."""

    def test_no_observations_returns_prior(self):
        """Test that no observations returns prior values with shrunk stdv."""
        from vs30.spatial import compute_spatial_adjustment_at_points

        points = np.array([[1500000, 5100000], [1501000, 5101000]])
        model_vs30 = np.array([300.0, 400.0])
        model_stdv = np.array([30.0, 40.0])

        # Empty observations
        obs_locations = np.empty((0, 2))
        obs_vs30 = np.empty(0)
        obs_model_vs30 = np.empty(0)
        obs_model_stdv = np.empty(0)
        obs_uncertainty = np.empty(0)

        mvn_vs30, mvn_stdv = compute_spatial_adjustment_at_points(
            points=points,
            model_vs30=model_vs30,
            model_stdv=model_stdv,
            obs_locations=obs_locations,
            obs_vs30=obs_vs30,
            obs_model_vs30=obs_model_vs30,
            obs_model_stdv=obs_model_stdv,
            obs_uncertainty=obs_uncertainty,
            model_type="geology",
        )

        # Should return prior vs30 unchanged
        np.testing.assert_array_equal(mvn_vs30, model_vs30)

    def test_with_nearby_observations(self):
        """Test spatial adjustment with nearby observations."""
        from vs30.spatial import compute_spatial_adjustment_at_points

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

        mvn_vs30, mvn_stdv = compute_spatial_adjustment_at_points(
            points=points,
            model_vs30=model_vs30,
            model_stdv=model_stdv,
            obs_locations=obs_locations,
            obs_vs30=obs_vs30,
            obs_model_vs30=obs_model_vs30,
            obs_model_stdv=obs_model_stdv,
            obs_uncertainty=obs_uncertainty,
            model_type="geology",
            max_dist_m=5000,
        )

        # Should adjust toward observation (increase vs30)
        assert mvn_vs30[0] > model_vs30[0], "Adjustment should pull vs30 toward observation"
        # Uncertainty should decrease
        assert mvn_stdv[0] < model_stdv[0], "Uncertainty should decrease with observation"

    def test_invalid_points_skipped(self):
        """Test that NaN points are skipped."""
        from vs30.spatial import compute_spatial_adjustment_at_points

        points = np.array([[1500000.0, 5100000.0], [1501000.0, 5101000.0]])
        model_vs30 = np.array([np.nan, 300.0])  # First point invalid
        model_stdv = np.array([30.0, 30.0])

        obs_locations = np.array([[1500500.0, 5100500.0]])
        obs_vs30 = np.array([350.0])
        obs_model_vs30 = np.array([300.0])
        obs_model_stdv = np.array([30.0])
        obs_uncertainty = np.array([25.0])

        mvn_vs30, mvn_stdv = compute_spatial_adjustment_at_points(
            points=points,
            model_vs30=model_vs30,
            model_stdv=model_stdv,
            obs_locations=obs_locations,
            obs_vs30=obs_vs30,
            obs_model_vs30=obs_model_vs30,
            obs_model_stdv=obs_model_stdv,
            obs_uncertainty=obs_uncertainty,
            model_type="terrain",
        )

        # First point should remain NaN
        assert np.isnan(mvn_vs30[0])
        # Second point should be adjusted
        assert not np.isnan(mvn_vs30[1])


class TestPrepareObservationDataErrors:
    """Tests for error paths in prepare_observation_data."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        import tempfile
        import shutil
        tmpdir = tempfile.mkdtemp(prefix="vs30_spatial_test_")
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def sample_vs30_raster(self, temp_dir):
        """Create a sample 2-band VS30 raster for testing."""
        import rasterio
        from rasterio.transform import from_bounds

        raster_path = temp_dir / "test_vs30.tif"

        width, height = 10, 10
        transform = from_bounds(1500000, 5100000, 1505000, 5105000, width, height)

        vs30_data = np.random.uniform(200, 600, (height, width)).astype(np.float32)
        stdv_data = np.random.uniform(20, 60, (height, width)).astype(np.float32)

        with rasterio.open(
            raster_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=2,
            dtype='float32',
            crs='EPSG:2193',
            transform=transform,
            nodata=-9999.0,
        ) as dst:
            dst.write(vs30_data, 1)
            dst.write(stdv_data, 2)

        return raster_path

    def test_unknown_model_type_raises(self, sample_vs30_raster, temp_dir):
        """Test that unknown model type raises ValueError."""
        import pandas as pd
        from vs30.spatial import RasterData, prepare_observation_data

        raster_data = RasterData.from_file(sample_vs30_raster)
        observations = pd.DataFrame({
            'easting': [1502000],
            'northing': [5102000],
            'vs30': [300],
            'uncertainty': [30],
        })
        model_table = np.array([[300, 30], [400, 40]])

        with pytest.raises(ValueError, match="Unknown model type"):
            prepare_observation_data(
                observations=observations,
                raster_data=raster_data,
                updated_model_table=model_table,
                model_type="invalid_model",
                output_dir=temp_dir,
            )


class TestSelectObservationsForPixelAdditional:
    """Additional tests for select_observations_for_pixel."""

    def test_select_no_nearby_observations(self):
        """Test when no observations are within max_dist_m."""
        pixel = PixelData(
            location=np.array([0.0, 0.0]),
            vs30=300.0,
            stdv=30.0,
            index=0,
        )

        obs_data = ObservationData(
            locations=np.array([[1000000.0, 1000000.0]]),  # Very far away
            vs30=np.array([350.0]),
            model_vs30=np.array([300.0]),
            model_stdv=np.array([30.0]),
            residuals=np.array([0.15]),
            omega=np.array([1.0]),
            uncertainty=np.array([25.0]),
        )

        selected = select_observations_for_pixel(
            pixel, obs_data, max_dist_m=1000, max_points=10
        )

        assert len(selected.locations) == 0

    def test_select_limited_by_max_points_additional(self):
        """Test that max_points limits selection."""
        pixel = PixelData(
            location=np.array([0.0, 0.0]),
            vs30=300.0,
            stdv=30.0,
            index=0,
        )

        # Create 10 nearby observations
        n_obs = 10
        np.random.seed(42)  # For reproducibility
        obs_data = ObservationData(
            locations=np.random.uniform(-100, 100, (n_obs, 2)),
            vs30=np.random.uniform(250, 350, n_obs),
            model_vs30=np.full(n_obs, 300.0),
            model_stdv=np.full(n_obs, 30.0),
            residuals=np.random.uniform(-0.1, 0.1, n_obs),
            omega=np.ones(n_obs),
            uncertainty=np.full(n_obs, 25.0),
        )

        selected = select_observations_for_pixel(
            pixel, obs_data, max_dist_m=10000, max_points=5
        )

        # Should select at most max_points
        assert len(selected.locations) <= 5


class TestBuildCovarianceMatrixAdditional:
    """Additional tests for build_covariance_matrix."""

    def test_build_basic_covariance(self):
        """Test building a basic covariance matrix."""
        pixel = PixelData(
            location=np.array([0.0, 0.0]),
            vs30=300.0,
            stdv=30.0,
            index=0,
        )

        obs_data = ObservationData(
            locations=np.array([[100.0, 0.0], [0.0, 100.0]]),
            vs30=np.array([310.0, 290.0]),
            model_vs30=np.array([300.0, 300.0]),
            model_stdv=np.array([30.0, 30.0]),
            residuals=np.array([0.033, -0.034]),
            omega=np.array([1.0, 1.0]),
            uncertainty=np.array([25.0, 25.0]),
        )

        cov = build_covariance_matrix(
            pixel, obs_data, "geology",
            phi=1000.0, noisy=False, cov_reduc=0.0
        )

        # Should be 3x3 (1 pixel + 2 observations)
        assert cov.shape == (3, 3)
        # Should be symmetric
        np.testing.assert_array_almost_equal(cov, cov.T)
        # Diagonal should be positive
        assert np.all(np.diag(cov) > 0)


class TestSubsampleByClusterAdditional:
    """Additional tests for subsample_by_cluster."""

    def test_subsample_unclustered_all_kept(self):
        """Test that unclustered observations (label=-1) are all kept."""
        # All unclustered
        labels = np.array([-1, -1, -1, -1, -1])
        selected = subsample_by_cluster(labels, step=2)

        # All should be kept
        assert len(selected) == 5
        np.testing.assert_array_equal(selected, [0, 1, 2, 3, 4])

    def test_subsample_clustered_subsampled(self):
        """Test that clustered observations are subsampled."""
        # Two clusters
        labels = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        selected = subsample_by_cluster(labels, step=2)

        # Should take every 2nd from each cluster: indices 0, 2, 4, 6
        assert len(selected) == 4

    def test_subsample_mixed(self):
        """Test mixed unclustered and clustered."""
        # Mix of unclustered (-1) and clustered (0, 1)
        labels = np.array([-1, 0, 0, 0, -1, 1, 1])
        selected = subsample_by_cluster(labels, step=2)

        # Unclustered indices [0, 4] all kept
        # Cluster 0 indices [1, 2, 3] -> [1, 3] (step=2)
        # Cluster 1 indices [5, 6] -> [5] (step=2)
        assert 0 in selected  # unclustered
        assert 4 in selected  # unclustered
        assert 1 in selected  # first of cluster 0


class TestFindAffectedPixels:
    """Tests for find_affected_pixels function."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        import tempfile
        import shutil
        tmpdir = tempfile.mkdtemp(prefix="vs30_spatial_test_")
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def sample_vs30_raster(self, temp_dir):
        """Create a sample 2-band VS30 raster for testing."""
        import rasterio
        from rasterio.transform import from_bounds

        raster_path = temp_dir / "test_vs30.tif"

        width, height = 10, 10
        transform = from_bounds(1500000, 5100000, 1505000, 5105000, width, height)

        vs30_data = np.random.uniform(200, 600, (height, width)).astype(np.float32)
        stdv_data = np.random.uniform(20, 60, (height, width)).astype(np.float32)

        with rasterio.open(
            raster_path,
            'w',
            driver='GTiff',
            height=height,
            width=width,
            count=2,
            dtype='float32',
            crs='EPSG:2193',
            transform=transform,
            nodata=-9999.0,
        ) as dst:
            dst.write(vs30_data, 1)
            dst.write(stdv_data, 2)

        return raster_path

    def test_find_affected_pixels_with_observations(self, sample_vs30_raster):
        """Test with some observations that affect pixels."""
        from vs30.spatial import RasterData, find_affected_pixels

        raster_data = RasterData.from_file(sample_vs30_raster)

        # Create observation near the raster center
        obs_data = ObservationData(
            locations=np.array([[1502500.0, 5102500.0]]),  # Center of raster
            vs30=np.array([350.0]),
            model_vs30=np.array([300.0]),
            model_stdv=np.array([30.0]),
            residuals=np.array([0.15]),
            omega=np.array([1.0]),
            uncertainty=np.array([25.0]),
        )

        result = find_affected_pixels(raster_data, obs_data, max_dist_m=5000)

        # Should find some affected pixels
        assert result.n_affected_pixels > 0


class TestGridPointsInBboxAdditional:
    """Additional tests for grid_points_in_bbox."""

    def test_grid_points_basic_additional(self):
        """Test basic bounding box calculation."""
        # Simple grid
        grid_locs = np.array([
            [0.0, 0.0],
            [100.0, 0.0],
            [200.0, 0.0],
            [0.0, 100.0],
            [100.0, 100.0],
        ])

        # Single observation at (100, 50) with max_dist=75
        obs_eastings_min = np.array([[25.0]])
        obs_eastings_max = np.array([[175.0]])
        obs_northings_min = np.array([[-25.0]])
        obs_northings_max = np.array([[125.0]])

        mask, obs_to_grid = grid_points_in_bbox(
            grid_locs,
            obs_eastings_min, obs_eastings_max,
            obs_northings_min, obs_northings_max
        )

        # Points 1 (100, 0) and 4 (100, 100) should be in bbox
        assert mask[1]  # (100, 0) is in bbox
        assert mask[4]  # (100, 100) is in bbox


class TestCalculateChunkSizeAdditional:
    """Additional tests for calculate_chunk_size."""

    def test_chunk_size_calculation(self):
        """Test chunk size calculation."""
        # With 1000 observations and 1GB memory
        chunk_size = calculate_chunk_size(1000, 1.0)

        # Should be approximately 1GB / 1000 bytes = 1M grid points
        assert chunk_size > 0
        assert chunk_size < 10_000_000_000  # Reasonable upper bound

    def test_chunk_size_minimum_one(self):
        """Test chunk size is at least 1."""
        # Very small memory, many observations
        chunk_size = calculate_chunk_size(10_000_000, 0.000001)

        assert chunk_size >= 1

"""Tests for the VS30 spatial module."""

import functools

import numpy as np
import pytest
import rasterio

from vs30 import constants, correlations, spatial

# Create a standard geology correlation callable for tests
geology_corr_fn = functools.partial(correlations.exponential, phi=1407)


class TestComputeSpatialAdjustmentForPixel:
    """Tests for the compute_spatial_adjustment_for_pixel function."""

    @pytest.fixture
    def pixel(self):
        """Create a test pixel."""
        return spatial.PixelData(
            location=np.array([1000.0, 1000.0]),
            vs30=250.0,
            stdv=0.4,
        )

    @pytest.fixture
    def nearby_observation(self):
        """Create a single nearby observation (vs30=280, model_vs30=260)."""
        return spatial.ObservationData(
            locations=np.array([[1100.0, 1000.0]]),  # 100m away
            model_stdv=np.array([0.4]),
            log_model_vs30=np.log(np.array([260.0])),
            residuals=np.array([np.log(280.0 / 260.0)]),
            noise_weights=np.ones(1),
        )

    def test_updates_toward_observation(self, pixel, nearby_observation):
        """Test that update shifts vs30 toward observation."""
        indices, distances = spatial.select_observations_for_pixel_batch(
            np.array([pixel.location]),
            nearby_observation,
            max_dist_m=5000.0,
            max_points=100,
        )
        result = spatial.compute_spatial_adjustment_for_pixel(
            pixel,
            nearby_observation,
            indices[0][np.isfinite(distances[0])],
            corr_fn=geology_corr_fn,
            corr_zero=geology_corr_fn(np.array([0.0]))[0],
            noisy=False,
        )

        # Observation is higher (280), prior is 250, update should increase
        assert result is not None
        updated_vs30, _ = result
        assert updated_vs30 > pixel.vs30

    def test_stdv_decreases_with_observation(self, pixel, nearby_observation):
        """Test that standard deviation decreases when observation is added."""
        indices, distances = spatial.select_observations_for_pixel_batch(
            np.array([pixel.location]),
            nearby_observation,
            max_dist_m=5000.0,
            max_points=100,
        )
        result = spatial.compute_spatial_adjustment_for_pixel(
            pixel,
            nearby_observation,
            indices[0][np.isfinite(distances[0])],
            corr_fn=geology_corr_fn,
            corr_zero=geology_corr_fn(np.array([0.0]))[0],
            noisy=False,
        )

        # Adding observation should reduce uncertainty
        assert result is not None
        _, updated_stdv = result
        assert updated_stdv < pixel.stdv

    def test_no_observations_returns_unchanged_vs30(self, pixel):
        """Test that no nearby observations returns unchanged vs30."""
        far_observation = spatial.ObservationData(
            locations=np.array([[100000.0, 100000.0]]),  # Very far
            model_stdv=np.array([0.4]),
            log_model_vs30=np.log(np.array([300.0])),
            residuals=np.zeros(1),
            noise_weights=np.ones(1),
        )
        indices, distances = spatial.select_observations_for_pixel_batch(
            np.array([pixel.location]),
            far_observation,
            max_dist_m=5000.0,
        )

        result = spatial.compute_spatial_adjustment_for_pixel(
            pixel,
            far_observation,
            indices[0][np.isfinite(distances[0])],
            corr_fn=geology_corr_fn,
            corr_zero=geology_corr_fn(np.array([0.0]))[0],
        )

        # VS30 should be unchanged when no nearby observations
        assert result is not None
        updated_vs30, _ = result
        assert updated_vs30 == pixel.vs30


class TestComputeMvnAtPoints:
    """Tests for compute_spatial_point_adjustments function."""

    def test_no_observations_returns_prior(self):
        """Test that no observations returns prior vs30 unchanged."""
        points = np.array([[1500000, 5100000], [1501000, 5101000]])
        model_vs30 = np.array([300.0, 400.0])
        model_stdv = np.array([30.0, 40.0])

        mvn_vs30, _ = spatial.compute_spatial_point_adjustments(
            points=points,
            model_vs30=model_vs30,
            model_stdv=model_stdv,
            obs_data=spatial.ObservationData.empty(),
            corr_fn=geology_corr_fn,
        )

        # Should return prior vs30 unchanged
        assert mvn_vs30 == pytest.approx(model_vs30)

    def test_with_nearby_observations(self):
        """Test spatial adjustment with nearby observations."""
        # Single query point
        points = np.array([[1500000.0, 5100000.0]])
        model_vs30 = np.array([300.0])
        model_stdv = np.array([30.0])

        # Nearby observation with higher vs30
        obs_model_vs30 = np.array([300.0])
        obs_data = spatial.ObservationData(
            locations=np.array([[1500100.0, 5100100.0]]),  # 141m away
            model_stdv=np.array([30.0]),
            log_model_vs30=np.log(obs_model_vs30),
            residuals=np.log(np.array([400.0]) / obs_model_vs30),
            noise_weights=np.ones(1),
        )

        mvn_vs30, mvn_stdv = spatial.compute_spatial_point_adjustments(
            points=points,
            model_vs30=model_vs30,
            model_stdv=model_stdv,
            obs_data=obs_data,
            corr_fn=geology_corr_fn,
            max_dist_m=5000,
        )

        # Should adjust toward observation (increase vs30)
        assert mvn_vs30[0] > model_vs30[0]
        # Uncertainty should decrease
        assert mvn_stdv[0] < model_stdv[0]


class TestFindAffectedPixels:
    """Tests for find_affected_pixels."""

    def build_raster_data(self, n_rows: int = 5, n_cols: int = 5) -> spatial.RasterData:
        """Build a small RasterData with valid pixels everywhere.

        Pixel size is 100m, origin (1500000, 5100000) at the top-left, so
        pixel (row, col) centre is at (1500050 + col*100, 5099950 - row*100).
        """
        vs30 = np.full((n_rows, n_cols), 300.0, dtype=np.float32)
        stdv = np.full((n_rows, n_cols), 0.5, dtype=np.float32)
        transform = rasterio.transform.Affine(100, 0, 1500000, 0, -100, 5100000)
        return spatial.RasterData.from_arrays(vs30=vs30, stdv=stdv, transform=transform)

    def test_find_affected_pixels_single_obs_in_centre(self):
        """One obs at the grid centre with 150m half-width covers a 3×3 = 9-pixel bbox (100m lattice)."""
        raster_data = self.build_raster_data(5, 5)
        obs_data = spatial.ObservationData(
            locations=np.array([[1500250.0, 5099750.0]]),
            model_stdv=np.array([0.5]),
            log_model_vs30=np.log(np.array([300.0])),
            residuals=np.zeros(1),
            noise_weights=np.ones(1),
        )

        affected_flat_indices, affected_locs = spatial.find_affected_pixels(
            raster_data,
            obs_data,
            max_spatial_intermediate_array_memory_gb=1.0,
            model_type=constants.ModelType.GEOLOGY,
            max_dist_m=150.0,
        )

        # 3×3 = 9 pixels in the obs bounding box.
        assert len(affected_flat_indices) == 9
        assert affected_locs.shape == (9, 2)

    def test_find_affected_pixels_far_obs_zero(self):
        """An observation far outside the raster bounds affects zero pixels."""
        raster_data = self.build_raster_data(5, 5)
        # Obs ~1 km outside the raster bounds.
        obs_data = spatial.ObservationData(
            locations=np.array([[1600000.0, 5200000.0]]),
            model_stdv=np.array([0.5]),
            log_model_vs30=np.log(np.array([300.0])),
            residuals=np.zeros(1),
            noise_weights=np.ones(1),
        )

        affected_flat_indices, affected_locs = spatial.find_affected_pixels(
            raster_data,
            obs_data,
            max_spatial_intermediate_array_memory_gb=1.0,
            model_type=constants.ModelType.GEOLOGY,
            max_dist_m=1000.0,
        )

        assert len(affected_flat_indices) == 0
        assert affected_locs.shape == (0, 2)

"""Tests for the VS30 spatial module."""

import functools

import numpy as np
import pandas as pd
import pytest
import rasterio

from conftest import load_fixed_model_config
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

    def test_colocated_observations_act_as_one_with_mean_residual(self, pixel):
        """With noisy=False, two observations at one location update like one observation with their mean residual."""
        obs_data = spatial.ObservationData(
            locations=np.array([[1100.0, 1000.0], [1100.0, 1000.0]]),  # same spot, 100 m away
            model_stdv=np.array([0.4, 0.4]),
            log_model_vs30=np.log(np.array([260.0, 260.0])),
            residuals=np.log(np.array([300.0, 280.0]) / 260.0),
            noise_weights=np.ones(2),
        )
        corr_zero = geology_corr_fn(np.array([0.0]))[0]

        result = spatial.compute_spatial_adjustment_for_pixel(
            pixel,
            obs_data,
            np.array([0, 1]),
            corr_fn=geology_corr_fn,
            corr_zero=corr_zero,
            noisy=False,
            cov_reduc=0.0,
        )

        # Both observations share the pixel covariance rho * 0.4**2, and their
        # block is 0.4**2 * corr_zero * [[1, 1], [1, 1]], so each gets weight
        # rho / corr_zero / 2.
        rho = geology_corr_fn(np.array([100.0]))[0]
        assert result is not None
        updated_vs30, updated_stdv = result
        assert updated_vs30 == pytest.approx(
            250.0 * np.exp(rho / corr_zero * np.mean(obs_data.residuals)), rel=1e-6
        )
        assert updated_stdv == pytest.approx(
            0.4 * np.sqrt(corr_zero - rho**2 / corr_zero), rel=1e-6
        )

    def test_non_finite_update_keeps_prior_and_warns(self, pixel, caplog):
        """An update that comes out non-finite keeps the prior values and logs a warning."""
        obs_data = spatial.ObservationData(
            locations=np.array([[1100.0, 1000.0]]),
            model_stdv=np.array([0.4]),
            log_model_vs30=np.log(np.array([260.0])),
            residuals=np.array([np.nan]),
            noise_weights=np.ones(1),
        )
        corr_zero = geology_corr_fn(np.array([0.0]))[0]

        with caplog.at_level("WARNING", logger="vs30.spatial"):
            result = spatial.compute_spatial_adjustment_for_pixel(
                pixel,
                obs_data,
                np.array([0]),
                corr_fn=geology_corr_fn,
                corr_zero=corr_zero,
            )

        assert result is not None
        updated_vs30, updated_stdv = result
        assert updated_vs30 == pixel.vs30
        assert updated_stdv == pytest.approx(pixel.stdv * np.sqrt(corr_zero))
        assert [record.levelname for record in caplog.records] == ["WARNING"]


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
        """One obs at the grid centre with 150m radius captures a 3×3 = 9-pixel neighbourhood (100m lattice)."""
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
            max_dist_m=150.0,
        )

        # 3×3 = 9 pixels within 150 m of the obs.
        assert len(affected_flat_indices) == 9
        assert affected_locs.shape == (9, 2)

    def test_find_affected_pixels_far_obs_zero(self):
        """An observation far outside the raster bounds affects zero pixels."""
        raster_data = self.build_raster_data(5, 5)
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
            max_dist_m=1000.0,
        )

        assert len(affected_flat_indices) == 0
        assert affected_locs.shape == (0, 2)


class TestComputeSpatialAdjustmentOnGrid:
    """Tests for compute_spatial_adjustment_on_grid."""

    def test_grid_with_no_valid_pixels_is_returned_unchanged(self):
        """A grid with no valid pixels (e.g. all sea) is returned as is instead of crashing."""
        vs30 = np.full((3, 3), float(constants.NODATA_VALUE))
        stdv = np.full((3, 3), float(constants.NODATA_VALUE))
        terrain_model_df = pd.read_csv(
            load_fixed_model_config(constants.FixedModelVersion.MODIFIED_FOSTER_2019)[
                "terrain_categorical_csv"
            ],
            comment="#",
            skipinitialspace=True,
        ).rename(columns=str.strip)
        # A real on-land observation in Wellington, so observation preparation succeeds.
        observations_df = pd.DataFrame(
            {
                constants.ObservationColumn.EASTING: [1749050.0],
                constants.ObservationColumn.NORTHING: [5427050.0],
                constants.ObservationColumn.VS30: [300.0],
                constants.ObservationColumn.UNCERTAINTY: [0.2],
            }
        )

        adjusted_vs30, adjusted_stdv = spatial.compute_spatial_adjustment_on_grid(
            vs30_array=vs30,
            stdv_array=stdv,
            profile={
                "transform": rasterio.transform.Affine(100, 0, 1748900, 0, -100, 5427200)
            },
            observations_df=observations_df,
            model_values_df=terrain_model_df,
            model_type=constants.ModelType.TERRAIN,
            corr_fn=geology_corr_fn,
            apply_alluvium_slope_mod=False,
            apply_coastal_distance_mod=False,
        )

        np.testing.assert_array_equal(adjusted_vs30, vs30)
        np.testing.assert_array_equal(adjusted_stdv, stdv)


class TestComputeSpatialPixelAdjustments:
    """Tests for compute_spatial_pixel_adjustments."""

    def test_show_progress_false_silences_progress_bar(self, capsys):
        """show_progress=False hides the per-pixel progress bar that is shown by default."""
        raster_data = spatial.RasterData.from_arrays(
            vs30=np.full((3, 3), 300.0),
            stdv=np.full((3, 3), 0.5),
            transform=rasterio.transform.Affine(100, 0, 1500000, 0, -100, 5100000),
        )
        obs_data = spatial.ObservationData(
            locations=np.array([[1500150.0, 5099850.0]]),
            model_stdv=np.array([0.5]),
            log_model_vs30=np.log(np.array([300.0])),
            residuals=np.zeros(1),
            noise_weights=np.ones(1),
        )
        affected_flat_indices, affected_locs = spatial.find_affected_pixels(
            raster_data, obs_data
        )

        spatial.compute_spatial_pixel_adjustments(
            raster_data, obs_data, affected_flat_indices, affected_locs, geology_corr_fn
        )
        # The observation is at the centre pixel, so all 3x3 = 9 pixels are adjusted.
        assert "9/9" in capsys.readouterr().err

        spatial.compute_spatial_pixel_adjustments(
            raster_data,
            obs_data,
            affected_flat_indices,
            affected_locs,
            geology_corr_fn,
            show_progress=False,
        )
        assert capsys.readouterr().err == ""

"""
Tests for the VS30 parallel processing module.

Tests cover:
- resolve_n_proc function
- single_threaded_blas context manager
- Parallel processing infrastructure
"""

import multiprocessing as mp

import numpy as np
import pandas as pd
import pytest

from vs30.parallel import (
    resolve_n_proc,
    single_threaded_blas,
)


class TestResolveNProc:
    """Tests for the resolve_n_proc function."""

    def test_none_returns_one(self):
        """Test that None returns 1 (single process)."""
        assert resolve_n_proc(None) == 1

    def test_one_returns_one(self):
        """Test that 1 returns 1."""
        assert resolve_n_proc(1) == 1

    def test_minus_one_returns_cpu_count(self):
        """Test that -1 returns CPU count."""
        expected = mp.cpu_count()
        assert resolve_n_proc(-1) == expected

    def test_explicit_value_returned(self):
        """Test that explicit values are returned (up to CPU count)."""
        # Small value should be returned as-is
        assert resolve_n_proc(2) == min(2, mp.cpu_count())
        assert resolve_n_proc(4) == min(4, mp.cpu_count())

    def test_large_value_capped_at_cpu_count(self):
        """Test that values larger than CPU count are capped."""
        large_value = mp.cpu_count() + 100
        assert resolve_n_proc(large_value) == mp.cpu_count()

    def test_zero_raises_error(self):
        """Test that 0 raises ValueError."""
        with pytest.raises(ValueError):
            resolve_n_proc(0)

    def test_negative_less_than_minus_one_raises(self):
        """Test that values < -1 raise ValueError."""
        with pytest.raises(ValueError):
            resolve_n_proc(-2)

        with pytest.raises(ValueError):
            resolve_n_proc(-100)


class TestSingleThreadedBlas:
    """Tests for the single_threaded_blas context manager."""

    def test_context_manager_runs(self):
        """Test that context manager runs without error."""
        with single_threaded_blas():
            # Just verify it executes
            result = np.dot(np.array([1, 2, 3]), np.array([4, 5, 6]))
            assert result == 32

    def test_numpy_operations_work(self):
        """Test that numpy operations work inside context."""
        with single_threaded_blas():
            # Matrix multiplication (uses BLAS)
            a = np.random.rand(100, 100)
            b = np.random.rand(100, 100)
            c = np.dot(a, b)

            assert c.shape == (100, 100)

    def test_context_manager_is_reentrant(self):
        """Test that context manager can be nested."""
        with single_threaded_blas():
            with single_threaded_blas():
                result = np.sum(np.array([1, 2, 3]))
                assert result == 6

    def test_eigenvalue_computation(self):
        """Test that eigenvalue computation works (BLAS-heavy)."""
        with single_threaded_blas():
            # Symmetric matrix for real eigenvalues
            a = np.random.rand(50, 50)
            a = (a + a.T) / 2  # Make symmetric

            eigenvalues, eigenvectors = np.linalg.eig(a)

            assert len(eigenvalues) == 50
            assert eigenvectors.shape == (50, 50)

    def test_linear_solve(self):
        """Test that linear system solving works."""
        with single_threaded_blas():
            # Solve Ax = b
            A = np.random.rand(20, 20)
            A = A + np.eye(20) * 10  # Make well-conditioned
            b = np.random.rand(20)

            x = np.linalg.solve(A, b)

            # Verify solution
            np.testing.assert_array_almost_equal(np.dot(A, x), b, decimal=10)


class TestParallelProcessingInfrastructure:
    """Tests for parallel processing infrastructure."""

    def test_spawn_context_available(self):
        """Test that spawn context is available."""
        from vs30.parallel import _spawn_context

        assert _spawn_context is not None
        # Should be able to create a Pool (just test creation, don't start)
        assert hasattr(_spawn_context, 'Pool')

    def test_multiprocessing_cpu_count(self):
        """Test that CPU count is available and reasonable."""
        cpu_count = mp.cpu_count()

        assert cpu_count >= 1
        assert cpu_count <= 1024  # Reasonable upper bound


# =============================================================================
# Additional Tests from Coverage Improvements
# =============================================================================


class TestLocationsChunkConfig:
    """Tests for LocationsChunkConfig dataclass."""

    def test_config_creation(self):
        """Test creating LocationsChunkConfig."""
        from vs30.parallel import LocationsChunkConfig

        config = LocationsChunkConfig(
            lon_column="longitude",
            lat_column="latitude",
            include_intermediate=True,
            combination_method="0.5",
            coast_distance_raster=None,
        )

        assert config.lon_column == "longitude"
        assert config.lat_column == "latitude"
        assert config.include_intermediate is True
        assert config.combination_method == "0.5"
        assert config.coast_distance_raster is None


class TestProcessLocationsChunkDirect:
    """Tests for _process_locations_chunk worker function (called directly)."""

    def test_process_empty_chunk(self):
        """Test processing an empty chunk."""
        from vs30.parallel import _process_locations_chunk, LocationsChunkConfig

        # Empty DataFrame
        chunk_df = pd.DataFrame({
            'longitude': [],
            'latitude': [],
        })

        observations_df = pd.DataFrame({
            'easting': [],
            'northing': [],
            'vs30': [],
            'uncertainty': [],
        })

        geol_model_df = pd.DataFrame({
            'id': [1, 2],
            'mean_vs30_km_per_s': [300.0, 400.0],
            'standard_deviation_vs30_km_per_s': [30.0, 40.0],
        })

        config = LocationsChunkConfig(
            lon_column='longitude',
            lat_column='latitude',
            include_intermediate=False,
            combination_method='0.5',
            coast_distance_raster=None,
        )

        # This should handle empty chunk gracefully
        args = (chunk_df, 0, observations_df, geol_model_df, geol_model_df, config)

        chunk_id, result_df = _process_locations_chunk(args)

        assert chunk_id == 0
        assert len(result_df) == 0


class TestProcessPixelsChunkDirect:
    """Tests for _process_pixels_chunk worker function (called directly)."""

    def test_process_single_pixel(self):
        """Test processing a single pixel chunk."""
        from vs30.parallel import _process_pixels_chunk

        # Simple pixel data
        pixel_data_dict = {
            0: {
                "location": np.array([1500000.0, 5100000.0]),
                "vs30": 300.0,
                "stdv": 30.0,
                "index": 0,
            }
        }

        # Observation data dict
        obs_data_dict = {
            "locations": np.array([[1500100.0, 5100100.0]]),
            "vs30": np.array([350.0]),
            "model_vs30": np.array([300.0]),
            "model_stdv": np.array([30.0]),
            "residuals": np.array([0.15]),
            "omega": np.array([1.0]),
            "uncertainty": np.array([25.0]),
        }

        config_params = {
            "model_type": "geology",
            "phi": 1000.0,
            "max_dist_m": 5000.0,
            "max_points": 50,
            "noisy": False,
            "cov_reduc": 0.0,
        }

        args = ([0], 0, pixel_data_dict, obs_data_dict, config_params)

        chunk_id, updates = _process_pixels_chunk(args)

        assert chunk_id == 0
        # Should produce one update for the pixel
        assert len(updates) == 1
        assert updates[0].pixel_index == 0

    def test_process_multiple_pixels(self):
        """Test processing multiple pixels."""
        from vs30.parallel import _process_pixels_chunk

        # Multiple pixels
        pixel_data_dict = {
            0: {
                "location": np.array([1500000.0, 5100000.0]),
                "vs30": 300.0,
                "stdv": 30.0,
                "index": 0,
            },
            1: {
                "location": np.array([1500500.0, 5100500.0]),
                "vs30": 350.0,
                "stdv": 35.0,
                "index": 1,
            },
        }

        obs_data_dict = {
            "locations": np.array([[1500100.0, 5100100.0]]),
            "vs30": np.array([320.0]),
            "model_vs30": np.array([300.0]),
            "model_stdv": np.array([30.0]),
            "residuals": np.array([0.065]),
            "omega": np.array([1.0]),
            "uncertainty": np.array([25.0]),
        }

        config_params = {
            "model_type": "terrain",
            "phi": 993.0,
            "max_dist_m": 5000.0,
            "max_points": 50,
            "noisy": True,
            "cov_reduc": 0.5,
        }

        args = ([0, 1], 0, pixel_data_dict, obs_data_dict, config_params)

        chunk_id, updates = _process_pixels_chunk(args)

        assert chunk_id == 0
        assert len(updates) == 2

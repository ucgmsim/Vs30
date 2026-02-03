"""
Tests for the VS30 parallel processing module.

Tests cover:
- resolve_n_proc function
- Parallel processing worker functions
"""

import multiprocessing as mp

import numpy as np
import pandas as pd
import pytest

from vs30 import parallel


class TestResolveNProc:
    """Tests for the resolve_n_proc function."""

    def test_none_returns_one(self):
        """Test that None returns 1 (single process)."""
        assert parallel.resolve_n_proc(None) == 1

    def test_one_returns_one(self):
        """Test that 1 returns 1."""
        assert parallel.resolve_n_proc(1) == 1

    def test_minus_one_returns_cpu_count(self):
        """Test that -1 returns CPU count."""
        expected = mp.cpu_count()
        assert parallel.resolve_n_proc(-1) == expected

    def test_explicit_value_returned(self):
        """Test that explicit values are returned (up to CPU count)."""
        # Small value should be returned as-is
        assert parallel.resolve_n_proc(2) == min(2, mp.cpu_count())
        assert parallel.resolve_n_proc(4) == min(4, mp.cpu_count())

    def test_large_value_capped_at_cpu_count(self):
        """Test that values larger than CPU count are capped."""
        large_value = mp.cpu_count() + 100
        assert parallel.resolve_n_proc(large_value) == mp.cpu_count()

    def test_zero_raises_error(self):
        """Test that 0 raises ValueError."""
        with pytest.raises(ValueError):
            parallel.resolve_n_proc(0)

    def test_negative_less_than_minus_one_raises(self):
        """Test that values < -1 raise ValueError."""
        with pytest.raises(ValueError):
            parallel.resolve_n_proc(-2)

        with pytest.raises(ValueError):
            parallel.resolve_n_proc(-100)


class TestProcessLocationsChunkDirect:
    """Tests for process_locations_chunk worker function (called directly)."""

    def test_process_empty_chunk(self):
        """Test processing an empty chunk."""
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

        loc_config = parallel.LocationsChunkConfig(
            lon_column='longitude',
            lat_column='latitude',
            include_intermediate=False,
            combination_method='0.5',
            coast_distance_raster=None,
            noisy=False,
        )

        # This should handle empty chunk gracefully
        args = (chunk_df, 0, observations_df, geol_model_df, geol_model_df, loc_config)

        chunk_id, result_df = parallel.process_locations_chunk(args)

        assert chunk_id == 0
        assert len(result_df) == 0


class TestProcessPixelsChunkDirect:
    """Tests for process_pixels_chunk worker function (called directly)."""

    def test_process_single_pixel(self):
        """Test processing a single pixel chunk."""
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

        chunk_id, updates = parallel.process_pixels_chunk(args)

        assert chunk_id == 0
        # Should produce one update for the pixel
        assert len(updates) == 1
        assert updates[0].pixel_index == 0

    def test_process_multiple_pixels(self):
        """Test processing multiple pixels."""
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

        chunk_id, updates = parallel.process_pixels_chunk(args)

        assert chunk_id == 0
        assert len(updates) == 2

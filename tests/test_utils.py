"""
Tests for the VS30 utils module.

Tests cover:
- euclidean_distance_matrix function
- correlation_function
- load_config function
- _resolve_base_path function
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import yaml

from vs30.utils import (
    euclidean_distance_matrix,
    correlation_function,
    load_config,
    _resolve_base_path,
)


class TestEuclideanDistanceMatrix:
    """Tests for the euclidean_distance_matrix function."""

    def test_basic_distance(self):
        """Test basic distance calculation."""
        points = np.array([
            [0, 0],
            [3, 4],  # Distance from origin = 5
        ])

        dist_matrix = euclidean_distance_matrix(points)

        assert dist_matrix.shape == (2, 2)
        assert dist_matrix[0, 0] == 0  # Distance to self
        assert dist_matrix[1, 1] == 0  # Distance to self
        assert np.isclose(dist_matrix[0, 1], 5.0)  # 3-4-5 triangle
        assert np.isclose(dist_matrix[1, 0], 5.0)  # Symmetric

    def test_symmetry(self):
        """Test that distance matrix is symmetric."""
        points = np.array([
            [0, 0],
            [100, 200],
            [500, 300],
        ])

        dist_matrix = euclidean_distance_matrix(points)

        # Should be symmetric
        np.testing.assert_array_almost_equal(dist_matrix, dist_matrix.T)

    def test_diagonal_is_zero(self):
        """Test that diagonal elements are zero."""
        points = np.random.rand(10, 2) * 1000

        dist_matrix = euclidean_distance_matrix(points)

        np.testing.assert_array_equal(np.diag(dist_matrix), np.zeros(10))

    def test_triangle_inequality(self):
        """Test that triangle inequality holds."""
        points = np.array([
            [0, 0],
            [1, 0],
            [0.5, 0.5],
        ])

        dist_matrix = euclidean_distance_matrix(points)

        # d(0,2) + d(2,1) >= d(0,1)
        assert dist_matrix[0, 2] + dist_matrix[2, 1] >= dist_matrix[0, 1] - 1e-10

    def test_single_point(self):
        """Test with single point."""
        points = np.array([[100, 200]])

        dist_matrix = euclidean_distance_matrix(points)

        assert dist_matrix.shape == (1, 1)
        assert dist_matrix[0, 0] == 0

    def test_large_distances(self):
        """Test with realistic NZ coordinates (large values)."""
        # NZTM coordinates
        points = np.array([
            [1570604, 5180029],  # Christchurch
            [1757209, 5920482],  # Auckland
        ])

        dist_matrix = euclidean_distance_matrix(points)

        # Distance should be roughly 750km
        expected_dist = np.sqrt((1757209 - 1570604)**2 + (5920482 - 5180029)**2)
        assert np.isclose(dist_matrix[0, 1], expected_dist)


class TestCorrelationFunction:
    """Tests for the correlation_function."""

    def test_zero_distance_high_correlation(self):
        """Test that near-zero distance gives high correlation."""
        distances = np.array([0.1])  # Very small distance
        phi = 1000

        corr = correlation_function(distances, phi)

        # Should be very close to 1
        assert corr[0] > 0.99

    def test_large_distance_low_correlation(self):
        """Test that large distance gives low correlation."""
        distances = np.array([10000])  # 10km
        phi = 1000  # 1km correlation length

        corr = correlation_function(distances, phi)

        # Should be close to exp(-10) ≈ 0.000045
        assert corr[0] < 0.01

    def test_correlation_at_phi(self):
        """Test correlation at distance = phi."""
        phi = 1000
        distances = np.array([phi])

        corr = correlation_function(distances, phi)

        # At distance = phi, correlation should be 1/e ≈ 0.368
        expected = 1 / np.exp(1)
        assert np.isclose(corr[0], expected, rtol=0.01)

    def test_correlation_decreases_with_distance(self):
        """Test that correlation decreases with distance."""
        distances = np.array([100, 500, 1000, 2000, 5000])
        phi = 1000

        corr = correlation_function(distances, phi)

        # Correlations should be monotonically decreasing
        for i in range(len(corr) - 1):
            assert corr[i] > corr[i + 1]

    def test_correlation_range(self):
        """Test that correlation is always between 0 and 1."""
        distances = np.random.rand(100) * 50000  # 0 to 50km
        phi = 1000

        corr = correlation_function(distances, phi)

        assert (corr >= 0).all()
        assert (corr <= 1).all()

    def test_matrix_input(self):
        """Test with 2D distance matrix input."""
        distances = np.array([
            [0, 1000, 2000],
            [1000, 0, 1000],
            [2000, 1000, 0],
        ])
        phi = 1000

        corr = correlation_function(distances, phi)

        assert corr.shape == (3, 3)
        # Diagonal should have highest correlation
        assert (np.diag(corr) > 0.99).all()

    def test_different_phi_values(self):
        """Test that larger phi gives higher correlation at same distance."""
        distance = np.array([1000])

        corr_small_phi = correlation_function(distance, phi=500)
        corr_large_phi = correlation_function(distance, phi=2000)

        # Larger phi means slower decay, so higher correlation at same distance
        assert corr_large_phi[0] > corr_small_phi[0]


class TestLoadConfig:
    """Tests for the load_config function."""

    def test_load_valid_yaml(self):
        """Test loading a valid YAML config."""
        config_data = {
            "key1": "value1",
            "key2": 42,
            "nested": {"a": 1, "b": 2},
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            loaded = load_config(config_path)

            assert loaded["key1"] == "value1"
            assert loaded["key2"] == 42
            assert loaded["nested"]["a"] == 1
        finally:
            config_path.unlink()

    def test_load_nonexistent_file_raises(self):
        """Test that loading nonexistent file raises error."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("/nonexistent/config.yaml"))


class TestResolveBasePath:
    """Tests for the _resolve_base_path function."""

    def test_vs30_config_path(self):
        """Test with standard vs30/config.yaml path."""
        config_path = Path("/some/project/vs30/config.yaml")

        base_path = _resolve_base_path(config_path)

        assert base_path == Path("/some/project")

    def test_custom_config_path(self):
        """Test with custom config path."""
        config_path = Path("/custom/location/my_config.yaml")

        base_path = _resolve_base_path(config_path)

        # Should return parent of config file
        assert base_path == Path("/custom/location")

    def test_config_in_different_dir(self):
        """Test with config.yaml not in vs30 directory."""
        config_path = Path("/project/configs/config.yaml")

        base_path = _resolve_base_path(config_path)

        # Parent is "configs", not "vs30", so return parent of config
        assert base_path == Path("/project/configs")

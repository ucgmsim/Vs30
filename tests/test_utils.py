"""
Tests for the VS30 utils module.

Tests cover:
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
    correlation_function,
    load_config,
    _resolve_base_path,
)


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


# =============================================================================
# Additional Tests from Coverage Improvements
# =============================================================================


class TestUtilsEdgeCases:
    """Additional edge case tests for utils module."""

    def test_correlation_at_very_large_distance(self):
        """Test correlation function at very large distances approaches zero."""
        distances = np.array([1000000.0])  # 1000 km
        phi = 1000.0  # 1 km correlation length

        corr = correlation_function(distances, phi)

        assert corr[0] < 0.001  # Should be essentially zero

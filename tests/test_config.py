"""
Tests for the VS30 Pydantic configuration system.

These tests verify that:
- Configuration loads correctly from YAML
- Validation works for required fields and types
- Default config can be loaded
- Custom configs can be provided via CLI
"""

import tempfile
from pathlib import Path

import pytest
import yaml

from vs30 import config
from conftest import reset_default_config


class TestVs30Config:
    """Tests for the Vs30Config Pydantic model."""

    def test_load_default_config(self):
        """Test that default config loads successfully."""
        reset_default_config()
        cfg = config.get_default_config()

        assert cfg is not None
        assert isinstance(cfg, config.Vs30Config)

    def test_load_from_yaml(self):
        """Test loading config from a YAML file."""
        # Create a minimal valid config (only fields still in config.py)
        config_data = {
            "n_proc": 1,
            "grid_xmin": 1000000,
            "grid_xmax": 1100000,
            "grid_ymin": 5000000,
            "grid_ymax": 5100000,
            "grid_dx": 100,
            "grid_dy": 100,
            "noisy": True,
            "max_spatial_boolean_array_memory_gb": 1.0,
            "obs_subsample_step_for_clustered": 100,
            "independent_observations_file": "none",
            "clustered_observations_file": "none",
            "output_dir": "/tmp/test",
            "combination_method": 1.0,
            "do_bayesian_update_of_geology_and_terrain_categorical_vs30_values": True,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            cfg = config.Vs30Config.from_yaml(config_path)
            assert cfg.grid_xmin == 1000000
            assert cfg.noisy is True
        finally:
            config_path.unlink()

    def test_missing_required_field_raises_error(self):
        """Test that missing required field raises validation error."""
        # Create config missing required field
        config_data = {
            "n_proc": 1,
            # Missing grid_xmin and other required fields
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            with pytest.raises(Exception):  # Pydantic ValidationError
                config.Vs30Config.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_file_not_found_raises_error(self):
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            config.Vs30Config.from_yaml(Path("/nonexistent/config.yaml"))


class TestConfigCaching:
    """Tests for config caching behavior."""

    def test_get_default_config_returns_same_instance(self):
        """Test that get_default_config returns cached instance."""
        reset_default_config()
        config1 = config.get_default_config()
        config2 = config.get_default_config()

        assert config1 is config2

    def test_reset_clears_cache(self):
        """Test that reset_default_config clears the cache."""
        config1 = config.get_default_config()
        reset_default_config()
        config2 = config.get_default_config()

        # After reset, a new instance should be created
        # (though values should be the same)
        assert config1 is not config2
        assert config1.grid_xmin == config2.grid_xmin

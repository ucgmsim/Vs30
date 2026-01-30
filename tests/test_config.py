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

from vs30.config import Vs30Config, get_default_config
from vs30.constants import (
    COV_REDUC,
    EPS,
    HYBRID_MOD6_DIST_MAX,
    HYBRID_MOD6_DIST_MIN,
    HYBRID_MOD6_VS30_MAX,
    HYBRID_MOD6_VS30_MIN,
    HYBRID_SIGMA_REDUCTION_FACTORS,
    HYBRID_VS30_PARAMS,
    K_VALUE,
    LOCATIONS_LAT_COLUMN,
    LOCATIONS_LON_COLUMN,
    MAX_DIST_M,
    MAX_POINTS,
    MIN_GROUP,
    MIN_SIGMA,
    N_PRIOR,
    NODATA_VALUE,
    OUTPUT_FILENAMES,
    PHI,
    RASTER_ID_NODATA_VALUE,
    WEIGHT_EPSILON_DIV_BY_ZERO,
)
from conftest import reset_default_config


class TestVs30Config:
    """Tests for the Vs30Config Pydantic model."""

    def test_load_default_config(self):
        """Test that default config loads successfully."""
        reset_default_config()
        config = get_default_config()

        assert config is not None
        assert isinstance(config, Vs30Config)

    def test_config_has_required_fields(self):
        """Test that config has all required fields."""
        config = get_default_config()

        # Grid parameters
        assert hasattr(config, "grid_xmin")
        assert hasattr(config, "grid_xmax")
        assert hasattr(config, "grid_ymin")
        assert hasattr(config, "grid_ymax")
        assert hasattr(config, "grid_dx")
        assert hasattr(config, "grid_dy")

        # Other required parameters
        assert hasattr(config, "noisy")
        assert hasattr(config, "combination_method")

    def test_config_types(self):
        """Test that config fields have correct types."""
        config = get_default_config()

        assert isinstance(config.grid_xmin, int)
        assert isinstance(config.grid_xmax, int)
        assert isinstance(config.noisy, bool)

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
            config = Vs30Config.from_yaml(config_path)
            assert config.grid_xmin == 1000000
            assert config.noisy is True
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
                Vs30Config.from_yaml(config_path)
        finally:
            config_path.unlink()

    def test_file_not_found_raises_error(self):
        """Test that missing config file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            Vs30Config.from_yaml(Path("/nonexistent/config.yaml"))


class TestConfigCaching:
    """Tests for config caching behavior."""

    def test_get_default_config_returns_same_instance(self):
        """Test that get_default_config returns cached instance."""
        reset_default_config()
        config1 = get_default_config()
        config2 = get_default_config()

        assert config1 is config2

    def test_reset_clears_cache(self):
        """Test that reset_default_config clears the cache."""
        config1 = get_default_config()
        reset_default_config()
        config2 = get_default_config()

        # After reset, a new instance should be created
        # (though values should be the same)
        assert config1 is not config2
        assert config1.grid_xmin == config2.grid_xmin


class TestConstants:
    """Tests for constants module values."""

    def test_phi_values(self):
        """Test that PHI constant has correct structure."""
        assert isinstance(PHI, dict)
        assert "geology" in PHI
        assert "terrain" in PHI
        assert PHI["geology"] > 0
        assert PHI["terrain"] > 0

    def test_output_filenames(self):
        """Test output_filenames constant."""
        assert isinstance(OUTPUT_FILENAMES, dict)
        assert "geology" in OUTPUT_FILENAMES
        assert "terrain" in OUTPUT_FILENAMES
        assert OUTPUT_FILENAMES["geology"].endswith(".tif")
        assert OUTPUT_FILENAMES["terrain"].endswith(".tif")

    def test_hybrid_params_structure(self):
        """Test that hybrid params have correct structure."""
        for param in HYBRID_VS30_PARAMS:
            assert hasattr(param, "gid")
            assert hasattr(param, "slope_limits")
            assert hasattr(param, "vs30_values")

            assert isinstance(param.gid, int)
            assert len(param.slope_limits) == 2
            assert len(param.vs30_values) == 2

    def test_hybrid_sigma_reduction_factors(self):
        """Test HYBRID_SIGMA_REDUCTION_FACTORS structure."""
        assert isinstance(HYBRID_SIGMA_REDUCTION_FACTORS, dict)
        for gid, factor in HYBRID_SIGMA_REDUCTION_FACTORS.items():
            assert isinstance(gid, int)
            assert isinstance(factor, float)
            assert 0 <= factor <= 1

    def test_bayesian_constants(self):
        """Test Bayesian update constants."""
        assert isinstance(N_PRIOR, int)
        assert N_PRIOR > 0
        assert isinstance(MIN_SIGMA, float)
        assert MIN_SIGMA > 0

    def test_dbscan_constants(self):
        """Test DBSCAN clustering constants."""
        assert isinstance(MIN_GROUP, int)
        assert MIN_GROUP > 0
        assert isinstance(EPS, float)
        assert EPS > 0

    def test_nodata_constants(self):
        """Test NoData constants."""
        assert isinstance(NODATA_VALUE, int)
        assert isinstance(RASTER_ID_NODATA_VALUE, int)

    def test_cov_reduc_constant(self):
        """Test covariance reduction constant."""
        assert isinstance(COV_REDUC, float)
        assert COV_REDUC >= 0

    def test_spatial_update_constants(self):
        """Test spatial update constants."""
        assert isinstance(MAX_DIST_M, int)
        assert MAX_DIST_M > 0
        assert isinstance(MAX_POINTS, int)
        assert MAX_POINTS > 0

    def test_model_combination_constants(self):
        """Test model combination constants."""
        assert isinstance(K_VALUE, float)
        assert K_VALUE > 0
        assert isinstance(WEIGHT_EPSILON_DIV_BY_ZERO, float)
        assert WEIGHT_EPSILON_DIV_BY_ZERO > 0

    def test_location_column_constants(self):
        """Test location column name constants."""
        assert isinstance(LOCATIONS_LON_COLUMN, str)
        assert isinstance(LOCATIONS_LAT_COLUMN, str)
        assert LOCATIONS_LON_COLUMN == "longitude"
        assert LOCATIONS_LAT_COLUMN == "latitude"


class TestConstantsEdgeCases:
    """Tests for constants module edge cases."""

    def test_hybrid_mod6_params(self):
        """Test accessing hybrid mod6 parameters."""
        assert HYBRID_MOD6_DIST_MIN is not None
        assert HYBRID_MOD6_DIST_MAX > HYBRID_MOD6_DIST_MIN
        assert HYBRID_MOD6_VS30_MIN is not None
        assert HYBRID_MOD6_VS30_MAX > HYBRID_MOD6_VS30_MIN

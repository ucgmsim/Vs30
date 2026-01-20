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

        # Spatial parameters
        assert hasattr(config, "max_dist_m")
        assert hasattr(config, "max_points")
        assert hasattr(config, "phi_geology")
        assert hasattr(config, "phi_terrain")

        # Bayesian parameters
        assert hasattr(config, "n_prior")
        assert hasattr(config, "min_sigma")

        # NoData values
        assert hasattr(config, "nodata_value")
        assert hasattr(config, "raster_id_nodata_value")

    def test_config_types(self):
        """Test that config fields have correct types."""
        config = get_default_config()

        assert isinstance(config.grid_xmin, int)
        assert isinstance(config.grid_xmax, int)
        assert isinstance(config.max_dist_m, int)
        assert isinstance(config.cov_reduc, float)
        assert isinstance(config.noisy, bool)
        assert isinstance(config.nztm_crs, str)
        assert isinstance(config.hybrid_vs30_params, list)

    def test_phi_property(self):
        """Test that phi property returns correct dictionary."""
        config = get_default_config()

        phi = config.phi
        assert isinstance(phi, dict)
        assert "geology" in phi
        assert "terrain" in phi
        assert phi["geology"] == config.phi_geology
        assert phi["terrain"] == config.phi_terrain

    def test_output_filenames_property(self):
        """Test output_filenames property."""
        config = get_default_config()

        filenames = config.output_filenames
        assert isinstance(filenames, dict)
        assert "geology" in filenames
        assert "terrain" in filenames

    def test_load_from_yaml(self):
        """Test loading config from a YAML file."""
        # Create a minimal valid config
        config_data = {
            "n_proc": 1,
            "grid_xmin": 1000000,
            "grid_xmax": 1100000,
            "grid_ymin": 5000000,
            "grid_ymax": 5100000,
            "grid_dx": 100,
            "grid_dy": 100,
            "full_nz_land_xmin": 1000000,
            "full_nz_land_xmax": 2000000,
            "full_nz_land_ymin": 4700000,
            "full_nz_land_ymax": 6300000,
            "max_dist_m": 10000,
            "max_points": 500,
            "cov_reduc": 1.5,
            "noisy": True,
            "max_spatial_boolean_array_memory_gb": 1.0,
            "obs_subsample_step_for_clustered": 100,
            "phi_geology": 1407,
            "phi_terrain": 993,
            "n_prior": 3,
            "min_sigma": 0.5,
            "min_group": 5,
            "eps": 15000.0,
            "raster_id_nodata_value": 255,
            "nodata_value": -32767,
            "independent_observations_file": "none",
            "clustered_observations_file": "none",
            "output_dir": "/tmp/test",
            "geology_mean_and_standard_deviation_per_category_file": "test.csv",
            "terrain_mean_and_standard_deviation_per_category_file": "test.csv",
            "posterior_prefix": "posterior_",
            "terrain_initial_vs30_filename": "terrain.tif",
            "geology_initial_vs30_filename": "geology.tif",
            "slope_raster_filename": "slope.tif",
            "coast_distance_raster_filename": "coast.tif",
            "geology_vs30_slope_and_coastal_distance_adjusted_filename": "hybrid.tif",
            "geology_id_filename": "gid.tif",
            "terrain_id_filename": "tid.tif",
            "terrain_vs30_mean_stddev_filename": "terrain_final.tif",
            "geology_vs30_mean_stddev_filename": "geology_final.tif",
            "combined_vs30_filename": "combined.tif",
            "combination_method": 1.0,
            "k_value": 3.0,
            "do_bayesian_update_of_geology_and_terrain_categorical_vs30_values": True,
            "hybrid_mod6_dist_min": 8000.0,
            "hybrid_mod6_dist_max": 20000.0,
            "hybrid_mod6_vs30_min": 240.0,
            "hybrid_mod6_vs30_max": 500.0,
            "hybrid_mod13_dist_min": 8000.0,
            "hybrid_mod13_dist_max": 20000.0,
            "hybrid_mod13_vs30_min": 197.0,
            "hybrid_mod13_vs30_max": 500.0,
            "hybrid_vs30_params": [
                {"gid": 2, "slope_limits": [-1.85, -1.22], "vs30_values": [242, 418]}
            ],
            "hybrid_sigma_reduction_factors": {2: 0.5},
            "min_slope_for_log": 1e-9,
            "min_dist_enforced": 0.1,
            "nztm_crs": "EPSG:2193",
            "plot_figsize": [12, 8],
            "plot_dpi": 300,
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".yaml", delete=False
        ) as f:
            yaml.dump(config_data, f)
            config_path = Path(f.name)

        try:
            config = Vs30Config.from_yaml(config_path)
            assert config.grid_xmin == 1000000
            assert config.max_dist_m == 10000
            assert config.phi_geology == 1407
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

    def test_backwards_compat_properties(self):
        """Test backwards compatibility property aliases."""
        config = get_default_config()

        # These should match the actual config values
        assert config.MODEL_NODATA == config.nodata_value
        assert config.RASTER_ID_NODATA_VALUE == config.raster_id_nodata_value
        assert config.GEOLOGY_MEAN_STDDEV_CSV == config.geology_mean_and_standard_deviation_per_category_file
        assert config.TERRAIN_MEAN_STDDEV_CSV == config.terrain_mean_and_standard_deviation_per_category_file


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
        assert config1.max_dist_m == config2.max_dist_m


class TestHybridVs30Param:
    """Tests for HybridVs30Param model."""

    def test_hybrid_params_structure(self):
        """Test that hybrid params have correct structure."""
        config = get_default_config()

        for param in config.hybrid_vs30_params:
            assert hasattr(param, "gid")
            assert hasattr(param, "slope_limits")
            assert hasattr(param, "vs30_values")

            assert isinstance(param.gid, int)
            assert len(param.slope_limits) == 2
            assert len(param.vs30_values) == 2

    def test_hybrid_sigma_reduction_factors(self):
        """Test hybrid_sigma_reduction_factors structure."""
        config = get_default_config()

        assert isinstance(config.hybrid_sigma_reduction_factors, dict)
        for gid, factor in config.hybrid_sigma_reduction_factors.items():
            # Keys are strings in YAML but should be usable as ints
            assert isinstance(factor, float)
            assert 0 <= factor <= 1


# =============================================================================
# Additional Tests from Coverage Improvements
# =============================================================================


class TestConfigEdgeCases:
    """Tests for config module edge cases."""

    def test_config_phi_access(self):
        """Test accessing phi values from config."""
        cfg = get_default_config()

        assert "geology" in cfg.phi
        assert "terrain" in cfg.phi
        assert cfg.phi["geology"] > 0
        assert cfg.phi["terrain"] > 0

    def test_config_hybrid_params(self):
        """Test accessing hybrid parameters."""
        cfg = get_default_config()

        assert cfg.hybrid_mod6_dist_min is not None
        assert cfg.hybrid_mod6_dist_max > cfg.hybrid_mod6_dist_min
        assert cfg.hybrid_mod6_vs30_min is not None
        assert cfg.hybrid_mod6_vs30_max > cfg.hybrid_mod6_vs30_min

    def test_config_output_filenames(self):
        """Test accessing output filenames."""
        cfg = get_default_config()

        assert "geology" in cfg.output_filenames
        assert "terrain" in cfg.output_filenames
        assert cfg.output_filenames["geology"].endswith(".tif")
        assert cfg.output_filenames["terrain"].endswith(".tif")

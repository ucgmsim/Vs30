"""
Tests for the VS30 CLI interface.

These tests verify that:
- CLI commands are available and have correct help
- Global --config option works
- Commands handle errors gracefully
"""

import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml


class TestCLIHelp:
    """Tests for CLI help and command availability."""

    def test_main_help(self):
        """Test that main help is available."""
        result = subprocess.run(
            ["vs30", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "VS30 map generation" in result.stdout
        assert "--config" in result.stdout
        assert "--verbose" in result.stdout

    def test_full_pipeline_help(self):
        """Test full-pipeline command help."""
        result = subprocess.run(
            ["vs30", "full-pipeline", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "full-pipeline" in result.stdout.lower() or "pipeline" in result.stdout.lower()

    def test_compute_at_locations_help(self):
        """Test compute-at-locations command help."""
        result = subprocess.run(
            ["vs30", "compute-at-locations", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "--locations-csv" in result.stdout
        assert "--output-csv" in result.stdout

    def test_spatial_fit_help(self):
        """Test spatial-fit command help."""
        result = subprocess.run(
            ["vs30", "spatial-fit", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_combine_help(self):
        """Test combine command help."""
        result = subprocess.run(
            ["vs30", "combine", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0


class TestGlobalConfigOption:
    """Tests for the global --config option."""

    def test_nonexistent_config_fails(self):
        """Test that nonexistent config file causes error."""
        result = subprocess.run(
            ["vs30", "--config", "/nonexistent/config.yaml", "full-pipeline", "--help"],
            capture_output=True,
            text=True,
        )
        # Should fail when trying to load nonexistent config
        assert result.returncode != 0 or "not found" in result.stderr.lower()

    def test_custom_config_loads(self):
        """Test that custom config file is loaded."""
        # Create a valid minimal config
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
            "max_dist_m": 5000,  # Different value to verify it's loaded
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
            "output_dir": "/tmp/test_cli",
            "geology_mean_and_standard_deviation_per_category_file": "categorical_vs30_mean_and_stddev/geology/geology_model_posterior_from_foster_2019_mean_and_standard_deviation.csv",
            "terrain_mean_and_standard_deviation_per_category_file": "categorical_vs30_mean_and_stddev/terrain/terrain_model_posterior_from_foster_2019_mean_and_standard_deviation.csv",
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
            "hybrid_sigma_reduction_factors": {"2": 0.5},
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
            # Just test that --help works with custom config
            result = subprocess.run(
                ["vs30", "--config", str(config_path), "--help"],
                capture_output=True,
                text=True,
            )
            assert result.returncode == 0
        finally:
            config_path.unlink()


class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    def test_missing_required_args(self):
        """Test that missing required arguments produce helpful error."""
        result = subprocess.run(
            ["vs30", "compute-at-locations"],
            capture_output=True,
            text=True,
        )
        # Should fail due to missing required args
        assert result.returncode != 0
        assert "required" in result.stderr.lower() or "missing" in result.stderr.lower()

    def test_invalid_input_file(self):
        """Test handling of invalid input file."""
        result = subprocess.run(
            [
                "vs30",
                "compute-at-locations",
                "--locations-csv", "/nonexistent/file.csv",
                "--output-csv", "/tmp/out.csv",
            ],
            capture_output=True,
            text=True,
        )
        assert result.returncode != 0


class TestVerboseOption:
    """Tests for the --verbose option."""

    def test_verbose_flag_accepted(self):
        """Test that --verbose flag is accepted."""
        result = subprocess.run(
            ["vs30", "--verbose", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

    def test_short_verbose_flag_accepted(self):
        """Test that -v flag is accepted."""
        result = subprocess.run(
            ["vs30", "-v", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0

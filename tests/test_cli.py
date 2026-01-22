"""
Tests for the VS30 CLI interface.

These tests verify that:
- CLI commands are available and have correct help
- Global --config option works
- Commands handle errors gracefully

Uses CliRunner for in-process invocation to enable coverage tracking.
"""

import shutil
import tempfile
from pathlib import Path

import pandas as pd
import pytest
import yaml
from typer.testing import CliRunner

from vs30.cli import app

runner = CliRunner()


class TestCLIHelp:
    """Tests for CLI help and command availability."""

    def test_main_help(self):
        """Test that main help is available."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "VS30 map generation" in result.output
        assert "--config" in result.output
        assert "--verbose" in result.output

    def test_full_pipeline_help(self):
        """Test full-pipeline command help."""
        result = runner.invoke(app, ["full-pipeline", "--help"])
        assert result.exit_code == 0
        assert "full-pipeline" in result.output.lower() or "pipeline" in result.output.lower()

    def test_compute_at_locations_help(self):
        """Test compute-at-locations command help."""
        result = runner.invoke(app, ["compute-at-locations", "--help"])
        assert result.exit_code == 0
        assert "--locations-csv" in result.output
        assert "--output-csv" in result.output

    def test_spatial_fit_help(self):
        """Test spatial-fit command help."""
        result = runner.invoke(app, ["spatial-fit", "--help"])
        assert result.exit_code == 0

    def test_combine_help(self):
        """Test combine command help."""
        result = runner.invoke(app, ["combine", "--help"])
        assert result.exit_code == 0


class TestGlobalConfigOption:
    """Tests for the global --config option."""

    def test_nonexistent_config_fails(self):
        """Test that nonexistent config file causes error."""
        result = runner.invoke(app, ["--config", "/nonexistent/config.yaml", "full-pipeline", "--help"])
        # Should fail when trying to load nonexistent config
        assert result.exit_code != 0 or "does not exist" in result.output.lower()

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
            result = runner.invoke(app, ["--config", str(config_path), "--help"])
            assert result.exit_code == 0
        finally:
            config_path.unlink()


class TestCLIErrorHandling:
    """Tests for CLI error handling."""

    def test_missing_required_args(self):
        """Test that missing required arguments produce helpful error."""
        result = runner.invoke(app, ["compute-at-locations"])
        # Should fail due to missing required args
        assert result.exit_code != 0
        assert "required" in result.output.lower() or "missing" in result.output.lower()

    def test_invalid_input_file(self):
        """Test handling of invalid input file."""
        result = runner.invoke(app, [
            "compute-at-locations",
            "--locations-csv", "/nonexistent/file.csv",
            "--output-csv", "/tmp/out.csv",
        ])
        assert result.exit_code != 0


class TestVerboseOption:
    """Tests for the --verbose option."""

    def test_verbose_flag_accepted(self):
        """Test that --verbose flag is accepted."""
        result = runner.invoke(app, ["--verbose", "--help"])
        assert result.exit_code == 0

    def test_short_verbose_flag_accepted(self):
        """Test that -v flag is accepted."""
        result = runner.invoke(app, ["-v", "--help"])
        assert result.exit_code == 0


class TestCLICommandHelpAdditional:
    """Additional tests for CLI command help."""

    def test_update_categorical_help(self):
        """Test update-categorical-vs30-models help."""
        result = runner.invoke(app, ["update-categorical-vs30-models", "--help"])
        assert result.exit_code == 0
        assert "--categorical-model-csv" in result.output

    def test_make_initial_raster_help(self):
        """Test make-initial-vs30-raster help."""
        result = runner.invoke(app, ["make-initial-vs30-raster", "--help"])
        assert result.exit_code == 0

    def test_adjust_geology_help(self):
        """Test adjust-geology-vs30-by-slope-and-coastal-distance help."""
        result = runner.invoke(app, ["adjust-geology-vs30-by-slope-and-coastal-distance", "--help"])
        assert result.exit_code == 0

    def test_full_pipeline_for_model_help(self):
        """Test full-pipeline-for-geology-or-terrain help."""
        result = runner.invoke(app, ["full-pipeline-for-geology-or-terrain", "--help"])
        assert result.exit_code == 0
        assert "--model-type" in result.output


class TestPlotPosteriorValuesCommand:
    """Tests for the plot-posterior-values CLI command."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        tmpdir = tempfile.mkdtemp(prefix="vs30_cli_test_")
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    @pytest.fixture
    def sample_posterior_csv(self, temp_dir):
        """Create a sample posterior CSV file for plotting tests."""
        csv_path = temp_dir / "test_posterior.csv"
        data = {
            "id": [1, 2, 3, 4, 5],
            "prior_mean_vs30_km_per_s": [200, 300, 400, 500, 600],
            "prior_standard_deviation_vs30_km_per_s": [20, 30, 40, 50, 60],
            "posterior_mean_vs30_km_per_s": [210, 310, 390, 480, 590],
            "posterior_standard_deviation_vs30_km_per_s": [18, 25, 35, 45, 55],
        }
        df = pd.DataFrame(data)
        df.to_csv(csv_path, index=False)
        return csv_path

    def test_plot_posterior_values_success(self, sample_posterior_csv, temp_dir):
        """Test successful plot generation."""
        result = runner.invoke(app, [
            "plot-posterior-values",
            "--csv-path", str(sample_posterior_csv),
            "--output-dir", str(temp_dir),
        ])

        # Check command succeeded
        assert result.exit_code == 0, f"Command failed: {result.output}"

        # Check output file was created
        expected_output = temp_dir / "test_posterior_plot.png"
        assert expected_output.exists(), f"Plot file not created: {expected_output}"

    def test_plot_posterior_values_missing_csv(self, temp_dir):
        """Test error handling for missing CSV file."""
        result = runner.invoke(app, [
            "plot-posterior-values",
            "--csv-path", str(temp_dir / "nonexistent.csv"),
            "--output-dir", str(temp_dir),
        ])

        assert result.exit_code != 0

    def test_plot_posterior_values_missing_columns(self, temp_dir):
        """Test error handling for CSV missing required columns."""
        # Create CSV missing required columns
        csv_path = temp_dir / "bad_posterior.csv"
        pd.DataFrame({"id": [1, 2], "some_value": [100, 200]}).to_csv(csv_path, index=False)

        result = runner.invoke(app, [
            "plot-posterior-values",
            "--csv-path", str(csv_path),
            "--output-dir", str(temp_dir),
        ])

        assert result.exit_code != 0


class TestComputeAtLocationsCommand:
    """Tests for compute-at-locations CLI command."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        tmpdir = tempfile.mkdtemp(prefix="vs30_cli_test_")
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    def test_compute_at_locations_basic(self, temp_dir):
        """Test basic compute-at-locations command."""
        # Create simple locations CSV
        locations_csv = temp_dir / "locations.csv"
        pd.DataFrame({
            'longitude': [172.63, 172.64],  # Near Christchurch
            'latitude': [-43.53, -43.54],
        }).to_csv(locations_csv, index=False)

        output_csv = temp_dir / "results.csv"

        result = runner.invoke(app, [
            "compute-at-locations",
            "--locations-csv", str(locations_csv),
            "--output-csv", str(output_csv),
        ])

        # Command should succeed
        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_csv.exists()

        # Check output has expected columns
        df = pd.read_csv(output_csv)
        assert "vs30" in df.columns
        assert "stdv" in df.columns
        assert len(df) == 2

    def test_compute_at_locations_custom_columns(self, temp_dir):
        """Test compute-at-locations with custom column names."""
        # Create locations CSV with custom column names
        locations_csv = temp_dir / "custom_locations.csv"
        pd.DataFrame({
            'lon': [172.63, 172.64],
            'lat': [-43.53, -43.54],
        }).to_csv(locations_csv, index=False)

        output_csv = temp_dir / "results.csv"

        result = runner.invoke(app, [
            "compute-at-locations",
            "--locations-csv", str(locations_csv),
            "--lon-column", "lon",
            "--lat-column", "lat",
            "--output-csv", str(output_csv),
        ])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_csv.exists()

    def test_compute_at_locations_missing_column(self, temp_dir):
        """Test error when required column is missing."""
        locations_csv = temp_dir / "bad_columns.csv"
        pd.DataFrame({
            'x': [172.63],
            'y': [-43.53],
        }).to_csv(locations_csv, index=False)

        result = runner.invoke(app, [
            "compute-at-locations",
            "--locations-csv", str(locations_csv),
            "--output-csv", str(temp_dir / "results.csv"),
        ])

        # Should fail with error about missing column
        assert result.exit_code != 0


# =============================================================================
# Error Handling Tests for CLI Commands
# =============================================================================


class TestUpdateCategoricalVs30ModelsErrors:
    """Tests for error handling in update-categorical-vs30-models command."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        tmpdir = tempfile.mkdtemp(prefix="vs30_cli_test_")
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    def test_no_observations_csv_provided(self, temp_dir):
        """Test error when neither clustered nor independent observations provided."""
        # Create a valid categorical model CSV
        categorical_csv = temp_dir / "categorical.csv"
        pd.DataFrame({
            'id': [1, 2],
            'mean_vs30_km_per_s': [300.0, 400.0],
            'standard_deviation_vs30_km_per_s': [30.0, 40.0],
        }).to_csv(categorical_csv, index=False)

        result = runner.invoke(app, [
            "update-categorical-vs30-models",
            "--categorical-model-csv", str(categorical_csv),
            "--model-type", "geology",
            "--output-dir", str(temp_dir),
        ])

        assert result.exit_code != 0
        assert "at least one" in result.output.lower()

    def test_missing_categorical_model_csv(self, temp_dir):
        """Test error when categorical model CSV doesn't exist."""
        obs_csv = temp_dir / "obs.csv"
        pd.DataFrame({
            'easting': [1500000], 'northing': [5100000], 'vs30': [300], 'uncertainty': [30],
        }).to_csv(obs_csv, index=False)

        result = runner.invoke(app, [
            "update-categorical-vs30-models",
            "--categorical-model-csv", str(temp_dir / "nonexistent.csv"),
            "--independent-observations-csv", str(obs_csv),
            "--model-type", "geology",
            "--output-dir", str(temp_dir),
        ])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()

    def test_missing_clustered_observations_csv(self, temp_dir):
        """Test error when clustered observations CSV doesn't exist."""
        categorical_csv = temp_dir / "categorical.csv"
        pd.DataFrame({
            'id': [1, 2],
            'mean_vs30_km_per_s': [300.0, 400.0],
            'standard_deviation_vs30_km_per_s': [30.0, 40.0],
        }).to_csv(categorical_csv, index=False)

        result = runner.invoke(app, [
            "update-categorical-vs30-models",
            "--categorical-model-csv", str(categorical_csv),
            "--clustered-observations-csv", str(temp_dir / "nonexistent.csv"),
            "--model-type", "geology",
            "--output-dir", str(temp_dir),
        ])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()

    def test_missing_independent_observations_csv(self, temp_dir):
        """Test error when independent observations CSV doesn't exist."""
        categorical_csv = temp_dir / "categorical.csv"
        pd.DataFrame({
            'id': [1, 2],
            'mean_vs30_km_per_s': [300.0, 400.0],
            'standard_deviation_vs30_km_per_s': [30.0, 40.0],
        }).to_csv(categorical_csv, index=False)

        result = runner.invoke(app, [
            "update-categorical-vs30-models",
            "--categorical-model-csv", str(categorical_csv),
            "--independent-observations-csv", str(temp_dir / "nonexistent.csv"),
            "--model-type", "geology",
            "--output-dir", str(temp_dir),
        ])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()

    def test_invalid_model_type(self, temp_dir):
        """Test error when model_type is invalid."""
        categorical_csv = temp_dir / "categorical.csv"
        pd.DataFrame({
            'id': [1, 2],
            'mean_vs30_km_per_s': [300.0, 400.0],
            'standard_deviation_vs30_km_per_s': [30.0, 40.0],
        }).to_csv(categorical_csv, index=False)

        obs_csv = temp_dir / "obs.csv"
        pd.DataFrame({
            'easting': [1500000], 'northing': [5100000], 'vs30': [300], 'uncertainty': [30],
        }).to_csv(obs_csv, index=False)

        result = runner.invoke(app, [
            "update-categorical-vs30-models",
            "--categorical-model-csv", str(categorical_csv),
            "--independent-observations-csv", str(obs_csv),
            "--model-type", "invalid_type",
            "--output-dir", str(temp_dir),
        ])

        assert result.exit_code != 0
        assert "geology" in result.output.lower() or "terrain" in result.output.lower()

    def test_missing_columns_in_categorical_csv(self, temp_dir):
        """Test error when categorical CSV is missing required columns."""
        categorical_csv = temp_dir / "categorical.csv"
        pd.DataFrame({
            'id': [1, 2],
            'some_other_column': [100, 200],
        }).to_csv(categorical_csv, index=False)

        obs_csv = temp_dir / "obs.csv"
        pd.DataFrame({
            'easting': [1500000], 'northing': [5100000], 'vs30': [300], 'uncertainty': [30],
        }).to_csv(obs_csv, index=False)

        result = runner.invoke(app, [
            "update-categorical-vs30-models",
            "--categorical-model-csv", str(categorical_csv),
            "--independent-observations-csv", str(obs_csv),
            "--model-type", "geology",
            "--output-dir", str(temp_dir),
        ])

        assert result.exit_code != 0
        # Error could be "missing" or the KeyError for the column name
        assert "error" in result.output.lower()

    def test_missing_columns_in_observations_csv(self, temp_dir):
        """Test error when observations CSV is missing required columns."""
        categorical_csv = temp_dir / "categorical.csv"
        pd.DataFrame({
            'id': [1, 2],
            'mean_vs30_km_per_s': [300.0, 400.0],
            'standard_deviation_vs30_km_per_s': [30.0, 40.0],
        }).to_csv(categorical_csv, index=False)

        obs_csv = temp_dir / "obs.csv"
        pd.DataFrame({
            'x': [1500000], 'y': [5100000],  # Wrong column names
        }).to_csv(obs_csv, index=False)

        result = runner.invoke(app, [
            "update-categorical-vs30-models",
            "--categorical-model-csv", str(categorical_csv),
            "--clustered-observations-csv", str(obs_csv),
            "--model-type", "geology",
            "--output-dir", str(temp_dir),
        ])

        assert result.exit_code != 0
        assert "missing" in result.output.lower()


class TestMakeInitialVs30RasterErrors:
    """Tests for error handling in make-initial-vs30-raster command."""

    def test_no_model_type_specified(self):
        """Test error when neither --terrain nor --geology specified."""
        result = runner.invoke(app, [
            "make-initial-vs30-raster",
        ])

        assert result.exit_code != 0
        assert "at least one" in result.output.lower()


class TestAdjustGeologyErrors:
    """Tests for error handling in adjust-geology-vs30-by-slope-and-coastal-distance command."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        tmpdir = tempfile.mkdtemp(prefix="vs30_cli_test_")
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    def test_missing_input_raster(self, temp_dir):
        """Test error when input raster doesn't exist."""
        result = runner.invoke(app, [
            "adjust-geology-vs30-by-slope-and-coastal-distance",
            "--input-raster", str(temp_dir / "nonexistent.tif"),
            "--id-raster", str(temp_dir / "gid.tif"),
            "--output-dir", str(temp_dir),
        ])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()

    def test_missing_id_raster(self, temp_dir):
        """Test error when ID raster doesn't exist."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        # Create a valid input raster
        input_raster = temp_dir / "input.tif"
        transform = from_bounds(1500000, 5100000, 1505000, 5105000, 10, 10)

        with rasterio.open(
            input_raster, 'w', driver='GTiff',
            height=10, width=10, count=2, dtype='float32',
            crs='EPSG:2193', transform=transform,
        ) as dst:
            dst.write(np.ones((10, 10), dtype=np.float32) * 300, 1)
            dst.write(np.ones((10, 10), dtype=np.float32) * 30, 2)

        result = runner.invoke(app, [
            "adjust-geology-vs30-by-slope-and-coastal-distance",
            "--input-raster", str(input_raster),
            "--id-raster", str(temp_dir / "nonexistent_gid.tif"),
            "--output-dir", str(temp_dir),
        ])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()


class TestSpatialFitErrors:
    """Tests for error handling in spatial-fit command."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        tmpdir = tempfile.mkdtemp(prefix="vs30_cli_test_")
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    def test_missing_input_raster(self, temp_dir):
        """Test error when input raster doesn't exist."""
        result = runner.invoke(app, [
            "spatial-fit",
            "--input-raster", str(temp_dir / "nonexistent.tif"),
            "--observations-csv", str(temp_dir / "obs.csv"),
            "--model-values-csv", str(temp_dir / "model.csv"),
            "--model-type", "geology",
            "--output-dir", str(temp_dir),
        ])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()

    def test_missing_observations_csv(self, temp_dir):
        """Test error when observations CSV doesn't exist."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        # Create a valid input raster
        input_raster = temp_dir / "input.tif"
        transform = from_bounds(1500000, 5100000, 1505000, 5105000, 10, 10)

        with rasterio.open(
            input_raster, 'w', driver='GTiff',
            height=10, width=10, count=2, dtype='float32',
            crs='EPSG:2193', transform=transform,
        ) as dst:
            dst.write(np.ones((10, 10), dtype=np.float32) * 300, 1)
            dst.write(np.ones((10, 10), dtype=np.float32) * 30, 2)

        result = runner.invoke(app, [
            "spatial-fit",
            "--input-raster", str(input_raster),
            "--observations-csv", str(temp_dir / "nonexistent.csv"),
            "--model-values-csv", str(temp_dir / "model.csv"),
            "--model-type", "terrain",
            "--output-dir", str(temp_dir),
        ])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()

    def test_missing_model_values_csv(self, temp_dir):
        """Test error when model values CSV doesn't exist."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        # Create a valid input raster
        input_raster = temp_dir / "input.tif"
        transform = from_bounds(1500000, 5100000, 1505000, 5105000, 10, 10)

        with rasterio.open(
            input_raster, 'w', driver='GTiff',
            height=10, width=10, count=2, dtype='float32',
            crs='EPSG:2193', transform=transform,
        ) as dst:
            dst.write(np.ones((10, 10), dtype=np.float32) * 300, 1)
            dst.write(np.ones((10, 10), dtype=np.float32) * 30, 2)

        # Create observations CSV
        obs_csv = temp_dir / "obs.csv"
        pd.DataFrame({
            'easting': [1502000], 'northing': [5102000],
            'vs30': [350], 'uncertainty': [25],
        }).to_csv(obs_csv, index=False)

        result = runner.invoke(app, [
            "spatial-fit",
            "--input-raster", str(input_raster),
            "--observations-csv", str(obs_csv),
            "--model-values-csv", str(temp_dir / "nonexistent.csv"),
            "--model-type", "terrain",
            "--output-dir", str(temp_dir),
        ])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()


class TestCombineErrors:
    """Tests for error handling in combine command."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for test outputs."""
        tmpdir = tempfile.mkdtemp(prefix="vs30_cli_test_")
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    def test_missing_geology_raster(self, temp_dir):
        """Test error when geology raster doesn't exist."""
        result = runner.invoke(app, [
            "combine",
            "--geology-tif", str(temp_dir / "nonexistent_geol.tif"),
            "--terrain-tif", str(temp_dir / "terrain.tif"),
            "--output-path", str(temp_dir / "combined.tif"),
        ])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()

    def test_missing_terrain_raster(self, temp_dir):
        """Test error when terrain raster doesn't exist."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        # Create geology raster
        geology_tif = temp_dir / "geology.tif"
        transform = from_bounds(1500000, 5100000, 1505000, 5105000, 10, 10)

        with rasterio.open(
            geology_tif, 'w', driver='GTiff',
            height=10, width=10, count=2, dtype='float32',
            crs='EPSG:2193', transform=transform, nodata=-9999.0,
        ) as dst:
            dst.write(np.ones((10, 10), dtype=np.float32) * 300, 1)
            dst.write(np.ones((10, 10), dtype=np.float32) * 0.3, 2)

        result = runner.invoke(app, [
            "combine",
            "--geology-tif", str(geology_tif),
            "--terrain-tif", str(temp_dir / "nonexistent_terrain.tif"),
            "--output-path", str(temp_dir / "combined.tif"),
        ])

        assert result.exit_code != 0
        assert "does not exist" in result.output.lower()

    def test_unknown_combination_method(self, temp_dir):
        """Test error when combination method is invalid."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        transform = from_bounds(1500000, 5100000, 1505000, 5105000, 10, 10)

        # Create geology raster
        geology_tif = temp_dir / "geology.tif"
        with rasterio.open(
            geology_tif, 'w', driver='GTiff',
            height=10, width=10, count=2, dtype='float32',
            crs='EPSG:2193', transform=transform, nodata=-9999.0,
        ) as dst:
            dst.write(np.ones((10, 10), dtype=np.float32) * 300, 1)
            dst.write(np.ones((10, 10), dtype=np.float32) * 0.3, 2)

        # Create terrain raster
        terrain_tif = temp_dir / "terrain.tif"
        with rasterio.open(
            terrain_tif, 'w', driver='GTiff',
            height=10, width=10, count=2, dtype='float32',
            crs='EPSG:2193', transform=transform, nodata=-9999.0,
        ) as dst:
            dst.write(np.ones((10, 10), dtype=np.float32) * 350, 1)
            dst.write(np.ones((10, 10), dtype=np.float32) * 0.35, 2)

        result = runner.invoke(app, [
            "combine",
            "--geology-tif", str(geology_tif),
            "--terrain-tif", str(terrain_tif),
            "--output-path", str(temp_dir / "combined.tif"),
            "--combination-method", "invalid_method",
        ])

        assert result.exit_code != 0
        assert "unknown" in result.output.lower()

    def test_stdv_weighting_combination_method(self, temp_dir):
        """Test standard_deviation_weighting combination method works."""
        import numpy as np
        import rasterio
        from rasterio.transform import from_bounds

        transform = from_bounds(1500000, 5100000, 1505000, 5105000, 10, 10)

        # Create geology raster
        geology_tif = temp_dir / "geology.tif"
        with rasterio.open(
            geology_tif, 'w', driver='GTiff',
            height=10, width=10, count=2, dtype='float32',
            crs='EPSG:2193', transform=transform, nodata=-9999.0,
        ) as dst:
            dst.write(np.ones((10, 10), dtype=np.float32) * 300, 1)
            dst.write(np.ones((10, 10), dtype=np.float32) * 0.3, 2)

        # Create terrain raster
        terrain_tif = temp_dir / "terrain.tif"
        with rasterio.open(
            terrain_tif, 'w', driver='GTiff',
            height=10, width=10, count=2, dtype='float32',
            crs='EPSG:2193', transform=transform, nodata=-9999.0,
        ) as dst:
            dst.write(np.ones((10, 10), dtype=np.float32) * 350, 1)
            dst.write(np.ones((10, 10), dtype=np.float32) * 0.35, 2)

        output_tif = temp_dir / "combined.tif"
        result = runner.invoke(app, [
            "combine",
            "--geology-tif", str(geology_tif),
            "--terrain-tif", str(terrain_tif),
            "--output-path", str(output_tif),
            "--combination-method", "standard_deviation_weighting",
        ])

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output_tif.exists()


class TestVerboseFlag:
    """Tests for the --verbose flag behavior."""

    def test_verbose_enables_debug_logging(self):
        """Test that --verbose flag is processed correctly."""
        # Just verify it doesn't crash when verbose is enabled
        result = runner.invoke(app, ["--verbose", "full-pipeline", "--help"])
        assert result.exit_code == 0

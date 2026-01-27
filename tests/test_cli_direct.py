"""
Direct CLI tests using Typer's CliRunner for coverage.

These tests invoke CLI commands directly in-process rather than
via subprocess, allowing proper coverage tracking.
"""

import re

from typer.testing import CliRunner

from vs30.cli import app, get_config

runner = CliRunner()

# Strip ANSI escape codes from output (Rich/Typer may emit them in CI)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")


def strip_ansi(text: str) -> str:
    return _ANSI_RE.sub("", text)


class TestCLICallback:
    """Tests for the main CLI callback."""

    def test_help_output(self):
        """Test that --help works."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "VS30" in result.stdout

    def test_verbose_flag(self):
        """Test that -v verbose flag is accepted."""
        result = runner.invoke(app, ["-v", "--help"])
        assert result.exit_code == 0

    def test_nonexistent_config_error(self):
        """Test that nonexistent config file gives error."""
        # Use a command that will actually try to use the config
        result = runner.invoke(app, ["--config", "/nonexistent/config.yaml", "full-pipeline"])
        # Should fail with non-zero exit code due to missing config
        # Typer validation returns exit code 2 with "does not exist" message
        # Note: message may wrap across lines, so check for "not exist" substring
        assert result.exit_code != 0
        assert "not exist" in result.output.lower()


class TestGetConfig:
    """Tests for the get_config function."""

    def test_get_config_returns_config(self):
        """Test that get_config returns a config object."""
        from vs30.config import Vs30Config

        # Reset config state
        import vs30.cli
        vs30.cli._cli_config = None

        config = get_config()
        assert config is not None
        assert isinstance(config, Vs30Config)


class TestFullPipelineCommand:
    """Tests for the full-pipeline command."""

    def test_full_pipeline_help(self):
        """Test full-pipeline --help."""
        result = runner.invoke(app, ["full-pipeline", "--help"])
        assert result.exit_code == 0
        assert "full-pipeline" in result.stdout.lower() or "pipeline" in result.stdout.lower()


class TestComputeAtLocationsCommand:
    """Tests for the compute-at-locations command."""

    def test_compute_at_locations_help(self):
        """Test compute-at-locations --help."""
        result = runner.invoke(app, ["compute-at-locations", "--help"])
        assert result.exit_code == 0
        output = strip_ansi(result.stdout)
        assert "--locations-csv" in output
        assert "--output-csv" in output

    def test_compute_at_locations_missing_args(self):
        """Test that missing required args gives error."""
        result = runner.invoke(app, ["compute-at-locations"])
        assert result.exit_code != 0

    def test_compute_at_locations_nonexistent_input(self):
        """Test that nonexistent input file gives error."""
        result = runner.invoke(app, [
            "compute-at-locations",
            "--locations-csv", "/nonexistent/locations.csv",
            "--output-csv", "/tmp/output.csv",
        ])
        assert result.exit_code != 0


class TestSpatialFitCommand:
    """Tests for the spatial-fit command."""

    def test_spatial_fit_help(self):
        """Test spatial-fit --help."""
        result = runner.invoke(app, ["spatial-fit", "--help"])
        assert result.exit_code == 0


class TestCombineCommand:
    """Tests for the combine command."""

    def test_combine_help(self):
        """Test combine --help."""
        result = runner.invoke(app, ["combine", "--help"])
        assert result.exit_code == 0


class TestUpdateCategoricalCommand:
    """Tests for the update-categorical-vs30-models command."""

    def test_update_categorical_help(self):
        """Test update-categorical-vs30-models --help."""
        result = runner.invoke(app, ["update-categorical-vs30-models", "--help"])
        assert result.exit_code == 0


class TestMakeInitialRasterCommand:
    """Tests for the make-initial-vs30-raster command."""

    def test_make_initial_raster_help(self):
        """Test make-initial-vs30-raster --help."""
        result = runner.invoke(app, ["make-initial-vs30-raster", "--help"])
        assert result.exit_code == 0


class TestAdjustGeologyCommand:
    """Tests for the adjust-geology-vs30-by-slope-and-coastal-distance command."""

    def test_adjust_geology_help(self):
        """Test adjust-geology-vs30-by-slope-and-coastal-distance --help."""
        result = runner.invoke(app, ["adjust-geology-vs30-by-slope-and-coastal-distance", "--help"])
        assert result.exit_code == 0


class TestPlotPosteriorCommand:
    """Tests for the plot-posterior-values command."""

    def test_plot_posterior_help(self):
        """Test plot-posterior-values --help."""
        result = runner.invoke(app, ["plot-posterior-values", "--help"])
        assert result.exit_code == 0

"""
End-to-end tests for the VS30 full-pipeline command.

These tests verify that the full pipeline produces consistent results
for different observation type combinations and processing modes.

Test scenarios (using small domain for fast execution):
- Independent observations only
- Clustered (CPT) observations only
- Both observation types combined

Each scenario is tested with both single-process (n_proc=1) and
multi-process (n_proc=cpu_count) modes to ensure parallel processing works correctly.
"""

import os
import shutil
import tempfile
import traceback
from pathlib import Path

import pytest
import yaml
from typer.testing import CliRunner

from conftest import BENCHMARKS_DIR
from conftest import compare_csvs
from conftest import compare_rasters
from conftest import FIXTURES_DIR
from vs30 import cli


def create_test_config(scenario: str, output_dir: Path, n_proc: int = 1) -> Path:
    """
    Create a test configuration file with the specified settings.

    Parameters
    ----------
    scenario : str
        One of "independent_only", "clustered_only", or "both".
    output_dir : Path
        Directory for pipeline output.
    n_proc : int
        Number of processes to use.

    Returns
    -------
    Path
        Path to the created config file.
    """
    # Load the base config for this scenario
    config_file = FIXTURES_DIR / f"test_config_{scenario}.yaml"
    with open(config_file) as f:
        config = yaml.safe_load(f)

    # Override settings for test
    config["n_proc"] = n_proc
    config["output_dir"] = str(output_dir)

    # Write to temp location
    test_config_path = output_dir / "test_config.yaml"
    with open(test_config_path, "w") as f:
        yaml.dump(config, f)

    return test_config_path


def run_full_pipeline(config_path: Path) -> None:
    """
    Run the full pipeline with the given config.

    Uses Typer's CliRunner to invoke the CLI in-process, enabling
    pytest coverage tracking of the executed code.

    Parameters
    ----------
    config_path : Path
        Path to the configuration file.
    """
    runner = CliRunner()
    result = runner.invoke(cli.app, ["--config", str(config_path), "full-pipeline"])
    if result.exit_code != 0:
        print(f"Output:\n{result.stdout}")
        if result.exception:
            print(f"Exception:\n{''.join(traceback.format_exception(type(result.exception), result.exception, result.exception.__traceback__))}")
        raise RuntimeError(f"Pipeline failed with exit code {result.exit_code}")


# Key output files to compare
KEY_OUTPUT_FILES = [
    "combined_vs30.tif",
    "geology_vs30_slope_and_coastal_distance_and_spatially_adjusted_with_uncertainty.tif",
    "terrain_vs30_spatially_adjusted_with_uncertainty.tif",
    "posterior_geology_model_posterior_from_foster_2019_mean_and_standard_deviation.csv",
    "posterior_terrain_model_posterior_from_foster_2019_mean_and_standard_deviation.csv",
]


class TestSmallDomainIndependentOnly:
    """Fast test with small domain and independent observations only."""

    SCENARIO = "small_independent_only"
    BENCHMARK_DIR = BENCHMARKS_DIR / "small_independent_only"

    @pytest.fixture
    def output_dir(self):
        """Create temporary output directory."""
        tmpdir = tempfile.mkdtemp(prefix="vs30_test_small_independent_")
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    def test_single_process(self, output_dir):
        """Test pipeline with n_proc=1."""
        config_path = create_test_config(self.SCENARIO, output_dir, n_proc=1)
        run_full_pipeline(config_path)

        for filename in KEY_OUTPUT_FILES:
            actual = output_dir / filename
            expected = self.BENCHMARK_DIR / filename
            assert actual.exists(), f"Missing output file: {filename}"

            if filename.endswith(".tif"):
                compare_rasters(actual, expected)
            elif filename.endswith(".csv"):
                compare_csvs(actual, expected)

    @pytest.mark.parametrize("n_proc", [os.cpu_count()])
    def test_multiprocess(self, output_dir, n_proc):
        """Test pipeline with multiple processes."""
        config_path = create_test_config(self.SCENARIO, output_dir, n_proc=n_proc)
        run_full_pipeline(config_path)

        for filename in KEY_OUTPUT_FILES:
            actual = output_dir / filename
            expected = self.BENCHMARK_DIR / filename
            assert actual.exists(), f"Missing output file: {filename}"

            if filename.endswith(".tif"):
                compare_rasters(actual, expected)
            elif filename.endswith(".csv"):
                compare_csvs(actual, expected)


class TestSmallDomainBothObservationTypes:
    """Fast test with small domain and both observation types."""

    SCENARIO = "small_both"
    BENCHMARK_DIR = BENCHMARKS_DIR / "small_both"

    @pytest.fixture
    def output_dir(self):
        """Create temporary output directory."""
        tmpdir = tempfile.mkdtemp(prefix="vs30_test_small_both_")
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    def test_single_process(self, output_dir):
        """Test pipeline with n_proc=1."""
        config_path = create_test_config(self.SCENARIO, output_dir, n_proc=1)
        run_full_pipeline(config_path)

        for filename in KEY_OUTPUT_FILES:
            actual = output_dir / filename
            expected = self.BENCHMARK_DIR / filename
            assert actual.exists(), f"Missing output file: {filename}"

            if filename.endswith(".tif"):
                compare_rasters(actual, expected)
            elif filename.endswith(".csv"):
                compare_csvs(actual, expected)

    @pytest.mark.parametrize("n_proc", [os.cpu_count()])
    def test_multiprocess(self, output_dir, n_proc):
        """Test pipeline with multiple processes."""
        config_path = create_test_config(self.SCENARIO, output_dir, n_proc=n_proc)
        run_full_pipeline(config_path)

        for filename in KEY_OUTPUT_FILES:
            actual = output_dir / filename
            expected = self.BENCHMARK_DIR / filename
            assert actual.exists(), f"Missing output file: {filename}"

            if filename.endswith(".tif"):
                compare_rasters(actual, expected)
            elif filename.endswith(".csv"):
                compare_csvs(actual, expected)


class TestSmallDomainClusteredOnly:
    """Fast test with clustered (CPT) observations only.

    Uses 10km domain at 500m resolution (400 pixels) with full test CPT dataset
    to ensure adequate observations per category for stable Bayesian updates.
    """

    SCENARIO = "small_clustered_only"
    BENCHMARK_DIR = BENCHMARKS_DIR / "small_clustered_only"

    @pytest.fixture
    def output_dir(self):
        """Create temporary output directory."""
        tmpdir = tempfile.mkdtemp(prefix="vs30_test_small_clustered_")
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    def test_single_process(self, output_dir):
        """Test pipeline with n_proc=1."""
        config_path = create_test_config(self.SCENARIO, output_dir, n_proc=1)
        run_full_pipeline(config_path)

        for filename in KEY_OUTPUT_FILES:
            actual = output_dir / filename
            expected = self.BENCHMARK_DIR / filename
            assert actual.exists(), f"Missing output file: {filename}"

            if filename.endswith(".tif"):
                compare_rasters(actual, expected)
            elif filename.endswith(".csv"):
                compare_csvs(actual, expected)

    @pytest.mark.parametrize("n_proc", [os.cpu_count()])
    def test_multiprocess(self, output_dir, n_proc):
        """Test pipeline with multiple processes."""
        config_path = create_test_config(self.SCENARIO, output_dir, n_proc=n_proc)
        run_full_pipeline(config_path)

        for filename in KEY_OUTPUT_FILES:
            actual = output_dir / filename
            expected = self.BENCHMARK_DIR / filename
            assert actual.exists(), f"Missing output file: {filename}"

            if filename.endswith(".tif"):
                compare_rasters(actual, expected)
            elif filename.endswith(".csv"):
                compare_csvs(actual, expected)

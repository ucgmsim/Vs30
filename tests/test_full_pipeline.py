"""
End-to-end tests for the VS30 full-pipeline command.

These tests verify that the full pipeline produces consistent results
for different observation type combinations and processing modes.

Test scenarios:
- Independent observations only
- Clustered (CPT) observations only
- Both observation types combined

Each scenario is tested with both single-process (n_proc=1) and
multi-process (n_proc=-1) modes to ensure parallel processing works correctly.
"""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import rasterio
import yaml

# Test fixtures directory
FIXTURES_DIR = Path(__file__).parent / "fixtures"
BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"


def compare_rasters(
    actual_path: Path,
    expected_path: Path,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Compare two raster files for equality within tolerance.

    Parameters
    ----------
    actual_path : Path
        Path to the actual output raster.
    expected_path : Path
        Path to the expected benchmark raster.
    rtol : float
        Relative tolerance for numpy.allclose.
    atol : float
        Absolute tolerance for numpy.allclose.

    Raises
    ------
    AssertionError
        If the rasters differ beyond tolerance.
    """
    with rasterio.open(actual_path) as actual, rasterio.open(expected_path) as expected:
        # Check metadata
        assert actual.count == expected.count, f"Band count mismatch: {actual.count} vs {expected.count}"
        assert actual.width == expected.width, f"Width mismatch: {actual.width} vs {expected.width}"
        assert actual.height == expected.height, f"Height mismatch: {actual.height} vs {expected.height}"

        # Check each band
        for band_idx in range(1, actual.count + 1):
            actual_data = actual.read(band_idx)
            expected_data = expected.read(band_idx)

            # Get nodata value
            nodata = actual.nodata

            # Create masks for valid data
            if nodata is not None:
                actual_valid = actual_data != nodata
                expected_valid = expected_data != nodata
                assert np.array_equal(actual_valid, expected_valid), (
                    f"Band {band_idx}: Valid data masks differ"
                )
                # Compare only valid data
                if np.any(actual_valid):
                    assert np.allclose(
                        actual_data[actual_valid],
                        expected_data[expected_valid],
                        rtol=rtol,
                        atol=atol,
                    ), f"Band {band_idx}: Data values differ beyond tolerance"
            else:
                assert np.allclose(actual_data, expected_data, rtol=rtol, atol=atol), (
                    f"Band {band_idx}: Data values differ beyond tolerance"
                )


def compare_csvs(
    actual_path: Path,
    expected_path: Path,
    rtol: float = 1e-5,
    atol: float = 1e-8,
) -> None:
    """
    Compare two CSV files for equality within tolerance.

    Parameters
    ----------
    actual_path : Path
        Path to the actual output CSV.
    expected_path : Path
        Path to the expected benchmark CSV.
    rtol : float
        Relative tolerance for numeric comparison.
    atol : float
        Absolute tolerance for numeric comparison.

    Raises
    ------
    AssertionError
        If the CSVs differ beyond tolerance.
    """
    actual_df = pd.read_csv(actual_path)
    expected_df = pd.read_csv(expected_path)

    # Check columns match
    assert set(actual_df.columns) == set(expected_df.columns), (
        f"Column mismatch: {set(actual_df.columns)} vs {set(expected_df.columns)}"
    )

    # Check row count
    assert len(actual_df) == len(expected_df), (
        f"Row count mismatch: {len(actual_df)} vs {len(expected_df)}"
    )

    # Compare each column
    for col in actual_df.columns:
        if pd.api.types.is_numeric_dtype(actual_df[col]):
            assert np.allclose(
                actual_df[col].values,
                expected_df[col].values,
                rtol=rtol,
                atol=atol,
                equal_nan=True,
            ), f"Column '{col}' values differ beyond tolerance"
        else:
            assert actual_df[col].equals(expected_df[col]), (
                f"Column '{col}' values differ"
            )


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

    Parameters
    ----------
    config_path : Path
        Path to the configuration file.
    """
    import subprocess

    result = subprocess.run(
        ["vs30", "--config", str(config_path), "full-pipeline"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        print(f"STDOUT:\n{result.stdout}")
        print(f"STDERR:\n{result.stderr}")
        raise RuntimeError(f"Pipeline failed with return code {result.returncode}")


# Key output files to compare
KEY_OUTPUT_FILES = [
    "combined_vs30.tif",
    "geology_vs30_slope_and_coastal_distance_and_spatially_adjusted_with_uncertainty.tif",
    "terrain_vs30_spatially_adjusted_with_uncertainty.tif",
    "posterior_geology_model_posterior_from_foster_2019_mean_and_standard_deviation.csv",
    "posterior_terrain_model_posterior_from_foster_2019_mean_and_standard_deviation.csv",
]


class TestFullPipelineIndependentOnly:
    """Test full pipeline with independent observations only."""

    SCENARIO = "independent_only"
    BENCHMARK_DIR = BENCHMARKS_DIR / "independent_only"

    @pytest.fixture
    def output_dir(self):
        """Create temporary output directory."""
        tmpdir = tempfile.mkdtemp(prefix="vs30_test_independent_")
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    def test_single_process(self, output_dir):
        """Test pipeline with n_proc=1."""
        config_path = create_test_config(self.SCENARIO, output_dir, n_proc=1)
        run_full_pipeline(config_path)

        # Compare key outputs
        for filename in KEY_OUTPUT_FILES:
            actual = output_dir / filename
            expected = self.BENCHMARK_DIR / filename
            assert actual.exists(), f"Missing output file: {filename}"

            if filename.endswith(".tif"):
                compare_rasters(actual, expected)
            elif filename.endswith(".csv"):
                compare_csvs(actual, expected)

    @pytest.mark.parametrize("n_proc", [-1, os.cpu_count()])
    def test_multiprocess(self, output_dir, n_proc):
        """Test pipeline with multiple processes."""
        config_path = create_test_config(self.SCENARIO, output_dir, n_proc=n_proc)
        run_full_pipeline(config_path)

        # Compare key outputs - multiprocess should produce same results
        for filename in KEY_OUTPUT_FILES:
            actual = output_dir / filename
            expected = self.BENCHMARK_DIR / filename
            assert actual.exists(), f"Missing output file: {filename}"

            if filename.endswith(".tif"):
                compare_rasters(actual, expected)
            elif filename.endswith(".csv"):
                compare_csvs(actual, expected)


class TestFullPipelineClusteredOnly:
    """Test full pipeline with clustered (CPT) observations only."""

    SCENARIO = "clustered_only"
    BENCHMARK_DIR = BENCHMARKS_DIR / "clustered_only"

    @pytest.fixture
    def output_dir(self):
        """Create temporary output directory."""
        tmpdir = tempfile.mkdtemp(prefix="vs30_test_clustered_")
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    def test_single_process(self, output_dir):
        """Test pipeline with n_proc=1."""
        config_path = create_test_config(self.SCENARIO, output_dir, n_proc=1)
        run_full_pipeline(config_path)

        # Compare key outputs
        for filename in KEY_OUTPUT_FILES:
            actual = output_dir / filename
            expected = self.BENCHMARK_DIR / filename
            assert actual.exists(), f"Missing output file: {filename}"

            if filename.endswith(".tif"):
                compare_rasters(actual, expected)
            elif filename.endswith(".csv"):
                compare_csvs(actual, expected)

    @pytest.mark.parametrize("n_proc", [-1, os.cpu_count()])
    def test_multiprocess(self, output_dir, n_proc):
        """Test pipeline with multiple processes."""
        config_path = create_test_config(self.SCENARIO, output_dir, n_proc=n_proc)
        run_full_pipeline(config_path)

        # Compare key outputs
        for filename in KEY_OUTPUT_FILES:
            actual = output_dir / filename
            expected = self.BENCHMARK_DIR / filename
            assert actual.exists(), f"Missing output file: {filename}"

            if filename.endswith(".tif"):
                compare_rasters(actual, expected)
            elif filename.endswith(".csv"):
                compare_csvs(actual, expected)


class TestFullPipelineBothObservationTypes:
    """Test full pipeline with both independent and clustered observations."""

    SCENARIO = "both"
    BENCHMARK_DIR = BENCHMARKS_DIR / "both"

    @pytest.fixture
    def output_dir(self):
        """Create temporary output directory."""
        tmpdir = tempfile.mkdtemp(prefix="vs30_test_both_")
        yield Path(tmpdir)
        shutil.rmtree(tmpdir)

    def test_single_process(self, output_dir):
        """Test pipeline with n_proc=1."""
        config_path = create_test_config(self.SCENARIO, output_dir, n_proc=1)
        run_full_pipeline(config_path)

        # Compare key outputs
        for filename in KEY_OUTPUT_FILES:
            actual = output_dir / filename
            expected = self.BENCHMARK_DIR / filename
            assert actual.exists(), f"Missing output file: {filename}"

            if filename.endswith(".tif"):
                compare_rasters(actual, expected)
            elif filename.endswith(".csv"):
                compare_csvs(actual, expected)

    @pytest.mark.parametrize("n_proc", [-1, os.cpu_count()])
    def test_multiprocess(self, output_dir, n_proc):
        """Test pipeline with multiple processes."""
        config_path = create_test_config(self.SCENARIO, output_dir, n_proc=n_proc)
        run_full_pipeline(config_path)

        # Compare key outputs
        for filename in KEY_OUTPUT_FILES:
            actual = output_dir / filename
            expected = self.BENCHMARK_DIR / filename
            assert actual.exists(), f"Missing output file: {filename}"

            if filename.endswith(".tif"):
                compare_rasters(actual, expected)
            elif filename.endswith(".csv"):
                compare_csvs(actual, expected)


# =============================================================================
# Small Domain Tests (Fast)
# =============================================================================
# These tests use a smaller 3km x 3km domain (vs 10km x 10km) and minimal
# observation subsets to run much faster (~20s vs ~5min per test).


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

    @pytest.mark.parametrize("n_proc", [-1, os.cpu_count()])
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

    @pytest.mark.parametrize("n_proc", [-1, os.cpu_count()])
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

    @pytest.mark.parametrize("n_proc", [-1, os.cpu_count()])
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

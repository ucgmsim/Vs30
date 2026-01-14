"""
Tests for the VS30 compute-at-locations command.

These tests verify that the compute-at-locations command produces
consistent Vs30 values for known locations (major NZ cities).
"""

import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

FIXTURES_DIR = Path(__file__).parent / "fixtures"
BENCHMARKS_DIR = Path(__file__).parent / "benchmarks"


class TestComputeAtLocations:
    """Tests for the compute-at-locations CLI command."""

    @pytest.fixture
    def output_csv(self):
        """Create temporary output file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            yield Path(f.name)
        Path(f.name).unlink(missing_ok=True)

    def test_nz_cities_single_process(self, output_csv):
        """Test compute-at-locations with major NZ cities."""
        input_csv = FIXTURES_DIR / "nz_cities.csv"
        expected_csv = BENCHMARKS_DIR / "nz_cities_vs30.csv"

        # Run command
        result = subprocess.run(
            [
                "vs30",
                "compute-at-locations",
                "--locations-csv", str(input_csv),
                "--output-csv", str(output_csv),
                "--lat-column", "latitude",
                "--lon-column", "longitude",
                "--include-intermediate",
                "--n-proc", "1",
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            pytest.fail(f"Command failed with return code {result.returncode}")

        # Compare outputs
        actual_df = pd.read_csv(output_csv)
        expected_df = pd.read_csv(expected_csv)

        # Check all rows are present
        assert len(actual_df) == len(expected_df), (
            f"Row count mismatch: {len(actual_df)} vs {len(expected_df)}"
        )

        # Check all columns are present
        assert set(actual_df.columns) == set(expected_df.columns), (
            f"Column mismatch: {set(actual_df.columns)} vs {set(expected_df.columns)}"
        )

        # Compare numeric columns with tolerance
        numeric_cols = [
            "easting", "northing",
            "geology_vs30", "geology_stdv",
            "geology_vs30_hybrid", "geology_stdv_hybrid",
            "geology_mvn_vs30", "geology_mvn_stdv",
            "terrain_vs30", "terrain_stdv",
            "terrain_mvn_vs30", "terrain_mvn_stdv",
            "vs30", "stdv",
        ]

        for col in numeric_cols:
            if col in actual_df.columns:
                assert np.allclose(
                    actual_df[col].values,
                    expected_df[col].values,
                    rtol=1e-5,
                    atol=1e-8,
                    equal_nan=True,
                ), f"Column '{col}' values differ beyond tolerance"

    def test_vs30_values_reasonable(self, output_csv):
        """Test that computed Vs30 values are within reasonable range."""
        input_csv = FIXTURES_DIR / "nz_cities.csv"

        # Run command
        result = subprocess.run(
            [
                "vs30",
                "compute-at-locations",
                "--locations-csv", str(input_csv),
                "--output-csv", str(output_csv),
                "--lat-column", "latitude",
                "--lon-column", "longitude",
                "--n-proc", "1",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        # Check Vs30 values are reasonable (typically 100-1000 m/s for NZ)
        df = pd.read_csv(output_csv)

        # Filter out rows with NaN values (some cities may fall outside coverage)
        valid_df = df.dropna(subset=["vs30", "stdv"])

        # Final Vs30 should be positive and within reasonable range
        assert (valid_df["vs30"] > 0).all(), "Vs30 values should be positive"
        assert (valid_df["vs30"] < 2000).all(), "Vs30 values should be < 2000 m/s"
        assert (valid_df["vs30"] > 50).all(), "Vs30 values should be > 50 m/s"

        # Standard deviations should be positive
        assert (valid_df["stdv"] > 0).all(), "Standard deviations should be positive"
        assert (valid_df["stdv"] < 2).all(), "Standard deviations should be reasonable"

        # At least most cities should have valid values
        assert len(valid_df) >= len(df) * 0.8, (
            f"Too many NaN values: {len(df) - len(valid_df)} of {len(df)}"
        )

    def test_christchurch_values(self, output_csv):
        """Test that Christchurch has expected low Vs30 (soft soils)."""
        input_csv = FIXTURES_DIR / "nz_cities.csv"

        # Run command
        result = subprocess.run(
            [
                "vs30",
                "compute-at-locations",
                "--locations-csv", str(input_csv),
                "--output-csv", str(output_csv),
                "--lat-column", "latitude",
                "--lon-column", "longitude",
                "--n-proc", "1",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Command failed: {result.stderr}"

        df = pd.read_csv(output_csv)
        chch_row = df[df["city"] == "Christchurch"].iloc[0]

        # Christchurch is known for soft soils, Vs30 typically < 250 m/s
        assert chch_row["vs30"] < 250, (
            f"Christchurch Vs30 should be low (soft soils), got {chch_row['vs30']}"
        )
        assert chch_row["vs30"] > 100, (
            f"Christchurch Vs30 should be > 100 m/s, got {chch_row['vs30']}"
        )


class TestComputeAtLocationsMultiprocess:
    """Tests for compute-at-locations with multiple processes."""

    @pytest.fixture
    def output_csv(self):
        """Create temporary output file."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            yield Path(f.name)
        Path(f.name).unlink(missing_ok=True)

    def test_multiprocess_matches_single(self, output_csv):
        """Test that multiprocess results match single-process results."""
        input_csv = FIXTURES_DIR / "nz_cities.csv"
        expected_csv = BENCHMARKS_DIR / "nz_cities_vs30.csv"

        # Run with multiple processes
        result = subprocess.run(
            [
                "vs30",
                "compute-at-locations",
                "--locations-csv", str(input_csv),
                "--output-csv", str(output_csv),
                "--lat-column", "latitude",
                "--lon-column", "longitude",
                "--include-intermediate",
                "--n-proc", "-1",  # Use all cores
            ],
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"STDOUT:\n{result.stdout}")
            print(f"STDERR:\n{result.stderr}")
            pytest.fail(f"Command failed with return code {result.returncode}")

        # Compare with single-process benchmark
        actual_df = pd.read_csv(output_csv)
        expected_df = pd.read_csv(expected_csv)

        # Compare key Vs30 values (equal_nan=True handles NaN values)
        assert np.allclose(
            actual_df["vs30"].values,
            expected_df["vs30"].values,
            rtol=1e-5,
            atol=1e-8,
            equal_nan=True,
        ), "Multiprocess Vs30 values should match single-process results"

        assert np.allclose(
            actual_df["stdv"].values,
            expected_df["stdv"].values,
            rtol=1e-5,
            atol=1e-8,
            equal_nan=True,
        ), "Multiprocess stdv values should match single-process results"

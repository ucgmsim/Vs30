"""
Tests for the VS30 compute-at-locations command.

These tests verify that the compute-at-locations command produces
consistent Vs30 values for known locations (major NZ cities).

Uses CliRunner for in-process invocation to enable coverage tracking.
"""

from conftest import BENCHMARKS_DIR
from conftest import compare_csvs
from conftest import FIXTURES_DIR
from conftest import run_cli

LOCATIONS_CLI_ARGS = [
    "compute-at-locations",
    "--locations-csv", str(FIXTURES_DIR / "nz_cities.csv"),
    "--lat-column", "latitude",
    "--lon-column", "longitude",
    "--include-intermediate",
]

EXPECTED_CSV = BENCHMARKS_DIR / "nz_cities_vs30.csv"


def test_single_process(tmp_path):
    """Test compute-at-locations with n_proc=1."""
    output_csv = tmp_path / "output.csv"
    run_cli([*LOCATIONS_CLI_ARGS, "--output-csv", str(output_csv), "--n-proc", "1"])
    compare_csvs(output_csv, EXPECTED_CSV)


def test_multiprocess(tmp_path):
    """Test that multiprocess results match single-process benchmark."""
    output_csv = tmp_path / "output.csv"
    run_cli([*LOCATIONS_CLI_ARGS, "--output-csv", str(output_csv), "--n-proc", "-1"])
    compare_csvs(output_csv, EXPECTED_CSV)

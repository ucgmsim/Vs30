"""Tests for the vs30 command-line interface."""

from pathlib import Path

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from vs30 import cli

WELLINGTON = "174.7762,-41.2865"


def run_points(tmp_path, csv_text: str, output_csv: Path | None = None):
    """Run ``vs30 points foster_2019_approx`` on a locations CSV; return the result and output path."""
    locations_csv = tmp_path / "sites.csv"
    locations_csv.write_text(csv_text)
    output_csv = output_csv or tmp_path / "out.csv"
    result = CliRunner().invoke(
        cli.app,
        ["points", "foster_2019_approx", str(locations_csv), str(output_csv)],
        env={"COLUMNS": "200"},
    )
    return result, output_csv


def test_rows_with_bad_coordinates_get_blank_results_and_a_warning(tmp_path, caplog):
    """Blank, non-numeric or out-of-range coordinates leave those rows blank, named in a warning, instead of stopping the run."""
    result, output_csv = run_points(
        tmp_path,
        "name,longitude,latitude\n"
        f"wellington,{WELLINGTON}\n"
        "blank,,\n"
        "text,abc,-41.2865\n"
        "swapped,-41.2865,174.7762\n",
    )

    assert result.exit_code == 0, result.output
    output = pd.read_csv(output_csv)
    assert np.isfinite(output.loc[0, "vs30"])
    assert output.loc[1:, "vs30"].isna().all()
    assert "line(s) 3, 4, 5" in caplog.text


def test_header_with_spaces_after_commas_is_read(tmp_path):
    """A header like 'name, longitude, latitude' is recognised."""
    result, output_csv = run_points(
        tmp_path, "name, longitude, latitude\nwellington, 174.7762, -41.2865\n"
    )

    assert result.exit_code == 0, result.output
    assert np.isfinite(pd.read_csv(output_csv).loc[0, "vs30"])


def test_missing_coordinate_column_error_lists_the_columns(tmp_path):
    """The error for a missing coordinate column lists the file's columns and the option to pick another."""
    result, _ = run_points(tmp_path, f"name,lon,lat\nwellington,{WELLINGTON}\n")

    assert result.exit_code == 2
    assert "name, lon, lat" in result.output
    assert "--lon-column" in result.output


def test_input_columns_named_like_outputs_are_kept_with_a_suffix(tmp_path, caplog):
    """An input column called vs30 is kept as vs30_input rather than silently replaced."""
    result, output_csv = run_points(
        tmp_path, f"name,longitude,latitude,vs30\nwellington,{WELLINGTON},250\n"
    )

    assert result.exit_code == 0, result.output
    output = pd.read_csv(output_csv)
    assert output.loc[0, "vs30_input"] == 250
    assert output.loc[0, "vs30"] != 250
    assert "vs30_input" in caplog.text


def test_sites_without_vs30_are_counted_in_a_warning(tmp_path, caplog):
    """Sites that get no Vs30 (here, Sydney) are counted in a warning."""
    result, _ = run_points(
        tmp_path,
        f"name,longitude,latitude\nwellington,{WELLINGTON}\nsydney,151.2093,-33.8688\n",
    )

    assert result.exit_code == 0, result.output
    assert "1 site(s) have no Vs30" in caplog.text


def test_unwritable_output_location_is_reported_before_computing(tmp_path):
    """points and grid refuse an output location they can't write to, before any computation."""
    read_only = tmp_path / "read_only"
    read_only.mkdir()
    read_only.chmod(0o500)

    points_result, _ = run_points(
        tmp_path, f"name,longitude,latitude\nwellington,{WELLINGTON}\n", read_only / "out.csv"
    )
    grid_result = CliRunner().invoke(
        cli.app,
        [
            "grid",
            "--model", "foster_2019_approx",
            "--grid-xmin", "1748100",
            "--grid-xmax", "1749100",
            "--grid-ymin", "5427100",
            "--grid-ymax", "5428100",
            "--grid-dx", "100",
            "--grid-dy", "100",
            "--output-dir", str(read_only),
        ],
        env={"COLUMNS": "200"},
    )

    for result in (points_result, grid_result):
        assert result.exit_code == 2, result.output
        assert "Can't write to output directory" in result.output

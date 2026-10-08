"""Tests for the vs30 command-line interface."""

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import threadpoolctl
from typer.testing import CliRunner

from vs30 import cli, config, pipeline

WELLINGTON = "174.7762,-41.2865"


@pytest.fixture(autouse=True)
def reset_package_logger():
    """Undo the CLI's logging setup so later tests don't log to a finished CliRunner's stream."""
    yield
    logging.getLogger("vs30").handlers = []
    logging.getLogger("vs30").setLevel(logging.NOTSET)


def run_points(
    tmp_path,
    csv_text: str,
    *extra_args: str,
    model: str = "foster_2019_approx",
    output_csv: Path | None = None,
):
    """Run ``vs30 points`` on a locations CSV; return the result and output path."""
    locations_csv = tmp_path / "sites.csv"
    locations_csv.write_text(csv_text)
    output_csv = output_csv or tmp_path / "out.csv"
    result = CliRunner().invoke(
        cli.app,
        ["points", model, str(locations_csv), str(output_csv), *extra_args],
        env={"COLUMNS": "200"},
    )
    return result, output_csv


def run_grid(*args: str):
    """Run ``vs30 grid`` with the given arguments."""
    return CliRunner().invoke(cli.app, ["grid", *args], env={"COLUMNS": "200"})


@pytest.fixture
def recorded_grid_runs(monkeypatch):
    """Replace the slow grid pipeline with a stub recording its arguments and BLAS thread count."""
    calls = []

    def fake_grid_pipeline(**kwargs):
        kwargs["blas_threads"] = [
            info["num_threads"]
            for info in threadpoolctl.threadpool_info()
            if info["user_api"] == "blas"
        ]
        calls.append(kwargs)

    monkeypatch.setattr(pipeline, "grid_pipeline", fake_grid_pipeline)
    return calls


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
        tmp_path,
        f"name,longitude,latitude\nwellington,{WELLINGTON}\n",
        output_csv=read_only / "out.csv",
    )
    grid_result = run_grid(
        "foster_2019_approx",
        str(read_only),
        "--grid-xmin", "1748100",
        "--grid-xmax", "1749100",
        "--grid-ymin", "5427100",
        "--grid-ymax", "5428100",
    )

    for result in (points_result, grid_result):
        assert result.exit_code == 2, result.output
        assert "Can't write to output directory" in result.output


def test_points_prints_progress_lines_but_not_detail_by_default(tmp_path):
    """By default, points prints a few progress lines but not the step-by-step detail."""
    result, output_csv = run_points(tmp_path, f"name,longitude,latitude\nwellington,{WELLINGTON}\n")

    assert result.exit_code == 0, result.output
    assert "Processing 1 location(s)" in result.output
    assert f"Results written to {output_csv}" in result.output
    assert "Loading observations from" not in result.output


def test_verbose_shows_the_detail(tmp_path):
    """--verbose adds the step-by-step detail."""
    result, _ = run_points(
        tmp_path, f"name,longitude,latitude\nwellington,{WELLINGTON}\n", "--verbose"
    )

    assert result.exit_code == 0, result.output
    assert "Loading observations from" in result.output


def test_gap_fill_local_grids_stay_quiet_by_default(tmp_path):
    """Gap-filling reports how many points it fills, without each local grid's stage messages."""
    # geology_gid9_outwash from tests/fixtures/consistency_test_points.csv: an on-land gap.
    result, _ = run_points(
        tmp_path,
        "name,longitude,latitude\noutwash,167.4593781323869,-46.14815700398977\n",
        model="jaehwi_v1p0",
    )

    assert result.exit_code == 0, result.output
    assert "Gap-filling 1 point(s) from local grids" in result.output
    assert "building the category raster" not in result.output


def test_grid_takes_model_and_output_dir_and_defaults_to_all_of_nz(
    tmp_path, recorded_grid_runs
):
    """`vs30 grid MODEL OUTPUT_DIR` computes the full-NZ grid and says how big it is."""
    result = run_grid("foster_2019_approx", str(tmp_path / "out"))

    assert result.exit_code == 0, result.output
    assert recorded_grid_runs[0]["grid_config"] == config.FULL_NZ_GRID_CONFIG
    assert recorded_grid_runs[0]["output_dir"] == tmp_path / "out"
    assert "10600 x 15200" in result.output


@pytest.mark.parametrize(
    "options, message",
    [
        (
            ["--grid-xmin", "1749100", "--grid-xmax", "1748100"],
            "--grid-xmin must be less than --grid-xmax",
        ),
        (["--grid-dx", "0"], "--grid-dx must be positive"),
        (["--grid-xmin", "1748100", "--grid-xmax", "1749150"], "whole number of"),
    ],
)
def test_grid_rejects_bad_bounds_before_computing(
    tmp_path, recorded_grid_runs, options, message
):
    """Bounds in the wrong order, a non-positive spacing, or a partial pixel are rejected up front."""
    result = run_grid("foster_2019_approx", str(tmp_path / "out"), *options)

    assert result.exit_code == 2
    assert message in result.output
    assert recorded_grid_runs == []


def test_grid_warns_when_bounds_are_off_the_full_nz_grid(tmp_path, recorded_grid_runs):
    """Bounds that don't line up with the full-NZ grid's pixels get a warning."""
    result = run_grid(
        "foster_2019_approx",
        str(tmp_path / "out"),
        "--grid-xmin", "1748150",
        "--grid-xmax", "1750150",
        "--grid-ymin", "5426100",
        "--grid-ymax", "5428100",
    )

    assert result.exit_code == 0, result.output
    assert "aren't aligned with the full-NZ grid" in result.output


def test_nproc_limits_dbscan_and_blas_threads(tmp_path, recorded_grid_runs):
    """--nproc caps both DBSCAN's processes and the BLAS threads used by the MVN."""
    result = run_grid("foster_2019_approx", str(tmp_path / "out"), "--nproc", "2")

    assert result.exit_code == 0, result.output
    assert recorded_grid_runs[0]["dbscan_nproc"] == 2
    assert set(recorded_grid_runs[0]["blas_threads"]) == {2}


def test_nproc_must_be_all_cores_or_positive(tmp_path, recorded_grid_runs):
    """--nproc 0 is rejected before computing."""
    result = run_grid("foster_2019_approx", str(tmp_path / "out"), "--nproc", "0")

    assert result.exit_code == 2
    assert "--nproc must be -1 (all cores) or a positive number" in result.output
    assert recorded_grid_runs == []


@pytest.mark.parametrize("command", ["points", "grid"])
def test_help_shows_option_names_in_full_at_80_columns(command):
    """No option name is cut short ('…') in a standard 80-column terminal."""
    result = CliRunner().invoke(cli.app, [command, "--help"], env={"COLUMNS": "80"})

    assert result.exit_code == 0
    assert "…" not in result.output


def test_mvn_chunk_memory_option_reaches_the_pipeline(tmp_path, recorded_grid_runs):
    """--mvn-chunk-memory-gb sets the MVN chunk memory cap."""
    result = run_grid(
        "foster_2019_approx", str(tmp_path / "out"), "--mvn-chunk-memory-gb", "2"
    )

    assert result.exit_code == 0, result.output
    assert recorded_grid_runs[0]["max_spatial_intermediate_array_memory_gb"] == 2

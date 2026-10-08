"""Tests for the gap-fill module."""

import numpy as np
import pandas as pd
import rasterio

from conftest import FIXTURES_DIR, load_fixed_model_config
from vs30 import config, constants, gapfill, pipeline


def test_classify_nodata_excludes_water_and_offshore():
    """classify_nodata should exclude GID=0 (water) and offshore pixels."""
    # Wellington CBD: known on-land location
    onland_location = np.array([[1749000.0, 5427000.0]])
    # Far offshore: known ocean location east of NZ
    offshore_location = np.array([[2200000.0, 5400000.0]])

    result = gapfill.classify_nodata(
        combined_vs30=np.array([np.nan]),
        geology_ids=np.array([5]),
        locations=onland_location,
    )
    assert result[0], "On-land nodata pixel with valid GID should be fillable"

    result = gapfill.classify_nodata(
        combined_vs30=np.array([np.nan]),
        geology_ids=np.array([0]),
        locations=onland_location,
    )
    assert not result[0], "Water pixel (GID=0) should not be fillable"

    result = gapfill.classify_nodata(
        combined_vs30=np.array([300.0]),
        geology_ids=np.array([5]),
        locations=onland_location,
    )
    assert not result[0], "Valid (non-NaN) pixel should not be fillable"

    result = gapfill.classify_nodata(
        combined_vs30=np.array([np.nan]),
        geology_ids=np.array([5]),
        locations=offshore_location,
    )
    assert not result[0], "Offshore nodata pixel should not be fillable"


def test_fill_nodata_grid_nearest_neighbor():
    """fill_nodata_grid fills an on-land NaN gap with the nearest valid value."""
    # 3x3 grid at 100m spacing, placed so pixel (1,1) center = (1749050, 5427050)
    # Origin is the top-left corner of pixel (0,0).
    # pixel center = origin + (index + 0.5) * pixel_size
    # For col 1: origin_x + 1.5 * 100 = 1749050 -> origin_x = 1748900
    # For row 1: origin_y + 1.5 * (-100) = 5427050 -> origin_y = 5427200
    pixel_size = 100
    transform = rasterio.transform.Affine(pixel_size, 0, 1748900, 0, -pixel_size, 5427200)
    profile = {"transform": transform}

    vs30 = np.array(
        [
            [200.0, 250.0, 275.0],
            [300.0, np.nan, 350.0],
            [375.0, 400.0, 450.0],
        ]
    )
    stdv = np.array(
        [
            [0.50, 0.60, 0.65],
            [0.70, np.nan, 0.80],
            [0.75, 0.90, 0.95],
        ]
    )
    # All non-zero GIDs so the center pixel passes the water filter
    geology_ids = np.full((3, 3), 5, dtype=int)

    filled_vs30, filled_stdv = gapfill.fill_nodata_grid(
        vs30, stdv, geology_ids, profile
    )

    # All four edge neighbors are equidistant; KDTree picks the first in row-major order.
    assert filled_vs30[1, 1] == 250.0
    assert filled_stdv[1, 1] == 0.6

    # All other pixels should be unchanged
    mask = np.ones((3, 3), dtype=bool)
    mask[1, 1] = False
    np.testing.assert_array_equal(filled_vs30[mask], vs30[mask])
    np.testing.assert_array_equal(filled_stdv[mask], stdv[mask])


def test_create_local_grid_config_expansion():
    """create_local_grid_config: bigger half_width → bigger grid, same centre, same pixel lattice as the full NZ grid."""
    dx = config.FULL_NZ_GRID_CONFIG.grid_dx
    dy = config.FULL_NZ_GRID_CONFIG.grid_dy

    # Pick an arbitrary pixel CENTRE on the full NZ grid (100 pixels from the
    # origin). Pixel centres are at grid_xmin + dx/2 + n*dx under the
    # pixel-edge bounds convention.
    n_pixels = 100
    easting = (
        config.FULL_NZ_GRID_CONFIG.grid_xmin
        + dx / 2
        + n_pixels * dx
    )
    northing = (
        config.FULL_NZ_GRID_CONFIG.grid_ymin
        + dy / 2
        + n_pixels * dy
    )

    # Production constants must satisfy create_local_grid_config's pixel-edge
    # constraint (half_width = k*dx + dx/2).
    assert (constants.GAPFILL_INITIAL_HALF_WIDTH_M - dx / 2) % dx == 0, (
        f"GAPFILL_INITIAL_HALF_WIDTH_M = {constants.GAPFILL_INITIAL_HALF_WIDTH_M} "
        f"must equal k*dx + dx/2 for create_local_grid_config to produce "
        f"pixel-aligned local grids."
    )
    assert constants.GAPFILL_HALF_WIDTH_EXPANSION_M % dx == 0, (
        f"GAPFILL_HALF_WIDTH_EXPANSION_M = {constants.GAPFILL_HALF_WIDTH_EXPANSION_M} "
        f"must be a multiple of dx so successive expansions stay aligned."
    )
    initial_half_width = constants.GAPFILL_INITIAL_HALF_WIDTH_M
    expanded_half_width = initial_half_width + constants.GAPFILL_HALF_WIDTH_EXPANSION_M

    initial_grid = gapfill.create_local_grid_config(
        easting,
        northing,
        config.FULL_NZ_GRID_CONFIG,
        initial_half_width,
    )
    expanded_grid = gapfill.create_local_grid_config(
        easting,
        northing,
        config.FULL_NZ_GRID_CONFIG,
        expanded_half_width,
    )

    # The expanded grid should be larger
    assert (expanded_grid.grid_xmax - expanded_grid.grid_xmin) > (
        initial_grid.grid_xmax - initial_grid.grid_xmin
    )

    # Both grids should still be centered on the snapped pixel centre
    assert (initial_grid.grid_xmin + initial_grid.grid_xmax) / 2 == (
        expanded_grid.grid_xmin + expanded_grid.grid_xmax
    ) / 2
    assert (initial_grid.grid_ymin + initial_grid.grid_ymax) / 2 == (
        expanded_grid.grid_ymin + expanded_grid.grid_ymax
    ) / 2

    # Both grids share the full NZ grid's pixel-centre lattice
    # (xmin offsets are an integer number of dx away from the full grid's xmin).
    assert initial_grid.grid_dx == dx
    assert expanded_grid.grid_dx == dx
    assert (
        initial_grid.grid_xmin - config.FULL_NZ_GRID_CONFIG.grid_xmin
    ) % dx == 0
    assert (
        expanded_grid.grid_xmin - config.FULL_NZ_GRID_CONFIG.grid_xmin
    ) % dx == 0


def test_points_gap_fill_shows_single_bar_over_points_to_fill(capsys):
    """Points-mode gap-fill shows one bar over the points being filled, not per-pixel bars from each local grid."""
    cfg = load_fixed_model_config(constants.FixedModelVersion.JAEHWI_V1P0)
    # geology_gid9_outwash is an on-land nodata gap (no terrain category) with no
    # observations near its local grid; auckland needs no filling.
    sites = (
        pd.read_csv(FIXTURES_DIR / "consistency_test_points.csv")
        .set_index("name")
        .loc[["auckland", "geology_gid9_outwash"]]
    )

    pipeline.points_pipeline(
        longitudes=sites["longitude"].to_numpy(),
        latitudes=sites["latitude"].to_numpy(),
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        geology_corr_fn=cfg["geology_corr_fn"],
        terrain_corr_fn=cfg["terrain_corr_fn"],
        geology_categorical_csv=cfg["geology_categorical_csv"],
        terrain_categorical_csv=cfg["terrain_categorical_csv"],
        independent_observations_csv=cfg["independent_observations_csv"],
        combination_method=constants.CombinationMethod(cfg["combination_method"]),
        combine_ratio=cfg["combine_ratio"],
        do_bayesian_update=cfg["do_bayesian_update"],
        apply_coastal_distance_mod=cfg["apply_coastal_distance_mod"],
        fill_gaps=True,
    )

    # tqdm redraws each bar in place with "\r" and ends it with "\n", so the last
    # "\r" segment of each "\n" line is what stays on screen (splitlines() would
    # also split on "\r").
    final_bar_states = [
        line.split("\r")[-1] for line in capsys.readouterr().err.split("\n") if line
    ]
    # Geology and terrain bars over both sites, then one bar over the single gap.
    assert len(final_bar_states) == 3, final_bar_states
    assert " 1/1 " in final_bar_states[2]


def local_grid_result(vs30: np.ndarray, stdv: np.ndarray) -> dict:
    """grid_pipeline-style result for the 3x3 grid whose pixel (1, 1) is centred on (1749050, 5427050)."""
    return {
        "combined_vs30": vs30,
        "combined_stdv": stdv,
        "profile": {
            "transform": rasterio.transform.Affine(100, 0, 1748900, 0, -100, 5427200)
        },
    }


def test_fill_one_point_via_local_grid_uses_nearest_valid_pixel(monkeypatch):
    """A nodata point takes the value of the valid pixel nearest its own pixel's centre."""
    vs30 = np.full((3, 3), np.nan)
    stdv = np.full((3, 3), np.nan)
    vs30[1, 0], stdv[1, 0] = 300.0, 0.7
    vs30[0, 2], stdv[0, 2] = 400.0, 0.9
    monkeypatch.setattr(
        pipeline,
        "grid_pipeline",
        lambda grid_config, output_dir, **kwargs: local_grid_result(vs30, stdv),
    )

    # The point sits 45 m east of its pixel centre: from the point, pixel (0, 2)
    # is nearer (114 m vs 145 m); from the pixel centre, as grid mode measures,
    # pixel (1, 0) is (100 m vs 141 m).
    assert pipeline.fill_one_point_via_local_grid(
        1749095.0, 5427050.0, config.FULL_NZ_GRID_CONFIG, {}
    ) == (300.0, 0.7)


def test_fill_one_point_via_local_grid_gives_up_at_points_limit(monkeypatch):
    """With no valid pixel in reach, local grids grow up to the points-mode limit, then the point stays nodata."""
    requested_half_widths = []

    def fake_grid_pipeline(grid_config, output_dir, **kwargs):
        requested_half_widths.append(
            (grid_config.grid_xmax - grid_config.grid_xmin) / 2
        )
        return local_grid_result(np.full((3, 3), np.nan), np.full((3, 3), np.nan))

    monkeypatch.setattr(pipeline, "grid_pipeline", fake_grid_pipeline)

    fill_vs30, fill_stdv = pipeline.fill_one_point_via_local_grid(
        1749050.0, 5427050.0, config.FULL_NZ_GRID_CONFIG, {}
    )

    assert np.isnan(fill_vs30) and np.isnan(fill_stdv)
    assert requested_half_widths == list(
        range(
            constants.GAPFILL_INITIAL_HALF_WIDTH_M,
            constants.GAPFILL_POINTS_MAX_HALF_WIDTH_M + 1,
            constants.GAPFILL_HALF_WIDTH_EXPANSION_M,
        )
    )

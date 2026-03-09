"""
Test that grid and points pipelines produce consistent Vs30 values.

Runs the grid pipeline on a small domain, then runs the points pipeline
at a pixel center coordinate and checks the results match.
"""

import numpy as np
import rasterio
import yaml

from conftest import FIXTURES_DIR

from vs30 import constants, pipeline
from vs30 import config as config_module


def test_grid_and_points_consistency(tmp_path):
    """Grid and points pipelines should produce the same Vs30 at a pixel center."""
    # Use the small test config
    config_file = FIXTURES_DIR / "test_config_small_independent_only.yaml"
    with open(config_file) as f:
        config_data = yaml.safe_load(f)

    grid_config = config_module.GridConfig.from_dict(config_data)
    grid_output_dir = tmp_path / "grid_output"

    # Resolve observation CSVs
    independent_observations_csv = None
    if config_data.get("independent_observations_file") not in (None, "none"):
        candidate = constants.RESOURCE_PATH / config_data["independent_observations_file"]
        if candidate.exists():
            independent_observations_csv = candidate

    clustered_observations_csv = None
    if config_data.get("clustered_observations_file") not in (None, "none"):
        candidate = constants.RESOURCE_PATH / config_data["clustered_observations_file"]
        if candidate.exists():
            clustered_observations_csv = candidate

    # Run grid pipeline
    pipeline.compute_grid(
        grid_config=grid_config,
        output_dir=grid_output_dir,
        combination_method=constants.CombinationMethod.RATIO,
        combine_ratio=float(config_data["combination_method"]),
        clustered_observations_csv=clustered_observations_csv,
        independent_observations_csv=independent_observations_csv,
        do_bayesian_update=config_data.get(
            "do_bayesian_update_of_geology_and_terrain_categorical_vs30_values", True
        ),
        noisy=config_data.get("noisy", True),
        n_proc=1,
        max_spatial_boolean_array_memory_gb=config_data.get(
            "max_spatial_boolean_array_memory_gb", 1.0
        ),
        obs_subsample_step_for_clustered=config_data.get(
            "obs_subsample_step_for_clustered", 100
        ),
    )

    # Read the combined raster and pick a pixel center coordinate
    combined_raster = grid_output_dir / constants.COMBINED_VS30_FILENAME
    with rasterio.open(combined_raster) as src:
        vs30_grid = src.read(1)
        stdv_grid = src.read(2)
        transform = src.transform

    # Pick a pixel near the center that has valid data
    row, col = vs30_grid.shape[0] // 2, vs30_grid.shape[1] // 2
    grid_vs30_value = vs30_grid[row, col]
    grid_stdv_value = stdv_grid[row, col]

    # Convert pixel center to NZTM coordinates
    easting, northing = rasterio.transform.xy(transform, row, col)

    # Convert NZTM to WGS84 for the points pipeline input
    from qcore import coordinates

    wgs = coordinates.nztm_to_wgs_depth(np.array([[northing, easting]]))
    lat, lon = wgs[0, 0], wgs[0, 1]

    # Run points pipeline with the same observation files
    result = pipeline.compute_at_locations(
        longitudes=np.array([lon]),
        latitudes=np.array([lat]),
        combination_method=constants.CombinationMethod.RATIO,
        combine_ratio=float(config_data["combination_method"]),
        clustered_observations_csv=clustered_observations_csv,
        independent_observations_csv=independent_observations_csv,
        include_intermediate=True,
        noisy=config_data.get("noisy", True),
        n_proc=1,
    )

    points_vs30_value = result[constants.COL_VS30].iloc[0]
    points_stdv_value = result[constants.COL_COMBINED_STDV].iloc[0]

    # Small Vs30 differences are expected due to raster resampling in the
    # grid pipeline vs direct source sampling in the points pipeline.
    np.testing.assert_allclose(
        points_vs30_value,
        grid_vs30_value,
        rtol=0.02,
        err_msg=f"Vs30 mismatch at ({easting}, {northing}): "
        f"grid={grid_vs30_value:.2f}, points={points_vs30_value:.2f}",
    )

    # Stdv has a wider tolerance because MVN posterior variance is sensitive
    # to small differences in how slope/coast data is sampled at observation
    # locations (grid-resampled rasters vs source data).
    np.testing.assert_allclose(
        points_stdv_value,
        grid_stdv_value,
        rtol=0.30,
        err_msg=f"Stdv mismatch at ({easting}, {northing}): "
        f"grid={grid_stdv_value:.2f}, points={points_stdv_value:.2f}",
    )

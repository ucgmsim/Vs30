"""
Test that grid and points pipelines produce consistent Vs30 values.

Runs the grid pipeline on a small domain, then runs the points pipeline
at three pixel center coordinates and checks the results match.
"""

import numpy as np
import rasterio
import yaml
from qcore import coordinates

from conftest import FIXTURES_DIR

from vs30 import constants, pipeline
from vs30 import config as config_module

def test_grid_and_points_consistency(tmp_path):
    """Grid and points pipelines should produce the same Vs30 at pixel centers."""
    # Use the small test config
    config_file = FIXTURES_DIR / "test_config_small_independent_only.yaml"
    with open(config_file) as f:
        config_data = yaml.safe_load(f)

    for key in constants.CSV_PATH_KEYS:
        if config_data[key]:
            if key in constants.OBSERVATION_CSV_KEYS:
                config_data[key] = FIXTURES_DIR / config_data[key]
            else:
                config_data[key] = constants.RESOURCE_PATH / constants.RESOURCE_SUBDIRS[key] / config_data[key]

    grid_output_dir = tmp_path / "grid_output"

    # Run grid pipeline
    pipeline.compute_grid(
        grid_config=config_module.GridConfig.from_dict(config_data),
        output_dir=grid_output_dir,
        geology_categorical_csv=config_data["geology_categorical_csv"],
        terrain_categorical_csv=config_data["terrain_categorical_csv"],
        clustered_observations_csv=config_data["clustered_observations_csv"],
        independent_observations_csv=config_data["independent_observations_csv"],
        combination_method=constants.CombinationMethod(config_data["combination_method"]),
        combine_ratio=config_data["combine_ratio"],
        noisy=config_data["noisy"],
        do_bayesian_update=config_data["do_bayesian_update"],
        n_proc=1,
    )

    # Read the combined raster and pick pixel center coordinates
    combined_raster = grid_output_dir / constants.COMBINED_VS30_FILENAME
    with rasterio.open(combined_raster) as src:
        vs30_grid = src.read(1)
        stdv_grid = src.read(2)
        transform = src.transform

    # Test at three pixel locations spread across the grid:
    # center, upper-left quarter, and lower-right quarter
    nrows, ncols = vs30_grid.shape
    test_pixels = [
        (nrows // 2, ncols // 2),
        (nrows // 4, ncols // 4),
        (3 * nrows // 4, 3 * ncols // 4),
    ]

    # Convert pixel centers to NZTM then WGS84
    eastings = []
    northings = []
    lons = []
    lats = []
    for row, col in test_pixels:
        e, n = rasterio.transform.xy(transform, row, col)
        eastings.append(e)
        northings.append(n)
        wgs = coordinates.nztm_to_wgs_depth(np.array([[n, e]]))
        lats.append(wgs[0, 0])
        lons.append(wgs[0, 1])

    # Run points pipeline at all three locations
    result = pipeline.compute_at_locations(
        longitudes=np.array(lons),
        latitudes=np.array(lats),
        geology_categorical_csv=config_data["geology_categorical_csv"],
        terrain_categorical_csv=config_data["terrain_categorical_csv"],
        clustered_observations_csv=config_data["clustered_observations_csv"],
        independent_observations_csv=config_data["independent_observations_csv"],
        combination_method=constants.CombinationMethod(config_data["combination_method"]),
        combine_ratio=config_data["combine_ratio"],
        noisy=config_data["noisy"],
        include_intermediate=True,
        n_proc=1,
    )

    for i, (row, col) in enumerate(test_pixels):
        grid_vs30_value = vs30_grid[row, col]
        grid_stdv_value = stdv_grid[row, col]
        points_vs30_value = result[constants.COL_VS30].iloc[i]
        points_stdv_value = result[constants.COL_COMBINED_STDV].iloc[i]
        easting, northing = eastings[i], northings[i]

        # Small Vs30 differences are expected because the grid pipeline samples
        # slope/coast from grid-resampled rasters while the points pipeline
        # samples directly from source data at each observation location.
        np.testing.assert_allclose(
            points_vs30_value,
            grid_vs30_value,
            rtol=0.03,
            err_msg=f"Vs30 mismatch at pixel ({row},{col}) ({easting}, {northing}): "
            f"grid={grid_vs30_value:.2f}, points={points_vs30_value:.2f}",
        )

        # Stdv has a wider tolerance because MVN posterior variance is sensitive
        # to small differences in how slope/coast data is sampled at observation
        # locations (grid-resampled rasters vs source data).
        np.testing.assert_allclose(
            points_stdv_value,
            grid_stdv_value,
            rtol=0.30,
            err_msg=f"Stdv mismatch at pixel ({row},{col}) ({easting}, {northing}): "
            f"grid={grid_stdv_value:.2f}, points={points_stdv_value:.2f}",
        )

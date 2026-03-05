"""
Test that grid and points pipelines produce consistent Vs30 values.

Runs the grid pipeline on a small domain, then runs the points pipeline
at a pixel center coordinate and checks the results match.
"""

import numpy as np
import pandas as pd
import rasterio
import yaml

from conftest import FIXTURES_DIR

from vs30 import constants, pipeline
from vs30.config import Vs30Config


def test_grid_and_points_consistency(tmp_path):
    """Grid and points pipelines should produce the same Vs30 at a pixel center."""
    # Use the small test config
    config_file = FIXTURES_DIR / "test_config_small_independent_only.yaml"
    with open(config_file) as f:
        config = yaml.safe_load(f)
    config["output_dir"] = str(tmp_path / "grid_output")
    config["n_proc"] = 1
    test_config = tmp_path / "config.yaml"
    with open(test_config, "w") as f:
        yaml.dump(config, f)

    # Run grid pipeline
    cfg = Vs30Config.from_yaml(test_config)
    pipeline.run_full_pipeline(cfg)

    # Read the combined raster and pick a pixel center coordinate
    combined_raster = tmp_path / "grid_output" / constants.COMBINED_VS30_FILENAME
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

    # Create a single-point CSV
    locations_csv = tmp_path / "point.csv"
    pd.DataFrame({"longitude": [lon], "latitude": [lat]}).to_csv(
        locations_csv, index=False
    )

    # Run points pipeline with the same config
    output_csv = tmp_path / "point_result.csv"
    pipeline.compute_at_locations(
        cfg=cfg,
        locations_csv=locations_csv,
        output_csv=output_csv,
        lon_column="longitude",
        lat_column="latitude",
        include_intermediate=True,
        n_proc=1,
    )

    result = pd.read_csv(output_csv)
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

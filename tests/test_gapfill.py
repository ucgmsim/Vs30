"""
Integration tests for gap-fill functionality.

Tests that on-land nodata gaps in the combined VS30 output are correctly
identified and filled. The classify_nodata test validates the core
classification at known coastal locations. The grid integration test
verifies pipeline integration on the existing small test domain.
"""

import numpy as np
import rasterio

from conftest import load_test_config

from vs30 import constants, pipeline, gapfill
from vs30 import config as config_module


def test_grid_pipeline_with_gapfill_integration(tmp_path):
    """Grid pipeline should run with gap-fill stage without errors.

    Uses the existing small_independent_only config (inland domain). Gap-fill
    is a no-op here (no coastal gaps), which validates the fast path. The
    classify_nodata tests below validate the core classification logic.
    """
    config_data = load_test_config("small_independent_only")
    grid_config = config_module.GridConfig.from_dict(config_data)

    output_dir = tmp_path / "grid_output"

    result = pipeline.grid_pipeline(
        grid_config=grid_config,
        output_dir=output_dir,
        geology_categorical_csv=config_data["geology_categorical_csv"],
        terrain_categorical_csv=config_data["terrain_categorical_csv"],
        clustered_observations_csv=config_data["clustered_observations_csv"],
        independent_observations_csv=config_data["independent_observations_csv"],
        combination_method=constants.CombinationMethod(config_data["combination_method"]),
        combine_ratio=config_data["combine_ratio"],
        noisy=config_data["noisy"],
        do_bayesian_update=config_data["do_bayesian_update"],
        n_proc=1,
        include_intermediate=True,
        apply_alluvium_slope_mod=config_data["apply_alluvium_slope_mod"],
        apply_coastal_distance_mod=config_data["apply_coastal_distance_mod"],
    )

    combined_vs30 = result["combined_vs30"]

    # Verify the final combined output exists and is readable
    combined_tif = output_dir / constants.COMBINED_VS30_FILENAME
    assert combined_tif.exists()

    with rasterio.open(combined_tif) as src:
        written_vs30 = src.read(1)
        nodata = src.nodata
        written_nan_count = np.sum(written_vs30 == nodata)
        inmem_nan_count = np.sum(np.isnan(combined_vs30))
        assert written_nan_count == inmem_nan_count

    # Intermediate pre-fill output should also exist
    prefill_tif = output_dir / constants.COMBINED_VS30_BEFORE_GAPFILL_FILENAME
    assert prefill_tif.exists()

    with rasterio.open(prefill_tif) as src:
        prefill_vs30 = src.read(1)
        prefill_nan_count = np.sum(prefill_vs30 == src.nodata)
    # Gap-fill should not increase nodata count
    assert written_nan_count <= prefill_nan_count


def test_classify_nodata_excludes_water_and_offshore():
    """classify_nodata should exclude GID=0 (water) and offshore pixels."""
    # Wellington CBD: known on-land location
    onland_location = np.array([[1749000.0, 5427000.0]])
    # Far offshore: known ocean location east of NZ
    offshore_location = np.array([[2200000.0, 5400000.0]])

    # On-land nodata with valid GID -> fillable
    result = gapfill.classify_nodata(
        combined_vs30=np.array([np.nan]),
        geology_ids=np.array([5]),
        locations=onland_location,
    )
    assert result[0] == True, "On-land nodata pixel with valid GID should be fillable"

    # Water pixel (GID=0) -> not fillable
    result = gapfill.classify_nodata(
        combined_vs30=np.array([np.nan]),
        geology_ids=np.array([0]),
        locations=onland_location,
    )
    assert result[0] == False, "Water pixel (GID=0) should not be fillable"

    # Valid pixel (not NaN) -> not fillable
    result = gapfill.classify_nodata(
        combined_vs30=np.array([300.0]),
        geology_ids=np.array([5]),
        locations=onland_location,
    )
    assert result[0] == False, "Valid (non-NaN) pixel should not be fillable"

    # Offshore nodata -> not fillable
    result = gapfill.classify_nodata(
        combined_vs30=np.array([np.nan]),
        geology_ids=np.array([5]),
        locations=offshore_location,
    )
    assert result[0] == False, "Offshore nodata pixel should not be fillable"

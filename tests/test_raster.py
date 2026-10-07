"""Tests for the VS30 raster module."""

import numpy as np
import pytest
import rasterio

from vs30 import config, constants, raster


def test_point_slope_matches_grid_slope_on_cell_edges():
    """Sampling slope at pixel centres picks the same slope.tif cell as the grid's resampling, including on cell edges."""
    # slope.tif has 270 m cells from x = 1060040, so the pixel-centre column
    # x = 1560350 lies exactly on a cell edge: (1560350 - 1060040) / 270 = 1853.
    transform = rasterio.transform.from_bounds(1560100, 5185100, 1561100, 5186100, 10, 10)
    grid_slope = raster.compute_slope_array(
        {"height": 10, "width": 10, "transform": transform, "crs": constants.NZTM_CRS}
    )
    rows, cols = np.mgrid[0:10, 0:10]
    xs, ys = rasterio.transform.xy(transform, rows.ravel(), cols.ravel())

    point_slope = raster.sample_slope_at_points(np.column_stack([xs, ys]))

    np.testing.assert_array_equal(point_slope.reshape(10, 10).astype(np.float32), grid_slope)



def test_geology_ids_for_a_grid_with_no_polygons_are_nodata():
    """A grid that no geology polygon reaches (open sea) gets nodata IDs, not an error."""
    ids, _ = raster.create_category_id_array(
        constants.ModelType.GEOLOGY,
        config.GridConfig(1699600, 1700600, 5899600, 5900600, 100, 100),
    )

    assert (ids == constants.RASTER_ID_NODATA_VALUE).all()



def test_coast_distance_is_capped_where_the_coast_is_out_of_range():
    """Far inland, coast distance is capped at the coastal modification's range instead of coming out as 0."""
    # A central North Island box more than 20 km from any coast.
    transform = rasterio.transform.from_bounds(1830100, 5700100, 1832100, 5702100, 20, 20)

    distances = raster.compute_coast_distance_raster(
        {"height": 20, "width": 20, "transform": transform}
    )

    assert (
        distances == max(constants.HYBRID_GID4_DIST_MAX, constants.HYBRID_GID10_DIST_MAX)
    ).all()


class TestApplyHybridGeologyModifications:
    """Tests for the apply_hybrid_geology_modifications function."""

    @pytest.fixture
    def sample_arrays(self):
        """Create sample arrays for testing.

        ``id_array`` uses GIDs not in ``HYBRID_GEOLOGY_PARAMS`` (i.e. not
        in ``{2, 3, 4, 6}``) by default so that pixels are untouched
        unless a test explicitly overwrites them with a hybrid GID.
        """
        id_array = np.array(
            [
                [1, 5, 7],
                [1, 5, 7],
                [1, 5, 7],
            ],
            dtype=np.uint8,
        )

        vs30_array = np.full((3, 3), 300.0, dtype=np.float32)
        stdv_array = np.full((3, 3), 0.5, dtype=np.float32)

        slope_array = np.array(
            [
                [0.01, 0.05, 0.1],
                [0.2, 0.5, 1.0],
                [2.0, 5.0, 10.0],
            ],
            dtype=np.float32,
        )

        coast_dist_array = np.array(
            [
                [5000.0, 8000.0, 12000.0],
                [15000.0, 20000.0, 25000.0],
                [30000.0, 10000.0, 5000.0],
            ],
            dtype=np.float32,
        )

        return id_array, vs30_array, stdv_array, slope_array, coast_dist_array

    def test_coastal_distance_mod_applies_to_gid4(self, sample_arrays):
        """Test that coastal distance modification applies to geology ID 4."""
        id_array, vs30_array, stdv_array, slope_array, coast_dist_array = sample_arrays

        id_array[1, 0] = 4  # Alluvium

        result_vs30, result_stdv = raster.apply_hybrid_geology_modifications(
            vs30_array.copy(),
            stdv_array.copy(),
            id_array,
            slope_array,
            coast_dist_array,
            apply_alluvium_slope_mod=True,
            apply_coastal_distance_mod=True,
        )

        # GID 4 pixel at (1,0) has coast_dist=15000 (within [8000, 20000]),
        # so the coastal-distance mod overwrites the slope-based vs30.
        # vs30 = 240 + (500 - 240) * (15000 - 8000) / (20000 - 8000) = 391.67
        expected_vs30 = 240 + (500 - 240) * (15000 - 8000) / (20000 - 8000)
        assert result_vs30[1, 0] == pytest.approx(expected_vs30)

    def test_coastal_distance_mod_applies_to_gid10(self, sample_arrays):
        """Test that coastal distance modification applies to geology ID 10."""
        id_array, vs30_array, stdv_array, slope_array, coast_dist_array = sample_arrays

        id_array[2, 1] = 10  # Floodplain

        result_vs30, result_stdv = raster.apply_hybrid_geology_modifications(
            vs30_array.copy(),
            stdv_array.copy(),
            id_array,
            slope_array,
            coast_dist_array,
            apply_alluvium_slope_mod=True,
            apply_coastal_distance_mod=True,
        )

        # GID 10 pixel at (2,1) has coast_dist=10000 (within [8000, 20000]).
        # GID 10 is not in HYBRID_GEOLOGY_PARAMS, so only the coastal-
        # distance mod runs. vs30 = 197 + (500 - 197) * (10000 - 8000) /
        # (20000 - 8000) = 247.5
        expected_vs30 = 197 + (500 - 197) * (10000 - 8000) / (20000 - 8000)
        assert result_vs30[2, 1] == pytest.approx(expected_vs30)

    def test_coastal_distance_mod_clamps_at_minimum(self, sample_arrays):
        """Test that coastal distance modification clamps vs30 at minimum value."""
        id_array, vs30_array, stdv_array, slope_array, coast_dist_array = sample_arrays

        id_array[0, 0] = 4  # Alluvium
        coast_dist_array[0, 0] = 1000.0  # Very close to coast

        result_vs30, _ = raster.apply_hybrid_geology_modifications(
            vs30_array.copy(),
            stdv_array.copy(),
            id_array,
            slope_array,
            coast_dist_array,
            apply_alluvium_slope_mod=True,
            apply_coastal_distance_mod=True,
        )

        # Should clamp at minimum (240)
        assert result_vs30[0, 0] == 240.0

    def test_coastal_distance_mod_clamps_at_maximum(self, sample_arrays):
        """Test that coastal distance modification clamps vs30 at maximum value."""
        id_array, vs30_array, stdv_array, slope_array, coast_dist_array = sample_arrays

        id_array[0, 0] = 4  # Alluvium
        coast_dist_array[0, 0] = 100000.0  # Very far inland

        result_vs30, _ = raster.apply_hybrid_geology_modifications(
            vs30_array.copy(),
            stdv_array.copy(),
            id_array,
            slope_array,
            coast_dist_array,
            apply_alluvium_slope_mod=True,
            apply_coastal_distance_mod=True,
        )

        # Should clamp at maximum (500)
        assert result_vs30[0, 0] == 500.0

    def test_no_modifications_returns_unchanged(self, sample_arrays):
        """All-non-hybrid IDs with coastal mod off leaves arrays untouched."""
        id_array, vs30_array, stdv_array, slope_array, coast_dist_array = sample_arrays

        result_vs30, result_stdv = raster.apply_hybrid_geology_modifications(
            vs30_array.copy(),
            stdv_array.copy(),
            id_array,
            slope_array,
            coast_dist_array,
            apply_alluvium_slope_mod=True,
            apply_coastal_distance_mod=False,
        )

        np.testing.assert_array_equal(result_vs30, vs30_array)
        np.testing.assert_array_equal(result_stdv, stdv_array)
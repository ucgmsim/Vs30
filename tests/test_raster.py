"""Tests for the VS30 raster module."""

import numpy as np
import pytest

from vs30 import raster


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
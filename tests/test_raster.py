"""
Tests for the VS30 raster module.

Tests cover:
- select_vs30_columns_by_priority function
- apply_hybrid_geology_modifications function
- Hybrid model calculations
"""

import tempfile
from pathlib import Path

import numpy as np
import pytest
import rasterio
from rasterio.transform import from_bounds

from vs30 import raster


class TestSelectVs30ColumnsByPriority:
    """Tests for the select_vs30_columns_by_priority function."""

    def test_independent_observations_priority(self):
        """Test that independent observations posterior is preferred."""
        columns = [
            "id",
            "mean_vs30_km_per_s",
            "standard_deviation_vs30_km_per_s",
            "posterior_mean_vs30_km_per_s_independent_observations",
            "posterior_standard_deviation_vs30_km_per_s_independent_observations",
            "posterior_mean_vs30_km_per_s_clustered_observations",
            "posterior_standard_deviation_vs30_km_per_s_clustered_observations",
        ]

        mean_col, std_col = raster.select_vs30_columns_by_priority(columns)

        assert mean_col == "posterior_mean_vs30_km_per_s_independent_observations"
        assert std_col == "posterior_standard_deviation_vs30_km_per_s_independent_observations"

    def test_clustered_observations_second_priority(self):
        """Test that clustered observations posterior is second priority."""
        columns = [
            "id",
            "mean_vs30_km_per_s",
            "standard_deviation_vs30_km_per_s",
            "posterior_mean_vs30_km_per_s_clustered_observations",
            "posterior_standard_deviation_vs30_km_per_s_clustered_observations",
        ]

        mean_col, std_col = raster.select_vs30_columns_by_priority(columns)

        assert mean_col == "posterior_mean_vs30_km_per_s_clustered_observations"
        assert std_col == "posterior_standard_deviation_vs30_km_per_s_clustered_observations"

    def test_standard_columns_fallback(self):
        """Test that standard columns are used as fallback."""
        columns = [
            "id",
            "mean_vs30_km_per_s",
            "standard_deviation_vs30_km_per_s",
        ]

        mean_col, std_col = raster.select_vs30_columns_by_priority(columns)

        assert mean_col == "mean_vs30_km_per_s"
        assert std_col == "standard_deviation_vs30_km_per_s"

    def test_missing_columns_raises_error(self):
        """Test that missing columns raises ValueError."""
        columns = ["id", "some_other_column"]

        with pytest.raises(ValueError, match="Could not find valid VS30"):
            raster.select_vs30_columns_by_priority(columns)

    def test_partial_pair_not_selected(self):
        """Test that having only mean or only std column doesn't match."""
        columns = [
            "id",
            "posterior_mean_vs30_km_per_s_independent_observations",
            # Missing posterior_standard_deviation_vs30_km_per_s_independent_observations
            "mean_vs30_km_per_s",
            "standard_deviation_vs30_km_per_s",
        ]

        mean_col, std_col = raster.select_vs30_columns_by_priority(columns)

        # Should fall back to standard columns since independent is incomplete
        assert mean_col == "mean_vs30_km_per_s"
        assert std_col == "standard_deviation_vs30_km_per_s"


class TestApplyHybridGeologyModifications:
    """Tests for the apply_hybrid_geology_modifications function."""

    @pytest.fixture
    def sample_arrays(self):
        """Create sample arrays for testing."""
        # 3x3 grid with different geology IDs
        id_array = np.array([
            [1, 2, 3],
            [4, 5, 6],
            [7, 10, 11],
        ], dtype=np.uint8)

        vs30_array = np.array([
            [300.0, 300.0, 300.0],
            [300.0, 300.0, 300.0],
            [300.0, 300.0, 300.0],
        ], dtype=np.float32)

        stdv_array = np.array([
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
        ], dtype=np.float32)

        slope_array = np.array([
            [0.01, 0.05, 0.1],
            [0.2, 0.5, 1.0],
            [2.0, 5.0, 10.0],
        ], dtype=np.float32)

        coast_dist_array = np.array([
            [5000.0, 8000.0, 12000.0],
            [15000.0, 20000.0, 25000.0],
            [30000.0, 10000.0, 5000.0],
        ], dtype=np.float32)

        return id_array, vs30_array, stdv_array, slope_array, coast_dist_array

    def test_mod6_applies_to_gid4(self, sample_arrays):
        """Test that mod6 modification applies to geology ID 4."""
        id_array, vs30_array, stdv_array, slope_array, coast_dist_array = sample_arrays

        # Set specific ID for testing
        id_array[1, 0] = 4  # Alluvium

        result_vs30, result_stdv = raster.apply_hybrid_geology_modifications(
            vs30_array.copy(),
            stdv_array.copy(),
            id_array,
            slope_array,
            coast_dist_array,
            mod6=True,
            mod13=False,
            hybrid=False,
            hybrid_mod6_dist_min=8000.0,
            hybrid_mod6_dist_max=20000.0,
            hybrid_mod6_vs30_min=240.0,
            hybrid_mod6_vs30_max=500.0,
        )

        # GID 4 pixel at (1,0) has coast_dist=15000
        # vs30 = 240 + (500-240) * (15000-8000) / (20000-8000)
        # vs30 = 240 + 260 * 7000/12000 = 240 + 151.67 = 391.67
        expected_vs30 = 240 + (500 - 240) * (15000 - 8000) / (20000 - 8000)
        assert np.isclose(result_vs30[1, 0], expected_vs30, rtol=0.01)

    def test_mod13_applies_to_gid10(self, sample_arrays):
        """Test that mod13 modification applies to geology ID 10."""
        id_array, vs30_array, stdv_array, slope_array, coast_dist_array = sample_arrays

        # GID 10 is at position (2, 1) with coast_dist=10000
        result_vs30, result_stdv = raster.apply_hybrid_geology_modifications(
            vs30_array.copy(),
            stdv_array.copy(),
            id_array,
            slope_array,
            coast_dist_array,
            mod6=False,
            mod13=True,
            hybrid=False,
            hybrid_mod13_dist_min=8000.0,
            hybrid_mod13_dist_max=20000.0,
            hybrid_mod13_vs30_min=197.0,
            hybrid_mod13_vs30_max=500.0,
        )

        # GID 10 pixel at (2,1) has coast_dist=10000
        # vs30 = 197 + (500-197) * (10000-8000) / (20000-8000)
        # vs30 = 197 + 303 * 2000/12000 = 197 + 50.5 = 247.5
        expected_vs30 = 197 + (500 - 197) * (10000 - 8000) / (20000 - 8000)
        assert np.isclose(result_vs30[2, 1], expected_vs30, rtol=0.01)

    def test_mod6_clamps_at_minimum(self, sample_arrays):
        """Test that mod6 clamps vs30 at minimum value."""
        id_array, vs30_array, stdv_array, slope_array, coast_dist_array = sample_arrays

        id_array[0, 0] = 4  # Alluvium
        coast_dist_array[0, 0] = 1000.0  # Very close to coast

        result_vs30, _ = raster.apply_hybrid_geology_modifications(
            vs30_array.copy(),
            stdv_array.copy(),
            id_array,
            slope_array,
            coast_dist_array,
            mod6=True,
            mod13=False,
            hybrid=False,
            hybrid_mod6_dist_min=8000.0,
            hybrid_mod6_dist_max=20000.0,
            hybrid_mod6_vs30_min=240.0,
            hybrid_mod6_vs30_max=500.0,
        )

        # Should clamp at minimum (240)
        assert result_vs30[0, 0] == 240.0

    def test_mod6_clamps_at_maximum(self, sample_arrays):
        """Test that mod6 clamps vs30 at maximum value."""
        id_array, vs30_array, stdv_array, slope_array, coast_dist_array = sample_arrays

        id_array[0, 0] = 4  # Alluvium
        coast_dist_array[0, 0] = 100000.0  # Very far inland

        result_vs30, _ = raster.apply_hybrid_geology_modifications(
            vs30_array.copy(),
            stdv_array.copy(),
            id_array,
            slope_array,
            coast_dist_array,
            mod6=True,
            mod13=False,
            hybrid=False,
            hybrid_mod6_dist_min=8000.0,
            hybrid_mod6_dist_max=20000.0,
            hybrid_mod6_vs30_min=240.0,
            hybrid_mod6_vs30_max=500.0,
        )

        # Should clamp at maximum (500)
        assert result_vs30[0, 0] == 500.0

    def test_no_modifications_returns_unchanged(self, sample_arrays):
        """Test that disabling all modifications returns unchanged arrays."""
        id_array, vs30_array, stdv_array, slope_array, coast_dist_array = sample_arrays

        original_vs30 = vs30_array.copy()
        original_stdv = stdv_array.copy()

        result_vs30, result_stdv = raster.apply_hybrid_geology_modifications(
            vs30_array.copy(),
            stdv_array.copy(),
            id_array,
            slope_array,
            coast_dist_array,
            mod6=False,
            mod13=False,
            hybrid=False,
        )

        np.testing.assert_array_equal(result_vs30, original_vs30)
        np.testing.assert_array_equal(result_stdv, original_stdv)


class TestLoadModelValuesFromCSV:
    """Tests for load_model_values_from_csv function."""

    def test_load_valid_csv(self):
        """Test loading a valid model CSV from resources."""
        # Load default geology model (correct path with subdirectory)
        values = raster.load_model_values_from_csv(
            "categorical_vs30_mean_and_stddev/geology/geology_model_prior_mean_and_standard_deviation.csv"
        )

        assert values.shape[1] == 2  # mean and stddev columns
        assert values.dtype == np.float64
        assert len(values) > 0

    def test_load_terrain_csv(self):
        """Test loading terrain model CSV."""
        values = raster.load_model_values_from_csv(
            "categorical_vs30_mean_and_stddev/terrain/terrain_model_prior_mean_and_standard_deviation.csv"
        )

        assert values.shape[1] == 2
        assert len(values) > 0

    def test_file_not_found_raises_error(self):
        """Test that missing CSV file raises FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="CSV file not found"):
            raster.load_model_values_from_csv("nonexistent/path/to/file.csv")


class TestCreateCategoryIdRaster:
    """Tests for create_category_id_raster function."""

    def test_invalid_model_type_raises(self):
        """Test that invalid model type raises ValueError."""
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="model_type must be"):
                raster.create_category_id_raster(
                    "invalid", Path(temp_dir),
                    xmin=1500000, xmax=1505000,
                    ymin=5100000, ymax=5105000,
                    dx=250, dy=250
                )


class TestApplyHybridModificationsAtPoints:
    """Tests for apply_hybrid_modifications_at_points function."""

    def test_requires_coast_raster_for_mod6(self):
        """Test that mod6 requires coast_distance_raster_path."""
        points = np.array([[1500000.0, 5100000.0]])
        vs30 = np.array([300.0])
        stdv = np.array([30.0])
        geology_ids = np.array([4])  # Alluvium (mod6 applies)

        with pytest.raises(ValueError, match="coast_distance_raster_path is required"):
            raster.apply_hybrid_modifications_at_points(
                points, vs30, stdv, geology_ids,
                mod6=True,
                mod13=False,
                coast_distance_raster_path=None,
            )

    @pytest.fixture
    def slope_raster(self, tmp_path):
        """Create a test slope raster."""
        slope_path = tmp_path / "slope.tif"

        # Create a simple slope raster covering a small test area
        xmin, xmax = 1740000, 1760000
        ymin, ymax = 5420000, 5440000
        width = 20
        height = 20

        transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

        # Create slope data varying from 0.01 to 10
        slope_data = np.linspace(0.01, 10, width * height).reshape(height, width).astype(np.float32)

        profile = {
            "driver": "GTiff",
            "width": width,
            "height": height,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:2193",
            "transform": transform,
            "nodata": -9999.0,
        }

        with rasterio.open(slope_path, "w", **profile) as dst:
            dst.write(slope_data, 1)

        return slope_path

    @pytest.fixture
    def coast_distance_raster(self, tmp_path):
        """Create a test coastal distance raster."""
        coast_path = tmp_path / "coast_dist.tif"

        # Same extent as slope raster
        xmin, xmax = 1740000, 1760000
        ymin, ymax = 5420000, 5440000
        width = 20
        height = 20

        transform = from_bounds(xmin, ymin, xmax, ymax, width, height)

        # Create coastal distance data varying from 1000 to 30000 meters
        coast_data = np.linspace(1000, 30000, width * height).reshape(height, width).astype(np.float32)

        profile = {
            "driver": "GTiff",
            "width": width,
            "height": height,
            "count": 1,
            "dtype": "float32",
            "crs": "EPSG:2193",
            "transform": transform,
            "nodata": -9999.0,
        }

        with rasterio.open(coast_path, "w", **profile) as dst:
            dst.write(coast_data, 1)

        return coast_path

    def test_hybrid_slope_modifications_at_points(self, slope_raster, coast_distance_raster):
        """Test slope-based modifications at points."""
        # Points within the test raster extent
        points = np.array([
            [1745000.0, 5425000.0],
            [1750000.0, 5430000.0],
            [1755000.0, 5435000.0],
        ])

        vs30 = np.array([300.0, 300.0, 300.0])
        stdv = np.array([0.5, 0.5, 0.5])
        # GID 2 has slope-based modifications
        geology_ids = np.array([2, 2, 2])

        modified_vs30, modified_stdv = raster.apply_hybrid_modifications_at_points(
            points, vs30.copy(), stdv.copy(), geology_ids,
            slope_raster_path=slope_raster,
            coast_distance_raster_path=coast_distance_raster,
            mod6=False,
            mod13=False,
            hybrid=True,
        )

        # VS30 values should have been modified by slope
        assert modified_vs30.shape == vs30.shape
        assert modified_stdv.shape == stdv.shape
        # Values should differ due to varying slope
        assert not np.allclose(modified_vs30, vs30)

    def test_mod6_coastal_modifications_at_points(self, slope_raster, coast_distance_raster):
        """Test mod6 (alluvium) coastal modifications at points."""
        # Points at different locations
        points = np.array([
            [1745000.0, 5425000.0],
            [1755000.0, 5435000.0],
        ])

        vs30 = np.array([300.0, 300.0])
        stdv = np.array([0.5, 0.5])
        # GID 4 = Alluvium (mod6 applies)
        geology_ids = np.array([4, 4])

        modified_vs30, modified_stdv = raster.apply_hybrid_modifications_at_points(
            points, vs30.copy(), stdv.copy(), geology_ids,
            slope_raster_path=slope_raster,
            coast_distance_raster_path=coast_distance_raster,
            mod6=True,
            mod13=False,
            hybrid=False,
        )

        # VS30 values should be modified (different from input)
        assert modified_vs30.shape == vs30.shape
        # Values should be modified to be within mod6 range [240, 500]
        assert np.all(modified_vs30 >= 240)
        assert np.all(modified_vs30 <= 500)

    def test_no_modifications_disabled(self, slope_raster, coast_distance_raster):
        """Test that disabling all modifications returns similar values."""
        points = np.array([[1750000.0, 5430000.0]])
        vs30 = np.array([300.0])
        stdv = np.array([0.5])
        # Use a geology ID that doesn't have special handling
        geology_ids = np.array([1])

        modified_vs30, modified_stdv = raster.apply_hybrid_modifications_at_points(
            points, vs30.copy(), stdv.copy(), geology_ids,
            slope_raster_path=slope_raster,
            coast_distance_raster_path=coast_distance_raster,
            mod6=False,
            mod13=False,
            hybrid=False,
        )

        # Values should be unchanged
        np.testing.assert_array_equal(modified_vs30, vs30)
        np.testing.assert_array_equal(modified_stdv, stdv)

"""Tests for vs_calc.utils module."""

import numpy as np
import pandas as pd
from vs_calc.utils import effective_stress_from_layers, split_layers_at_depths


def test_empty_layers():
    """Test that empty layers DataFrame returns empty DataFrame."""
    empty_df = pd.DataFrame(
        columns=[
            "layer_thickness_m",
            "unsaturated_unit_weight_kN/m3",
            "saturated_unit_weight_kN/m3",
        ]
    )
    depths = np.array([1.0, 2.0])
    result = split_layers_at_depths(empty_df, depths)
    assert result.empty
    assert len(result) == 0


def test_empty_depths():
    """Test that empty depths array returns original layers unchanged."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [1.0, 2.0],
            "unsaturated_unit_weight_kN/m3": [18.0, 19.0],
            "saturated_unit_weight_kN/m3": [20.0, 21.0],
        }
    )
    depths = np.array([])
    result = split_layers_at_depths(layers, depths)
    pd.testing.assert_frame_equal(result, layers)


def test_single_depth_split_middle_of_layer():
    """Test splitting a single layer at a depth in the middle."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [2.0],
            "unsaturated_unit_weight_kN/m3": [18.0],
            "saturated_unit_weight_kN/m3": [20.0],
        }
    )
    depths = np.array([1.0])
    result = split_layers_at_depths(layers, depths)

    expected = pd.DataFrame(
        {
            "layer_thickness_m": [1.0, 1.0],
            "unsaturated_unit_weight_kN/m3": [18.0, 18.0],
            "saturated_unit_weight_kN/m3": [20.0, 20.0],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_single_depth_at_layer_boundary():
    """Test depth exactly at layer boundary (should not split)."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [1.0, 2.0],
            "unsaturated_unit_weight_kN/m3": [18.0, 19.0],
            "saturated_unit_weight_kN/m3": [20.0, 21.0],
        }
    )
    depths = np.array([1.0])
    result = split_layers_at_depths(layers, depths)

    # Depth at boundary should not split (depth <= top or depth >= bottom)
    # At depth 1.0, first layer is 0-1, second is 1-3
    # For first layer: depth=1.0 >= bottom=1.0, so no split
    # For second layer: depth=1.0 <= top=1.0, so no split
    pd.testing.assert_frame_equal(result, layers)


def test_single_depth_split_multiple_layers():
    """Test splitting multiple layers with a single depth."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [1.0, 2.0, 1.5],
            "unsaturated_unit_weight_kN/m3": [18.0, 19.0, 20.0],
            "saturated_unit_weight_kN/m3": [20.0, 21.0, 22.0],
        }
    )
    depths = np.array([2.5])
    result = split_layers_at_depths(layers, depths)

    # Depth 2.5 is in the second layer (1.0-3.0)
    # Layer 0: 0-1.0 (no split)
    # Layer 1: 1.0-3.0 splits at 2.5 -> 1.0-2.5 and 2.5-3.0
    # Layer 2: 3.0-4.5 (no split)
    expected = pd.DataFrame(
        {
            "layer_thickness_m": [1.0, 1.5, 0.5, 1.5],
            "unsaturated_unit_weight_kN/m3": [18.0, 19.0, 19.0, 20.0],
            "saturated_unit_weight_kN/m3": [20.0, 21.0, 21.0, 22.0],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_multiple_depths_same_layer():
    """Test multiple depths splitting the same layer."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [5.0],
            "unsaturated_unit_weight_kN/m3": [18.0],
            "saturated_unit_weight_kN/m3": [20.0],
        }
    )
    depths = np.array([1.0, 3.0])
    result = split_layers_at_depths(layers, depths)

    # Should split at 1.0 first: 0-1.0, 1.0-5.0
    # Then split at 3.0: 0-1.0, 1.0-3.0, 3.0-5.0
    expected = pd.DataFrame(
        {
            "layer_thickness_m": [1.0, 2.0, 2.0],
            "unsaturated_unit_weight_kN/m3": [18.0, 18.0, 18.0],
            "saturated_unit_weight_kN/m3": [20.0, 20.0, 20.0],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_multiple_depths_different_layers():
    """Test multiple depths splitting different layers."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [1.0, 2.0, 1.5],
            "unsaturated_unit_weight_kN/m3": [18.0, 19.0, 20.0],
            "saturated_unit_weight_kN/m3": [20.0, 21.0, 22.0],
        }
    )
    depths = np.array([0.5, 2.5, 3.5])
    result = split_layers_at_depths(layers, depths)

    # Depth 0.5 splits layer 0: 0-0.5, 0.5-1.0
    # Depth 2.5 splits layer 1: 1.0-2.5, 2.5-3.0
    # Depth 3.5 splits layer 2: 3.0-3.5, 3.5-4.5
    expected = pd.DataFrame(
        {
            "layer_thickness_m": [0.5, 0.5, 1.5, 0.5, 0.5, 1.0],
            "unsaturated_unit_weight_kN/m3": [18.0, 18.0, 19.0, 19.0, 20.0, 20.0],
            "saturated_unit_weight_kN/m3": [20.0, 20.0, 21.0, 21.0, 22.0, 22.0],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_depth_above_all_layers():
    """Test depth above all layers (should not split anything)."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [1.0, 2.0],
            "unsaturated_unit_weight_kN/m3": [18.0, 19.0],
            "saturated_unit_weight_kN/m3": [20.0, 21.0],
        }
    )
    depths = np.array([-0.5])
    result = split_layers_at_depths(layers, depths)

    # Depth above all layers should not split
    pd.testing.assert_frame_equal(result, layers)


def test_depth_below_all_layers():
    """Test depth below all layers (should not split anything)."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [1.0, 2.0],
            "unsaturated_unit_weight_kN/m3": [18.0, 19.0],
            "saturated_unit_weight_kN/m3": [20.0, 21.0],
        }
    )
    depths = np.array([5.0])
    result = split_layers_at_depths(layers, depths)

    # Depth below all layers should not split
    pd.testing.assert_frame_equal(result, layers)


def test_depth_at_zero():
    """Test depth at zero (surface)."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [2.0],
            "unsaturated_unit_weight_kN/m3": [18.0],
            "saturated_unit_weight_kN/m3": [20.0],
        }
    )
    depths = np.array([0.0])
    result = split_layers_at_depths(layers, depths)

    # Depth at 0.0: top=0, depth=0, so depth <= top, no split occurs
    # Original layer should remain unchanged
    pd.testing.assert_frame_equal(result, layers)


def test_single_float_input():
    """Test backward compatibility with single float input."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [2.0],
            "unsaturated_unit_weight_kN/m3": [18.0],
            "saturated_unit_weight_kN/m3": [20.0],
        }
    )
    depths = 1.0  # Single float instead of array
    result = split_layers_at_depths(layers, depths)

    expected = pd.DataFrame(
        {
            "layer_thickness_m": [1.0, 1.0],
            "unsaturated_unit_weight_kN/m3": [18.0, 18.0],
            "saturated_unit_weight_kN/m3": [20.0, 20.0],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_single_int_input():
    """Test backward compatibility with single int input."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [2.0],
            "unsaturated_unit_weight_kN/m3": [18.0],
            "saturated_unit_weight_kN/m3": [20.0],
        }
    )
    depths = 1  # Single int instead of array
    result = split_layers_at_depths(layers, depths)

    expected = pd.DataFrame(
        {
            "layer_thickness_m": [1.0, 1.0],
            "unsaturated_unit_weight_kN/m3": [18.0, 18.0],
            "saturated_unit_weight_kN/m3": [20.0, 20.0],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_unsorted_depths():
    """Test that unsorted depths are sorted before processing."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [5.0],
            "unsaturated_unit_weight_kN/m3": [18.0],
            "saturated_unit_weight_kN/m3": [20.0],
        }
    )
    depths = np.array([3.0, 1.0])  # Unsorted
    result = split_layers_at_depths(layers, depths)

    # Should be sorted internally and split at 1.0 then 3.0
    expected = pd.DataFrame(
        {
            "layer_thickness_m": [1.0, 2.0, 2.0],
            "unsaturated_unit_weight_kN/m3": [18.0, 18.0, 18.0],
            "saturated_unit_weight_kN/m3": [20.0, 20.0, 20.0],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_duplicate_depths():
    """Test that duplicate depths are removed."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [3.0],
            "unsaturated_unit_weight_kN/m3": [18.0],
            "saturated_unit_weight_kN/m3": [20.0],
        }
    )
    depths = np.array([1.0, 1.0, 2.0, 1.0])  # Duplicates
    result = split_layers_at_depths(layers, depths)

    # Should remove duplicates and split at 1.0 and 2.0
    expected = pd.DataFrame(
        {
            "layer_thickness_m": [1.0, 1.0, 1.0],
            "unsaturated_unit_weight_kN/m3": [18.0, 18.0, 18.0],
            "saturated_unit_weight_kN/m3": [20.0, 20.0, 20.0],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


def test_preserve_unit_weights():
    """Test that unit weights are preserved correctly after splitting."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [2.0],
            "unsaturated_unit_weight_kN/m3": [18.5],
            "saturated_unit_weight_kN/m3": [20.5],
        }
    )
    depths = np.array([1.0])
    result = split_layers_at_depths(layers, depths)

    # Both sublayers should have same unit weights as original
    assert all(result["unsaturated_unit_weight_kN/m3"] == 18.5)
    assert all(result["saturated_unit_weight_kN/m3"] == 20.5)


def test_cumulative_thickness_preserved():
    """Test that total thickness remains the same after splitting."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [1.0, 2.0, 1.5],
            "unsaturated_unit_weight_kN/m3": [18.0, 19.0, 20.0],
            "saturated_unit_weight_kN/m3": [20.0, 21.0, 22.0],
        }
    )
    original_total = layers["layer_thickness_m"].sum()

    depths = np.array([0.5, 2.5, 3.5])
    result = split_layers_at_depths(layers, depths)

    result_total = result["layer_thickness_m"].sum()
    assert np.isclose(result_total, original_total)


def test_2d_array_flattened():
    """Test that 2D array is properly flattened. Same as test_multiple_depths_different_layers but with 2D array input."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [1.0, 2.0, 1.5],
            "unsaturated_unit_weight_kN/m3": [18.0, 19.0, 20.0],
            "saturated_unit_weight_kN/m3": [20.0, 21.0, 22.0],
        }
    )
    depths = np.array([[0.5, 2.5, 3.5]])
    result = split_layers_at_depths(layers, depths)

    # Depth 0.5 splits layer 0: 0-0.5, 0.5-1.0
    # Depth 2.5 splits layer 1: 1.0-2.5, 2.5-3.0
    # Depth 3.5 splits layer 2: 3.0-3.5, 3.5-4.5
    expected = pd.DataFrame(
        {
            "layer_thickness_m": [0.5, 0.5, 1.5, 0.5, 0.5, 1.0],
            "unsaturated_unit_weight_kN/m3": [18.0, 18.0, 19.0, 19.0, 20.0, 20.0],
            "saturated_unit_weight_kN/m3": [20.0, 20.0, 21.0, 21.0, 22.0, 22.0],
        }
    )
    pd.testing.assert_frame_equal(result, expected)


# Tests for effective_stress_from_layers function

WATER_UNIT_WEIGHT = 9.81  # kN/m³


def test_effective_stress_empty_layers():
    """Test that empty layers returns empty array."""
    empty_df = pd.DataFrame(
        columns=[
            "layer_thickness_m",
            "unsaturated_unit_weight_kN/m3",
            "saturated_unit_weight_kN/m3",
        ]
    )
    result = effective_stress_from_layers(empty_df, 2.0)
    assert len(result) == 0
    assert isinstance(result, np.ndarray)


def test_effective_stress_none_layers():
    """Test that None layers returns empty array."""
    result = effective_stress_from_layers(None, 2.0)
    assert len(result) == 0
    assert isinstance(result, np.ndarray)


def test_effective_stress_all_above_gwl():
    """Test effective stress when all layers are above groundwater level."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [1.0, 2.0],
            "unsaturated_unit_weight_kN/m3": [18.0, 19.0],
            "saturated_unit_weight_kN/m3": [20.0, 21.0],
        }
    )
    groundwater_level = 5.0  # Below all layers
    result = effective_stress_from_layers(layers, groundwater_level)

    # All layers above GWL, so:
    # - Use unsaturated weights
    # - No pore water pressure
    # Layer 1 bottom at 1.0m: σ' = 18.0 * 1.0 = 18.0 kPa
    # Layer 2 bottom at 3.0m: σ' = 18.0 * 1.0 + 19.0 * 2.0 = 18.0 + 38.0 = 56.0 kPa
    expected = np.array([18.0, 56.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_effective_stress_all_below_gwl():
    """Test effective stress when all layers are below groundwater level."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [1.0, 2.0],
            "unsaturated_unit_weight_kN/m3": [18.0, 19.0],
            "saturated_unit_weight_kN/m3": [20.0, 21.0],
        }
    )
    groundwater_level = 0.0  # At surface
    result = effective_stress_from_layers(layers, groundwater_level)

    # All layers below GWL, so:
    # - Use saturated weights
    # - Full pore water pressure
    # Layer 1 bottom at 1.0m: σ_total = 20.0 * 1.0 = 20.0 kPa
    #   u = 9.81 * 1.0 = 9.81 kPa
    #   σ' = 20.0 - 9.81 = 10.19 kPa
    # Layer 2 bottom at 3.0m: σ_total = 20.0 * 1.0 + 21.0 * 2.0 = 20.0 + 42.0 = 62.0 kPa
    #   u = 9.81 * 3.0 = 29.43 kPa
    #   σ' = 62.0 - 29.43 = 32.57 kPa
    expected = np.array([10.19, 32.57])
    np.testing.assert_array_almost_equal(result, expected, decimal=2)


def test_effective_stress_gwl_in_middle():
    """Test effective stress when groundwater level is in the middle of layers."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [2.0],
            "unsaturated_unit_weight_kN/m3": [18.0],
            "saturated_unit_weight_kN/m3": [20.0],
        }
    )
    groundwater_level = 1.0  # In middle of layer
    result = effective_stress_from_layers(layers, groundwater_level)

    # Layer splits at 1.0m:
    # Layer 1 (0-1.0m): unsaturated, above GWL
    #   σ' = 18.0 * 1.0 = 18.0 kPa (no pore pressure)
    # Layer 2 (1.0-2.0m): saturated, below GWL
    #   σ_total = 18.0 * 1.0 + 20.0 * 1.0 = 38.0 kPa
    #   u = 9.81 * (2.0 - 1.0) = 9.81 kPa
    #   σ' = 38.0 - 9.81 = 28.19 kPa
    expected = np.array([18.0, 28.19])
    np.testing.assert_array_almost_equal(result, expected, decimal=2)


def test_effective_stress_multiple_layers_with_gwl():
    """Test effective stress with multiple layers and GWL in middle."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [1.0, 2.0, 1.5],
            "unsaturated_unit_weight_kN/m3": [18.0, 19.0, 20.0],
            "saturated_unit_weight_kN/m3": [20.0, 21.0, 22.0],
        }
    )
    groundwater_level = 2.5  # In middle of second layer
    result = effective_stress_from_layers(layers, groundwater_level)

    # Layers split at 2.5m:
    # Layer 1 (0-1.0m): unsaturated, above GWL
    #   σ' = 18.0 * 1.0 = 18.0 kPa
    # Layer 2a (1.0-2.5m): unsaturated, above GWL
    #   σ' = 18.0 * 1.0 + 19.0 * 1.5 = 18.0 + 28.5 = 46.5 kPa
    # Layer 2b (2.5-3.0m): saturated, below GWL
    #   σ_total = 18.0 * 1.0 + 19.0 * 1.5 + 21.0 * 0.5 = 18.0 + 28.5 + 10.5 = 57.0 kPa
    #   u = 9.81 * (3.0 - 2.5) = 4.905 kPa
    #   σ' = 57.0 - 4.905 = 52.095 kPa
    # Layer 3 (3.0-4.5m): saturated, below GWL
    #   σ_total = 57.0 + 22.0 * 1.5 = 57.0 + 33.0 = 90.0 kPa
    #   u = 9.81 * (4.5 - 2.5) = 19.62 kPa
    #   σ' = 90.0 - 19.62 = 70.38 kPa
    expected = np.array([18.0, 46.5, 52.095, 70.38])
    np.testing.assert_array_almost_equal(result, expected, decimal=2)


def test_effective_stress_gwl_at_boundary():
    """Test effective stress when GWL is exactly at layer boundary."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [1.0, 2.0],
            "unsaturated_unit_weight_kN/m3": [18.0, 19.0],
            "saturated_unit_weight_kN/m3": [20.0, 21.0],
        }
    )
    groundwater_level = 1.0  # At boundary between layers
    result = effective_stress_from_layers(layers, groundwater_level)

    # GWL at boundary, so no split:
    # Layer 1 (0-1.0m): bottom <= GWL, so treated as above (unsaturated)
    #   σ' = 18.0 * 1.0 = 18.0 kPa
    # Layer 2 (1.0-3.0m): entirely below GWL (saturated)
    #   σ_total = 18.0 * 1.0 + 21.0 * 2.0 = 18.0 + 42.0 = 60.0 kPa
    #   u = 9.81 * (3.0 - 1.0) = 19.62 kPa
    #   σ' = 60.0 - 19.62 = 40.38 kPa
    expected = np.array([18.0, 40.38])
    np.testing.assert_array_almost_equal(result, expected, decimal=2)


def test_effective_stress_single_layer_above_gwl():
    """Test effective stress with single layer above GWL."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [1.0],
            "unsaturated_unit_weight_kN/m3": [18.0],
            "saturated_unit_weight_kN/m3": [20.0],
        }
    )
    groundwater_level = 2.0
    result = effective_stress_from_layers(layers, groundwater_level)

    # Single layer above GWL: unsaturated, no pore pressure
    expected = np.array([18.0])
    np.testing.assert_array_almost_equal(result, expected)


def test_effective_stress_single_layer_below_gwl():
    """Test effective stress with single layer below GWL."""
    layers = pd.DataFrame(
        {
            "layer_thickness_m": [1.0],
            "unsaturated_unit_weight_kN/m3": [18.0],
            "saturated_unit_weight_kN/m3": [20.0],
        }
    )
    groundwater_level = 0.0
    result = effective_stress_from_layers(layers, groundwater_level)

    # Single layer below GWL: saturated with pore pressure
    # σ_total = 20.0 * 1.0 = 20.0 kPa
    # u = 9.81 * 1.0 = 9.81 kPa
    # σ' = 20.0 - 9.81 = 10.19 kPa
    expected = np.array([10.19])
    np.testing.assert_array_almost_equal(result, expected, decimal=2)

import numpy as np
import pandas as pd
from nzgd.constants import WATER_UNIT_WEIGHT_kN_m3


def convert_to_midpoint(
    measures: np.ndarray, depths: np.ndarray, layered: bool = False
):
    """
    Converts the given values using the midpoint method
    Useful for a staggered line plot and integration
    """
    new_depths, new_measures, prev_depth, prev_measure = [], [], None, None
    for ix, depth in enumerate(depths):
        measure = measures[ix]
        if ix == 0:
            new_depths.append(float(0))
            new_measures.append(float(measures[1]) if measure == 0 else float(measure))
        else:
            if prev_depth is not None:
                new_depths.append(
                    float(prev_depth) if layered else float((depth + prev_depth) / 2)
                )
                new_measures.append(float(prev_measure))
                new_depths.append(
                    float(prev_depth) if layered else float((depth + prev_depth) / 2)
                )
                new_measures.append(float(measure))
        if ix == len(depths) - 1:
            # Add extra depth for last value in array
            new_depths.append(float(depth))
            new_measures.append(float(measure))
        if ix != 0 or measure != 0:
            prev_depth = depth
            prev_measure = measure

    return new_measures, new_depths


def normalise_weights(weights: dict):
    """
    Normalises the weights within an error of 0.02 from 1 otherwise throws a ValueError
    """
    if len(weights) != 0:
        inital_sum = sum(weights.values())
        if inital_sum < 0.98 or inital_sum > 1.02:
            raise ValueError("Weights sum is not close enough to 1")
        elif inital_sum != 1:
            new_weights = dict()
            for k, v in weights.items():
                new_weights[k] = v / inital_sum
            return new_weights
        else:
            return weights
    else:
        return weights


def split_layers_at_depths(
    layers: pd.DataFrame, depths_to_split_at: np.ndarray
) -> pd.DataFrame:
    """
    Split layers at the specified depth values. Each depth value will split the layer containing it
    into two parts: one above and one below the depth.

    Parameters
    ----------
    layers : pandas.DataFrame
        DataFrame with columns: ['layer_thickness_m', 'unsaturated_unit_weight_kN/m3', 'saturated_unit_weight_kN/m3']
    depths_to_split_at : np.ndarray
        1D numpy array of depths from surface in meters at which to split layers. Values will be sorted internally.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the same columns, where any layer intersected by any of the depth values is
        split into sublayers, retaining original unsaturated and saturated unit weights for later selection.
    """
    if layers.empty:
        return layers

    # Convert to numpy array and handle single value for backward compatibility
    if isinstance(depths_to_split_at, (int, float)):
        depth_values = np.array([float(depths_to_split_at)])
    else:
        depth_values = np.asarray(depths_to_split_at, dtype=float).flatten()

    # Remove duplicates and sort (np.unique returns sorted unique values)
    depth_values = np.unique(depth_values)

    if len(depth_values) == 0:
        return layers

    # Start with original layers
    current_layers = layers.copy()

    # Apply each depth split sequentially
    for depth in depth_values:
        sublayers = []
        cumulative_depth = 0.0

        for _, row in current_layers.iterrows():
            thickness = float(row["layer_thickness_m"])
            unsat_w = float(row["unsaturated_unit_weight_kN/m3"])
            sat_w = float(row["saturated_unit_weight_kN/m3"])

            top = cumulative_depth
            bottom = cumulative_depth + thickness

            if depth <= top or depth >= bottom:
                # Entirely above or below; keep as-is
                sublayers.append(
                    {
                        "layer_thickness_m": thickness,
                        "unsaturated_unit_weight_kN/m3": unsat_w,
                        "saturated_unit_weight_kN/m3": sat_w,
                    }
                )
            else:
                # Depth intersects this layer; split into two
                above_thickness = depth - top
                below_thickness = bottom - depth
                if above_thickness > 0:
                    sublayers.append(
                        {
                            "layer_thickness_m": above_thickness,
                            "unsaturated_unit_weight_kN/m3": unsat_w,
                            "saturated_unit_weight_kN/m3": sat_w,
                        }
                    )
                if below_thickness > 0:
                    sublayers.append(
                        {
                            "layer_thickness_m": below_thickness,
                            "unsaturated_unit_weight_kN/m3": unsat_w,
                            "saturated_unit_weight_kN/m3": sat_w,
                        }
                    )

            cumulative_depth = bottom

        current_layers = pd.DataFrame(
            sublayers,
            columns=[
                "layer_thickness_m",
                "unsaturated_unit_weight_kN/m3",
                "saturated_unit_weight_kN/m3",
            ],
        )

    return current_layers


def effective_stress_from_layers(
    layers_df: pd.DataFrame, groundwater_level: float
) -> np.ndarray:
    """
    Calculate effective stress using 3-column layer input with groundwater splitting.

    Parameters
    ----------
    layers : pandas.DataFrame
        DataFrame with columns: ['layer_thickness_m', 'unsaturated_unit_weight_kN/m3', 'saturated_unit_weight_kN/m3'].
    groundwater_level : float
        Depth to groundwater level from surface in meters.

    Returns
    -------
    np.ndarray
        Effective stress at the bottom of each resulting sublayer (kPa), after splitting
        the groundwater-intersected layer. Unit weights are chosen here based on position
        relative to groundwater level (unsaturated above, saturated below).

    Raises
    ------
    ValueError
        If the groundwater level is not at the bottom of a layer in layers_df.
    """
    if layers_df is None or len(layers_df) == 0:
        return np.array([])

    # The layer containing the groundwater level in layers_df should have already
    # been split so that the groundwater level is at the bottom of the layer.
    # If this split has been correctly done, calling split_layers_at_depths again should
    # not change the layers_df.
    gwl_split_layers_df = split_layers_at_depths(layers_df, groundwater_level)

    print()

    if not layers_df.equals(split_layers_at_depths(layers_df, groundwater_level)):
        raise ValueError(
            "The groundwater level is not at the bottom of a layer in layers_df."
            "Please use the split_layers_at_depths function to split the layers at the groundwater level first."
        )

    # Compute cumulative bottom depths and top depths
    bottoms = np.cumsum(layers_df["layer_thickness_m"].to_numpy(dtype=float))

    # Mask for layers entirely above groundwater (bottom <= gwl)
    above_mask = bottoms <= groundwater_level

    # Choose unit weights in two steps for clarity:
    # 1) select saturated unit weights for all layers
    # 2) override above groundwater level layers with unsaturated unit weights
    unit_w = layers_df["saturated_unit_weight_kN/m3"].to_numpy(dtype=float)
    unsat_arr = layers_df["unsaturated_unit_weight_kN/m3"].to_numpy(dtype=float)
    unit_w[above_mask] = unsat_arr[above_mask]

    # Layer stresses and cumulative total stress
    layer_stress = layers_df["layer_thickness_m"].to_numpy(dtype=float) * unit_w
    total_stress = np.cumsum(layer_stress)

    # Pore water pressure at layer bottoms given by 9.81 * depth_below_gwl.
    # Set to zero above the groundwater level.
    depth_below_gwl = bottoms - groundwater_level
    depth_below_gwl[above_mask] = 0.0
    pore_water_pressure = WATER_UNIT_WEIGHT_kN_m3 * depth_below_gwl

    # Effective stress at bottoms
    effective_stresses = total_stress - pore_water_pressure

    return effective_stresses

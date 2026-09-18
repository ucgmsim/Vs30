import numpy as np
import pandas as pd

from vs_calc import constants


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
        return layers.copy()

    columns = [
        "layer_thickness_m",
        "unsaturated_unit_weight_kN/m3",
        "saturated_unit_weight_kN/m3",
    ]
    if not set(columns).issubset(layers.columns):
        raise ValueError(f"Layers must contain columns {columns}")
    values = layers[columns].to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError(
            "Layer thicknesses and unit weights must be finite and positive"
        )

    depth_values = np.asarray(depths_to_split_at, dtype=float).ravel()
    if not np.isfinite(depth_values).all():
        raise ValueError("Split depths must be finite")

    bottoms = np.cumsum(values[:, 0])
    interior_depths = depth_values[(depth_values > 0) & (depth_values < bottoms[-1])]
    split_bottoms = np.union1d(bottoms, interior_depths)
    indices = np.searchsorted(bottoms, split_bottoms, side="left")
    result = layers.iloc[indices].copy().reset_index(drop=True)
    result["layer_thickness_m"] = np.diff(np.r_[0.0, split_bottoms])
    return result


def effective_stress_from_layers(
    layers_df: pd.DataFrame, groundwater_level: float
) -> np.ndarray:
    """
    Calculate effective stress at layer bottoms, splitting at groundwater first.

    Parameters
    ----------
    layers_df : pandas.DataFrame
        DataFrame with columns: ['layer_thickness_m', 'unsaturated_unit_weight_kN/m3', 'saturated_unit_weight_kN/m3'].
        Layers must be contiguous and ordered from the ground surface downwards.
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
        If groundwater is negative or non-finite, or layers have invalid thicknesses
        or unit weights. The input DataFrame is not modified.
    """
    if layers_df is None or len(layers_df) == 0:
        return np.array([])

    if (
        groundwater_level is None
        or not np.isfinite(groundwater_level)
        or groundwater_level < 0
    ):
        raise ValueError("Groundwater level must be finite and non-negative")
    layers = split_layers_at_depths(layers_df, groundwater_level)
    thicknesses = layers["layer_thickness_m"].to_numpy(dtype=float)
    bottoms = np.cumsum(thicknesses)
    unit_weights = np.where(
        bottoms <= groundwater_level,
        layers["unsaturated_unit_weight_kN/m3"].to_numpy(dtype=float),
        layers["saturated_unit_weight_kN/m3"].to_numpy(dtype=float),
    )
    total_stress = np.cumsum(thicknesses * unit_weights)
    pore_water_pressure = constants.WATER_UNIT_WEIGHT_KN_M3 * np.maximum(
        bottoms - groundwater_level, 0.0
    )
    return total_stress - pore_water_pressure

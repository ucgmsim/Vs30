import numpy as np
import pandas as pd
from vs_calc import SPT
from vs_calc.constants import SoilType
from nzgd.constants import WATER_UNIT_WEIGHT_kN_m3

def split_layer_at_groundwater_level(layers: pd.DataFrame, groundwater_level: float) -> pd.DataFrame:
    """
    Split the layer containing the groundwater level into two parts: one above the groundwater level and one below.

    Parameters
    ----------
    layers : pandas.DataFrame
        DataFrame with columns: ['layer_thickness_m', 'unsaturated_unit_weight_kN/m3', 'saturated_unit_weight_kN/m3']
    groundwater_level : float
        Depth to groundwater level from surface in meters.

    Returns
    -------
    pandas.DataFrame
        DataFrame with the same columns, where any layer intersected by the groundwater level is
        split into two sublayers, retaining original unsaturated and saturated unit weights for later selection.
    """
    if layers.empty:
        return layers

    sublayers = []
    cumulative_depth = 0.0

    print()

    for _, row in layers.iterrows():
        print()
        thickness = float(row["layer_thickness_m"])
        unsat_w = float(row["unsaturated_unit_weight_kN/m3"])
        sat_w = float(row["saturated_unit_weight_kN/m3"])

        top = cumulative_depth
        bottom = cumulative_depth + thickness

        if groundwater_level <= top or groundwater_level >= bottom:
            # Entirely above or below; keep as-is
            sublayers.append({
                "layer_thickness_m": thickness,
                "unsaturated_unit_weight_kN/m3": unsat_w,
                "saturated_unit_weight_kN/m3": sat_w,
            })
        else:
            # Groundwater intersects this layer; split into two
            above_thk = groundwater_level - top
            below_thk = bottom - groundwater_level
            if above_thk > 0:
                sublayers.append({
                    "layer_thickness_m": above_thk,
                    "unsaturated_unit_weight_kN/m3": unsat_w,
                    "saturated_unit_weight_kN/m3": sat_w,
                })
            if below_thk > 0:
                sublayers.append({
                    "layer_thickness_m": below_thk,
                    "unsaturated_unit_weight_kN/m3": unsat_w,
                    "saturated_unit_weight_kN/m3": sat_w,
                })

        cumulative_depth = bottom

    return pd.DataFrame(sublayers, columns=[
        "layer_thickness_m", "unsaturated_unit_weight_kN/m3", "saturated_unit_weight_kN/m3"
    ])


def effective_stress_from_layers(
    layers: pd.DataFrame, groundwater_level: float
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
    """
    if layers is None or len(layers) == 0:
        return np.array([])

    # Split by groundwater without selecting weights
    layers_df = split_layer_at_groundwater_level(layers, groundwater_level)

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


def calculate_effective_stress(depth: float, soil_type: SoilType, spt: SPT, correlation_func):
    """
    Unified effective stress dispatcher that chooses between simple and advanced calculations.
    
    Parameters
    ----------
    depth : float
        The depth to calculate effective stress for
    soil_type : SoilType
        The soil type at the measurement point
    spt : SPT
        The SPT object containing layer data and groundwater level
    correlation_func : callable
        The correlation-specific effective stress function to use (e.g., effective_stress_brandenberg)
        
    Returns
    -------
    stress : float
        The effective stress value
    sigma : float
        The sigma value for the correlation
    tao : float
        The tao value for the correlation
    b0 : float
        The b0 coefficient
    b1 : float
        The b1 coefficient
    b2 : float
        The b2 coefficient
    """
    if spt.layers is not None:
        # Use improved Jaehwi calculation with layer data
        # Calculate effective stress using the refactored layer-based method
        effective_stresses = effective_stress_from_layers(spt.layers, spt.groundwater_level)
        
        # Get the correlation-specific coefficients using the provided function
        stress, sigma, tao, b0, b1, b2 = correlation_func(depth, soil_type, spt.groundwater_level)
        
        # Replace the stress value with the improved calculation
        # Find the appropriate stress value for this depth using the split sublayers
        # (must use split layers to match the effective_stresses array length)
        split_layers = split_layer_at_groundwater_level(spt.layers, spt.groundwater_level)
        bottoms = np.cumsum(split_layers["layer_thickness_m"].to_numpy(dtype=float))
        layer_index = int(np.searchsorted(bottoms, depth, side="right") - 1)
        if 0 <= layer_index < len(effective_stresses):
            stress = effective_stresses[layer_index]
        
        return stress, sigma, tao, b0, b1, b2
    else:
        # Use existing simple calculation (backward compatibility)
        return correlation_func(depth, soil_type)


def brandenberg_2010(spt: SPT):
    """
    SPT-Vs correlation developed by Brandenberg et al. (2010).

    Parameters
    ----------
    spt : SPT
        The SPT object to use for the correlation.

    Returns
    -------
    vs : np.ndarray
        The Vs values for the given SPT object.
    vs_sd : np.ndarray
        The standard deviation of the Vs values for the given SPT object.
    depth_values : np.ndarray
        The depth values for the Vs values.
    eff_stress : np.ndarray
        The effective stress values for the Vs values.
    """
    # Ensures N60 is calculated before trying to get Vs
    N60 = spt.N60
    vs = []
    vs_sd = []
    depth_values = []
    eff_stress = []
    for depth_idx, depth in enumerate(spt.depth):
        true_d = (
            depth + 0.3048
        )  # Spt testing driven a pile 18 inches into the ground in 3 incremental steps. the
        # number of blows is ignored and only consider the total of the second and third increments. We interests
        # in the vertical effective stress after second increments hence add 12 inches(0.3 m) on top of the start
        # depth given
        cur_N60 = N60[depth_idx]
        if cur_N60 > 0:
            stress, sigma, tao, b0, b1, b2 = calculate_effective_stress(
                true_d, spt.soil_type[depth_idx], spt, effective_stress_brandenberg
            )
            lnVs = (
                b0 + b1 * np.log(cur_N60) + b2 * np.log(stress)
            )  # (Brandendberg et al, 2010)
            total_std = np.sqrt(tao**2 + sigma**2)
            vs.append(np.exp(lnVs))
            vs_sd.append(total_std)
            depth_values.append(depth)
            eff_stress.append(stress)
    return (
        np.asarray(vs),
        np.asarray(vs_sd),
        np.asarray(depth_values),
        np.asarray(eff_stress),
    )


def effective_stress_brandenberg(
    depth: float, soiltype: SoilType = SoilType.Clay, water_table_depth: float = 2
):
    """
    Gets the effective stress for the given depth and soil type.

    Parameters
    ----------
    depth : float
        The depth to get the effective stress for.
    soiltype : SoilType
        The soil type to use for the effective stress calculation.
    water_table_depth : float (optional) default 2
        The depth of the water table, default is 2 m below the ground surface.

    Returns
    -------
    stress : float
        The effective stress for the given depth / soil type.
    sigma : float
        The sigma value for the given depth / soil type.
    tao : float
        The tao value for the given depth / soil type.
    b0 : float
        The b0 value for the given depth / soil type.
    b1 : float
        The b1 value for the given depth / soil type.
    b2 : float
        The b2 value for the given depth / soil type.
    """
    if soiltype == SoilType.Sand:
        b0 = 4.045
        b1 = 0.096
        b2 = 0.236
        tao = 0.217
        # (Brandendberg et al, 2010)
        if depth > water_table_depth:
            stress = water_table_depth * 18 + (depth - water_table_depth) * (20 - 9.81)
        else:
            stress = depth * 18
        if stress <= 200:
            sigma = 0.57 - 0.07 * np.log(stress)
        else:
            sigma = 0.2
        return stress, sigma, tao, b0, b1, b2
    elif soiltype == SoilType.Silt:
        b0 = 3.783
        b1 = 0.178
        b2 = 0.231
        tao = 0.227
        # (Brandendberg et al, 2010)
        if depth > water_table_depth:
            # TODO Ensure correct method is used to calculate effective stress
            stress = water_table_depth * 17 + (depth - water_table_depth) * (19 - 9.81)
        else:
            stress = depth * 17
        if stress <= 200:
            sigma = 0.31 - 0.03 * np.log(stress)
        else:
            sigma = 0.15
        return stress, sigma, tao, b0, b1, b2
    else:
        # default is clay
        b0 = 3.996
        b1 = 0.230
        b2 = 0.164
        tao = 0.227
        # (Brandendberg et al, 2010)
        if depth > water_table_depth:
            stress = water_table_depth * 16 + (depth - water_table_depth) * (18 - 9.81)
        else:
            stress = depth * 16
        if stress <= 200:
            sigma = 0.21 - 0.01 * np.log(stress)
        else:
            sigma = 0.16
        return stress, sigma, tao, b0, b1, b2


def kwak_2015(spt: SPT):
    """
    Baseline SPT-Vs correlation developed by Kwak et al. (2015).

    Parameters
    ----------
    spt : SPT
        The SPT object to use for the correlation.

    Returns
    -------
    vs : np.ndarray
        The Vs values for the given SPT object.
    vs_sd : np.ndarray
        The standard deviation of the Vs values for the given SPT object.
    depth_values : np.ndarray
        The depth values for the Vs values.
    eff_stress : np.ndarray
        The effective stress values for the Vs values.
    """
    # Ensures N60 is calculated before trying to get Vs
    N60 = spt.N60
    vs = []
    vs_sd = []
    depth_values = []
    eff_stress = []
    for depth_idx, depth in enumerate(spt.depth):
        true_d = (
            depth + 0.3048
        )  # Spt testing drives a pile 18 inches into the ground in 3 incremental steps. The
        # number of blows is ignored and we only consider the total of the second and third increments. We are interested
        # in the vertical effective stress after the second increment, hence we add 12 inches (0.3 m) on top of the start
        # depth given
        cur_N60 = N60[depth_idx]
        if cur_N60 > 0:
            stress, sigma, tao, b0, b1, b2 = calculate_effective_stress(
                true_d, spt.soil_type[depth_idx], spt, effective_stress_kwak
            )
            lnVs = b0 + b1 * np.log(cur_N60) + b2 * np.log(stress)  # (Kwak et al, 2015)
            # TODO Calculate the correct standard deviation (Currently using Brandenberg)
            total_std = np.sqrt(tao**2 + sigma**2)
            vs.append(np.exp(lnVs))
            vs_sd.append(total_std)
            depth_values.append(depth)
            eff_stress.append(stress)
    return (
        np.asarray(vs),
        np.asarray(vs_sd),
        np.asarray(depth_values),
        np.asarray(eff_stress),
    )


def effective_stress_kwak(
    depth: float, soiltype: SoilType = SoilType.Clay, water_table_depth: float = 2
):
    """
    Gets the effective stress for the given depth and soil type.

    Parameters
    ----------
    depth : float
        The depth to get the effective stress for.
    soiltype : SoilType
        The soil type to use for the effective stress calculation.
    water_table_depth : float  (optional) default 2
        The depth of the water table, default is 2 m below the ground surface.

    Returns
    -------
    stress : float
        The effective stress for the given depth / soil type.
    sigma : float
        The sigma value for the given depth / soil type.
    tao : float
        The tao value for the given depth / soil type.
    b0 : float
        The b0 value for the given depth / soil type.
    b1 : float
        The b1 value for the given depth / soil type.
    b2 : float
        The b2 value for the given depth / soil type.
    """
    if soiltype == SoilType.Sand:
        b0 = 3.913
        b1 = 0.167
        b2 = 0.216
        tao = 0.217
        # (Brandendberg et al, 2010)
        if depth > water_table_depth:
            stress = water_table_depth * 18 + (depth - water_table_depth) * (20 - 9.81)
        else:
            stress = depth * 18
        if stress <= 200:
            sigma = 0.57 - 0.07 * np.log(stress)
        else:
            sigma = 0.2
        return stress, sigma, tao, b0, b1, b2
    elif soiltype == SoilType.Silt:
        b0 = 3.879
        b1 = 0.255
        b2 = 0.168
        tao = 0.227
        # (Brandendberg et al, 2010)
        if depth > water_table_depth:
            stress = water_table_depth * 17 + (depth - water_table_depth) * (19 - 9.81)
        else:
            stress = depth * 17
        if stress <= 200:
            sigma = 0.31 - 0.03 * np.log(stress)
        else:
            sigma = 0.15
        return stress, sigma, tao, b0, b1, b2
    elif soiltype == SoilType.Gravel:
        b0 = 3.840
        b1 = 0.154
        b2 = 0.285
        tao = 0.369
        # (Brandendberg et al, 2010)
        if depth > water_table_depth:
            stress = water_table_depth * 19 + (depth - water_table_depth) * (21 - 9.81)
        else:
            stress = depth * 19
        if stress <= 200:
            sigma = 0.31 - 0.03 * np.log(stress)
        else:
            sigma = 0.15
        return stress, sigma, tao, b0, b1, b2
    else:
        # default is clay
        b0 = 4.119
        b1 = 0.209
        b2 = 0.165
        tao = 0.227
        # (Brandendberg et al, 2010)
        if depth > water_table_depth:
            stress = water_table_depth * 16 + (depth - water_table_depth) * (18 - 9.81)
        else:
            stress = depth * 16
        if stress <= 200:
            sigma = 0.21 - 0.01 * np.log(stress)
        else:
            sigma = 0.16
        return stress, sigma, tao, b0, b1, b2


def brandenberg_2010_layered(spt: SPT):
    """
    SPT-Vs correlation using Brandenberg et al. (2010) with layer-based effective stress.
    
    This variant requires layer data to be provided in the SPT object. It uses the improved
    Jaehwi effective stress calculation method that accounts for heterogeneous soil layers.
    
    Parameters
    ----------
    spt : SPT
        The SPT object to use for the correlation. Must have spt.layers defined.
        
    Returns
    -------
    vs : np.ndarray
        The Vs values for the given SPT object.
    vs_sd : np.ndarray
        The standard deviation of the Vs values for the given SPT object.
    depth_values : np.ndarray
        The depth values for the Vs values.
    eff_stress : np.ndarray
        The effective stress values for the Vs values.
        
    Raises
    ------
    ValueError
        If spt.layers is None.
    """
    if spt.layers is None:
        raise ValueError("brandenberg_2010_layered requires layer data in SPT object")
    
    return brandenberg_2010(spt)


def kwak_2015_layered(spt: SPT):
    """
    SPT-Vs correlation using Kwak et al. (2015) with layer-based effective stress.
    
    This variant requires layer data to be provided in the SPT object. It uses the improved
    Jaehwi effective stress calculation method that accounts for heterogeneous soil layers.
    
    Parameters
    ----------
    spt : SPT
        The SPT object to use for the correlation. Must have spt.layers defined.
        
    Returns
    -------
    vs : np.ndarray
        The Vs values for the given SPT object.
    vs_sd : np.ndarray
        The standard deviation of the Vs values for the given SPT object.
    depth_values : np.ndarray
        The depth values for the Vs values.
    eff_stress : np.ndarray
        The effective stress values for the Vs values.
        
    Raises
    ------
    ValueError
        If spt.layers is None.
    """
    if spt.layers is None:
        raise ValueError("kwak_2015_layered requires layer data in SPT object")
    
    return kwak_2015(spt)


SPT_CORRELATIONS = {
    "brandenberg_2010": brandenberg_2010,
    "kwak_2015": kwak_2015,
    "brandenberg_2010_layered": brandenberg_2010_layered,
    "kwak_2015_layered": kwak_2015_layered,
}

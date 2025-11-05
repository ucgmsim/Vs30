import numpy as np

from vs_calc import SPT
from vs_calc.constants import SoilType


def calculate_effective_stress(
    depth: float, soil_type: SoilType, spt: SPT, correlation_func
):
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
        # Get the correlation-specific coefficients using the provided function
        stress, sigma, tao, b0, b1, b2 = correlation_func(
            depth, soil_type, spt.groundwater_level
        )

        # Correct the inaccurate effective stress values by replacing the
        # effective stress value at this depth with the result of the
        # layer-based effective stress calculation
        layer_index = int(np.searchsorted(spt._layer_bottoms, depth, side="right"))

        if 0 <= layer_index < len(spt._effective_stresses):
            stress = spt._effective_stresses[layer_index]
        return stress, sigma, tao, b0, b1, b2
    else:
        # If layers are not provided, just return the
        return correlation_func(depth, soil_type)


def brandenberg_2010(spt: SPT):
    """
    SPT-Vs correlation developed by Brandenberg et al. (2010).

    Uses the equation: lnVs = b0 + b1*log(N60) + b2*log(stress)
    This correlation supports Sand, Silt, and Clay soil types only (no Gravel support).

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
    Gets the effective stress and regression coefficients for Brandenberg et al. (2010) correlation.

    The effective stress and sigma calculation formulas are the same as in effective_stress_kwak,
    but the regression coefficients (b0, b1, b2, tao) are specific to Brandenberg et al. (2010).
    This function supports Sand, Silt, and Clay soil types only (no Gravel support).

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

    Uses the same equation as brandenberg_2010 (lnVs = b0 + b1*log(N60) + b2*log(stress)),
    but with different regression coefficients (b0, b1, b2, tao) specific to Kwak et al. (2015).
    The main difference from brandenberg_2010 is the coefficient values and support for Gravel soil type.

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
    Gets the effective stress and regression coefficients for Kwak et al. (2015) correlation.

    The effective stress and sigma calculation formulas are the same as in effective_stress_brandenberg,
    but the regression coefficients (b0, b1, b2, tao) are specific to Kwak et al. (2015).
    This function supports Sand, Silt, Clay, and Gravel soil types (Gravel is not supported by Brandenberg).

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
        # (Kwak et al, 2015)
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
        # (Kwak et al, 2015)
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
        # (Kwak et al, 2015)
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
        # (Kwak et al, 2015)
        if depth > water_table_depth:
            stress = water_table_depth * 16 + (depth - water_table_depth) * (18 - 9.81)
        else:
            stress = depth * 16
        if stress <= 200:
            sigma = 0.21 - 0.01 * np.log(stress)
        else:
            sigma = 0.16
        return stress, sigma, tao, b0, b1, b2


SPT_CORRELATIONS = {
    "brandenberg_2010": brandenberg_2010,
    "kwak_2015": kwak_2015,
}

import numpy as np
from vs_calc import SPT
from vs_calc.constants import SoilType


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
            stress, sigma, tao, b0, b1, b2 = effective_stress_brandenberg(
                true_d, spt.soil_type[depth_idx]
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
            stress, sigma, tao, b0, b1, b2 = effective_stress_kwak(
                true_d, spt.soil_type[depth_idx]
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


SPT_CORRELATIONS = {
    "brandenberg_2010": brandenberg_2010,
    "kwak_2015": kwak_2015,
}

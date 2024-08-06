import numpy as np

from vs_calc.constants import SoilType


def brandenberg_2010(spt):
    """
    SPT-Vs correlation developed by Brandenberg et al. (2010).
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
    return np.asarray(vs), np.asarray(vs_sd), np.asarray(depth_values), np.asarray(eff_stress)


def effective_stress_brandenberg(
    depth: np.ndarray, soiltype: SoilType = SoilType.Clay, water_table_depth=2
):
    """
    water_table_depth: default set to 2 m below ground surface
    Returns stress, sigma, b0, b1, b2 factors
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


def kwak_2015(spt):
    """
    Baseline SPT-Vs correlation developed by Kwak et al. (2015).
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
            stress, sigma, tao, b0, b1, b2 = effective_stress_kwak(
                true_d, spt.soil_type[depth_idx]
            )
            lnVs = (
                b0 + b1 * np.log(cur_N60) + b2 * np.log(stress)
            )  # (Kwak et al, 2015)
            total_std = np.sqrt(tao**2 + sigma**2) #NOTE I HAVE NOT CODED THE CORRECT STDEV YET, THIS IS STILL BRANDENBERG
            vs.append(np.exp(lnVs))
            vs_sd.append(total_std)
            depth_values.append(depth)
            eff_stress.append(stress)
    return np.asarray(vs), np.asarray(vs_sd), np.asarray(depth_values), np.asarray(eff_stress)


def effective_stress_kwak(
    depth: np.ndarray, soiltype: SoilType = SoilType.Clay, water_table_depth=2
):
    """
    water_table_depth: default set to 2 m below ground surface
    Returns stress, sigma, b0, b1, b2 factors
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

import numpy as np

from vs_calc import constants
from vs_calc.SPT import SPT
from vs_calc.constants import SoilType


def calculate_effective_stress(
    depth: float, soil_type: SoilType, spt: SPT, correlation_func
):
    """
    Evaluate the correlation using layer stress when a layer profile is supplied.

    Stress is linear between layer boundaries once the groundwater interface has
    been inserted. Interpolation therefore evaluates the requested depth exactly,
    including the penetration offset used by SPT correlations. Queries beyond
    the supplied profile raise ValueError rather than switching stress models.

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

    if spt.layers is None:
        return correlation_func(depth, soil_type, spt.groundwater_level)

    max_depth = spt._layer_bottoms[-1]
    if not np.isfinite(depth) or depth < 0 or depth > max_depth + 1e-10:
        raise ValueError(
            f"Stress depth {depth:g} m is outside the layer profile (0 to {max_depth:g} m)"
        )
    stress = np.interp(
        depth,
        np.r_[0.0, spt._layer_bottoms],
        np.r_[0.0, spt._effective_stresses],
    )
    return correlation_func(
        depth, soil_type, spt.groundwater_level, effective_stress=stress
    )


def effective_stress_and_sigma(depth, soiltype, water_table_depth, effective_stress):
    """Evaluate the existing soil-specific stress and within-site uncertainty."""
    parameters = {
        SoilType.Sand: (18, 20, 0.57, 0.07, 0.2),
        SoilType.Silt: (17, 19, 0.31, 0.03, 0.15),
        SoilType.Gravel: (19, 21, 0.31, 0.03, 0.15),
        SoilType.Clay: (16, 18, 0.21, 0.01, 0.16),
    }
    unsaturated, saturated, intercept, gradient, sigma_above_200 = parameters.get(
        soiltype, parameters[SoilType.Clay]
    )
    if effective_stress is None:
        stress = min(depth, water_table_depth) * unsaturated + max(
            depth - water_table_depth, 0.0
        ) * (saturated - constants.WATER_UNIT_WEIGHT_KN_M3)
    else:
        stress = effective_stress
    if not np.isfinite(stress) or stress <= 0:
        raise ValueError(
            "Effective stress must be finite and positive for SPT correlations"
        )
    sigma = intercept - gradient * np.log(stress) if stress <= 200 else sigma_above_200
    return stress, sigma


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
            depth + constants.SPT_DEPTH_OFFSET_M
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
    depth: float,
    soiltype: SoilType = SoilType.Clay,
    water_table_depth: float = constants.DEFAULT_GROUNDWATER_LEVEL_M,
    effective_stress: float = None,
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
    effective_stress : float, optional
        Layer-derived stress in kPa. When supplied, both Vs and sigma use it.

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
    elif soiltype == SoilType.Silt:
        b0 = 3.783
        b1 = 0.178
        b2 = 0.231
        tao = 0.227
    else:
        # Preserve the existing clay fallback for unsupported soil types.
        soiltype = SoilType.Clay
        b0 = 3.996
        b1 = 0.230
        b2 = 0.164
        tao = 0.227
    stress, sigma = effective_stress_and_sigma(
        depth, soiltype, water_table_depth, effective_stress
    )
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
            depth + constants.SPT_DEPTH_OFFSET_M
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
    depth: float,
    soiltype: SoilType = SoilType.Clay,
    water_table_depth: float = constants.DEFAULT_GROUNDWATER_LEVEL_M,
    effective_stress: float = None,
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
    effective_stress : float, optional
        Layer-derived stress in kPa. When supplied, both Vs and sigma use it.

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
    elif soiltype == SoilType.Silt:
        b0 = 3.879
        b1 = 0.255
        b2 = 0.168
        tao = 0.227
    elif soiltype == SoilType.Gravel:
        b0 = 3.840
        b1 = 0.154
        b2 = 0.285
        tao = 0.369
    else:
        # default is clay
        b0 = 4.119
        b1 = 0.209
        b2 = 0.165
        tao = 0.227
    stress, sigma = effective_stress_and_sigma(
        depth, soiltype, water_table_depth, effective_stress
    )
    return stress, sigma, tao, b0, b1, b2


SPT_CORRELATIONS = {
    "brandenberg_2010": brandenberg_2010,
    "kwak_2015": kwak_2015,
}

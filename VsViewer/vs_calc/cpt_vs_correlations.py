import numpy as np
import numpy.typing as npt

from .CPT import CPT


def mcgann_2015(cpt: CPT) -> tuple[npt.NDArray, npt.NDArray]:
    """
    CPT-Vs correlation developed by McGann et al. (2015b).
    qc, fs in kPa

    Parameters
    ----------
    cpt : CPT
        A CPT object containing the cone penetration test data.

    Returns
    -------
    tuple of npt.NDArray
        A tuple containing two numpy arrays:
        - VsMcGann: The calculated shear wave velocity values.
        - Vs_SD: The standard deviation of the calculated Vs values.
    """

    VsMcGann = np.array(
        [18.4 * (cpt.Qc * 1000) ** 0.144 * (cpt.Fs * 1000) ** 0.083 * cpt.depth**0.278]
    ).T
    # standard deviation
    Vs_SD = np.zeros([len(cpt.depth), 1])
    for i in range(0, len(cpt.depth)):
        if cpt.depth[i] <= 5:
            Vs_SD[i] = 0.162
        elif 5 < cpt.depth[i] < 10:
            Vs_SD[i] = 0.216 - 0.0108 * (cpt.depth[i])
        else:
            Vs_SD[i] = 0.108
    return VsMcGann, Vs_SD


def mcgann_2018(cpt: CPT) -> tuple[npt.NDArray, npt.NDArray]:
    """
    CPT-Vs correlation developed for Loess soil by McGann et al. (2018)

    Parameters
    ----------
    cpt : CPT
        A CPT object containing the cone penetration test data.

    Returns
    -------
    tuple of npt.NDArray
        A tuple containing two numpy arrays:
        - VsMcGann2: The calculated shear wave velocity values.
        - Vs_SD: The standard deviation of the calculated Vs values.
    """
    VsMcGann2 = np.array(
        [
            103.6
            * (cpt.Qc * 1000) ** 0.0074
            * (cpt.Fs * 1000) ** 0.130
            * cpt.depth**0.253
        ]
    ).T
    Vs_SD = np.ones([len(cpt.depth), 1]) * 0.2367

    return VsMcGann2, Vs_SD


def andrus_2007_holocene(cpt: CPT) -> tuple[npt.NDArray, npt.NDArray]:
    """
    CPT-Vs correlation developed by Andrus et al. (2007) for Holocene-age soils (Equation 5).
    qt in kPa.

    Parameters
    ----------
    cpt : CPT
        A CPT object containing the cone penetration test data.

    Returns
    -------
    tuple of npt.NDArray
        A tuple containing two numpy arrays:
        - VsAnd: The calculated shear wave velocity values.
        - Vs_SD: The standard deviation of the calculated Vs values.
    """
    # Holocene-Age Soils, where ASF = 1
    cpt.qt[cpt.qt <= 0] = 0.0001  # adjust for possible negative qt
    VsAnd = np.array(
        [2.27 * ((cpt.qt * 1000) ** 0.412) * (cpt.Ic**0.989) * (cpt.depth**0.033)]
    ).T
    # residual standard deviation: suggests that 68% of the data fall within 22m/s of the model
    # (see Table 2 for statistics of Equation 6).

    Vs_SD = np.log(22 / VsAnd + 1)
    # Manages when there is a 0 depth in the CPT
    Vs_SD[Vs_SD == np.inf] = 0
    return VsAnd, Vs_SD


def andrus_2007_pleistocene(cpt: CPT) -> tuple[npt.NDArray, npt.NDArray]:
    """
    CPT-Vs correlation developed by Andrus et al. (2007) for Pleistocene-age soils (Equation 7).
    qt in kPa.

    Parameters
    ----------
    cpt : CPT
        A CPT object containing the cone penetration test data.

    Returns
    -------
    tuple of npt.NDArray
        A tuple containing two numpy arrays:
        - VsAnd: The calculated shear wave velocity values.
        - Vs_SD: The standard deviation of the calculated Vs values.
    """
    # Pleistocene-Age Soils, where SF = 1 in
    cpt.qt[cpt.qt <= 0] = 0.0001  # adjust for possible negative qt
    VsAnd = np.array(
        [
            2.62
            * ((cpt.qt * 1000) ** 0.395)
            * (cpt.Ic**0.912)
            * (cpt.depth**0.124)
            * 1.12
        ]
    ).T
    # residual standard deviation suggests that 68% of the data fall within 45m/s of the model
    # (see Table 2 for statistics for Equation 7).
    Vs_SD = np.log(45 / VsAnd + 1)
    # Manages when there is a 0 depth in the CPT
    Vs_SD[Vs_SD == np.inf] = 0
    return VsAnd, Vs_SD


def andrus_2007_tertiary_age_cooper_marl(cpt: CPT) -> tuple[npt.NDArray, npt.NDArray]:
    """
    CPT-Vs correlation developed by Andrus et al. (2007) for Tertiary-Age Cooper Marl deposits (Equation 9).
    qt in kPa.

    Parameters
    ----------
    cpt : CPT
        A CPT object containing the cone penetration test data.

    Returns
    -------
    tuple of npt.NDArray
        A tuple containing two numpy arrays:
        - VsAnd: The calculated shear wave velocity values.
        - Vs_SD: The standard deviation of the calculated Vs values.
    """
    cpt.qt[cpt.qt <= 0] = 0.0001  # adjust for possible negative qt
    VsAnd = np.array([13.0 * ((cpt.qt * 1000) ** 0.382) * (cpt.depth**0.099)]).T
    # residual standard deviation suggests that 68% of the data fall within 67m/s of the model
    # (see Table 2 for statistics for Equation 9).
    Vs_SD = np.log(67 / VsAnd + 1)
    # Manages when there is a 0 depth in the CPT
    Vs_SD[Vs_SD == np.inf] = 0
    return VsAnd, Vs_SD


def robertson_2009(cpt: CPT) -> tuple[npt.NDArray, npt.NDArray]:
    """
    CPT-Vs correlation developed by Robertson (2009).

    Parameters
    ----------
    cpt : CPT
        A CPT object containing the cone penetration test data.

    Returns
    -------
    tuple of npt.NDArray
        A tuple containing two numpy arrays:
        - VsRob: The calculated shear wave velocity values.
        - Vs_SD: The standard deviation of the calculated Vs values.
    """
    cpt.Qtn[cpt.Qtn <= 0] = 0.0001  # adjust for possible negative Qtn
    pa = 0.1
    alpha = 10 ** (0.55 * cpt.Ic + 1.68)
    VsRob = np.array([(alpha * cpt.Qtn) ** 0.5 * (cpt.effStress / pa) ** 0.25]).T
    # standard deviation(not available), set to 0.2
    Vs_SD = np.full((len(VsRob), 1), 0.2)
    return VsRob, Vs_SD


def hegazy_2006(cpt: CPT) -> tuple[npt.NDArray, npt.NDArray]:
    """
    CPT-Vs correlation developed by Hegazy & Mayne(2006).

    Parameters
    ----------
    cpt : CPT
        A CPT object containing the cone penetration test data.

    Returns
    -------
    tuple of npt.NDArray
        A tuple containing two numpy arrays:
        - VsHegazy: The calculated shear wave velocity values.
        - Vs_SD: The standard deviation of the calculated Vs values.
    """
    cpt.Qtn[cpt.Qtn <= 0] = 0.0001  # adjust for possible negative qc1n
    pa = 0.1
    VsHegazy = np.array(
        [0.0831 * cpt.Qtn * (cpt.effStress / pa) ** 0.25 * np.exp(1.786 * cpt.Ic)]
    ).T
    # standard deviation(not available), set to 0.2
    Vs_SD = np.full((len(VsHegazy), 1), 0.2)
    return VsHegazy, Vs_SD


CPT_CORRELATIONS = {
    "andrus_2007_holocene": andrus_2007_holocene,
    "andrus_2007_pleistocene": andrus_2007_pleistocene,
    "andrus_2007_tertiary_age_cooper_marl": andrus_2007_tertiary_age_cooper_marl,
    "robertson_2009": robertson_2009,
    "hegazy_2006": hegazy_2006,
    "mcgann_2015": mcgann_2015,
    "mcgann_2018": mcgann_2018,
}

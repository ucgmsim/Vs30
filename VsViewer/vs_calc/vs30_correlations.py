import numpy as np

# Coefficients from the Boore et al. (2011) paper for conversion from VsZ to Vs30
VS30_BOORE_2011_COEFFS = np.array(
    [
        [0.2046, 1.318, -0.1174, 0.119],
        [-0.06072, 1.482, -0.1423, 0.111],
        [-0.2744, 1.607, -0.1600, 0.103],
        [-0.3723, 1.649, -0.1634, 0.097],
        [-0.4941, 1.707, -0.1692, 0.090],
        [-0.5438, 1.715, -0.1667, 0.084],
        [-0.6006, 1.727, -0.1649, 0.078],
        [-0.6082, 1.707, -0.1576, 0.072],
        [-0.6322, 1.698, -0.1524, 0.067],
        [-0.6118, 1.659, -0.1421, 0.062],
        [-0.5780, 1.611, -0.1303, 0.056],
        [-0.5430, 1.565, -0.1193, 0.052],
        [-0.5282, 1.535, -0.1115, 0.047],
        [-0.4960, 1.494, -0.1020, 0.043],
        [-0.4552, 1.447, -0.09156, 0.038],
        [-0.4059, 1.396, -0.08064, 0.035],
        [-0.3827, 1.365, -0.07338, 0.030],
        [-0.3531, 1.331, -0.06585, 0.027],
        [-0.3158, 1.291, -0.05751, 0.023],
        [-0.2736, 1.250, -0.04896, 0.019],
        [-0.2227, 1.202, -0.03943, 0.016],
        [-0.1768, 1.159, -0.03087, 0.013],
        [-0.1349, 1.120, -0.02310, 0.009],
        [-0.09038, 1.080, -0.01527, 0.006],
        [-0.04612, 1.040, -0.007618, 0.003],
    ]
)

# Coefficients from the Boore et al. (2004) paper for conversion from VsZ to Vs30
VS30_BOORE_2004_COEFFS = [
    [0.042062, 1.0292, 0.07126],
    [0.02214, 1.0341, 0.064722],
    [0.012571, 1.0352, 0.059352999999999996],
    [0.014186, 1.0318, 0.054754],
    [0.0123, 1.0297, 0.050086000000000006],
    [0.013795, 1.0263, 0.045925],
    [0.013893, 1.0237, 0.042219],
    [0.019565, 1.019, 0.039422000000000006],
    [0.024879, 1.0144, 0.036365],
    [0.025613999999999998, 1.0117, 0.033233],
    [0.025439, 1.0095, 0.030181],
    [0.025311, 1.0072, 0.027001],
    [0.0269, 1.0044, 0.024087],
    [0.022207, 1.0042, 0.020825999999999997],
    [0.016891, 1.0043, 0.017676],
    [0.011483000000000002, 1.0045, 0.014691000000000001],
    [0.0065646, 1.0045, 0.011452],
    [0.002519, 1.0043, 0.0083871],
    [0.00077322, 1.0031, 0.0055264],
    [0.00043143000000000006, 1.0015, 0.0027355],
]


def boore_2011(vs_profile):
    """
    VsZ-Vs30 correlation developed by Boore et al. (2011).
    """
    # Get Coeffs from max depth
    max_depth = int(vs_profile.max_depth)
    index = max_depth - 5
    if index < 0:
        raise IndexError("VsProfile is not deep enough")
    C0, C1, C2, SD = VS30_BOORE_2011_COEFFS[index]

    # Compute Vs30 and Vs30_sd
    vs30 = 10 ** (
        C0 + C1 * np.log10(vs_profile.vsz) + C2 * (np.log10(vs_profile.vsz)) ** 2
    )
    log_vsz = np.log(vs_profile.vsz)
    d_vs30 = (
        C1 * 10 ** (C1 * np.log10(log_vsz))
        + 2 * C2 * np.log10(log_vsz) * 10 ** (C2 * np.log10(log_vsz) ** 2)
    ) / log_vsz
    vs30_sd = np.sqrt(SD**2 + (d_vs30**2))
    return vs30, vs30_sd


def boore_2004(vs_profile):
    """
    VsZ-Vs30 correlation developed by Boore et al. (2004).
    """
    # Get Coeffs from max depth
    max_depth = int(vs_profile.max_depth)
    index = max_depth - 10
    if index < 0:
        raise IndexError("VsProfile is not deep enough")
    a, b, sigma = VS30_BOORE_2004_COEFFS[index]
    vs30 = 10 ** (a + b * np.log10(vs_profile.vsz))
    return vs30, sigma


VS30_CORRELATIONS = {"boore_2011": boore_2011, "boore_2004": boore_2004}

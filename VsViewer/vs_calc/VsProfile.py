import numpy as np

from .CPT import CPT
from .utils import convert_to_midpoint
from .constants import CORRELATIONS


# Coefficients from the Boore et al. (2011) paper for conversion from VsZ to Vs30
VS30_COEFFS = np.array(
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


class VsProfile:
    """
    Contains the data for a Vs Profile
    """

    def __init__(
        self,
        cpt_name: str,
        correlation: str,
        vs: np.ndarray,
        vs_sd: np.ndarray,
        depth: np.ndarray,
    ):
        self.cpt_name = cpt_name
        self.correlation = correlation
        # Ensures the max depth does not go below 30m
        # Also cut to the highest int depth
        self.max_depth = int(depth[-1]) if int(depth[-1]) < 30 else 30
        # Ensures that the VsZ calculation will be done using the highest int depth
        # for correlations to Vs30
        int_depth_mask = depth <= self.max_depth
        self.vs = vs[int_depth_mask]
        self.vs_sd = vs_sd[int_depth_mask]
        self.depth = depth[int_depth_mask]

        # VsZ and Vs30 info init for lazy loading
        self._vsz = None
        self._vs30 = None
        self._vs30_sd = None

    @staticmethod
    def from_cpt(cpt: CPT, correlation: str):
        """
        Creates a VsProfile from a CPT and correlation
        """
        # Check Correlation string
        if correlation not in CORRELATIONS.keys():
            raise KeyError(
                f"{correlation} not found in set of correlations {CORRELATIONS.keys()}"
            )
        vs, vs_sd = CORRELATIONS[correlation](cpt)
        return VsProfile(
            cpt.name, correlation, vs.squeeze(), vs_sd.squeeze(), cpt.depth
        )

    @property
    def vsz(self):
        """
        Gets the VsZ value and computes the value if not set
        """
        if self._vsz is None:
            self._vsz = self.calc_vsz()
        return self._vsz

    @property
    def vs30(self):
        """
        Gets the Vs30 value and computes the value if not set
        Will grab value from VsZ if max depth is 30m already
        then no conversion is needed
        """
        if self._vs30 is None:
            self._vs30, self._vs30_sd = self.calc_vs30()
        return self._vs30

    @property
    def vs30_sd(self):
        """
        Gets the Vs30 Standard Deviation value and computes the value if not set
        Will grab value from VsZ if max depth is 30m already
        then no conversion is needed
        """
        if self._vs30_sd is None:
            self._vs30, self._vs30_sd = self.calc_vs30()
        return self._vs30_sd

    def calc_vsz(self):
        """
        Calculates the average Vs at the max Z depth for the given VsProfile
        """
        vs_midpoint, depth_midpoint = convert_to_midpoint(self.vs, self.depth)
        time = 0
        for ix in range(1, len(vs_midpoint), 2):
            change_in_z = depth_midpoint[ix] - depth_midpoint[ix - 1]
            time += change_in_z / vs_midpoint[ix]
        return self.max_depth / time

    def calc_vs30(self):
        """
        Calculates Vs30 and Vs30_sd for the given VsProfile based on VsZ and max depth
        By Boore et al. (2011)
        """
        if self.max_depth == 30:
            # Set Vs30 to VsZ as Z is 30
            vs30, vs30_sd = self.vsz, 0
        else:
            # Get Coeffs from max depth
            max_depth = int(self.max_depth)
            index = max_depth - 5
            if index < 0:
                raise IndexError("CPT is not deep enough")
            C0, C1, C2, SD = VS30_COEFFS[index]

            # Compute Vs30 and Vs30_sd
            vs30 = 10 ** (C0 + C1 * np.log10(self.vsz) + C2 * (np.log10(self.vsz)) ** 2)
            log_vsz = np.log(self.vsz)
            d_vs30 = (
                C1 * 10 ** (C1 * np.log10(log_vsz))
                + 2 * C2 * np.log10(log_vsz) * 10 ** (C2 * np.log10(log_vsz) ** 2)
            ) / log_vsz
            vs30_sd = np.sqrt(SD**2 + (d_vs30**2))

        return vs30, vs30_sd

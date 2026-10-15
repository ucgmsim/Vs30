from enum import Enum, auto

WATER_UNIT_WEIGHT_KN_M3 = 9.81
DEFAULT_GROUNDWATER_LEVEL_M = 2.0
DEFAULT_BOREHOLE_DIAMETER_MM = 150.0
SPT_DEPTH_OFFSET_M = 0.3048

# Provisional standard deviation of ln(Vs30) for a Vs profile before any
# extrapolation to 30 m: the measurement uncertainty Foster et al. (2019) gave
# CPT-derived and other Vs30 data. A stopgap until uncertainty is propagated
# through each correlation; see dev/docs/vs_calc/vs30_uncertainty_followup.md
# on the Vs30 repository's dev_scripts_archive branch.
PROFILE_SIGMA_LN_VS30 = 0.2


class HammerType(Enum):
    Auto = auto()
    Safety = auto()
    Standard = auto()


class SoilType(Enum):
    Clay = auto()
    Silt = auto()
    Sand = auto()
    Gravel = auto()

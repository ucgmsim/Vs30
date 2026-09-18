from enum import Enum, auto

WATER_UNIT_WEIGHT_KN_M3 = 9.81
DEFAULT_GROUNDWATER_LEVEL_M = 2.0
DEFAULT_BOREHOLE_DIAMETER_MM = 150.0
SPT_DEPTH_OFFSET_M = 0.3048


class HammerType(Enum):
    Auto = auto()
    Safety = auto()
    Standard = auto()


class SoilType(Enum):
    Clay = auto()
    Silt = auto()
    Sand = auto()
    Gravel = auto()

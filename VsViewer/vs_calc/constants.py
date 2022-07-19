from enum import Enum, auto


class HammerType(Enum):
    Auto = auto()
    Safety = auto()
    Standard = auto()


class SoilType(Enum):
    Clay = auto()
    Silt = auto()
    Sand = auto()

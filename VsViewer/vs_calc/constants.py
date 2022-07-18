from enum import Enum, auto

from .get_vs_correlations import andrus_2007, robertson_2009, hegazy_2006, mcgann_2015, mcgann_2018

CORRELATIONS = {
    "andrus_2007": andrus_2007,
    "robertson_2009": robertson_2009,
    "hegazy_2006": hegazy_2006,
    "mcgann_2015": mcgann_2015,
    "mcgann_2018": mcgann_2018,
}


class HammerType(Enum):
    Auto = auto()
    Safety = auto()
    Standard = auto()


class SoilType(Enum):
    Clay = auto()
    Silt = auto()
    Sand = auto()

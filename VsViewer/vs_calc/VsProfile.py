from .CPT import CPT
from .constants import CORRELATIONS


class VsProfile:
    """
    Contains the data for a Vs Profile
    """

    def __init__(self, cpt: CPT, correlation: str):
        self.cpt = cpt
        self.correlation = correlation
        self.depth = cpt.depth

        # Check Correlation string
        if correlation not in CORRELATIONS.keys():
            raise KeyError(f"{correlation} not found in set of correlations {CORRELATIONS.keys()}")

        vs, vs_sd = CORRELATIONS[correlation](cpt)
        self.vs = vs.squeeze()
        self.vs_sd = vs_sd.squeeze()

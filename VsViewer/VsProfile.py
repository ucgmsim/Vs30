from CPT import CPT
import constants as const


class VsProfile:
    """
    Contains the data for a Vs Profile
    """

    def __init__(self, cpt: CPT, correlation: str):
        self.cpt = cpt
        self.correlation = correlation
        self.depth = cpt.depth
        vs, vs_sd = const.CORRELATIONS[correlation](cpt)
        self.vs = vs.squeeze()
        self.vs_sd = vs_sd.squeeze()

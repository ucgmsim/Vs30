from VsViewer.CPT import CPT
import VsViewer.constants as const
import VsViewer.get_vs_correlations as get_vs_correlations


class VsProfile:
    """
    Contains the data for a Vs Profile
    """

    def __init__(self, cpt: CPT, correlation: const.Correlation):
        self.cpt = cpt
        self.correlation = correlation
        depth, vs, vs_sd = self.calc_vs_profile()
        self.depth = depth
        self.vs = vs
        self.vs_sd = vs_sd

    def calc_vs_profile(self):
        """
        Calculates the Vs Profile information based on the cpt and specified correlation
        """
        (qt, Ic, Qtn, qc1n, _, effStress) = self.cpt.get_cpt_params()

        if self.correlation == const.Correlation.mcgann:
            (depth, vs, vs_sd) = get_vs_correlations.mcgann(
                self.cpt.depth, self.cpt.Qc, self.cpt.Fs
            )

        elif self.correlation == const.Correlation.andrus:
            (depth, vs, vs_sd) = get_vs_correlations.andrus(Ic, self.cpt.depth, qt)

        elif self.correlation == const.Correlation.robertson:
            (depth, vs, vs_sd) = get_vs_correlations.robertson(
                self.cpt.depth, Ic, Qtn, effStress
            )

        elif self.correlation == const.Correlation.hegazy:
            (depth, vs, vs_sd) = get_vs_correlations.hegazy(
                self.cpt.depth, Ic, qc1n, effStress
            )

        elif self.correlation == const.Correlation.mcgann2:
            (depth, vs, vs_sd) = get_vs_correlations.mcgann2(
                self.cpt.depth, self.cpt.Qc, self.cpt.Fs
            )

        else:
            raise ValueError(
                f"Correlation {self.correlation.name} does not have a correlation function"
            )

        return depth, vs, vs_sd

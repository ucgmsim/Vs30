from typing import Iterable

from .VsProfile import VsProfile


def calculate_weighted_vs30(
    vs_profiles: Iterable[VsProfile], cpt_weights: dict, correlation_weights: dict
):
    """
    Calculates the weighted Vs30 by combining each of the Vs30Profiles
    with set weights for the CPT's and the Correlations
    """
    average_vs30, average_vs30_sd = 0, 0
    for vs_profile in vs_profiles:
        weight = (
            cpt_weights[vs_profile.cpt.cpt_ffp.stem]
            * correlation_weights[vs_profile.correlation]
        )
        average_vs30 += vs_profile.vs30 * weight
        average_vs30_sd += vs_profile.vs30_sd * weight

    return average_vs30, average_vs30_sd

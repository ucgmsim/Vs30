from typing import Iterable

import numpy as np

from .VsProfile import VsProfile


def calculate_weighted_vs30(
    vs_profiles: Iterable[VsProfile], cpt_weights: dict, correlation_weights: dict
):
    """
    Calculates the weighted Vs30 by combining each of the Vs30Profiles
    with set weights for the CPT's and the Correlations
    """
    average_vs30 = sum(
        cpt_weights[vs_profile.cpt_name]
        * correlation_weights[vs_profile.correlation]
        * vs_profile.vs30
        for vs_profile in vs_profiles
    )
    # Calculate variance / sigma
    # Based on the equation variance = sum(W_i*Sigma_i^2) + sum(W_i*(x - mu)^2)
    average_vs30_variance = 0
    for vs_profile in vs_profiles:
        weight = (
            cpt_weights[vs_profile.cpt_name]
            * correlation_weights[vs_profile.correlation]
        )
        average_vs30_variance += weight * np.square(vs_profile.vs30_sd)
        average_vs30_variance += weight * np.square(vs_profile.vs30 - average_vs30)
    average_vs30_sd = np.sqrt(average_vs30_variance)
    return average_vs30, average_vs30_sd

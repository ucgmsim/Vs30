from typing import Iterable, Dict

import numpy as np

from VsViewer.vs_calc.VsProfile import VsProfile


def calculate_weighted_vs30(
    vs_profiles: Iterable[VsProfile], vs_weights: Dict, correlation_weights: Dict
):
    """
    Calculates the weighted Vs30 by combining each of the Vs30Profiles
    with set weights for the VsProfile's and the Correlations
    """
    # To deal with VsProfiles straight that have no correlation
    correlation_weights[""] = 1
    average_vs30 = sum(
        vs_weights[vs_profile.name]
        * correlation_weights[vs_profile.correlation]
        * vs_profile.vs30
        for vs_profile in vs_profiles
    )
    # Calculate variance / sigma
    # Based on the equation variance = sum(W_i*Sigma_i^2) + sum(W_i*(x - mu)^2)
    average_vs30_variance = 0
    for vs_profile in vs_profiles:
        weight = (
            vs_weights[vs_profile.name]
            * correlation_weights[vs_profile.correlation]
        )
        average_vs30_variance += weight * np.square(vs_profile.vs30_sd)
        average_vs30_variance += weight * np.square(vs_profile.vs30 - average_vs30)
    average_vs30_sd = np.sqrt(average_vs30_variance)
    return average_vs30, average_vs30_sd

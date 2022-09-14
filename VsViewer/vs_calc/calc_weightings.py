from typing import Iterable, Dict

import numpy as np

from vs_calc.VsProfile import VsProfile
from vs_calc.utils import convert_to_midpoint, normalise_weights


def get_weight(
    vs_profile: VsProfile,
    vs_weights: Dict,
    vs_correlations: Dict,
    vs30_correlations: Dict,
):
    return (
        vs_weights[vs_profile.name]
        * (
            1
            if vs_profile.vs_correlation is None
            else vs_correlations[vs_profile.vs_correlation]
        )
        * (
            1
            if vs_profile.vs30_correlation is None
            else vs30_correlations[vs_profile.vs30_correlation]
        )
    )


def calculate_weighted_vs30(
    vs_profiles: Iterable[VsProfile],
    vs_weights: dict,
    cpt_vs_correlation_weights: dict,
    spt_vs_correlation_weights: dict,
    vs30_correlation_weights: dict,
):
    """
    Calculates the weighted Vs30 by combining each of the Vs30Profiles
    with set weights for the VsProfile's and the Correlations
    """
    vs_weights, vs_correlation_weights, spt_vs_correlation_weights, vs30_correlation_weights = (
        normalise_weights(vs_weights),
        normalise_weights(cpt_vs_correlation_weights),
        normalise_weights(spt_vs_correlation_weights),
        normalise_weights(vs30_correlation_weights),
    )
    vs_correlation_weights.update(spt_vs_correlation_weights)
    average_vs30 = sum(
        get_weight(
            vs_profile, vs_weights, vs_correlation_weights, vs30_correlation_weights
        )
        * vs_profile.vs30
        for vs_profile in vs_profiles
    )
    # Calculate variance / sigma
    # Based on the equation variance = sum(W_i*Sigma_i^2) + sum(W_i*(x - mu)^2)
    average_vs30_variance = 0
    for vs_profile in vs_profiles:
        weight = get_weight(
            vs_profile, vs_weights, vs_correlation_weights, vs30_correlation_weights
        )
        average_vs30_variance += weight * np.square(vs_profile.vs30_sd)
        average_vs30_variance += weight * np.square(vs_profile.vs30 - average_vs30)
    average_vs30_sd = 0 if average_vs30_variance == 0 else np.log(np.sqrt(average_vs30_variance))
    return average_vs30, average_vs30_sd


def calc_average_vs_midpoint(
    vs_profiles: Iterable[VsProfile],
    vs_weights: dict,
    vs_correlation_weights: dict,
    vs30_correlation_weights: dict,
):
    """
    Calculates the weighted average midpoint plot for different VsProfiles with different weightings
    Re-Samples at 0.005 as depth may not be consistent with each VsProfile
    Ignores 0 values and starting constants when not starting from depth of 0
    """
    vs_weights, vs_correlation_weights, vs30_correlation_weights = (
        normalise_weights(vs_weights),
        normalise_weights(vs_correlation_weights),
        normalise_weights(vs30_correlation_weights),
    )
    depth_values = []
    weighted_vs = []
    weighted_sd = []
    max_depth = max([vs_profile.max_depth for vs_profile in vs_profiles])
    cur_depth = 0
    increment = 0.005
    cur_zero = False
    midpoint_vs_profiles = dict()
    for vs_profile in vs_profiles:
        vs, depth = convert_to_midpoint(vs_profile.vs, vs_profile.depth)
        vs_sd, _ = convert_to_midpoint(vs_profile.vs_sd, vs_profile.depth)
        midpoint_vs_profiles[vs_profile.name] = {
            "vs": vs,
            "depth": depth,
            "vs_sd": vs_sd,
        }
    while cur_depth <= max_depth:
        weighted_value = 0
        total_weight = 0
        vs_sds = dict()
        for vs_profile in vs_profiles:
            if vs_profile.depth[0] <= cur_depth <= vs_profile.depth[-1]:
                # Depth is within valid range for this vsProfile
                depth = midpoint_vs_profiles[vs_profile.name]["depth"]
                vs = midpoint_vs_profiles[vs_profile.name]["vs"]
                vs_sd = midpoint_vs_profiles[vs_profile.name]["vs_sd"]
                lower_value_idx = np.flatnonzero(cur_depth >= np.asarray(depth))[-1]
                value = vs[lower_value_idx]
                if value != 0:
                    weighted_value += value * get_weight(
                        vs_profile,
                        vs_weights,
                        vs_correlation_weights,
                        vs30_correlation_weights,
                    )
                    total_weight += get_weight(
                        vs_profile,
                        vs_weights,
                        vs_correlation_weights,
                        vs30_correlation_weights,
                    )
                    vs_sds[vs_profile.name] = {
                        "vs_sd": vs_sd[lower_value_idx],
                        "vs_profile": vs_profile,
                        "vs": value,
                    }
        if weighted_value != 0:
            # Calculate variance / sigma
            # Based on the equation variance = sum(W_i*Sigma_i^2) + sum(W_i*(x - mu)^2)
            average_vs_variance = 0
            for vs_profile_name, vs_sd_data in vs_sds.items():
                weight = get_weight(
                    vs_sd_data["vs_profile"],
                    vs_weights,
                    vs_correlation_weights,
                    vs30_correlation_weights,
                ) * (1 / total_weight)
                average_vs_variance += weight * np.square(vs_sd_data["vs_sd"])
            average_vs_sd = np.sqrt(average_vs_variance)
            # Check if the values are in a zero state
            if cur_zero:
                if len(weighted_vs) > 0:
                    middle_depth = (depth_values[-1] + cur_depth) / 2
                    depth_values.extend([middle_depth, middle_depth])
                    weighted_vs.extend(
                        [weighted_vs[-1], weighted_value * (1 / total_weight)]
                    )
                    weighted_sd.extend([weighted_sd[-1], average_vs_sd])
                cur_zero = False
            # Ensures any missing weights re-balance the value to a 1 weighting
            weighted_vs.append(weighted_value * (1 / total_weight))
            weighted_sd.append(average_vs_sd)
            depth_values.append(cur_depth)
        else:
            # Keep a record of when the next point is to ensure averaging between values when there is no data
            cur_zero = True
        cur_depth += increment
    # Add zero value if not found
    if depth_values[0] != 0:
        depth_values.insert(0, 0)
        weighted_vs.insert(0, weighted_vs[0])
        weighted_sd.insert(0, weighted_sd[0])
    return depth_values, weighted_vs, weighted_sd

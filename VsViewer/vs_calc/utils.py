import numpy as np

from .VsProfile import VsProfile


def convert_to_midpoint(measures: np.ndarray, depths: np.ndarray):
    """
    Converts the given values using the midpoint method
    Useful for a staggered line plot and integration
    """
    new_depths, new_measures, prev_depth, prev_measure = [], [], None, None
    for ix, depth in enumerate(depths):
        measure = measures[ix]
        if ix == 0 and depth != 0:
            new_depths.append(0)
            new_measures.append(measure)
        else:
            if prev_depth is not None:
                new_depths.append((depth + prev_depth) / 2)
                new_measures.append(prev_measure)
                new_depths.append((depth + prev_depth) / 2)
                new_measures.append(measure)
        if ix == len(depths) - 1:
            # Add extra depth for last value in array
            new_depths.append(depth)
            new_measures.append(measure)
        prev_depth = depth
        prev_measure = measure
    return new_measures, new_depths


def calc_vsz(vs_profile: VsProfile):
    """
    Calculates the average Vs at the max Z depth for the given VsProfile
    """
    vs_midpoint, depth_midpoint = convert_to_midpoint(vs_profile.vs, vs_profile.depth)
    time = 0
    for ix in range(1, len(vs_midpoint), 2):
        change_in_z = depth_midpoint[ix] - depth_midpoint[ix - 1]
        time += change_in_z / vs_midpoint[ix]
    vsz = vs_profile.depth[-1] / time
    return vsz
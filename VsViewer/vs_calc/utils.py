import numpy as np


def convert_to_midpoint(measures: np.ndarray, depths: np.ndarray):
    """
    Converts the given values using the midpoint method
    Useful for a staggered line plot and integration
    """
    new_depths, new_measures, prev_depth, prev_measure = [], [], None, None
    for ix, depth in enumerate(depths):
        measure = measures[ix]
        if ix == 0:
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

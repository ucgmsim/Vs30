import numpy as np


def convert_to_midpoint(measures: np.ndarray, depths: np.ndarray, layered: bool = False):
    """
    Converts the given values using the midpoint method
    Useful for a staggered line plot and integration
    """
    new_depths, new_measures, prev_depth, prev_measure = [], [], None, None
    for ix, depth in enumerate(depths):
        measure = measures[ix]
        if ix == 0:
            new_depths.append(float(0))
            new_measures.append(float(measures[1]) if measure == 0 else float(measure))
        else:
            if prev_depth is not None:
                new_depths.append(float(prev_depth) if layered else float((depth + prev_depth) / 2))
                new_measures.append(float(prev_measure))
                new_depths.append(float(prev_depth) if layered else float((depth + prev_depth) / 2))
                new_measures.append(float(measure))
        if ix == len(depths) - 1:
            # Add extra depth for last value in array
            new_depths.append(float(depth))
            new_measures.append(float(measure))
        if ix != 0 or measure != 0:
            prev_depth = depth
            prev_measure = measure

    return new_measures, new_depths


def normalise_weights(weights: dict):
    """
    Normalises the weights within an error of 0.02 from 1 otherwise throws a ValueError
    """
    if len(weights) != 0:
        inital_sum = sum(weights.values())
        if inital_sum < 0.98 or inital_sum > 1.02:
            raise ValueError("Weights sum is not close enough to 1")
        elif inital_sum != 1:
            new_weights = dict()
            for k, v in weights.items():
                new_weights[k] = v / inital_sum
            return new_weights
        else:
            return weights
    else:
        return weights

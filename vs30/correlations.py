"""Spatial correlation functions for MVN spatial adjustment."""

import functools
from collections.abc import Callable

import numpy as np
import scipy.special

from vs30 import constants


def exponential(
    distances: np.ndarray,
    phi: float,
    min_dist: float = constants.MIN_DIST_ENFORCED,
) -> np.ndarray:
    """
    Calculate exponential correlation from distances.

    Parameters
    ----------
    distances : ndarray
        Array of distances in meters.
    phi : float
        Correlation length parameter in meters.
    min_dist : float, optional
        Minimum distance enforced to prevent division issues. Default is MIN_DIST_ENFORCED.

    Returns
    -------
    ndarray
        Correlation values between 0 and 1. Same shape as distances.
    """
    return np.exp(-np.maximum(min_dist, distances) / phi)


def matern(
    distances: np.ndarray,
    range_m: float,
    kappa: float,
    min_dist: float = constants.MIN_DIST_ENFORCED,
) -> np.ndarray:
    """
    Calculate Matérn correlation from distances, as in Foster et al. (2019).

    Implemented to match R's gstat::vgm() (the tool used by Foster et al., 2019).

    This implementation uses the gstat parameterization: `range_m` is used
    directly as the kernel scale, with no sqrt(2 kappa) factor that some other
    libraries apply. Range values fit with gstat are therefore not
    interchangeable with other libraries' Matérn implementations.

    Parameters
    ----------
    distances : ndarray
        Array of distances in meters.
    range_m : float
        Matérn range (scale) parameter in meters (gstat convention).
    kappa : float
        Matérn smoothness parameter.
    min_dist : float, optional
        Minimum distance enforced to prevent numerical issues.

    Returns
    -------
    ndarray
        Correlation values in [0, 1]. Same shape as distances.
    """
    d = np.maximum(min_dist, distances)
    scaled = d / range_m
    # scipy.special: gamma = Gamma function; kv = modified Bessel function (2nd kind).
    rho = (
        (2 ** (1 - kappa) / scipy.special.gamma(kappa))
        * (scaled ** kappa)
        * scipy.special.kv(kappa, scaled)
    )
    # Clamp NaN from numerical edge cases (kv can overflow for very small d)
    return np.where(np.isfinite(rho), rho, 1.0)


def resolve_correlation_function(
    config_section: dict,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Resolve a correlation config section into a callable.

    Parameters
    ----------
    config_section : dict
        Must contain a "model" key ("exponential" or "matern") plus the
        model-specific parameters.

    Returns
    -------
    callable
        Function with signature (distances: ndarray) -> ndarray.
    """
    model = config_section["model"]
    if model == "exponential":
        return functools.partial(
            exponential,
            phi=config_section["phi"],
        )
    elif model == "matern":
        return functools.partial(
            matern,
            range_m=config_section["range"],
            kappa=config_section["kappa"],
        )
    else:
        raise ValueError(f"Unknown correlation model: {model}")

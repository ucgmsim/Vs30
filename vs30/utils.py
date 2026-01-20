from pathlib import Path

import numpy as np
import yaml

from vs30.config import get_default_config

# ============================================================================
# Coordinate Conversion Functions
# ============================================================================


def _resolve_base_path(config_path: Path) -> Path:
    """
    Resolve base path from config file location.

    The base path is the parent directory of the vs30 package directory.
    For example, if config is at vs30/config.yaml, base_path is the workspace root.

    Parameters
    ----------
    config_path : Path
        Path to config.yaml file.

    Returns
    -------
    Path
        Base path for input/output files.
    """
    if config_path.name == "config.yaml" and config_path.parent.name == "vs30":
        return config_path.parent.parent
    else:
        return config_path.parent


def load_config(config_path: Path) -> dict:
    """
    Load configuration from YAML file.

    Parameters
    ----------
    config_path : Path
        Path to config.yaml file.

    Returns
    -------
    dict
        Configuration dictionary.
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


# ============================================================================
# Helper Functions for Correlation
# ============================================================================


def correlation_function(distances: np.ndarray, phi: float) -> np.ndarray:
    """
    Calculate correlation function from distances.

    Parameters
    ----------
    distances : ndarray
        Array of distances in meters. Can be scalar, 1D, or 2D (distance matrix).
    phi : float
        Correlation length parameter in meters.

    Returns
    -------
    correlations : ndarray
        Correlation values between 0 and 1. Same shape as distances.

    Notes
    -----
    Uses exponential correlation function: 1 / exp(distance / phi)
    """
    cfg = get_default_config()
    return 1 / np.exp(np.maximum(cfg.min_dist_enforced, distances) / phi)


# ============================================================================
# Model Combination Functions
# ============================================================================


def combine_models_at_points(
    geol_vs30: np.ndarray,
    geol_stdv: np.ndarray,
    terr_vs30: np.ndarray,
    terr_stdv: np.ndarray,
    combination_method: str | float,
    epsilon: float = 1e-10,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combine geology and terrain Vs30 models at points using weighted average.

    This function implements the combination logic used by both the sequential
    and parallel point-based computation paths.

    Parameters
    ----------
    geol_vs30 : ndarray
        Geology model Vs30 values at points.
    geol_stdv : ndarray
        Geology model standard deviation at points.
    terr_vs30 : ndarray
        Terrain model Vs30 values at points.
    terr_stdv : ndarray
        Terrain model standard deviation at points.
    combination_method : str or float
        Either a ratio (float, where 1.0 means equal weighting) or
        "standard_deviation_weighting" for inverse variance weighting.
    epsilon : float, optional
        Small value to prevent division by zero in variance weighting.
        Default is 1e-10.

    Returns
    -------
    combined_vs30 : ndarray
        Combined Vs30 values.
    combined_stdv : ndarray
        Combined standard deviation values.
    """
    try:
        # Try parsing as float (ratio-based combination)
        ratio = float(combination_method)
        combined_vs30 = geol_vs30 * ratio + terr_vs30 * (1 - ratio)
        combined_stdv = np.sqrt(
            (geol_stdv * ratio) ** 2 + (terr_stdv * (1 - ratio)) ** 2
        )
    except (ValueError, TypeError):
        # String-based combination method
        if combination_method == "standard_deviation_weighting":
            # Inverse variance weighting
            geol_weight = 1 / (geol_stdv**2 + epsilon)
            terr_weight = 1 / (terr_stdv**2 + epsilon)
            total_weight = geol_weight + terr_weight

            combined_vs30 = (
                geol_vs30 * geol_weight + terr_vs30 * terr_weight
            ) / total_weight
            combined_stdv = np.sqrt(1 / total_weight)
        else:
            # Default to 0.5 ratio (equal weighting)
            combined_vs30 = (geol_vs30 + terr_vs30) / 2
            combined_stdv = np.sqrt((geol_stdv**2 + terr_stdv**2) / 4)

    return combined_vs30, combined_stdv

from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.distance import cdist, euclidean


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


def _find_config_file(config: Path | None) -> Path:
    """
    Find config.yaml file if not specified.

    Parameters
    ----------
    config : Path | None
        Config path if provided, None otherwise.

    Returns
    -------
    Path
        Path to config.yaml file.

    Raises
    ------
    FileNotFoundError
        If config file cannot be found.
    """
    if config is not None:
        return config

    cwd = Path.cwd()
    # Try common locations
    candidates = [
        cwd / "vs30" / "config.yaml",
        cwd / "config.yaml",
        cwd.parent / "vs30" / "config.yaml",
    ]

    for candidate in candidates:
        if candidate.exists():
            return candidate

    raise FileNotFoundError(
        "Could not find config.yaml. Tried:\n"
        + "\n".join(f"  - {c}" for c in candidates)
        + "\nPlease specify --config PATH"
    )


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
# Helper Functions for Distance and Correlation
# ============================================================================


def euclidean_distance(p1: np.ndarray, p2: np.ndarray) -> float:
    """
    Calculate Euclidean distance between two points.

    Parameters
    ----------
    p1 : ndarray
        First point coordinates (2,) array [easting, northing].
    p2 : ndarray
        Second point coordinates (2,) array [easting, northing].

    Returns
    -------
    float
        Euclidean distance in meters.
    """
    return euclidean(p1, p2)


def euclidean_distance_matrix(points: np.ndarray) -> np.ndarray:
    """
    Calculate full distance matrix between points.

    Parameters
    ----------
    points : ndarray
        Array of point coordinates (N, 2) where each row is [easting, northing].

    Returns
    -------
    ndarray
        Distance matrix (N, N) where element [i, j] is distance between points i and j.
    """
    return cdist(points, points, metric="euclidean")


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
    return 1 / np.exp(np.maximum(0.1, distances) / phi)

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist, euclidean

from vs30 import constants
from vs30.category import (
    ID_NODATA,
    _assign_to_category_geology,
    _assign_to_category_terrain,
)


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


def prepare_observation_data(
    observations: pd.DataFrame,
    raster_data: "spatial.RasterData",
    updated_model_table: np.ndarray,
    model_name: str,
) -> "spatial.ObservationData":
    """
    Prepare observation data for MVN processing.

    Parameters
    ----------
    observations : DataFrame
        Observations with vs30, uncertainty, easting, northing, and model ID.
    raster_data : RasterData
        Raster data object.
    updated_model_table : ndarray
        Updated model table (n_categories, 2) array of [vs30, stdv].
    model_name : str
        Model name ("geology" or "terrain").

    Returns
    -------
    ObservationData
        Prepared observation data object.
    """
    # Get observation locations
    obs_locs = observations[["easting", "northing"]].values

    # Interpolate model values at observation locations
    if model_name == "geology":
        model_ids = _assign_to_category_geology(obs_locs)
    elif model_name == "terrain":
        model_ids = _assign_to_category_terrain(obs_locs)
    else:
        raise ValueError(f"Unknown model name: {model_name}")

    # Get model vs30 and stdv from updated model table
    # Model IDs are 1-indexed in the raster, but 0-indexed in the model table
    # ID_NODATA is 255, valid IDs are 1-15 for geology, 1-16 for terrain
    valid_mask = (
        (model_ids != ID_NODATA)
        & (model_ids > 0)
        & (model_ids <= len(updated_model_table))
    )
    model_vs30 = np.full(len(observations), np.nan)
    model_stdv = np.full(len(observations), np.nan)

    # Convert 1-indexed model IDs to 0-indexed array indices
    valid_model_ids = model_ids[valid_mask] - 1
    model_vs30[valid_mask] = updated_model_table[valid_model_ids, 0]
    model_stdv[valid_mask] = updated_model_table[valid_model_ids, 1]

    # Filter out observations where model values are NaN/NoData
    valid_obs_mask = ~np.isnan(model_vs30) & ~np.isnan(model_stdv)
    obs_locs = obs_locs[valid_obs_mask]
    vs30_obs = observations.vs30.values[valid_obs_mask]
    model_vs30 = model_vs30[valid_obs_mask]
    model_stdv = model_stdv[valid_obs_mask]
    uncertainty = observations.uncertainty.values[valid_obs_mask]

    # Calculate log residuals
    residuals = np.log(vs30_obs / model_vs30)

    # Apply noise weighting (if noisy=True)
    if constants.NOISY:
        omega = np.sqrt(model_stdv**2 / (model_stdv**2 + uncertainty**2))
        residuals *= omega
    else:
        omega = np.ones(len(residuals))

    from vs30 import spatial

    return spatial.ObservationData(
        locations=obs_locs,
        vs30=vs30_obs,
        model_vs30=model_vs30,
        model_stdv=model_stdv,
        residuals=residuals,
        omega=omega,
        uncertainty=uncertainty,
    )


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
    Uses exponential correlation function: 1 / exp(max(0.1, distance) / phi)
    Minimum distance of 0.1 meters is enforced to prevent division issues.
    """
    return 1 / np.exp(np.maximum(0.1, distances) / phi)

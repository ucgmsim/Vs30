"""Shared utility functions: correlation function and model combination."""

from pathlib import Path

import numpy as np
import pandas as pd

from vs30 import constants


def correlation_function(
    distances: np.ndarray,
    phi: float,
    min_dist: float = constants.MIN_DIST_ENFORCED,
) -> np.ndarray:
    """
    Calculate exponential correlation from distances.

    Parameters
    ----------
    distances : ndarray
        Array of distances in meters. Can be scalar, 1D, or 2D (distance matrix).
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


def combine_vs30_models(
    geol_vs30: np.ndarray,
    geol_stdv: np.ndarray,
    terr_vs30: np.ndarray,
    terr_stdv: np.ndarray,
    combination_method: constants.CombinationMethod,
    combine_ratio: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combine geology and terrain Vs30 models using log-space weighted mixture.

    This function implements the correct combination algorithm that matches
    the raster-based `combine` CLI command. Models are combined in log-space
    (geometric weighting) with standard deviation computed using the mixture
    of log-normals formula.

    Parameters
    ----------
    geol_vs30 : ndarray
        Geology model Vs30 values.
    geol_stdv : ndarray
        Geology model standard deviation (in log-space).
    terr_vs30 : ndarray
        Terrain model Vs30 values.
    terr_stdv : ndarray
        Terrain model standard deviation (in log-space).
    combination_method : CombinationMethod
        Method for combining models: STANDARD_DEVIATION_WEIGHTING for
        variance-based weighting, or RATIO for fixed-ratio weighting.
    combine_ratio : float, optional
        Geology-to-terrain weight ratio (e.g., 1.0 for equal weighting,
        2.0 for geology having twice the weight). Required when
        combination_method is RATIO.

    Returns
    -------
    combined_vs30 : ndarray
        Combined Vs30 values (geometric weighted mean).
    combined_stdv : ndarray
        Combined standard deviation (mixture of log-normals formula).

    Notes
    -----
    The combination is performed in log-space:
    - Weights are computed based on the combination method
    - Log-space combination: log_comb = w_g * log(geol) + w_t * log(terr)
    - Combined Vs30 = exp(log_comb)
    - Combined stdv uses mixture of log-normals: sqrt(w_g*(diff_g² + σ_g²) + w_t*(diff_t² + σ_t²))

    For ratio=1.0 with inputs (200, 400), the result is ~283 (geometric mean),
    not 300 (arithmetic mean).

    Uses K_VALUE and WEIGHT_EPSILON_DIV_BY_ZERO constants from constants.py
    for standard deviation weighting calculations.
    """
    # Determine weights based on combination method
    if combination_method == constants.CombinationMethod.STANDARD_DEVIATION_WEIGHTING:
        # Variance-based weighting: lower stdv gets higher weight
        m_g = (
            geol_stdv**2 + constants.WEIGHT_EPSILON_DIV_BY_ZERO
        ) ** -constants.K_VALUE
        m_t = (
            terr_stdv**2 + constants.WEIGHT_EPSILON_DIV_BY_ZERO
        ) ** -constants.K_VALUE
        total_m = m_g + m_t
        w_g = m_g / total_m
        w_t = m_t / total_m
    elif combination_method == constants.CombinationMethod.RATIO:
        if combine_ratio is None:
            raise ValueError(
                "combine_ratio is required when combination_method is RATIO"
            )
        total_w = combine_ratio + 1.0
        w_g = combine_ratio / total_w
        w_t = 1.0 / total_w
    else:
        raise ValueError(f"Unknown combination method: {combination_method}")

    # Combine in log-space (geometric weighting)
    log_g = np.log(geol_vs30)
    log_t = np.log(terr_vs30)

    log_comb = log_g * w_g + log_t * w_t
    combined_vs30 = np.exp(log_comb)

    # Combined stdv using mixture of log-normals formula
    # Each component contributes: weight * (squared_diff_from_mean + variance)
    combined_stdv = np.sqrt(
        w_g * ((log_g - log_comb) ** 2 + geol_stdv**2)
        + w_t * ((log_t - log_comb) ** 2 + terr_stdv**2)
    )

    return combined_vs30, combined_stdv


def resolve_observation_csv(
    csv_path: Path | None,
    config_filename: str | None,
    res_dir: Path,
) -> Path | None:
    """
    Resolve an observation CSV path from an explicit argument or config default.

    Parameters
    ----------
    csv_path : Path or None
        Explicitly provided path.
    config_filename : str or None
        Filename from config (e.g., cfg.clustered_observations_file).
    res_dir : Path
        Resources directory to look for config-specified files.

    Returns
    -------
    Path or None
        Resolved path, or None if no valid file found.
    """
    if csv_path is not None:
        return csv_path
    if config_filename is not None:
        candidate = res_dir / config_filename
        if candidate.exists():
            return candidate
    return None


def validate_csv_columns(
    df: pd.DataFrame, required_cols: list[str], label: str
) -> None:
    """
    Raise ValueError if the DataFrame is missing any required columns.

    Parameters
    ----------
    df : DataFrame
        DataFrame to validate.
    required_cols : list[str]
        Column names that must be present.
    label : str
        Descriptive label used in the error message (e.g. "Clustered observations CSV").

    Raises
    ------
    ValueError
        If any required columns are missing.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{label} missing required columns: {missing}")

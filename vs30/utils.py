import numpy as np

from vs30.config import get_default_config


# ============================================================================
# Correlation Functions
# ============================================================================


def correlation_function(
    distances: np.ndarray,
    phi: float,
    min_dist: float | None = None,
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
        Minimum distance enforced to prevent division issues. Default from config.

    Returns
    -------
    correlations : ndarray
        Correlation values between 0 and 1. Same shape as distances.

    Notes
    -----
    Uses exponential correlation function: exp(-distance / phi).
    A small minimum distance (default 0.1 m) is enforced to prevent
    exact-zero distances from producing correlation = 1.0.
    """
    if min_dist is None:
        min_dist = get_default_config().min_dist_enforced
    return np.exp(-np.maximum(min_dist, distances) / phi)


# ============================================================================
# Model Combination Functions
# ============================================================================


def combine_vs30_models(
    geol_vs30: np.ndarray,
    geol_stdv: np.ndarray,
    terr_vs30: np.ndarray,
    terr_stdv: np.ndarray,
    combination_method: str | float,
    k_value: float = 3.0,
    epsilon: float = 1e-10,
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
    combination_method : str or float
        Either a ratio (float) where ratio = geology_weight / terrain_weight,
        so ratio=1.0 gives equal weighting, ratio=2.0 gives geology twice
        the weight of terrain. Or "standard_deviation_weighting" for
        variance-based weighting using k_value exponent.
    k_value : float, optional
        Exponent for standard deviation weighting. Higher values give more
        weight to the model with lower uncertainty. Default is 3.0.
    epsilon : float, optional
        Small value to prevent division by zero in variance weighting.
        Default is 1e-10.

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
    """
    # Determine weights based on combination method
    method_str = str(combination_method).strip()
    if method_str == "standard_deviation_weighting":
        # Variance-based weighting: lower stdv gets higher weight
        m_g = (geol_stdv**2 + epsilon) ** -k_value
        m_t = (terr_stdv**2 + epsilon) ** -k_value
        total_m = m_g + m_t
        w_g = m_g / total_m
        w_t = m_t / total_m
    else:
        try:
            # Ratio-based: ratio = geology_weight / terrain_weight
            ratio = float(combination_method)
        except (ValueError, TypeError):
            raise ValueError(f"Unknown combination method: {combination_method}")
        total_w = ratio + 1.0
        w_g = ratio / total_w
        w_t = 1.0 / total_w

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

"""Shared utility functions: model combination and data helpers."""

import numpy as np
import pandas as pd

from vs30 import constants


def combine_vs30_models(
    geol_vs30: np.ndarray,
    geol_stdv: np.ndarray,
    terr_vs30: np.ndarray,
    terr_stdv: np.ndarray,
    combination_method: constants.CombinationMethod,
    combine_ratio: float | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Combine geology and terrain Vs30 models using a weighted geometric mean.

    Parameters
    ----------
    geol_vs30 : ndarray
        Geology model Vs30 values.
    geol_stdv : ndarray
        Geology model standard deviation, in log-space.
    terr_vs30 : ndarray
        Terrain model Vs30 values.
    terr_stdv : ndarray
        Terrain model standard deviation, in log-space.
    combination_method : CombinationMethod
        STANDARD_DEVIATION_WEIGHTING (lower-stdv model gets higher weight) or
        RATIO (fixed geology-to-terrain ratio).
    combine_ratio : float, optional
        Geology-to-terrain weight ratio (e.g., 2.0 weights geology twice as
        heavily as terrain); required when method is RATIO.

    Returns
    -------
    combined_vs30 : ndarray
        Combined Vs30 values.
    combined_stdv : ndarray
        Combined log-space standard deviation.

    Raises
    ------
    ValueError
        If combine_ratio is None when combination_method is RATIO, or if
        combination_method is not a recognized CombinationMethod value.
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


def validate_csv_columns(
    df: pd.DataFrame, required_cols: list[str], label: str
) -> None:
    """
    Validate that the DataFrame contains all required columns.

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


def nan_to_nodata(arr: np.ndarray) -> np.ndarray:
    """
    Replace NaNs with ``constants.NODATA_VALUE`` for raster output.

    Parameters
    ----------
    arr : ndarray
        Input array, possibly containing NaN values.

    Returns
    -------
    ndarray
        Copy of ``arr`` with NaN entries replaced by
        ``constants.NODATA_VALUE``. Same shape as input.
    """
    return np.where(np.isnan(arr), constants.NODATA_VALUE, arr)

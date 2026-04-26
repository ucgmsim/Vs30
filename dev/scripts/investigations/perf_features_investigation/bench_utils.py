"""Helpers for the perf-features-investigation benchmarking harness."""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
VIKTOR_OBS_PATH = (
    REPO_ROOT / "vs30/resources/observations/viktor_inferred_vs30_from_cpt.csv"
)


def subsample_observations(n: int, seed: int = 42) -> pd.DataFrame:
    """Return a deterministic subsample of viktor_cpt observations.

    Parameters
    ----------
    n
        Number of observations to return.
    seed
        Seed for the numpy random generator.

    Returns
    -------
    pd.DataFrame
        Subsampled observations with the standard required columns.

    Raises
    ------
    ValueError
        If ``n`` exceeds the number of available observations.
    """
    df = pd.read_csv(VIKTOR_OBS_PATH, comment="#", skipinitialspace=True)
    if n > len(df):
        raise ValueError(
            f"n ({n}) exceeds available observations ({len(df)})"
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[idx].reset_index(drop=True)

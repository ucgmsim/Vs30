"""Helpers for the perf-features-investigation benchmarking harness."""

import contextlib
from pathlib import Path

import numpy as np
import pandas as pd

from vs30 import config, constants, pipeline, spatial


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
        raise ValueError(f"n ({n}) exceeds available observations ({len(df)})")
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[idx].reset_index(drop=True)


# (dx, dy, x_extent_m, y_extent_m) — chosen empirically to hit
# the targets within ~2× tolerance. Actual N_valid is logged at runtime so
# analysis can use the true value instead of the target.
_GRID_PRESETS: dict[int, tuple[int, int, int, int]] = {
    1_000: (2000, 2000, 90_000, 90_000),
    10_000: (1000, 1000, 150_000, 150_000),
    100_000: (500, 500, 250_000, 250_000),
    1_000_000: (200, 200, 350_000, 350_000),
}

# Centre of the subdomain — chosen near central NZ so the box always lands
# on land. NZTM (easting, northing).
_DOMAIN_CENTRE = (1_580_000, 5_180_000)


def make_raster_data(n_target: int):
    """Build a real RasterData of approximately ``n_target`` valid pixels.

    Uses the production ``pipeline.create_initial_vs30_arrays`` with
    ``model_type=TERRAIN`` to populate a sub-region of NZ. The exact
    valid-pixel count varies with the underlying terrain raster; callers
    should log ``raster_data.valid_flat_indices.size`` rather than rely on
    ``n_target`` exactly.

    Parameters
    ----------
    n_target
        Approximate number of valid pixels to return.

    Returns
    -------
    raster_data : spatial.RasterData
        Real raster data backed by the IwahashiPike terrain raster.
    profile : dict
        Rasterio profile (transform, crs, nodata).
    """
    if n_target not in _GRID_PRESETS:
        raise ValueError(
            f"n_target must be one of {sorted(_GRID_PRESETS)}, got {n_target}"
        )
    dx, dy, x_extent, y_extent = _GRID_PRESETS[n_target]
    cx, cy = _DOMAIN_CENTRE
    grid_config = config.GridConfig(
        grid_xmin=cx - x_extent // 2,
        grid_xmax=cx + x_extent // 2,
        grid_ymin=cy - y_extent // 2,
        grid_ymax=cy + y_extent // 2,
        grid_dx=dx,
        grid_dy=dy,
    )
    vs30_array, stdv_array, _, profile = pipeline.create_initial_vs30_arrays(
        grid_config,
        constants.ModelType.TERRAIN,
        # Read the canonical terrain prior CSV, which is bundled.
        pipeline.read_categorical_csv(
            constants.RESOURCE_PATH
            / constants.RESOURCE_SUBDIRS["terrain_categorical_csv"]
            / "terrain_model_prior_mean_and_standard_deviation.csv"
        ),
    )
    raster_data = spatial.RasterData.from_arrays(
        vs30=vs30_array,
        stdv=stdv_array,
        transform=profile["transform"],
        crs=profile.get("crs", constants.NZTM_CRS),
        nodata=constants.NODATA_VALUE,
    )
    return raster_data, profile


def make_full_bbox_result(
    raster_data: spatial.RasterData, n_obs: int
) -> spatial.BoundingBoxResult:
    """Build a BoundingBoxResult that marks every valid pixel as affected.

    Used to disable the ``find_affected_pixels`` pre-filter for the OFF
    condition. Each observation's index list is set to the full valid-pixel
    set so the parallel path (which uses obs_to_grid_indices) still works.

    Parameters
    ----------
    raster_data
        Raster whose valid pixels become the affected set.
    n_obs
        Number of observations — needed to size obs_to_grid_indices.

    Returns
    -------
    spatial.BoundingBoxResult
        Mask covers every valid pixel; per-observation index lists each
        contain the full valid_flat_indices array.
    """
    mask = np.zeros(raster_data.vs30.size, dtype=bool)
    mask[raster_data.valid_flat_indices] = True
    obs_to_grid_indices = [raster_data.valid_flat_indices.copy() for _ in range(n_obs)]
    return spatial.BoundingBoxResult(
        mask=mask,
        obs_to_grid_indices=obs_to_grid_indices,
        n_affected_pixels=int(mask.sum()),
    )


@contextlib.contextmanager
def bypass_observation_threshold():
    """Disable the n_obs > 1000 fallback for one measurement.

    The production guard at ``pipeline.compute_spatial_adjustment_on_grid``
    forces ``nproc=1`` whenever the observation count exceeds
    ``MULTIPROCESS_OBSERVATION_THRESHOLD``. To measure the multiproc path
    in that regime we temporarily raise the threshold to a value larger
    than any conceivable observation count, then restore it.
    """
    original = constants.MULTIPROCESS_OBSERVATION_THRESHOLD
    constants.MULTIPROCESS_OBSERVATION_THRESHOLD = 10**12
    try:
        yield
    finally:
        constants.MULTIPROCESS_OBSERVATION_THRESHOLD = original

"""Helpers for the points-perf-investigation harness."""

import datetime as _dt
import importlib.util
import resource
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qcore import coordinates

from vs30 import category, constants, pipeline

# Reuse subsample_observations from the grid harness via path import.
# This avoids duplicating the canonical subsampling routine.
_GRID_HARNESS_DIR = Path(__file__).resolve().parents[1] / "perf_features_investigation"
_grid_bench_spec = importlib.util.spec_from_file_location(
    "perf_bench_utils", _GRID_HARNESS_DIR / "bench_utils.py"
)
_grid_bench_mod = importlib.util.module_from_spec(_grid_bench_spec)
_grid_bench_spec.loader.exec_module(_grid_bench_mod)
subsample_observations = _grid_bench_mod.subsample_observations  # noqa: F401  -- re-exported


# WGS84 bounding box that comfortably covers all NZ land. Slightly looser than
# tight to allow for the rejection sampler's land-mask filtering to do its job.
_NZ_LON_MIN, _NZ_LON_MAX = 165.0, 180.0
_NZ_LAT_MIN, _NZ_LAT_MAX = -48.0, -34.0


def generate_nz_land_points(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return ``n`` random (lon, lat) points uniformly distributed over NZ land.

    Rejection samples WGS84 lon/lat within the NZ bounding box and keeps only
    points that land on a valid (non-nodata) IwahashiPike terrain raster pixel.

    Parameters
    ----------
    n
        Number of points to return.
    seed
        Seed for the numpy random generator.

    Returns
    -------
    lons, lats : np.ndarray
        Two ``(n,)`` arrays of WGS84 longitudes and latitudes.
    """
    rng = np.random.default_rng(seed)
    kept_lons: list[float] = []
    kept_lats: list[float] = []
    while len(kept_lons) < n:
        # Over-sample by ~3x; about 30-40% of the NZ bbox is land.
        batch_size = max(3 * n, 1000)
        lons = rng.uniform(_NZ_LON_MIN, _NZ_LON_MAX, batch_size)
        lats = rng.uniform(_NZ_LAT_MIN, _NZ_LAT_MAX, batch_size)
        nztm = coordinates.wgs_depth_to_nztm(np.column_stack([lats, lons]))
        eastings = nztm[:, 1]
        northings = nztm[:, 0]
        points = np.column_stack([eastings, northings])
        terrain_ids = category.assign_to_category_terrain(points)
        land_mask = terrain_ids != constants.RASTER_ID_NODATA_VALUE
        kept_lons.extend(lons[land_mask].tolist())
        kept_lats.extend(lats[land_mask].tolist())
    return np.array(kept_lons[:n]), np.array(kept_lats[:n])


def materialize_obs_csvs(
    out_dir: Path, n_obs_values: list[int], seed: int = 42
) -> dict[int, Path]:
    """Subsample viktor_cpt observations and write one CSV per N_obs value.

    Pre-generating these once-per-sweep avoids the cost of a temp-file write
    per cell. The returned paths are usable as ``independent_observations_csv``
    arguments to ``pipeline.points_pipeline``.

    Parameters
    ----------
    out_dir
        Directory to write the CSVs into. Created if it does not exist.
    n_obs_values
        Distinct N_obs values to materialise.
    seed
        Seed forwarded to ``subsample_observations``.

    Returns
    -------
    dict[int, Path]
        Map of N_obs → path of the CSV containing that many observations.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    for n_obs in n_obs_values:
        df = subsample_observations(n_obs, seed=seed)
        path = out_dir / f"obs_subsampled_n{n_obs}.csv"
        df.to_csv(path, index=False)
        paths[n_obs] = path
    return paths


def load_modified_foster_2019_config() -> dict:
    """Return the resolved modified_foster_2019 model config.

    Wraps ``cli.load_model_config`` (which builds correlation-function partials
    and resolves resource paths) for the harness.
    """
    from vs30 import cli  # local import — cli has heavy transitive imports

    return cli.load_model_config(constants.FixedModelVersion.MODIFIED_FOSTER_2019)


def _peak_rss_mb() -> float:
    """Peak resident-set size of the current process in MB.

    Linux ``ru_maxrss`` is in kibibytes, so divide by 1024 for MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def time_one_run(
    lons: np.ndarray,
    lats: np.ndarray,
    obs_csv_path: Path,
    nproc: int,
    rep: int,
    cfg: dict,
) -> dict:
    """Time one ``pipeline.points_pipeline`` call end-to-end.

    The harness uses the modified_foster_2019 config but injects the
    sweep's subsampled observations as ``independent_observations_csv``.
    With ``do_bayesian_update=False`` (the modified_foster_2019 default),
    the points pipeline does not invoke DBSCAN clustering, so the
    "independent vs clustered" distinction is moot for the spatial
    adjustment — both go through the same ``compute_spatial_adjustment_at_points``
    code path. Forced to ``False`` here to remove ambiguity.

    Parameters
    ----------
    lons, lats
        WGS84 query coordinates.
    obs_csv_path
        Path to the materialised observation CSV for this cell.
    nproc
        ``nproc`` passed through to ``points_pipeline``. Strategy endpoints
        for the sweep are 1 (BLAS multi-threaded) and 8 (BLAS single-threaded
        inside workers).
    rep
        Repetition index, recorded for downstream median computation.
    cfg
        Resolved model config from ``load_modified_foster_2019_config``.

    Returns
    -------
    dict
        CSV-row-shaped fields recording the timing.
    """
    n_obs = len(pd.read_csv(obs_csv_path))
    t0 = time.perf_counter()
    pipeline.points_pipeline(
        longitudes=lons,
        latitudes=lats,
        geology_categorical_csv=cfg["geology_categorical_csv"],
        terrain_categorical_csv=cfg["terrain_categorical_csv"],
        clustered_observations_csv=None,
        independent_observations_csv=obs_csv_path,
        combination_method=constants.CombinationMethod(cfg["combination_method"]),
        combine_ratio=cfg["combine_ratio"],
        noisy=cfg["noisy"],
        mvn=cfg["mvn"],
        do_bayesian_update=False,
        include_intermediate=False,
        nproc=nproc,
        geology_corr_fn=cfg["geology_corr_fn"],
        terrain_corr_fn=cfg["terrain_corr_fn"],
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        apply_coastal_distance_mod=cfg["apply_coastal_distance_mod"],
        fill_gaps=False,
    )
    t_total = time.perf_counter() - t0
    return {
        "N_query": int(len(lons)),
        "N_obs": n_obs,
        "nproc": nproc,
        "rep": rep,
        "t_total_s": t_total,
        "peak_rss_mb": _peak_rss_mb(),
        "timestamp_iso": _dt.datetime.now().isoformat(timespec="seconds"),
    }

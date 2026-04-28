"""Multiprocessing support for parallel spatial adjustment."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from vs30 import category, constants, multiprocess, raster, spatial, utils


@dataclass
class PointsObsData:
    """Precomputed observation arrays for use in points-pipeline workers.

    All arrays have length N_obs. ``model_vs30`` and ``model_stdv`` carry
    post-hybrid-mods values for geology, raw categorical values for terrain.
    """

    locations: np.ndarray
    vs30: np.ndarray
    uncertainty: np.ndarray
    model_vs30: np.ndarray
    model_stdv: np.ndarray

    @classmethod
    def empty(cls) -> "PointsObsData":
        return cls(
            locations=np.empty((0, 2)),
            vs30=np.empty(0),
            uncertainty=np.empty(0),
            model_vs30=np.empty(0),
            model_stdv=np.empty(0),
        )


def prepare_geology_obs_data(
    observations_df: pd.DataFrame,
    geol_model_df: pd.DataFrame,
    apply_alluvium_slope_mod: bool,
    apply_coastal_distance_mod: bool,
) -> PointsObsData:
    """Precompute observation-side geology values for points_pipeline.

    Parameters
    ----------
    observations_df
        Combined observations DataFrame (easting, northing, vs30, uncertainty).
    geol_model_df
        Categorical geology model with Vs30 mean and standard deviation per category.
    apply_alluvium_slope_mod
        Whether to apply the alluvium slope modification.
    apply_coastal_distance_mod
        Whether to apply the coastal distance modification.

    Returns
    -------
    PointsObsData
        Precomputed observation arrays (model_vs30/stdv are post-hybrid-mods).
        Returns ``PointsObsData.empty()`` if ``observations_df`` is empty.
    """
    if len(observations_df) == 0:
        return PointsObsData.empty()

    obs_locs = observations_df[
        [constants.ObservationColumn.EASTING, constants.ObservationColumn.NORTHING]
    ].values
    obs_geol_ids = category.assign_to_category_geology(obs_locs)
    obs_geol_vs30_df = category.get_vs30_for_ids(obs_geol_ids, geol_model_df)

    # Apply hybrid modifications to observation model values so residuals
    # match the grid pipeline; see spatial.prepare_observation_data.
    obs_slope = raster.sample_slope_at_points(obs_locs)
    # Legacy parity: NODATA slope samples at observations are replaced with
    # the 255 sentinel so log10(255) ≈ 2.41 feeds np.interp and returns the
    # MAX Vs30 for the gid; the equivalent grid-pixel handling uses 1e-9
    # and returns the MIN Vs30. See constants.LEGACY_OBS_SLOPE_NODATA_SENTINEL.
    obs_slope = np.where(
        obs_slope < 0, constants.LEGACY_OBS_SLOPE_NODATA_SENTINEL, obs_slope
    )
    obs_coast_dist = (
        raster.compute_coastal_distance_at_points(obs_locs)
        if apply_coastal_distance_mod
        else np.zeros(len(obs_locs))
    )
    obs_model_vs30, obs_model_stdv = raster.apply_hybrid_geology_modifications(
        obs_geol_vs30_df[constants.COL_CATEGORY_VS30_MEAN].values,
        obs_geol_vs30_df[constants.COL_CATEGORY_VS30_STDV].values,
        obs_geol_ids,
        obs_slope,
        obs_coast_dist,
        apply_alluvium_slope_mod=apply_alluvium_slope_mod,
        apply_coastal_distance_mod=apply_coastal_distance_mod,
    )

    return PointsObsData(
        locations=obs_locs,
        vs30=observations_df[constants.ObservationColumn.VS30].values,
        uncertainty=observations_df[constants.ObservationColumn.UNCERTAINTY].values,
        model_vs30=obs_model_vs30,
        model_stdv=obs_model_stdv,
    )


def prepare_terrain_obs_data(
    observations_df: pd.DataFrame,
    terr_model_df: pd.DataFrame,
) -> PointsObsData:
    """Precompute observation-side terrain values for points_pipeline.

    Parameters
    ----------
    observations_df
        Combined observations DataFrame (easting, northing, vs30, uncertainty).
    terr_model_df
        Categorical terrain model with Vs30 mean and standard deviation per category.

    Returns
    -------
    PointsObsData
        Precomputed observation arrays.
        Returns ``PointsObsData.empty()`` if ``observations_df`` is empty.
    """
    if len(observations_df) == 0:
        return PointsObsData.empty()

    obs_locs = observations_df[
        [constants.ObservationColumn.EASTING, constants.ObservationColumn.NORTHING]
    ].values
    obs_terr_ids = category.assign_to_category_terrain(obs_locs)
    obs_terr_vs30_df = category.get_vs30_for_ids(obs_terr_ids, terr_model_df)

    return PointsObsData(
        locations=obs_locs,
        vs30=observations_df[constants.ObservationColumn.VS30].values,
        uncertainty=observations_df[constants.ObservationColumn.UNCERTAINTY].values,
        model_vs30=obs_terr_vs30_df[constants.COL_CATEGORY_VS30_MEAN].values,
        model_stdv=obs_terr_vs30_df[constants.COL_CATEGORY_VS30_STDV].values,
    )


def process_geology_at_points(
    points: np.ndarray,
    model_df: pd.DataFrame,
    geology_obs_data: PointsObsData,
    corr_fn: Callable,
    apply_alluvium_slope_mod: bool,
    apply_coastal_distance_mod: bool,
    noisy: bool = False,
    progress_bar: tqdm | None = None,
) -> tuple[
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray
]:
    """
    Process geology model at points, including hybrid modifications and spatial adjustment.

    Parameters
    ----------
    points : ndarray
        Array of shape (n_points, 2) with (easting, northing) coordinates.
    model_df : DataFrame
        Categorical geology model with Vs30 mean and standard deviation per category.
    geology_obs_data : PointsObsData
        Precomputed observation-side geology values from
        ``prepare_geology_obs_data``.
    corr_fn : Callable
        Correlation function for spatial adjustment.
    apply_alluvium_slope_mod : bool
        Whether to apply the alluvium slope modification (to query-point hybrid mods).
    apply_coastal_distance_mod : bool
        Whether to apply the coastal distance modification (to query-point hybrid mods).
    noisy : bool
        Whether to apply noise weighting in spatial adjustment.
    progress_bar : tqdm, optional
        External progress bar to update per point during spatial adjustment.

    Returns
    -------
    geol_ids : ndarray
        Geology category IDs at each point.
    geol_vs30 : ndarray
        Initial geology Vs30 values (before hybrid mods).
    geol_stdv : ndarray
        Initial geology standard deviation (before hybrid mods).
    geol_vs30_hybrid : ndarray
        Geology Vs30 after hybrid modifications.
    geol_stdv_hybrid : ndarray
        Geology standard deviation after hybrid modifications.
    geol_mvn_vs30 : ndarray
        Final geology Vs30 after spatial adjustment.
    geol_mvn_stdv : ndarray
        Final geology standard deviation after spatial adjustment.
    """
    geol_ids = category.assign_to_category_geology(points)

    geol_vs30_df = category.get_vs30_for_ids(geol_ids, model_df)
    geol_vs30 = geol_vs30_df[constants.COL_CATEGORY_VS30_MEAN].values
    geol_stdv = geol_vs30_df[constants.COL_CATEGORY_VS30_STDV].values

    slope_at_points = raster.sample_slope_at_points(points)
    coast_dist_at_points = (
        raster.compute_coastal_distance_at_points(points)
        if apply_coastal_distance_mod
        else np.zeros(len(points))
    )

    geol_vs30_hybrid, geol_stdv_hybrid = raster.apply_hybrid_geology_modifications(
        geol_vs30,
        geol_stdv,
        geol_ids,
        slope_at_points,
        coast_dist_at_points,
        apply_alluvium_slope_mod=apply_alluvium_slope_mod,
        apply_coastal_distance_mod=apply_coastal_distance_mod,
    )

    if len(geology_obs_data.locations) > 0:
        geol_mvn_vs30, geol_mvn_stdv = spatial.compute_spatial_adjustment_at_points(
            points=points,
            model_vs30=geol_vs30_hybrid,
            model_stdv=geol_stdv_hybrid,
            obs_locations=geology_obs_data.locations,
            obs_vs30=geology_obs_data.vs30,
            obs_model_vs30=geology_obs_data.model_vs30,
            obs_model_stdv=geology_obs_data.model_stdv,
            obs_uncertainty=geology_obs_data.uncertainty,
            corr_fn=corr_fn,
            noisy=noisy,
            progress_bar=progress_bar,
        )
    else:
        geol_mvn_vs30 = geol_vs30_hybrid
        geol_mvn_stdv = geol_stdv_hybrid

    return (
        geol_ids,
        geol_vs30,
        geol_stdv,
        geol_vs30_hybrid,
        geol_stdv_hybrid,
        geol_mvn_vs30,
        geol_mvn_stdv,
    )


def process_terrain_at_points(
    points: np.ndarray,
    model_df: pd.DataFrame,
    terrain_obs_data: PointsObsData,
    corr_fn: Callable,
    noisy: bool = False,
    progress_bar: tqdm | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process terrain model at points, including spatial adjustment.

    Parameters
    ----------
    points : ndarray
        Array of shape (n_points, 2) with (easting, northing) coordinates.
    model_df : DataFrame
        Categorical terrain model with Vs30 mean and standard deviation per category.
    terrain_obs_data : PointsObsData
        Precomputed observation-side terrain values from
        ``prepare_terrain_obs_data``.
    corr_fn : Callable
        Correlation function for spatial adjustment.
    noisy : bool
        Whether to apply noise weighting in spatial adjustment.
    progress_bar : tqdm, optional
        External progress bar to update per point during spatial adjustment.

    Returns
    -------
    terr_ids : ndarray
        Terrain category IDs at each point.
    terr_vs30 : ndarray
        Initial terrain Vs30 values.
    terr_stdv : ndarray
        Initial terrain standard deviation.
    terr_mvn_vs30 : ndarray
        Final terrain Vs30 after spatial adjustment.
    terr_mvn_stdv : ndarray
        Final terrain standard deviation after spatial adjustment.
    """
    terr_ids = category.assign_to_category_terrain(points)

    terr_vs30_df = category.get_vs30_for_ids(terr_ids, model_df)
    terr_vs30 = terr_vs30_df[constants.COL_CATEGORY_VS30_MEAN].values
    terr_stdv = terr_vs30_df[constants.COL_CATEGORY_VS30_STDV].values

    if len(terrain_obs_data.locations) > 0:
        terr_mvn_vs30, terr_mvn_stdv = spatial.compute_spatial_adjustment_at_points(
            points=points,
            model_vs30=terr_vs30,
            model_stdv=terr_stdv,
            obs_locations=terrain_obs_data.locations,
            obs_vs30=terrain_obs_data.vs30,
            obs_model_vs30=terrain_obs_data.model_vs30,
            obs_model_stdv=terrain_obs_data.model_stdv,
            obs_uncertainty=terrain_obs_data.uncertainty,
            corr_fn=corr_fn,
            noisy=noisy,
            progress_bar=progress_bar,
        )
    else:
        terr_mvn_vs30 = terr_vs30
        terr_mvn_stdv = terr_stdv

    return (
        terr_ids,
        terr_vs30,
        terr_stdv,
        terr_mvn_vs30,
        terr_mvn_stdv,
    )


@dataclass
class LocationsChunkConfig:
    """
    Configuration parameters for processing a locations chunk.

    Attributes
    ----------
    include_intermediate : bool
        Whether to include intermediate values in output.
    combination_method : CombinationMethod
        Method for combining geology and terrain models.
    combine_ratio : float or None
        Geology-to-terrain weight ratio. Required when combination_method is RATIO.
    noisy : bool
        Whether to apply noise weighting in spatial adjustment.
    """

    include_intermediate: bool
    model_type: constants.ModelType
    combination_method: constants.CombinationMethod
    combine_ratio: float | None
    noisy: bool
    geology_corr_fn: Callable | None
    terrain_corr_fn: Callable | None
    apply_alluvium_slope_mod: bool
    apply_coastal_distance_mod: bool


def process_locations_chunk(
    args: tuple,
) -> tuple[int, pd.DataFrame]:  # pragma: no cover
    """
    Worker function: process a chunk of locations through the full pipeline.

    Note: This function is excluded from coverage because it runs in a
    spawned subprocess which cannot be tracked by pytest-cov.

    Parameters
    ----------
    args : tuple
        (points, chunk_id, geology_obs_data, terrain_obs_data, geol_model_df, terr_model_df, config)
        where points is an (N, 2) array of NZTM (easting, northing) coordinates,
        and geology_obs_data / terrain_obs_data are PointsObsData instances
        (or None if that branch is not running).

    Returns
    -------
    tuple
        (chunk_id, result_df) where result_df has all computed columns
    """
    (
        points,
        chunk_id,
        geology_obs_data,
        terrain_obs_data,
        geol_model_df,
        terr_model_df,
        config,
    ) = args

    result = {}

    run_geology = config.model_type in (
        constants.ModelType.GEOLOGY,
        constants.ModelType.COMBINED,
    )
    run_terrain = config.model_type in (
        constants.ModelType.TERRAIN,
        constants.ModelType.COMBINED,
    )

    if run_geology:
        (
            geol_ids,
            geol_vs30,
            geol_stdv,
            geol_vs30_hybrid,
            geol_stdv_hybrid,
            geol_mvn_vs30,
            geol_mvn_stdv,
        ) = process_geology_at_points(
            points,
            geol_model_df,
            geology_obs_data,
            config.geology_corr_fn,
            apply_alluvium_slope_mod=config.apply_alluvium_slope_mod,
            apply_coastal_distance_mod=config.apply_coastal_distance_mod,
            noisy=config.noisy,
        )

        if config.include_intermediate:
            result[constants.COL_GEOLOGY_ID] = geol_ids
            result[constants.COL_GEOLOGY_VS30] = geol_vs30
            result[constants.COL_GEOLOGY_STDV] = geol_stdv
            result[constants.COL_GEOLOGY_VS30_HYBRID] = geol_vs30_hybrid
            result[constants.COL_GEOLOGY_STDV_HYBRID] = geol_stdv_hybrid
            result[constants.COL_GEOLOGY_MVN_VS30] = geol_mvn_vs30
            result[constants.COL_GEOLOGY_MVN_STDV] = geol_mvn_stdv

    if run_terrain:
        (
            terr_ids,
            terr_vs30,
            terr_stdv,
            terr_mvn_vs30,
            terr_mvn_stdv,
        ) = process_terrain_at_points(
            points,
            terr_model_df,
            terrain_obs_data,
            config.terrain_corr_fn,
            config.noisy,
        )

        if config.include_intermediate:
            result[constants.COL_TERRAIN_ID] = terr_ids
            result[constants.COL_TERRAIN_VS30] = terr_vs30
            result[constants.COL_TERRAIN_STDV] = terr_stdv
            result[constants.COL_TERRAIN_MVN_VS30] = terr_mvn_vs30
            result[constants.COL_TERRAIN_MVN_STDV] = terr_mvn_stdv

    if run_geology and run_terrain:
        combined_vs30, combined_stdv = utils.combine_vs30_models(
            geol_mvn_vs30,
            geol_mvn_stdv,
            terr_mvn_vs30,
            terr_mvn_stdv,
            config.combination_method,
            config.combine_ratio,
        )
        result[constants.ObservationColumn.VS30] = combined_vs30
        result[constants.COL_COMBINED_STDV] = combined_stdv
    elif run_geology:
        result[constants.ObservationColumn.VS30] = geol_mvn_vs30
        result[constants.COL_COMBINED_STDV] = geol_mvn_stdv
    elif run_terrain:
        result[constants.ObservationColumn.VS30] = terr_mvn_vs30
        result[constants.COL_COMBINED_STDV] = terr_mvn_stdv

    return chunk_id, pd.DataFrame(result)


def run_parallel_locations(
    points: np.ndarray,
    geology_obs_data: PointsObsData | None,
    terrain_obs_data: PointsObsData | None,
    geol_model_df: pd.DataFrame | None,
    terr_model_df: pd.DataFrame | None,
    config: LocationsChunkConfig,
    nproc: int,
) -> pd.DataFrame:
    """
    Process locations in parallel.

    Divides the points array into chunks and processes each chunk
    in a separate process using the full VS30 pipeline. Pass None for
    the model and obs-data fields whose branch is not used by
    ``config.model_type``.

    Parameters
    ----------
    points : ndarray
        Array of shape (N, 2) with NZTM (easting, northing) coordinates.
    geology_obs_data : PointsObsData or None
        Precomputed observation-side geology values from
        ``prepare_geology_obs_data``. Required when running geology;
        otherwise None.
    terrain_obs_data : PointsObsData or None
        Precomputed observation-side terrain values from
        ``prepare_terrain_obs_data``. Required when running terrain;
        otherwise None.
    geol_model_df : DataFrame or None
        Geology categorical model. Required when running geology; otherwise None.
    terr_model_df : DataFrame or None
        Terrain categorical model. Required when running terrain; otherwise None.
    config : LocationsChunkConfig
        Configuration parameters for processing
    nproc : int
        Number of processes to use (must be > 1)

    Returns
    -------
    DataFrame
        Results with vs30, stdv, and intermediate columns (if requested)
    """
    n_chunks = min(len(points), constants.N_PROGRESS_CHUNKS)
    split_indices = np.array_split(range(len(points)), n_chunks)
    chunk_args = [
        (
            points[idx],
            i,
            geology_obs_data,
            terrain_obs_data,
            geol_model_df,
            terr_model_df,
            config,
        )
        for i, idx in enumerate(split_indices)
        if len(idx) > 0
    ]

    with multiprocess.single_threaded_blas():
        with multiprocess.spawn_context.Pool(processes=nproc) as pool:
            results = []
            with tqdm(total=len(points), unit="point") as pbar:
                for chunk_id, result_df in pool.imap(
                    process_locations_chunk, chunk_args
                ):
                    results.append((chunk_id, result_df))
                    pbar.update(len(result_df))

    results.sort(key=lambda x: x[0])
    return pd.concat([r[1] for r in results], ignore_index=True)

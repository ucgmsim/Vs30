"""Multiprocessing support for parallel spatial adjustment."""

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
import pandas as pd
from tqdm import tqdm

from vs30 import category, constants, multiprocess, raster, spatial, utils


def process_geology_at_points(
    points: np.ndarray,
    model_df: pd.DataFrame,
    observations_df: pd.DataFrame,
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

    This function encapsulates the full geology processing pipeline:
    1. Get initial Vs30 values from categorical model
    2. Apply hybrid modifications (slope and coastal distance)
    3. Apply spatial adjustment using observations

    Parameters
    ----------
    points : ndarray
        Array of shape (n_points, 2) with (easting, northing) coordinates.
    model_df : DataFrame
        Categorical geology model with Vs30 mean and standard deviation per category.
    observations_df : DataFrame
        Observation data with columns: easting, northing, vs30, uncertainty.
    corr_fn : Callable
        Correlation function for spatial adjustment.
    apply_alluvium_slope_mod : bool
        Whether to apply the alluvium slope modification.
    apply_coastal_distance_mod : bool
        Whether to apply the coastal distance modification.
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

    if len(observations_df) > 0:
        obs_locs = observations_df[
            [constants.ObservationColumn.EASTING, constants.ObservationColumn.NORTHING]
        ].values
        obs_geol_ids = category.assign_to_category_geology(obs_locs)
        obs_geol_vs30_df = category.get_vs30_for_ids(obs_geol_ids, model_df)

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

        geol_mvn_vs30, geol_mvn_stdv = spatial.compute_spatial_adjustment_at_points(
            points=points,
            model_vs30=geol_vs30_hybrid,
            model_stdv=geol_stdv_hybrid,
            obs_locations=obs_locs,
            obs_vs30=observations_df[constants.ObservationColumn.VS30].values,
            obs_model_vs30=obs_model_vs30,
            obs_model_stdv=obs_model_stdv,
            obs_uncertainty=observations_df[constants.ObservationColumn.UNCERTAINTY].values,
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
    observations_df: pd.DataFrame,
    corr_fn: Callable,
    noisy: bool = False,
    progress_bar: tqdm | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Process terrain model at points, including spatial adjustment.

    This function encapsulates the full terrain processing pipeline:
    1. Get initial Vs30 values from categorical model
    2. Apply spatial adjustment using observations (no hybrid modifications for terrain)

    Parameters
    ----------
    points : ndarray
        Array of shape (n_points, 2) with (easting, northing) coordinates.
    model_df : DataFrame
        Categorical terrain model with Vs30 mean and standard deviation per category.
    observations_df : DataFrame
        Observation data with columns: easting, northing, vs30, uncertainty.
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

    if len(observations_df) > 0:
        obs_locs = observations_df[
            [constants.ObservationColumn.EASTING, constants.ObservationColumn.NORTHING]
        ].values
        obs_terr_ids = category.assign_to_category_terrain(obs_locs)
        obs_terr_vs30_df = category.get_vs30_for_ids(obs_terr_ids, model_df)
        terr_mvn_vs30, terr_mvn_stdv = spatial.compute_spatial_adjustment_at_points(
            points=points,
            model_vs30=terr_vs30,
            model_stdv=terr_stdv,
            obs_locations=obs_locs,
            obs_vs30=observations_df[constants.ObservationColumn.VS30].values,
            obs_model_vs30=obs_terr_vs30_df[constants.COL_CATEGORY_VS30_MEAN].values,
            obs_model_stdv=obs_terr_vs30_df[constants.COL_CATEGORY_VS30_STDV].values,
            obs_uncertainty=observations_df[constants.ObservationColumn.UNCERTAINTY].values,
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

    This function runs in a separate process and processes a subset of
    locations through the complete VS30 pipeline (geology + terrain + combine).

    Note: This function is excluded from coverage because it runs in a
    spawned subprocess which cannot be tracked by pytest-cov.

    Parameters
    ----------
    args : tuple
        (points, chunk_id, observations_df, geol_model_df, terr_model_df, config)
        where points is an (N, 2) array of NZTM (easting, northing) coordinates.

    Returns
    -------
    tuple
        (chunk_id, result_df) where result_df has all computed columns
    """
    (
        points,
        chunk_id,
        observations_df,
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
            observations_df,
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
            points, terr_model_df, observations_df, config.terrain_corr_fn, config.noisy
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


def process_pixels_chunk(
    args: tuple,
) -> tuple[int, list[tuple[int, float, float]]]:  # pragma: no cover
    """
    Worker function: compute spatial adjustments for a chunk of affected pixels.

    This function runs in a separate process and computes spatial adjustments
    for a subset of affected pixels.

    Note: This function is excluded from coverage because it runs in a
    spawned subprocess which cannot be tracked by pytest-cov.

    Parameters
    ----------
    args : tuple
        (pixels, chunk_id, obs_data, corr_fn, max_dist_m, max_points,
        noisy, cov_reduc, corr_zero)

    Returns
    -------
    tuple
        (chunk_id, list of (flat_index, updated_vs30, updated_stdv) tuples)
    """
    (
        pixels,
        chunk_id,
        obs_data,
        corr_fn,
        max_dist_m,
        max_points,
        noisy,
        cov_reduc,
        corr_zero,
    ) = args

    updates = []
    for pixel in pixels:
        result = spatial.compute_spatial_adjustment_for_pixel(
            pixel,
            obs_data,
            corr_fn,
            max_dist_m=max_dist_m,
            max_points=max_points,
            noisy=noisy,
            cov_reduc=cov_reduc,
            corr_zero=corr_zero,
        )
        if result is not None:
            vs30, stdv, _ = result
            updates.append((pixel.index, vs30, stdv))

    return chunk_id, updates


def run_parallel_locations(
    points: np.ndarray,
    observations_df: pd.DataFrame,
    geol_model_df: pd.DataFrame | None,
    terr_model_df: pd.DataFrame | None,
    config: LocationsChunkConfig,
    nproc: int,
) -> pd.DataFrame:
    """
    Process locations in parallel.

    Divides the points array into chunks and processes each chunk
    in a separate process using the full VS30 pipeline. Pass None for
    the model that is not used by ``config.model_type``.

    Parameters
    ----------
    points : ndarray
        Array of shape (N, 2) with NZTM (easting, northing) coordinates.
    observations_df : DataFrame
        Observation data for spatial adjustment (must have easting, northing, vs30, uncertainty)
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
    # Split into many small chunks for smooth progress bar updates.
    # pool.imap distributes chunks to nproc workers automatically.
    n_chunks = min(len(points), constants.N_PROGRESS_CHUNKS)
    split_indices = np.array_split(range(len(points)), n_chunks)
    chunk_args = [
        (
            points[idx],
            i,
            observations_df,
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


def run_parallel_spatial_fit(
    affected_flat_indices: np.ndarray,
    raster_data,  # RasterData - avoid import cycle
    obs_data: spatial.ObservationData,
    corr_fn: Callable,
    model_type: constants.ModelType,
    max_dist_m: float,
    max_points: int,
    noisy: bool,
    cov_reduc: float,
    nproc: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute spatial adjustments for affected pixels in parallel.

    Divides the affected pixels into chunks and processes each chunk
    in a separate process.

    Parameters
    ----------
    affected_flat_indices : ndarray
        Flat indices of affected pixels in the raster
    raster_data : RasterData
        Raster data object with vs30, stdv, and coordinate info
    obs_data : ObservationData
        Observation data for spatial adjustment
    corr_fn : Callable
        Correlation function for spatial adjustment
    model_type : constants.ModelType
        Model type (ModelType.GEOLOGY or ModelType.TERRAIN)
    max_dist_m : float
        Maximum distance for considering observations
    max_points : int
        Maximum number of observations per pixel
    noisy : bool
        Whether to apply noise weighting
    cov_reduc : float
        Covariance reduction factor
    nproc : int
        Number of processes to use (must be > 1)

    Returns
    -------
    tuple of ndarray
        (updated_vs30, updated_stdv) arrays with spatial adjustments applied.
    """
    if len(affected_flat_indices) == 0:
        return raster_data.vs30.copy(), raster_data.stdv.copy()

    grid_locs = raster_data.get_coordinates()

    # Build a flat-index → valid-index map. searchsorted returns an insertion
    # point, so we additionally check equality to confirm the flat_idx really
    # is in valid_flat_indices and skip otherwise.
    pixels = []
    for flat_idx in affected_flat_indices:
        valid_idx = np.searchsorted(raster_data.valid_flat_indices, flat_idx)
        if (
            valid_idx < len(raster_data.valid_flat_indices)
            and raster_data.valid_flat_indices[valid_idx] == flat_idx
        ):
            pixels.append(
                spatial.PixelData(
                    location=grid_locs[valid_idx],
                    vs30=float(raster_data.vs30.flat[flat_idx]),
                    stdv=float(raster_data.stdv.flat[flat_idx]),
                    index=int(flat_idx),
                )
            )

    corr_zero = corr_fn(np.array([0.0]))[0]

    # Split into many small chunks for smooth progress bar updates.
    # pool.imap distributes chunks to nproc workers automatically.
    n_chunks = min(len(pixels), constants.N_PROGRESS_CHUNKS)
    chunks = np.array_split(np.arange(len(pixels)), n_chunks)
    chunk_args = [
        (
            [pixels[i] for i in chunk],
            chunk_id,
            obs_data,
            corr_fn,
            max_dist_m,
            max_points,
            noisy,
            cov_reduc,
            corr_zero,
        )
        for chunk_id, chunk in enumerate(chunks)
        if len(chunk) > 0
    ]

    label = str(model_type).capitalize()
    with multiprocess.single_threaded_blas():
        with multiprocess.spawn_context.Pool(processes=min(nproc, len(chunk_args))) as pool:
            results = []
            with tqdm(
                total=len(pixels),
                desc=f"{label}: spatial adjustment",
                unit="pixel",
            ) as pbar:
                for chunk_id, chunk_updates in pool.imap(
                    process_pixels_chunk, chunk_args
                ):
                    results.append((chunk_id, chunk_updates))
                    pbar.update(len(chunks[chunk_id]))

    updated_vs30 = raster_data.vs30.copy()
    updated_stdv = raster_data.stdv.copy()
    for _, chunk_updates in results:
        for flat_idx, vs30, stdv in chunk_updates:
            updated_vs30.flat[flat_idx] = vs30
            updated_stdv.flat[flat_idx] = stdv

    return updated_vs30, updated_stdv

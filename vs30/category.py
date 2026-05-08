"""
Functions for relating measurement vs30 values to geology/terrain categories
and performing Bayesian updates of categorical mean and standard deviation values.
"""

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
import sklearn.cluster

from vs30 import constants, raster, utils


def assign_to_category_geology(points: np.ndarray) -> np.ndarray:
    """
    Assign geology category IDs to points using QMAP polygon spatial join.

    Parameters
    ----------
    points : ndarray
        Array of NZTM (easting, northing) coordinates.

    Returns
    -------
    ndarray
        Category IDs (1-indexed, or constants.RASTER_ID_NODATA_VALUE for
        points outside the polygons).
    """
    gdf = raster.load_qmap_shapefile()[
        [constants.SHAPEFILE_GEOLOGY_ID_COLUMN, constants.SHAPEFILE_GEOMETRY_COLUMN]
    ]
    joined = gpd.sjoin(
        gpd.GeoDataFrame(geometry=shapely.points(points), crs=gdf.crs),
        gdf,
        how="left",
        predicate="within",
    )

    # Default to ID_NODATA where the spatial join returned no match.
    values = np.full(len(points), constants.RASTER_ID_NODATA_VALUE, dtype=np.uint8)
    value_mask = ~joined[constants.SHAPEFILE_GEOLOGY_ID_COLUMN].isna()
    values[value_mask] = joined.loc[
        value_mask, constants.SHAPEFILE_GEOLOGY_ID_COLUMN
    ].to_numpy()

    return values


def assign_to_category_terrain(points: np.ndarray) -> np.ndarray:
    """
    Assign terrain category IDs to points using IwahashiPike raster lookup.

    Parameters
    ----------
    points : ndarray
        Array of NZTM (easting, northing) coordinates.

    Returns
    -------
    ndarray
        Category IDs (1-indexed, or constants.RASTER_ID_NODATA_VALUE for
        points outside the raster).
    """
    data, transform, nodata = raster.load_terrain_raster_array()
    rows, cols = rasterio.transform.rowcol(transform, points[:, 0], points[:, 1])
    rows = np.asarray(rows)
    cols = np.asarray(cols)
    in_bounds = (
        (rows >= 0) & (rows < data.shape[0]) & (cols >= 0) & (cols < data.shape[1])
    )
    terrain_ids = np.full(
        len(points), constants.RASTER_ID_NODATA_VALUE, dtype=data.dtype
    )
    terrain_ids[in_bounds] = data[rows[in_bounds], cols[in_bounds]]
    if nodata is not None:
        terrain_ids[terrain_ids == nodata] = constants.RASTER_ID_NODATA_VALUE
    return terrain_ids


def assign_to_category(
    points: np.ndarray, model_type: constants.ModelType
) -> np.ndarray:
    """
    Assign category IDs to points using the geology or terrain model.

    Parameters
    ----------
    points : ndarray
        Array of NZTM (easting, northing) coordinates.
    model_type : constants.ModelType
        ``ModelType.GEOLOGY`` or ``ModelType.TERRAIN``.

    Returns
    -------
    ndarray
        Category IDs.

    Raises
    ------
    ValueError
        If ``model_type`` is not GEOLOGY or TERRAIN.
    """
    if model_type == constants.ModelType.GEOLOGY:
        return assign_to_category_geology(points)
    if model_type == constants.ModelType.TERRAIN:
        return assign_to_category_terrain(points)
    raise ValueError(f"Unsupported model_type for category assignment: {model_type}")


def update_with_independent_data(
    categorical_model_df: pd.DataFrame,
    observations_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform Bayesian update of category mean and standard deviation values.

    Parameters
    ----------
    categorical_model_df : DataFrame
        Prior categorical model — either initial (with COL_MEAN, COL_STDV) or
        a posterior from a previous clustered update (with
        COL_POSTERIOR_*_CLUSTERED).
    observations_df : DataFrame
        Observations with vs30, uncertainty, and category ID columns.

    Returns
    -------
    DataFrame
        Categorical model with new posterior columns added.

    Raises
    ------
    ValueError
        If `categorical_model_df` has neither posterior-clustered columns
        nor COL_MEAN/COL_STDV columns.
    """
    posterior_df = categorical_model_df.copy()

    if constants.COL_POSTERIOR_MEAN_CLUSTERED in posterior_df.columns:
        posterior_df[constants.COL_PRIOR_MEAN] = posterior_df[
            constants.COL_POSTERIOR_MEAN_CLUSTERED
        ]
        posterior_df[constants.COL_PRIOR_STDV] = posterior_df[
            constants.COL_POSTERIOR_STDV_CLUSTERED
        ]
    elif constants.COL_MEAN in posterior_df.columns:
        posterior_df = posterior_df.rename(
            columns={
                constants.COL_MEAN: constants.COL_PRIOR_MEAN,
                constants.COL_STDV: constants.COL_PRIOR_STDV,
            }
        )
    else:
        raise ValueError(
            f"No usable prior columns. Need either "
            f"{constants.COL_POSTERIOR_MEAN_CLUSTERED}+{constants.COL_POSTERIOR_STDV_CLUSTERED} "
            f"or {constants.COL_MEAN}+{constants.COL_STDV}."
        )

    posterior_df[constants.COL_PRIOR_STDV] = np.clip(
        posterior_df[constants.COL_PRIOR_STDV].to_numpy(),
        constants.MIN_SIGMA,
        None,
    )

    posterior_df[constants.COL_ASSUMED_NUM_PRIOR_OBS] = constants.N_PRIOR
    posterior_df[constants.COL_ENFORCED_MIN_SIGMA] = constants.MIN_SIGMA

    obs_ids = observations_df[constants.STANDARD_ID_COLUMN].to_numpy()
    obs_unc = observations_df[constants.ObservationColumn.UNCERTAINTY].to_numpy()
    # Cache log(obs_vs30) once; the inner update is performed entirely in
    # log space, so we never need to re-take the log of an observation or
    # round-trip current_mean through exp/log per iteration.
    log_obs_vs30 = np.log(observations_df[constants.ObservationColumn.VS30].to_numpy())

    # Posterior arrays start equal to the prior; categories with no matching
    # observations keep these prior values unchanged.
    new_means = posterior_df[constants.COL_PRIOR_MEAN].to_numpy(dtype=float, copy=True)
    new_stds = posterior_df[constants.COL_PRIOR_STDV].to_numpy(dtype=float, copy=True)
    new_ns = np.full(len(posterior_df), constants.N_PRIOR, dtype=float)

    cat_id_to_row = {
        int(cat_id): i
        for i, cat_id in enumerate(posterior_df[constants.STANDARD_ID_COLUMN])
    }

    for cat_id in np.unique(obs_ids):
        row_idx = cat_id_to_row.get(int(cat_id))
        if row_idx is None:
            continue
        cat_mask = obs_ids == cat_id

        log_current_mean = np.log(new_means[row_idx])
        current_std = new_stds[row_idx]
        current_n = new_ns[row_idx]

        for log_obs, unc in zip(log_obs_vs30[cat_mask], obs_unc[cat_mask]):
            # Variance update reads the OLD mean — compute current_std before
            # updating log_current_mean below.
            current_std = np.sqrt(
                (
                    current_n * current_std**2
                    + unc**2
                    + (current_n / (current_n + 1)) * (log_obs - log_current_mean) ** 2
                )
                / (current_n + 1)
            )
            log_current_mean = (current_n * log_current_mean + log_obs) / (
                current_n + 1
            )
            current_n += 1

        new_means[row_idx] = float(np.exp(log_current_mean))
        new_stds[row_idx] = current_std
        new_ns[row_idx] = current_n

    posterior_df[constants.COL_POSTERIOR_MEAN_INDEPENDENT] = new_means
    posterior_df[constants.COL_POSTERIOR_STDV_INDEPENDENT] = new_stds
    posterior_df[constants.COL_POSTERIOR_NOBS_INDEPENDENT] = new_ns

    return posterior_df


def perform_clustering(
    sites_df: pd.DataFrame,
    nproc: int = -1,
) -> pd.DataFrame:
    """
    Apply per-category DBSCAN clustering to sites and add cluster labels.

    Parameters
    ----------
    sites_df : DataFrame
        Observations with category ID, easting, and northing columns.
    nproc : int, optional
        Number of parallel processes for DBSCAN (-1 = all cores).

    Returns
    -------
    DataFrame
        Copy of sites_df with a "cluster" column. -1 means unclustered.
    """
    sites_df = sites_df.copy()
    sites_df[constants.ObservationColumn.CLUSTER] = constants.CLUSTER_UNCLUSTERED_LABEL

    features = np.column_stack(
        (
            sites_df[constants.ObservationColumn.EASTING].to_numpy(),
            sites_df[constants.ObservationColumn.NORTHING].to_numpy(),
        )
    )
    model_ids = sites_df[constants.STANDARD_ID_COLUMN].to_numpy()
    ids = np.unique(model_ids[model_ids != constants.RASTER_ID_NODATA_VALUE]).astype(
        int
    )

    for category_id in ids:
        category_mask = model_ids == category_id
        if category_mask.sum() < constants.MIN_GROUP:
            continue
        sites_df.loc[category_mask, constants.ObservationColumn.CLUSTER] = (
            sklearn.cluster.DBSCAN(
                eps=constants.EPS, min_samples=constants.MIN_GROUP, n_jobs=nproc
            ).fit_predict(features[category_mask])
        )

    return sites_df


def compute_cluster_weighted_mean_and_stddev(
    category_sites: pd.DataFrame,
    cluster_counts: pd.Series,
    effective_n: int,
) -> tuple[float, float]:
    """Compute cluster-weighted geometric mean and log-space standard deviation.

    Parameters
    ----------
    category_sites : DataFrame
        Observation sites for one category, with vs30 and cluster columns.
    cluster_counts : Series
        Value counts of cluster labels for this category.
    effective_n : int
        Number of effective independent observations.

    Returns
    -------
    tuple[float, float]
        (geometric_mean_vs30, log_space_standard_deviation)
    """
    log_vs30_all = np.log(category_sites[constants.ObservationColumn.VS30].to_numpy())
    cluster_labels = category_sites[constants.ObservationColumn.CLUSTER].to_numpy()
    weights = np.repeat(1.0 / effective_n, len(category_sites))
    weighted_log_vs30_sum = 0.0

    for cluster_label in cluster_counts.index:
        cluster_log = log_vs30_all[cluster_labels == cluster_label]
        if cluster_label == constants.CLUSTER_UNCLUSTERED_LABEL:
            weighted_log_vs30_sum += cluster_log.sum()
        else:
            weighted_log_vs30_sum += cluster_log.mean()
            weights[cluster_labels == cluster_label] /= cluster_log.size

    return float(np.exp(weighted_log_vs30_sum / effective_n)), float(
        np.sqrt(
            np.sum(weights * (log_vs30_all - weighted_log_vs30_sum / effective_n) ** 2)
        )
    )


def update_with_clustered_data(
    prior_df: pd.DataFrame,
    sites_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform Bayesian update for clustered data.

    Parameters
    ----------
    prior_df : DataFrame
        Prior categorical model with mean and standard deviation columns.
    sites_df : DataFrame
        Clustered observation sites with vs30, cluster, and category ID columns.

    Returns
    -------
    DataFrame
        Updated DataFrame with posterior mean and standard deviation columns.

    Raises
    ------
    ValueError
        If `prior_df` has neither posterior-clustered columns nor
        COL_MEAN/COL_STDV columns.
    """
    posterior_df = prior_df.copy()

    if constants.COL_PRIOR_MEAN in posterior_df.columns:
        pass  # already normalized
    elif constants.COL_MEAN in posterior_df.columns:
        posterior_df = posterior_df.rename(
            columns={
                constants.COL_MEAN: constants.COL_PRIOR_MEAN,
                constants.COL_STDV: constants.COL_PRIOR_STDV,
            }
        )
    else:
        raise ValueError(
            f"No usable prior columns. Need either "
            f"{constants.COL_PRIOR_MEAN}+{constants.COL_PRIOR_STDV} "
            f"or {constants.COL_MEAN}+{constants.COL_STDV}."
        )

    # Cast to float so .at[] assignments below don't downcast.
    posterior_df[constants.COL_POSTERIOR_MEAN_CLUSTERED] = posterior_df[
        constants.COL_PRIOR_MEAN
    ].astype(float)
    posterior_df[constants.COL_POSTERIOR_STDV_CLUSTERED] = posterior_df[
        constants.COL_PRIOR_STDV
    ].astype(float)

    valid_sites = sites_df[
        sites_df[constants.STANDARD_ID_COLUMN] != constants.RASTER_ID_NODATA_VALUE
    ]

    id_to_idx = dict(
        zip(
            posterior_df[constants.STANDARD_ID_COLUMN].astype(int),
            posterior_df.index,
        )
    )

    for category_id in valid_sites[constants.STANDARD_ID_COLUMN].unique():
        category_id_int = int(category_id)
        if category_id_int not in id_to_idx:
            continue

        category_sites = valid_sites[
            valid_sites[constants.STANDARD_ID_COLUMN] == category_id_int
        ]
        cluster_counts = category_sites[
            constants.ObservationColumn.CLUSTER
        ].value_counts()

        # Effective independent observations: one per cluster, plus each unclustered point.
        effective_n = len(cluster_counts)
        if constants.CLUSTER_UNCLUSTERED_LABEL in cluster_counts.index:
            effective_n += cluster_counts[constants.CLUSTER_UNCLUSTERED_LABEL] - 1
        if effective_n == 0:
            continue

        mean_vs30, stddev = compute_cluster_weighted_mean_and_stddev(
            category_sites, cluster_counts, effective_n
        )
        posterior_df.at[
            id_to_idx[category_id_int], constants.COL_POSTERIOR_MEAN_CLUSTERED
        ] = mean_vs30
        posterior_df.at[
            id_to_idx[category_id_int], constants.COL_POSTERIOR_STDV_CLUSTERED
        ] = stddev

    return posterior_df


def get_vs30_for_ids(
    category_ids: np.ndarray,
    categorical_model_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Get Vs30 mean and stddev for category IDs from a categorical model.

    Parameters
    ----------
    category_ids : np.ndarray
        Array of category IDs (e.g. from assign_to_category_geology or
        assign_to_category_terrain).
    categorical_model_df : pd.DataFrame
        DataFrame with category ID, Vs30 mean, and Vs30 stddev columns.
        Column names auto-detected via utils.select_vs30_columns_by_priority.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        ``(vs30_mean, vs30_stdv)`` float64 arrays. NaN for IDs not found.
    """
    mean_col, stdv_col = utils.select_vs30_columns_by_priority(
        list(categorical_model_df.columns)
    )
    reindexed = categorical_model_df.set_index(constants.STANDARD_ID_COLUMN).reindex(
        category_ids
    )
    return (
        reindexed[mean_col].to_numpy(dtype=np.float64),
        reindexed[stdv_col].to_numpy(dtype=np.float64),
    )

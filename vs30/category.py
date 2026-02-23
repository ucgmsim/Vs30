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

from vs30 import constants, raster


def assign_to_category_geology(points: np.ndarray) -> np.ndarray:
    """
    Assign geology category IDs to points using polygon spatial join.

    Uses QMAP shapefile polygons to determine which geology category
    each point belongs to.

    Parameters
    ----------
    points : ndarray
        2D numpy array of NZTM coordinates (easting, northing).

    Returns
    -------
    ndarray
        Array of category IDs (1-indexed, or constants.RASTER_ID_NODATA_VALUE if outside polygons).
    """
    # load QMAP polygons
    gdf = gpd.read_file(constants.DATA_DIR / constants.GEOLOGY_SHAPEFILE_PATH)[
        [constants.SHAPEFILE_GEOLOGY_ID_COLUMN, constants.SHAPEFILE_GEOMETRY_COLUMN]
    ]

    # Build point GeoDataFrame
    points_shapely = shapely.points(points)
    points_gdf = gpd.GeoDataFrame(geometry=points_shapely, crs=gdf.crs)

    # Spatial join
    joined = gpd.sjoin(points_gdf, gdf, how="left", predicate="within")

    # Default to ID_NODATA, fill with gid where available
    values = np.full(len(points), constants.RASTER_ID_NODATA_VALUE, dtype=np.uint8)
    value_mask = ~joined[constants.SHAPEFILE_GEOLOGY_ID_COLUMN].isna()
    values[value_mask] = joined.loc[
        value_mask, constants.SHAPEFILE_GEOLOGY_ID_COLUMN
    ].values

    return values


def assign_to_category_terrain(points: np.ndarray) -> np.ndarray:
    """
    Assign terrain category IDs to points using raster nearest neighbor lookup.

    Uses IwahashiPike terrain raster to determine which terrain category
    each point belongs to. Reads the category ID value from the pixel
    containing each point.

    Parameters
    ----------
    points : ndarray
        2D numpy array of NZTM coordinates (easting, northing).

    Returns
    -------
    ndarray
        Array of category IDs (1-indexed, or constants.RASTER_ID_NODATA_VALUE if outside raster).
    """
    with rasterio.open(constants.DATA_DIR / constants.TERRAIN_RASTER_FILENAME) as src:
        terrain_ids = np.array([s[0] for s in src.sample(points)], dtype=src.dtypes[0])

        # Handle nodata values
        if src.nodata is not None:
            terrain_ids[terrain_ids == src.nodata] = constants.RASTER_ID_NODATA_VALUE

    return terrain_ids


def compute_bayesian_posterior_mean(
    prior_mean: float,
    num_prior_observations: float,
    observation_value: float,
) -> float:
    """
    Compute posterior mean using Bayesian update formula.

    Parameters
    ----------
    prior_mean : float
        Prior mean (in linear space, not log space).
    num_prior_observations : float
        Effective number of prior observations.
    observation_value : float
        New observation value (in linear space).

    Returns
    -------
    float
        Posterior mean (in linear space).
    """

    weighted_log_mean = (
        num_prior_observations * np.log(prior_mean) + np.log(observation_value)
    ) / (num_prior_observations + 1)
    return np.exp(weighted_log_mean)


def compute_bayesian_posterior_variance(
    prior_stdv: float,
    num_prior_observations: float,
    observation_uncertainty: float,
    prior_mean: float,
    observation_value: float,
) -> float:
    """
    Compute posterior variance using Bayesian update formula.

    Parameters
    ----------
    prior_stdv : float
        Prior standard deviation.
    num_prior_observations : float
        Effective number of prior observations.
    observation_uncertainty : float
        Uncertainty (standard deviation) of new observation.
    prior_mean : float
        Prior mean.
    observation_value : float
        New observation value (in linear space).

    Returns
    -------
    float
        Posterior variance.
    """
    log_residual = np.log(observation_value) - np.log(prior_mean)
    mean_shift = (
        num_prior_observations / (num_prior_observations + 1)
    ) * log_residual**2
    pooled_variance = (
        num_prior_observations * prior_stdv**2 + observation_uncertainty**2 + mean_shift
    )
    return pooled_variance / (num_prior_observations + 1)


def update_with_independent_data(
    categorical_model_df: pd.DataFrame,
    observations_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform Bayesian update of category mean and standard deviation values.

    Finds the posterior model with observations, updating each category's
    mean and standard deviation based on measurements assigned to that category.

    Parameters
    ----------
    categorical_model_df : DataFrame
        DataFrame with prior mean and standard deviation columns.
        Can handle various column naming conventions.
    observations_df : DataFrame
        Observations containing vs30, uncertainty, and category ID column.

    Returns
    -------
    DataFrame
        Updated DataFrame with posterior mean and standard deviation values.
        Columns:
        - constants.COL_POSTERIOR_MEAN_INDEPENDENT
        - constants.COL_POSTERIOR_STDV_INDEPENDENT
        - constants.COL_POSTERIOR_NOBS_INDEPENDENT
        - constants.COL_ASSUMED_NUM_PRIOR_OBS
        - constants.COL_ENFORCED_MIN_SIGMA
    """
    # Make a working copy to avoid modifying the input DataFrame
    updated_categorical_model_df = categorical_model_df.copy()

    # Setup Bayesian prior from categorical data, if this is not the first time we are updating. 
    if constants.COL_POSTERIOR_MEAN_CLUSTERED in updated_categorical_model_df.columns:
        updated_categorical_model_df[constants.COL_PRIOR_MEAN] = (
            updated_categorical_model_df[constants.COL_POSTERIOR_MEAN_CLUSTERED]
        )
        updated_categorical_model_df[constants.COL_PRIOR_STDV] = (
            updated_categorical_model_df[constants.COL_POSTERIOR_STDV_CLUSTERED]
        )
    elif constants.COL_MEAN in updated_categorical_model_df.columns:
            # Initial prior format - rename to prior_ columns
            updated_categorical_model_df = updated_categorical_model_df.rename(
                columns={
                    constants.COL_MEAN: constants.COL_PRIOR_MEAN,
                    constants.COL_STDV: constants.COL_PRIOR_STDV,
                }
            )
     else:
            raise ValueError(
                f"No usable prior information found. Expected either posterior columns from "
                f"previous Bayesian update or initial categorical model columns ('{constants.COL_MEAN}', "
                f"'{constants.COL_STDV}')."
            )

    # Enforce minimum sigma value on prior
    mask = updated_categorical_model_df[constants.COL_PRIOR_STDV] < constants.MIN_SIGMA
    updated_categorical_model_df.loc[mask, constants.COL_PRIOR_STDV] = (
        constants.MIN_SIGMA
    )

    # Initialize posterior columns
    updated_categorical_model_df[constants.COL_ASSUMED_NUM_PRIOR_OBS] = (
        constants.N_PRIOR
    )
    updated_categorical_model_df[constants.COL_ENFORCED_MIN_SIGMA] = constants.MIN_SIGMA
    updated_categorical_model_df[constants.COL_POSTERIOR_MEAN_INDEPENDENT] = (
        updated_categorical_model_df[constants.COL_PRIOR_MEAN]
    )
    updated_categorical_model_df[constants.COL_POSTERIOR_STDV_INDEPENDENT] = (
        updated_categorical_model_df[constants.COL_PRIOR_STDV]
    )
    updated_categorical_model_df[constants.COL_POSTERIOR_NOBS_INDEPENDENT] = (
        constants.N_PRIOR
    )

    for category_row_idx, category_row in updated_categorical_model_df.iterrows():
        # Match observations to this category using model_id
        category_id = category_row[constants.STANDARD_ID_COLUMN]
        observations_for_category_df = observations_df[
            observations_df[constants.STANDARD_ID_COLUMN] == category_id
        ]

        # Initialize running values for sequential update
        current_mean = category_row[constants.COL_POSTERIOR_MEAN_INDEPENDENT]
        current_std = category_row[constants.COL_POSTERIOR_STDV_INDEPENDENT]
        current_n = category_row[constants.COL_POSTERIOR_NOBS_INDEPENDENT]

        for _, observation_row in observations_for_category_df.iterrows():
            new_variance = compute_bayesian_posterior_variance(
                current_std,
                current_n,
                observation_row[constants.COL_UNCERTAINTY],
                current_mean,
                observation_row[constants.COL_VS30],
            )

            new_mean = compute_bayesian_posterior_mean(
                current_mean,
                current_n,
                observation_row[constants.COL_VS30],
            )

            # Update running values for next iteration
            current_mean = new_mean
            current_std = np.sqrt(new_variance)
            current_n += 1

        # Write final posterior values for this category
        updated_categorical_model_df.at[
            category_row_idx, constants.COL_POSTERIOR_MEAN_INDEPENDENT
        ] = current_mean
        updated_categorical_model_df.at[
            category_row_idx, constants.COL_POSTERIOR_STDV_INDEPENDENT
        ] = current_std
        updated_categorical_model_df.at[
            category_row_idx, constants.COL_POSTERIOR_NOBS_INDEPENDENT
        ] = current_n

    return updated_categorical_model_df


def perform_clustering(
    sites_df: pd.DataFrame,
    nproc: int = -1,
) -> pd.DataFrame:
    """
    Apply DBSCAN clustering to sites DataFrame, adding cluster assignments.

    Clusters sites spatially within each category to avoid over-weighting
    dense measurement clusters. Clustering is performed separately for each
    category ID.

    Parameters
    ----------
    sites_df : DataFrame
        Observations DataFrame with columns: constants.STANDARD_ID_COLUMN, easting, northing.
        Must have category IDs already assigned.
    nproc : int, optional
        Number of processes for DBSCAN. -1 to use all available cores.
        Default is -1.

    Returns
    -------
    DataFrame
        Modified DataFrame with added "cluster" column containing cluster IDs.
        -1 indicates unclustered points.

    Notes
    -----
    Uses MIN_GROUP and EPS constants from constants.py for DBSCAN parameters.
    """
    sites_df = sites_df.copy()
    # Default not a member of any cluster
    sites_df[constants.COL_CLUSTER] = constants.CLUSTER_UNCLUSTERED_LABEL

    features = np.column_stack(
        (
            sites_df[constants.COL_EASTING].values,
            sites_df[constants.COL_NORTHING].values,
        )
    )
    model_ids = sites_df[constants.STANDARD_ID_COLUMN].values
    ids = np.unique(model_ids)
    ids = ids[ids != constants.RASTER_ID_NODATA_VALUE].astype(int)

    for category_id in ids:
        subset_mask = model_ids == category_id
        subset = features[subset_mask]
        if subset.shape[0] < constants.MIN_GROUP:
            # Can't form any groups
            continue

        dbscan = sklearn.cluster.DBSCAN(
            eps=constants.EPS, min_samples=constants.MIN_GROUP, n_jobs=nproc
        )
        dbscan.fit(subset)

        # Save labels
        sites_df.loc[subset_mask, constants.COL_CLUSTER] = dbscan.labels_

    return sites_df


def compute_effective_sample_size(cluster_counts: pd.Series) -> int:
    """Count effective independent observations from cluster assignments.

    Each DBSCAN cluster contributes one effective observation regardless of
    size. Each unclustered point (label = -1) counts individually.

    Parameters
    ----------
    cluster_counts : Series
        Value counts of cluster labels for one category.

    Returns
    -------
    int
        Number of effective independent observations.
    """
    effective_n = len(cluster_counts)
    if constants.CLUSTER_UNCLUSTERED_LABEL in cluster_counts.index:
        effective_n += cluster_counts[constants.CLUSTER_UNCLUSTERED_LABEL] - 1
    return effective_n


def compute_cluster_weighted_mean_and_stddev(
    category_sites: pd.DataFrame,
    cluster_counts: pd.Series,
    effective_n: int,
) -> tuple[float, float]:
    """Compute cluster-weighted geometric mean and log-space standard deviation.

    Clusters contribute their geometric mean as a single pseudo-observation.
    Unclustered points each contribute individually.

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
    weighted_log_vs30_sum = 0.0
    weights = np.repeat(1.0 / effective_n, len(category_sites))

    for cluster_label in cluster_counts.index:
        cluster_mask = category_sites[constants.COL_CLUSTER] == cluster_label
        cluster_sites = category_sites[cluster_mask]
        if cluster_label == constants.CLUSTER_UNCLUSTERED_LABEL:
            weighted_log_vs30_sum += np.sum(
                np.log(cluster_sites[constants.COL_VS30].values)
            )
        else:
            weighted_log_vs30_sum += np.sum(
                np.log(cluster_sites[constants.COL_VS30].values)
            ) / len(cluster_sites)
            weights[cluster_mask] /= len(cluster_sites)

    log_geometric_mean = weighted_log_vs30_sum / effective_n
    geometric_mean_vs30 = np.exp(log_geometric_mean)
    log_stddev = np.sqrt(
        np.sum(
            weights
            * (np.log(category_sites[constants.COL_VS30].values) - log_geometric_mean)
            ** 2
        )
    )
    return geometric_mean_vs30, log_stddev


def update_with_clustered_data(
    prior_df: pd.DataFrame,
    sites_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Perform Bayesian update for clustered CPT data.

    Each DBSCAN cluster contributes a single effective observation
    (the geometric mean of its members), preventing spatially dense
    geotechnical investigations from dominating the posterior.
    Unclustered points (label = -1) each count as one observation.

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
    """
    # Create a copy to update
    posterior_df = prior_df.copy()

    if constants.COL_PRIOR_MEAN not in posterior_df.columns:
        if constants.COL_MEAN in posterior_df.columns:
            # Initial prior format - rename to prior_ columns
            posterior_df = posterior_df.rename(
                columns={
                    constants.COL_MEAN: constants.COL_PRIOR_MEAN,
                    constants.COL_STDV: constants.COL_PRIOR_STDV,
                }
            )

    # Initialize posterior columns with suffix
    posterior_df[constants.COL_POSTERIOR_MEAN_CLUSTERED] = posterior_df[
        constants.COL_PRIOR_MEAN
    ]
    posterior_df[constants.COL_POSTERIOR_STDV_CLUSTERED] = posterior_df[
        constants.COL_PRIOR_STDV
    ]

    # Filter out sites with ID_NODATA
    valid_sites = sites_df[
        sites_df[constants.STANDARD_ID_COLUMN] != constants.RASTER_ID_NODATA_VALUE
    ].copy()

    # Build a mapping from category ID to DataFrame index for direct updates
    id_to_idx = dict(
        zip(
            posterior_df[constants.STANDARD_ID_COLUMN].astype(int),
            posterior_df.index,
        )
    )

    # Process each category ID that exists in the sites
    unique_ids = valid_sites[constants.STANDARD_ID_COLUMN].unique()

    for category_id in unique_ids:
        category_id_int = int(category_id)
        if category_id_int not in id_to_idx:
            continue

        category_sites = valid_sites[
            valid_sites[constants.STANDARD_ID_COLUMN] == category_id_int
        ]
        cluster_counts = category_sites[constants.COL_CLUSTER].value_counts()

        effective_n = compute_effective_sample_size(cluster_counts)
        if effective_n == 0:
            continue

        mean_vs30, stddev = compute_cluster_weighted_mean_and_stddev(
            category_sites, cluster_counts, effective_n
        )
        df_idx = id_to_idx[category_id_int]
        posterior_df.at[df_idx, constants.COL_POSTERIOR_MEAN_CLUSTERED] = mean_vs30
        posterior_df.at[df_idx, constants.COL_POSTERIOR_STDV_CLUSTERED] = stddev

    return posterior_df


def get_vs30_for_ids(
    category_ids: np.ndarray,
    categorical_model_df: pd.DataFrame,
) -> pd.DataFrame:
    """Get Vs30 mean and standard deviation for category IDs from categorical model.

    Looks up the Vs30 mean and standard deviation for each category ID
    from the categorical model DataFrame.

    Parameters
    ----------
    category_ids : np.ndarray
        Array of category IDs (e.g. from assign_to_category_geology or
        assign_to_category_terrain).
    categorical_model_df : pd.DataFrame
        DataFrame with columns for category ID, Vs30 mean, and Vs30 standard deviation.
        Supports various column naming conventions (see Notes).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - constants.COL_CATEGORY_VS30_MEAN: Vs30 mean values (m/s) for each
          category ID. NaN for IDs not found in the model.
        - constants.COL_CATEGORY_VS30_STDV: Vs30 standard deviation values for
          each category ID. NaN for IDs not found in the model.

    Notes
    -----
    The function automatically detects the column naming convention in the
    categorical model DataFrame. See raster.select_vs30_columns_by_priority
    for the priority order.
    """
    mean_col, stdv_col = raster.select_vs30_columns_by_priority(
        list(categorical_model_df.columns)
    )

    # Build lookup dictionaries from category ID to Vs30 values
    id_to_vs30 = dict(
        zip(
            categorical_model_df[constants.STANDARD_ID_COLUMN],
            categorical_model_df[mean_col],
        )
    )
    id_to_stdv = dict(
        zip(
            categorical_model_df[constants.STANDARD_ID_COLUMN],
            categorical_model_df[stdv_col],
        )
    )

    # Look up Vs30 values for each point
    vs30_mean = np.array(
        [id_to_vs30.get(cid, np.nan) for cid in category_ids], dtype=np.float64
    )
    vs30_stdv = np.array(
        [id_to_stdv.get(cid, np.nan) for cid in category_ids], dtype=np.float64
    )

    return pd.DataFrame(
        {
            constants.COL_CATEGORY_VS30_MEAN: vs30_mean,
            constants.COL_CATEGORY_VS30_STDV: vs30_stdv,
        }
    )

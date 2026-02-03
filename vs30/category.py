"""
Functions for relating measurement vs30 values to geology/terrain categories
and performing Bayesian updates of categorical mean and standard deviation values.

This module is self-contained and includes all functionality needed to:
1. Relate measurement locations to category IDs
2. Compute category statistics from measurements
3. Perform Bayesian updates of category mean and standard deviation values
"""

import math

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
import sklearn.cluster

from vs30 import constants
from vs30 import raster


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
    # load QMAP polygons (keeps CRS from file)
    gdf = gpd.read_file(constants.DATA_DIR / constants.GEOLOGY_SHAPEFILE_PATH)[
        [constants.SHAPEFILE_GEOLOGY_ID_COLUMN, constants.SHAPEFILE_GEOMETRY_COLUMN]
    ]

    # Build point GeoDataFrame (ensure float64)
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
    containing each point (nearest neighbor, not interpolation).

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
        sampled = list(src.sample(points))
        terrain_ids = np.array([s[0] for s in sampled], dtype=src.dtypes[0])

        # Handle nodata values
        if src.nodata is not None:
            terrain_ids[terrain_ids == src.nodata] = constants.RASTER_ID_NODATA_VALUE

    return terrain_ids


def compute_bayesian_posterior_mean(
    prior_mean: float,
    num_prior_observations: float,
    posterior_variance: float,
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
    posterior_variance : float
        Posterior variance (computed from compute_bayesian_posterior_variance).
    observation_value : float
        New observation value (in linear space).

    Returns
    -------
    float
        Posterior mean (in linear space).
    """

    weighted_log_mean = (
        num_prior_observations * math.log(prior_mean) + math.log(observation_value)
    ) / (num_prior_observations + 1)
    return math.exp(weighted_log_mean)


def compute_bayesian_posterior_variance(
    prior_stdv: float,
    num_prior_observations: float,
    uncertainty: float,
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
    uncertainty : float
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
    log_residual = math.log(observation_value) - math.log(prior_mean)
    mean_shift = (
        num_prior_observations / (num_prior_observations + 1)
    ) * log_residual**2
    pooled_variance = (
        num_prior_observations * prior_stdv**2 + uncertainty**2 + mean_shift
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
        - "posterior_mean_vs30_km_per_s_independent_observations"
        - "posterior_standard_deviation_vs30_km_per_s_independent_observations"
        - "posterior_num_observations_independent_observations"
        - "assumed_num_prior_observations"
        - "enforced_min_sigma"

    Notes
    -----
    Uses N_PRIOR and MIN_SIGMA constants from constants.py.
    """
    n_prior = constants.N_PRIOR
    min_sigma = constants.MIN_SIGMA
    # Make a working copy to avoid modifying the input DataFrame
    updated_categorical_model_df = categorical_model_df.copy()

    # Identify prior columns
    prior_mean_col = constants.COL_PRIOR_MEAN
    prior_std_col = constants.COL_PRIOR_STDV

    # If a Bayesian update was previously performed, use the posterior values as priors
    # for subsequent updates. Otherwise, use the raw categorical model data as priors.
    #
    # This ensures sequential Bayesian updates: when both clustered and independent observations
    # are processed, independent observations use the spatially bias-corrected clustered posterior
    # as their prior, rather than the original (potentially biased) categorical model priors.
    if constants.COL_POSTERIOR_MEAN_CLUSTERED in updated_categorical_model_df.columns:
        # Use clustered posterior as prior for independent updates
        # This implements the sequential Bayesian update: clustered → independent
        updated_categorical_model_df[prior_mean_col] = updated_categorical_model_df[
            constants.COL_POSTERIOR_MEAN_CLUSTERED
        ]
        updated_categorical_model_df[prior_std_col] = updated_categorical_model_df[
            constants.COL_POSTERIOR_STDV_CLUSTERED
        ]
    else:
        # No posterior available - must have raw categorical data to use as priors
        if constants.COL_MEAN in updated_categorical_model_df.columns:
            # Initial prior format - rename to prior_ columns
            updated_categorical_model_df = updated_categorical_model_df.rename(
                columns={
                    constants.COL_MEAN: prior_mean_col,
                    constants.COL_STDV: prior_std_col,
                }
            )
        else:
            # Fail fast - no usable prior information available
            raise ValueError(
                f"No usable prior information found. Expected either posterior columns from "
                f"previous Bayesian update or initial categorical model columns ('{constants.COL_MEAN}', "
                f"'{constants.COL_STDV}')."
            )

    # Enforce minimum sigma value on prior
    mask = updated_categorical_model_df[prior_std_col] < min_sigma
    updated_categorical_model_df.loc[mask, prior_std_col] = min_sigma

    # Initialize posterior columns
    post_mean_col = constants.COL_POSTERIOR_MEAN_INDEPENDENT
    post_std_col = constants.COL_POSTERIOR_STDV_INDEPENDENT
    post_n_col = constants.COL_POSTERIOR_NOBS_INDEPENDENT

    updated_categorical_model_df["assumed_num_prior_observations"] = n_prior
    updated_categorical_model_df["enforced_min_sigma"] = min_sigma
    updated_categorical_model_df[post_mean_col] = updated_categorical_model_df[
        prior_mean_col
    ]
    updated_categorical_model_df[post_std_col] = updated_categorical_model_df[
        prior_std_col
    ]
    updated_categorical_model_df[post_n_col] = n_prior

    for category_row_idx, category_row in updated_categorical_model_df.iterrows():
        # Match observations to this category using model_id
        category_id = category_row[constants.STANDARD_ID_COLUMN]
        observations_for_category_df = observations_df[
            observations_df[constants.STANDARD_ID_COLUMN] == category_id
        ]

        # Initialize running values for sequential update
        current_mean = category_row[post_mean_col]
        current_std = category_row[post_std_col]
        current_n = category_row[post_n_col]

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
                new_variance,
                observation_row[constants.COL_VS30],
            )

            # Update running values for next iteration
            current_mean = new_mean
            current_std = math.sqrt(new_variance)
            current_n += 1

        # Write final posterior values for this category
        updated_categorical_model_df.at[category_row_idx, post_mean_col] = current_mean
        updated_categorical_model_df.at[category_row_idx, post_std_col] = current_std
        updated_categorical_model_df.at[category_row_idx, post_n_col] = current_n

    return updated_categorical_model_df


def perform_clustering(
    sites_df: pd.DataFrame,
    model_type: str,
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
    model_type : str
        Model type: "geology" or "terrain".
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
        (sites_df[constants.COL_EASTING].values, sites_df[constants.COL_NORTHING].values)
    )
    model_ids = sites_df[constants.STANDARD_ID_COLUMN].values
    ids = np.array(sorted(set(model_ids)))
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

    # Identify prior columns
    prior_mean_col = constants.COL_PRIOR_MEAN
    prior_std_col = constants.COL_PRIOR_STDV

    if prior_mean_col not in posterior_df.columns:
        if constants.COL_MEAN in posterior_df.columns:
            # Initial prior format - rename to prior_ columns
            posterior_df = posterior_df.rename(
                columns={
                    constants.COL_MEAN: prior_mean_col,
                    constants.COL_STDV: prior_std_col,
                }
            )

    # Initialize posterior columns with suffix
    post_mean_col = constants.COL_POSTERIOR_MEAN_CLUSTERED
    post_std_col = constants.COL_POSTERIOR_STDV_CLUSTERED

    posterior_df[post_mean_col] = posterior_df[prior_mean_col]
    posterior_df[post_std_col] = posterior_df[prior_std_col]

    # Convert to numpy array format for computation
    max_id_prior = (
        int(posterior_df[constants.STANDARD_ID_COLUMN].max())
        if len(posterior_df) > 0
        else 0
    )

    # Filter out sites with ID_NODATA
    valid_sites = sites_df[
        sites_df[constants.STANDARD_ID_COLUMN] != constants.RASTER_ID_NODATA_VALUE
    ].copy()
    max_id_sites = (
        int(valid_sites[constants.STANDARD_ID_COLUMN].max())
        if len(valid_sites) > 0
        else 0
    )

    max_id = max(max_id_prior, max_id_sites)

    posterior_array = np.full((max_id + 1, 2), np.nan)
    id_to_idx = {}
    for idx, row in posterior_df.iterrows():
        cat_id = int(row[constants.STANDARD_ID_COLUMN])
        if cat_id <= max_id:
            # Use prior values as starting point
            posterior_array[cat_id, 0] = row[prior_mean_col]
            posterior_array[cat_id, 1] = row[prior_std_col]
            id_to_idx[cat_id] = idx

    # Process each category ID that exists in the sites
    unique_ids = valid_sites[constants.STANDARD_ID_COLUMN].unique()

    for category_id in unique_ids:
        category_id_int = int(category_id)
        if category_id_int not in id_to_idx or category_id_int > max_id:
            continue

        category_sites = valid_sites[
            valid_sites[constants.STANDARD_ID_COLUMN] == category_id_int
        ]
        cluster_counts = category_sites[constants.COL_CLUSTER].value_counts()

        # Effective sample size: one per cluster, but each noise point (-1) counts individually.
        # len(cluster_counts) counts distinct cluster IDs. If CLUSTER_UNCLUSTERED_LABEL is present,
        # it was counted once but represents cluster_counts[CLUSTER_UNCLUSTERED_LABEL] individual
        # observations, so add the extra.
        effective_n = len(cluster_counts)
        if constants.CLUSTER_UNCLUSTERED_LABEL in cluster_counts.index:
            # CLUSTER_UNCLUSTERED_LABEL was already counted once
            effective_n += cluster_counts[constants.CLUSTER_UNCLUSTERED_LABEL] - 1

        if effective_n == 0:
            continue

        weighted_log_vs30_sum = 0.0
        weights = np.repeat(1.0 / effective_n, len(category_sites))

        for cluster_label in cluster_counts.index:
            cluster_mask = category_sites[constants.COL_CLUSTER] == cluster_label
            cluster_sites = category_sites[cluster_mask]
            if cluster_label == constants.CLUSTER_UNCLUSTERED_LABEL:
                # Unclustered points: each counts as one observation
                weighted_log_vs30_sum += np.sum(
                    np.log(cluster_sites[constants.COL_VS30].values)
                )
            else:
                # Clustered points: entire cluster counts as one observation
                weighted_log_vs30_sum += np.sum(
                    np.log(cluster_sites[constants.COL_VS30].values)
                ) / len(cluster_sites)
                weights[cluster_mask] /= len(cluster_sites)

        # Compute geometric mean and weighted standard deviation
        log_geometric_mean = weighted_log_vs30_sum / effective_n
        posterior_array[category_id_int, 0] = math.exp(log_geometric_mean)
        posterior_array[category_id_int, 1] = np.sqrt(
            np.sum(
                weights
                * (
                    np.log(category_sites[constants.COL_VS30].values)
                    - log_geometric_mean
                )
                ** 2
            )
        )

    # Convert back to DataFrame format
    for category_id, df_idx in id_to_idx.items():
        mean_val = posterior_array[category_id, 0]
        std_val = posterior_array[category_id, 1]

        if np.isnan(mean_val) or np.isnan(std_val):
            # Keep original prior values if somehow NaN
            continue

        posterior_df.at[df_idx, post_mean_col] = mean_val
        posterior_df.at[df_idx, post_std_col] = std_val

    return posterior_df


def posterior_from_bayesian_update(
    categorical_model_df: pd.DataFrame,
    independent_observations_df: pd.DataFrame | None = None,
    clustered_observations_df: pd.DataFrame | None = None,
    model_type: str = "geology",
) -> pd.DataFrame:
    """
    Dispatcher function to perform Bayesian updates with clustered and/or independent data.

    When both clustered and independent observations are provided, the order matters:
    1. Clustered observations (typically CPT data) are processed first with spatial clustering
       to correct for sampling biases that may arise from dense geotechnical investigations.
    2. Independent observations (typically direct Vs30 measurements) then update the
       bias-corrected model.

    This order is scientifically motivated because:
    - Clustered data may have spatial biases (urban/infrastructure-focused sampling)
    - Independent data are often higher-quality and more representative
    - Processing clustered data first corrects biases, then independent data refines the model

    Parameters
    ----------
    categorical_model_df : DataFrame
        DataFrame with prior mean and standard deviation columns.
    independent_observations_df : DataFrame, optional
        Independent observations for Bayesian update.
    clustered_observations_df : DataFrame, optional
        Clustered observations for Bayesian update.
    model_type : str, optional
        Model type: "geology" or "terrain". Default is "geology".

    Returns
    -------
    DataFrame
        Updated DataFrame with posterior values.

    Notes
    -----
    Uses N_PRIOR and MIN_SIGMA constants from constants.py for the independent
    observations Bayesian update.
    """
    df = categorical_model_df.copy()

    if clustered_observations_df is not None:
        df = update_with_clustered_data(df, clustered_observations_df)

    if independent_observations_df is not None:
        df = update_with_independent_data(df, independent_observations_df)

    return df


def get_vs30_for_points(
    points: np.ndarray,
    model_type: str,
    categorical_model_df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Get Vs30 mean and standard deviation at points from categorical model.

    This function assigns each point to a geology or terrain category, then
    looks up the Vs30 mean and standard deviation from the categorical model.

    Parameters
    ----------
    points : np.ndarray
        (N, 2) array of [easting, northing] coordinates in NZTM.
    model_type : str
        Either "geology" or "terrain".
    categorical_model_df : pd.DataFrame
        DataFrame with columns for category ID, Vs30 mean, and Vs30 standard deviation.
        Supports various column naming conventions (see Notes).

    Returns
    -------
    vs30_mean : np.ndarray
        Array of Vs30 mean values (m/s) at each point. NaN for points outside
        valid categories.
    vs30_stdv : np.ndarray
        Array of Vs30 standard deviation values at each point. NaN for points
        outside valid categories.
    category_ids : np.ndarray
        Array of category IDs assigned to each point.

    Notes
    -----
    The function automatically detects the column naming convention in the
    categorical model DataFrame. It looks for columns in this priority order
    (names defined in constants.py):

    For mean: col_posterior_mean_independent, col_posterior_mean_clustered,
              col_prior_mean, col_mean

    For stddev: col_posterior_stdv_independent, col_posterior_stdv_clustered,
                col_prior_stdv, col_stdv
    """
    # Assign category IDs to points
    if model_type == "geology":
        category_ids = assign_to_category_geology(points)
    elif model_type == "terrain":
        category_ids = assign_to_category_terrain(points)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Must be 'geology' or 'terrain'."
        )

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

    return vs30_mean, vs30_stdv, category_ids

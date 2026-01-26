"""
Functions for relating measurement vs30 values to geology/terrain categories
and performing Bayesian updates of categorical mean and standard deviation values.

This module is self-contained and includes all functionality needed to:
1. Relate measurement locations to category IDs
2. Compute category statistics from measurements
3. Perform Bayesian updates of category mean and standard deviation values
"""

from math import exp, log, sqrt
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from sklearn.cluster import DBSCAN

from vs30.config import get_default_config

_cfg = get_default_config()
RASTER_ID_NODATA_VALUE = _cfg.raster_id_nodata_value

STANDARD_ID_COLUMN = "id"

_data_dir = Path(__file__).parent / "data"


def _get_qmap_path() -> Path:
    """Get path to QMAP geology shapefile from config."""
    cfg = get_default_config()
    return _data_dir / cfg.geology_shapefile_path


def _get_terrain_raster_path() -> Path:
    """Get path to terrain classification raster from config."""
    cfg = get_default_config()
    return _data_dir / cfg.terrain_raster_filename


# ============================================================================
# Category Assignment Functions
# ============================================================================


def _assign_to_category_geology(points: np.ndarray) -> np.ndarray:
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
        Array of category IDs (1-indexed, or RASTER_ID_NODATA_VALUE if outside polygons).
    """
    # load QMAP polygons (keeps CRS from file)
    gdf = gpd.read_file(_get_qmap_path())[["gid", "geometry"]]

    # Build point GeoDataFrame (ensure float64)
    points_shapely = shapely.points(points)
    points_gdf = gpd.GeoDataFrame(geometry=points_shapely, crs=gdf.crs)

    # Spatial join
    joined = gpd.sjoin(points_gdf, gdf, how="left", predicate="within")

    # Default to ID_NODATA, fill with gid where available
    values = np.full(len(points), RASTER_ID_NODATA_VALUE, dtype=np.uint8)
    value_mask = ~joined["gid"].isna()
    values[value_mask] = joined.loc[value_mask, "gid"].values

    return values


def _assign_to_category_terrain(points: np.ndarray) -> np.ndarray:
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
        Array of category IDs (1-indexed, or RASTER_ID_NODATA_VALUE if outside raster).
    """
    with rasterio.open(_get_terrain_raster_path()) as src:
        nodata = src.nodata
        dtype = src.dtypes[0]

        sampled = list(src.sample(points))
        terrain_ids = np.array([s[0] for s in sampled], dtype=dtype)

        # Handle nodata values
        if nodata is not None:
            terrain_ids[terrain_ids == nodata] = RASTER_ID_NODATA_VALUE

    return terrain_ids


# ============================================================================
# Helper Functions for Bayesian Update
# ============================================================================


def _compute_bayesian_posterior_mean(prior_mean, num_prior_observations, posterior_variance, observation_value):
    """
    Compute posterior mean using Bayesian update formula.

    Parameters
    ----------
    prior_mean : float
        Prior mean (in linear space, not log space).
    num_prior_observations : float
        Effective number of prior observations.
    posterior_variance : float
        Posterior variance (computed from _compute_bayesian_posterior_variance).
    observation_value : float
        New observation value (in linear space).

    Returns
    -------
    float
        Posterior mean (in linear space).
    """
    return exp(
        (num_prior_observations / posterior_variance * log(prior_mean) + log(observation_value) / posterior_variance)
        / (num_prior_observations / posterior_variance + 1 / posterior_variance)
    )


def _compute_bayesian_posterior_variance(prior_stdv, num_prior_observations, uncertainty, prior_mean, observation_value):
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
    mean_shift = (num_prior_observations / (num_prior_observations + 1)) * (log(observation_value) - log(prior_mean)) ** 2
    return (num_prior_observations * prior_stdv**2 + uncertainty**2 + mean_shift) / (num_prior_observations + 1)


# ============================================================================
# Bayesian Update Functions
# ============================================================================


def update_with_independent_data(
    categorical_model_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    n_prior: int = 3,
    min_sigma: float = 0.5,
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
    n_prior : int, optional
        Assume prior model made up of n_prior measurements. Default is 3.
    min_sigma : float, optional
        Minimum model_stdv allowed. Default is 0.5.

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
    """
    cfg = get_default_config()

    # Make a working copy to avoid modifying the input DataFrame
    updated_categorical_model_df = categorical_model_df.copy()

    # Identify prior columns
    prior_mean_col = cfg.col_prior_mean
    prior_std_col = cfg.col_prior_stdv

    # If a Bayesian update was previously performed, use the posterior values as priors
    # for subsequent updates. Otherwise, use the raw categorical model data as priors.
    #
    # This ensures sequential Bayesian updates: when both clustered and independent observations
    # are processed, independent observations use the spatially bias-corrected clustered posterior
    # as their prior, rather than the original (potentially biased) categorical model priors.
    if cfg.col_posterior_mean_clustered in updated_categorical_model_df.columns:
        # Use clustered posterior as prior for independent updates
        # This implements the sequential Bayesian update: clustered → independent
        updated_categorical_model_df[prior_mean_col] = updated_categorical_model_df[
            cfg.col_posterior_mean_clustered
        ]
        updated_categorical_model_df[prior_std_col] = updated_categorical_model_df[
            cfg.col_posterior_stdv_clustered
        ]
    else:
        # No posterior available - must have raw categorical data to use as priors
        if cfg.col_mean in updated_categorical_model_df.columns:
            # Initial prior format - rename to prior_ columns
            updated_categorical_model_df = updated_categorical_model_df.rename(
                columns={
                    cfg.col_mean: prior_mean_col,
                    cfg.col_stdv: prior_std_col,
                }
            )
        else:
            # Fail fast - no usable prior information available
            raise ValueError(
                f"No usable prior information found. Expected either posterior columns from "
                f"previous Bayesian update or initial categorical model columns ('{cfg.col_mean}', "
                f"'{cfg.col_stdv}')."
            )

    # Enforce minimum sigma value on prior
    mask = updated_categorical_model_df[prior_std_col] < min_sigma
    updated_categorical_model_df.loc[mask, prior_std_col] = min_sigma

    # Initialize posterior columns
    post_mean_col = cfg.col_posterior_mean_independent
    post_std_col = cfg.col_posterior_stdv_independent
    post_n_col = "posterior_num_observations_independent_observations"

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
        category_id = category_row[STANDARD_ID_COLUMN]
        observations_for_category_df = observations_df[
            observations_df[STANDARD_ID_COLUMN] == category_id
        ]

        # Initialize running values for sequential update
        current_mean = category_row[post_mean_col]
        current_std = category_row[post_std_col]
        current_n = category_row[post_n_col]

        for _, observation_row in observations_for_category_df.iterrows():
            new_variance = _compute_bayesian_posterior_variance(
                current_std,
                current_n,
                observation_row["uncertainty"],
                current_mean,
                observation_row["vs30"],
            )

            new_mean = _compute_bayesian_posterior_mean(
                current_mean,
                current_n,
                new_variance,
                observation_row["vs30"],
            )

            # Update running values for next iteration
            current_mean = new_mean
            current_std = sqrt(new_variance)
            current_n += 1

            # Update the DataFrame with the latest posterior values
            updated_categorical_model_df.at[category_row_idx, post_mean_col] = (
                current_mean
            )

            updated_categorical_model_df.at[category_row_idx, post_std_col] = (
                current_std
            )

            updated_categorical_model_df.at[category_row_idx, post_n_col] = current_n

    return updated_categorical_model_df


def perform_clustering(
    sites_df: pd.DataFrame,
    model_type: str,
    min_group: int = 5,
    eps: float = 15000,
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
        Observations DataFrame with columns: STANDARD_ID_COLUMN, easting, northing.
        Must have category IDs already assigned.
    model_type : str
        Model type: "geology" or "terrain".
    min_group : int, optional
        Minimum group size for DBSCAN clustering. Default is 5.
    eps : float, optional
        Maximum distance (metres) for points to be considered in same cluster.
        Default is 15000.
    nproc : int, optional
        Number of processes for DBSCAN. -1 to use all available cores.
        Default is -1.

    Returns
    -------
    DataFrame
        Modified DataFrame with added "cluster" column containing cluster IDs.
        -1 indicates unclustered points.
    """
    sites_df = sites_df.copy()
    # Default not a member of any cluster (-1)
    sites_df["cluster"] = -1

    features = np.column_stack((sites_df.easting.values, sites_df.northing.values))
    model_ids = sites_df[STANDARD_ID_COLUMN].values
    ids = np.array(sorted(set(model_ids)))
    ids = ids[ids != RASTER_ID_NODATA_VALUE].astype(int)

    for category_id in ids:
        subset_mask = model_ids == category_id
        subset = features[subset_mask]
        if subset.shape[0] < min_group:
            # Can't form any groups
            continue

        dbscan = DBSCAN(eps=eps, min_samples=min_group, n_jobs=nproc)
        dbscan.fit(subset)

        # Save labels
        sites_df.loc[subset_mask, "cluster"] = dbscan.labels_

    return sites_df


def update_with_clustered_data(
    prior_df: pd.DataFrame, sites_df: pd.DataFrame, model_type: str
) -> pd.DataFrame:
    """
    Perform Bayesian update for clustered CPT data.

    model_type is deprecated but kept for compatibility.
    """
    cfg = get_default_config()

    # Create a copy to update
    posterior_df = prior_df.copy()

    # Identify prior columns
    prior_mean_col = cfg.col_prior_mean
    prior_std_col = cfg.col_prior_stdv

    if prior_mean_col not in posterior_df.columns:
        if cfg.col_mean in posterior_df.columns:
            # Initial prior format - rename to prior_ columns
            posterior_df = posterior_df.rename(
                columns={
                    cfg.col_mean: prior_mean_col,
                    cfg.col_stdv: prior_std_col,
                }
            )

    # Initialize posterior columns with suffix
    post_mean_col = cfg.col_posterior_mean_clustered
    post_std_col = cfg.col_posterior_stdv_clustered

    posterior_df[post_mean_col] = posterior_df[prior_mean_col]
    posterior_df[post_std_col] = posterior_df[prior_std_col]

    # Convert to numpy array format for computation
    max_id_prior = (
        int(posterior_df[STANDARD_ID_COLUMN].max()) if len(posterior_df) > 0 else 0
    )

    # Filter out sites with ID_NODATA
    valid_sites = sites_df[
        sites_df[STANDARD_ID_COLUMN] != RASTER_ID_NODATA_VALUE
    ].copy()
    max_id_sites = (
        int(valid_sites[STANDARD_ID_COLUMN].max()) if len(valid_sites) > 0 else 0
    )

    max_id = max(max_id_prior, max_id_sites)

    posterior_array = np.full((max_id + 1, 2), np.nan)
    id_to_idx = {}
    for idx, row in posterior_df.iterrows():
        cat_id = int(row[STANDARD_ID_COLUMN])
        if cat_id <= max_id:
            # Use prior values as starting point
            posterior_array[cat_id, 0] = row[prior_mean_col]
            posterior_array[cat_id, 1] = row[prior_std_col]
            id_to_idx[cat_id] = idx

    # Process each category ID that exists in the sites
    unique_ids = valid_sites[STANDARD_ID_COLUMN].unique()

    for category_id in unique_ids:
        category_id_int = int(category_id)
        if category_id_int not in id_to_idx or category_id_int > max_id:
            continue

        idtable = valid_sites[valid_sites[STANDARD_ID_COLUMN] == category_id_int]
        clusters = idtable["cluster"].value_counts()

        # Overall N is one per cluster, clusters labeled -1 are individual clusters
        n = len(clusters)
        if -1 in clusters.index:
            n += clusters[-1] - 1

        if n == 0:
            continue

        vs_sum = 0
        w = np.repeat(1 / n, len(idtable))

        for c in clusters.index:
            cidx = idtable["cluster"] == c
            ctable = idtable[cidx]
            if c == -1:
                # Values not part of cluster, weight = 1 per value
                vs_sum += sum(np.log(ctable.vs30.values))
            else:
                # Values in cluster, weight = 1 / cluster_size per value
                vs_sum += sum(np.log(ctable.vs30.values)) / len(ctable)
                w[cidx] /= len(ctable)

        # Update posterior array
        posterior_array[category_id_int, 0] = exp(vs_sum / n)
        posterior_array[category_id_int, 1] = np.sqrt(
            sum(w * (np.log(idtable.vs30.values) - vs_sum / n) ** 2)
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
    n_prior: int = 3,
    min_sigma: float = 0.5,
    model_type: str = "geology",  # Passed to cluster update
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
    """
    df = categorical_model_df.copy()

    if clustered_observations_df is not None:
        df = update_with_clustered_data(df, clustered_observations_df, model_type)

    # Step 2: Update with independent data (if provided)
    # Independent observations refine the already bias-corrected model from clustered data
    if independent_observations_df is not None:
        df = update_with_independent_data(
            df, independent_observations_df, n_prior, min_sigma
        )

    return df


# ============================================================================
# Point-Based Query Functions
# ============================================================================


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
    (names defined in config.yaml):

    For mean: col_posterior_mean_independent, col_posterior_mean_clustered,
              col_prior_mean, col_mean

    For stddev: col_posterior_stdv_independent, col_posterior_stdv_clustered,
                col_prior_stdv, col_stdv
    """
    # Assign category IDs to points
    if model_type == "geology":
        category_ids = _assign_to_category_geology(points)
    elif model_type == "terrain":
        category_ids = _assign_to_category_terrain(points)
    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Must be 'geology' or 'terrain'."
        )

    from vs30.raster import _select_vs30_columns_by_priority

    mean_col, stdv_col = _select_vs30_columns_by_priority(
        list(categorical_model_df.columns)
    )

    # Build lookup dictionaries from category ID to Vs30 values
    id_to_vs30 = dict(
        zip(categorical_model_df[STANDARD_ID_COLUMN], categorical_model_df[mean_col])
    )
    id_to_stdv = dict(
        zip(categorical_model_df[STANDARD_ID_COLUMN], categorical_model_df[stdv_col])
    )

    # Look up Vs30 values for each point
    vs30_mean = np.array(
        [id_to_vs30.get(cid, np.nan) for cid in category_ids], dtype=np.float64
    )
    vs30_stdv = np.array(
        [id_to_stdv.get(cid, np.nan) for cid in category_ids], dtype=np.float64
    )

    return vs30_mean, vs30_stdv, category_ids

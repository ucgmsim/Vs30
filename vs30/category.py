"""
Functions for relating measurement vs30 values to geology/terrain categories
and performing Bayesian updates of categorical mean and standard deviation values.

This module is self-contained and includes all functionality needed to:
1. Relate measurement locations to category IDs
2. Compute category statistics from measurements
3. Perform Bayesian updates of category mean and standard deviation values
"""

import os
from math import exp, log, sqrt

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from sklearn.cluster import DBSCAN

from vs30 import constants

# Constant for no-data category ID from constants
ID_NODATA = constants.ID_NODATA

# Standard column name used for model IDs in DataFrames
STANDARD_ID_COLUMN = (
    constants.STANDARD_ID_COLUMN if hasattr(constants, "STANDARD_ID_COLUMN") else "id"
)

# Data paths for category assignment from constants/internal logic
_data_dir = os.path.join(os.path.dirname(__file__), "data")
QMAP = os.path.join(_data_dir, "qmap", "qmap.shp")
MODEL_RASTER = os.path.join(_data_dir, "IwahashiPike.tif")


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
        Array of category IDs (1-indexed, or ID_NODATA if outside polygons).
    """
    # load QMAP polygons (keeps CRS from file)
    gdf = gpd.read_file(QMAP)[["gid", "geometry"]]

    # Build point GeoDataFrame (ensure float64)
    points_shapely = shapely.points(points)
    points_gdf = gpd.GeoDataFrame(geometry=points_shapely, crs=gdf.crs)

    # Spatial join
    joined = gpd.sjoin(points_gdf, gdf, how="left", predicate="within")

    # Default to ID_NODATA, fill with gid where available
    values = np.full(len(points), ID_NODATA, dtype=np.uint8)
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
        Array of category IDs (1-indexed, or ID_NODATA if outside raster).
    """
    with rasterio.open(MODEL_RASTER) as src:
        nodata = src.nodata
        dtype = src.dtypes[0]

        # Convert points to list of (x, y) tuples for rasterio.sample()
        # rasterio expects coordinates as (x, y) pairs
        coords = [(points[i, 0], points[i, 1]) for i in range(len(points))]

        # Sample values at point locations (nearest neighbor)
        # sample() returns an iterator of arrays, one per band
        # For single-band raster, each element is a 1-element array
        sampled = list(src.sample(coords))

        # Extract values from first band (assuming single band raster)
        # sampled[i] is a numpy array with shape (1,) for single-band raster
        terrain_ids = np.array([s[0] for s in sampled], dtype=dtype)

        # Handle nodata values - convert raster nodata to ID_NODATA
        if nodata is not None:
            nodata_mask = terrain_ids == nodata
            terrain_ids[nodata_mask] = ID_NODATA
        # Also handle any NaN values that might occur outside raster bounds
        nan_mask = np.isnan(terrain_ids)
        terrain_ids[nan_mask] = ID_NODATA

    return terrain_ids


# ============================================================================
# Helper Functions for Bayesian Update
# ============================================================================


def _new_mean(mu_0, n0, var, y):
    """
    Compute updated mean using Bayesian update formula.

    Parameters
    ----------
    mu_0 : float
        Prior mean (in linear space, not log space).
    n0 : float
        Effective number of prior observations.
    var : float
        Updated variance (computed from _new_var).
    y : float
        New observation value (in linear space).

    Returns
    -------
    float
        Updated mean (in linear space).
    """

    return exp((n0 / var * log(mu_0) + log(y) / var) / (n0 / var + 1 / var))


def _new_var(sigma_0, n0, uncertainty, mu_0, y):
    """
    Compute updated variance using Bayesian update formula.

    Parameters
    ----------
    sigma_0 : float
        Prior standard deviation.
    n0 : float
        Effective number of prior observations.
    uncertainty : float
        Uncertainty (standard deviation) of new observation.
    mu_0 : float
        Prior mean.
    y : float
        New observation value (in linear space).

    Returns
    -------
    float
        Updated variance.
    """
    mean_shift = (n0 / (n0 + 1)) * (log(y) - log(mu_0)) ** 2
    return (n0 * sigma_0**2 + uncertainty**2 + mean_shift) / (n0 + 1)


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
    # Handle input format
    categorical_model_df = categorical_model_df.copy()

    # Identify prior columns
    prior_mean_col = "prior_mean_vs30_km_per_s"
    prior_std_col = "prior_standard_deviation_vs30_km_per_s"

    # Check for existing prior columns, otherwise fallback to generic names or clustered posterior
    if prior_mean_col not in categorical_model_df.columns:
        if "mean_vs30_km_per_s" in categorical_model_df.columns:
            # Initial prior format - rename to prior_ columns
            categorical_model_df = categorical_model_df.rename(
                columns={
                    "mean_vs30_km_per_s": prior_mean_col,
                    "standard_deviation_vs30_km_per_s": prior_std_col,
                }
            )
        elif (
            "posterior_mean_vs30_km_per_s_clustered_observations"
            in categorical_model_df.columns
        ):
            # Use clustered posterior as prior
            categorical_model_df[prior_mean_col] = categorical_model_df[
                "posterior_mean_vs30_km_per_s_clustered_observations"
            ]
            categorical_model_df[prior_std_col] = categorical_model_df[
                "posterior_standard_deviation_vs30_km_per_s_clustered_observations"
            ]
        # Add else if needed for other cases, but this covers the main ones

    updated_categorical_model_df = categorical_model_df.copy()

    # Enforce minimum sigma value on prior
    mask = updated_categorical_model_df[prior_std_col] < min_sigma
    updated_categorical_model_df.loc[mask, prior_std_col] = min_sigma

    # Initialize posterior columns
    post_mean_col = "posterior_mean_vs30_km_per_s_independent_observations"
    post_std_col = "posterior_standard_deviation_vs30_km_per_s_independent_observations"
    post_n_col = "posterior_num_observations_independent_observations"

    updated_categorical_model_df["assumed_num_prior_observations"] = n_prior
    updated_categorical_model_df["enforced_min_sigma"] = min_sigma
    updated_categorical_model_df[post_mean_col] = updated_categorical_model_df[
        prior_mean_col
    ].astype(np.float64)
    updated_categorical_model_df[post_std_col] = updated_categorical_model_df[
        prior_std_col
    ].astype(np.float64)
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
            new_variance = _new_var(
                current_std,
                current_n,
                observation_row["uncertainty"],
                current_mean,
                observation_row["vs30"],
            )

            new_mean = _new_mean(
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
    ids = ids[ids != ID_NODATA].astype(np.int32)

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
    # Create a copy to update
    posterior_df = prior_df.copy()

    # Identify prior columns
    prior_mean_col = "prior_mean_vs30_km_per_s"
    prior_std_col = "prior_standard_deviation_vs30_km_per_s"

    if prior_mean_col not in posterior_df.columns:
        if "mean_vs30_km_per_s" in posterior_df.columns:
            # Initial prior format - rename to prior_ columns
            posterior_df = posterior_df.rename(
                columns={
                    "mean_vs30_km_per_s": prior_mean_col,
                    "standard_deviation_vs30_km_per_s": prior_std_col,
                }
            )
        # Note: If passing in output from independent update, we might need to handle checks here,
        # but usually clustered update comes first or stands alone.

    # Initialize posterior columns with suffix
    post_mean_col = "posterior_mean_vs30_km_per_s_clustered_observations"
    post_std_col = "posterior_standard_deviation_vs30_km_per_s_clustered_observations"

    posterior_df[post_mean_col] = posterior_df[prior_mean_col].astype(np.float64)
    posterior_df[post_std_col] = posterior_df[prior_std_col].astype(np.float64)

    # Convert to numpy array format for computation
    max_id_prior = (
        int(posterior_df[STANDARD_ID_COLUMN].max()) if len(posterior_df) > 0 else 0
    )

    # Filter out sites with ID_NODATA
    valid_sites = sites_df[sites_df[STANDARD_ID_COLUMN] != ID_NODATA].copy()
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
    categories_updated = 0

    # ... (skipping logging for brevity of replacement, logic remains similar)

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
        categories_updated += 1

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
    model_type: str = "geology",  # Passed to cluster update, though unused logic-wise
) -> pd.DataFrame:
    """
    Dispatcher function to perform Bayesian updates with clustered and/or independent data.
    """
    df = categorical_model_df.copy()

    # 1. Update with clustered data if provided
    if clustered_observations_df is not None:
        df = update_with_clustered_data(df, clustered_observations_df, model_type)

    # 2. Update with independent data if provided
    if independent_observations_df is not None:
        df = update_with_independent_data(
            df, independent_observations_df, n_prior, min_sigma
        )

    return df

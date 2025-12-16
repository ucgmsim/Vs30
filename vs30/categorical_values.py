"""
Functions for relating measurement vs30 values to geology/terrain categories
and performing Bayesian updates of categorical mean and standard deviation values.

This module is self-contained and includes all functionality needed to:
1. Relate measurement locations to category IDs
2. Compute category statistics from measurements
3. Perform Bayesian updates of category mean and standard deviation values
"""

import os
from math import exp, log

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
import shapely
from sklearn.cluster import DBSCAN

# Constant for no-data category ID
ID_NODATA = 255

# Standard column name used for model IDs in DataFrames
STANDARD_ID_COLUMN = "id"

# Data paths for category assignment
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


def find_posterior(
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
        DataFrame with columns: prior_mean_vs30_km_per_s, prior_standard_deviation_vs30_km_per_s
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
        - "posterior_mean_vs30_km_per_s"
        - "posterior_standard_deviation_vs30_km_per_s"
        - "posterior_num_observations"
        - "assumed_num_prior_observations"
        - "enforced_min_sigma"
    """
    # Handle input format - could be initial prior or output from cluster_update/find_posterior
    categorical_model_df = categorical_model_df.copy()

    # Check if input has posterior_ columns (from previous update)
    has_posterior_columns = (
        "posterior_mean_vs30_km_per_s" in categorical_model_df.columns
    )

    if has_posterior_columns:
        # Input is from a previous update (cluster_update or find_posterior)
        # Use posterior_ columns as the new prior
        updated_categorical_model_df = categorical_model_df.copy()
        updated_categorical_model_df["prior_mean_vs30_km_per_s"] = (
            updated_categorical_model_df["posterior_mean_vs30_km_per_s"].copy()
        )
        updated_categorical_model_df["prior_standard_deviation_vs30_km_per_s"] = (
            updated_categorical_model_df[
                "posterior_standard_deviation_vs30_km_per_s"
            ].copy()
        )
    else:
        # Initial prior format - rename to prior_ columns
        updated_categorical_model_df = categorical_model_df.copy()
        updated_categorical_model_df = updated_categorical_model_df.rename(
            columns={
                "mean_vs30_km_per_s": "prior_mean_vs30_km_per_s",
                "standard_deviation_vs30_km_per_s": "prior_standard_deviation_vs30_km_per_s",
            }
        )

    # Enforce minimum sigma value on prior
    mask = (
        updated_categorical_model_df["prior_standard_deviation_vs30_km_per_s"]
        < min_sigma
    )
    updated_categorical_model_df.loc[mask, "prior_standard_deviation_vs30_km_per_s"] = (
        min_sigma
    )

    # Initialize posterior columns
    updated_categorical_model_df["assumed_num_prior_observations"] = n_prior
    updated_categorical_model_df["enforced_min_sigma"] = min_sigma
    updated_categorical_model_df["posterior_mean_vs30_km_per_s"] = (
        updated_categorical_model_df["prior_mean_vs30_km_per_s"].astype(np.float64)
    )
    updated_categorical_model_df["posterior_standard_deviation_vs30_km_per_s"] = (
        updated_categorical_model_df["prior_standard_deviation_vs30_km_per_s"].astype(
            np.float64
        )
    )
    updated_categorical_model_df["posterior_num_observations"] = n_prior

    for category_row_idx, category_row in updated_categorical_model_df.iterrows():
        # Match observations to this category using model_id
        category_id = category_row[STANDARD_ID_COLUMN]
        observations_for_category_df = observations_df[
            observations_df[STANDARD_ID_COLUMN] == category_id
        ]

        for _, observation_row in observations_for_category_df.iterrows():
            new_variance = _new_var(
                category_row["posterior_standard_deviation_vs30_km_per_s"],
                category_row["posterior_num_observations"],
                observation_row["uncertainty"],
                category_row["posterior_mean_vs30_km_per_s"],
                observation_row["vs30"],
            )

            updated_categorical_model_df.at[
                category_row_idx, "posterior_mean_vs30_km_per_s"
            ] = _new_mean(
                category_row["posterior_mean_vs30_km_per_s"],
                category_row["posterior_num_observations"],
                new_variance,
                observation_row["vs30"],
            )

            updated_categorical_model_df.at[
                category_row_idx, "posterior_num_observations"
            ] += 1

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


def cluster_update(
    prior_df: pd.DataFrame, sites_df: pd.DataFrame, model_type: str
) -> pd.DataFrame:
    """
    Perform Bayesian update for clustered CPT data.

    Creates a model from the distribution of measured sites as clustered.
    This handles spatial clustering of measurements to avoid over-weighting
    dense measurement clusters.

    Parameters
    ----------
    prior_df : DataFrame
        Prior model DataFrame with columns:
        - STANDARD_ID_COLUMN: category IDs
        - Either "mean_vs30_km_per_s" and "standard_deviation_vs30_km_per_s" (initial prior),
          or "posterior_mean_vs30_km_per_s" and "posterior_standard_deviation_vs30_km_per_s"
          (from previous update - cluster_update or find_posterior)
    sites_df : DataFrame
        Observations with vs30, category ID, and cluster assignments.
        Must have columns: STANDARD_ID_COLUMN, "cluster", "vs30".
    model_type : str
        Model type: "geology" or "terrain" (for compatibility, not currently used).

    Returns
    -------
    DataFrame
        Updated model DataFrame containing:
        - STANDARD_ID_COLUMN: category IDs
        - "prior_mean_vs30_km_per_s": prior mean VS30 (from input)
        - "prior_standard_deviation_vs30_km_per_s": prior standard deviation (from input)
        - "posterior_mean_vs30_km_per_s": posterior mean VS30 (updated)
        - "posterior_standard_deviation_vs30_km_per_s": posterior standard deviation (updated)
        All other columns from prior_df are preserved.
    """
    # Create a copy to update
    posterior_df = prior_df.copy()

    # Rename columns to match find_posterior format (create prior_ and posterior_ columns)
    # This ensures consistency when chaining cluster_update with find_posterior
    has_posterior_columns = "posterior_mean_vs30_km_per_s" in posterior_df.columns

    if has_posterior_columns:
        # Input is from a previous update - use posterior_ as the new prior
        posterior_df["prior_mean_vs30_km_per_s"] = posterior_df[
            "posterior_mean_vs30_km_per_s"
        ].copy()
        posterior_df["prior_standard_deviation_vs30_km_per_s"] = posterior_df[
            "posterior_standard_deviation_vs30_km_per_s"
        ].copy()
    else:
        # First time through - rename existing columns to prior_ and create posterior_ columns
        posterior_df = posterior_df.rename(
            columns={
                "mean_vs30_km_per_s": "prior_mean_vs30_km_per_s",
                "standard_deviation_vs30_km_per_s": "prior_standard_deviation_vs30_km_per_s",
            }
        )
        posterior_df["posterior_mean_vs30_km_per_s"] = posterior_df[
            "prior_mean_vs30_km_per_s"
        ].astype(np.float64)
        posterior_df["posterior_standard_deviation_vs30_km_per_s"] = posterior_df[
            "prior_standard_deviation_vs30_km_per_s"
        ].astype(np.float64)

    # Convert to numpy array format for computation (matching old codebase structure)
    # We need to map DataFrame IDs to array indices
    # Find the maximum ID from both the prior and the sites to ensure array is large enough
    max_id_prior = (
        int(posterior_df[STANDARD_ID_COLUMN].max()) if len(posterior_df) > 0 else 0
    )

    # Filter out sites with ID_NODATA
    valid_sites = sites_df[sites_df[STANDARD_ID_COLUMN] != ID_NODATA].copy()
    max_id_sites = (
        int(valid_sites[STANDARD_ID_COLUMN].max()) if len(valid_sites) > 0 else 0
    )

    max_id = max(max_id_prior, max_id_sites)
    # Create array with size max_id + 1 to accommodate all IDs
    # Initialize with NaN to detect uninitialized values
    posterior_array = np.full((max_id + 1, 2), np.nan)
    id_to_idx = {}
    for idx, row in posterior_df.iterrows():
        cat_id = int(row[STANDARD_ID_COLUMN])
        if cat_id <= max_id:
            # Use prior values as starting point
            posterior_array[cat_id, 0] = row["prior_mean_vs30_km_per_s"]
            posterior_array[cat_id, 1] = row["prior_standard_deviation_vs30_km_per_s"]
            id_to_idx[cat_id] = idx

    # Process each category ID that exists in the sites
    unique_ids = valid_sites[STANDARD_ID_COLUMN].unique()
    categories_updated = 0
    categories_skipped_no_match = 0
    categories_skipped_no_clusters = 0

    for category_id in unique_ids:
        category_id_int = int(category_id)
        if category_id_int not in id_to_idx or category_id_int > max_id:
            # Category ID not in prior model or out of range, skip
            categories_skipped_no_match += 1
            continue

        idtable = valid_sites[valid_sites[STANDARD_ID_COLUMN] == category_id_int]
        clusters = idtable["cluster"].value_counts()

        # Overall N is one per cluster, clusters labeled -1 are individual clusters
        n = len(clusters)
        if -1 in clusters.index:
            n += clusters[-1] - 1

        if n == 0:
            categories_skipped_no_clusters += 1
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

    # Log statistics for debugging
    import logging

    logger = logging.getLogger(__name__)
    logger.info(
        f"cluster_update: Updated {categories_updated} categories, "
        f"skipped {categories_skipped_no_match} (no match in prior), "
        f"skipped {categories_skipped_no_clusters} (no valid clusters)"
    )
    if len(unique_ids) > 0:
        logger.info(
            f"cluster_update: Found {len(unique_ids)} unique category IDs in observations: {sorted(unique_ids)[:10]}..."
        )
    logger.info(
        f"cluster_update: Prior has {len(id_to_idx)} categories with IDs: {sorted(id_to_idx.keys())}"
    )

    # Convert back to DataFrame format
    # Update all categories that were in the prior (whether or not they had observations)
    for category_id, df_idx in id_to_idx.items():
        # All categories in prior should have values (either prior or updated)
        # Categories without observations keep their prior values
        # Categories with observations get updated values
        # We check for NaN as a safety measure, but all categories should have values
        mean_val = posterior_array[category_id, 0]
        std_val = posterior_array[category_id, 1]

        if np.isnan(mean_val) or np.isnan(std_val):
            logger.warning(
                f"Category {category_id} has NaN values in posterior array - this should not happen!"
            )
            # Keep original prior values if somehow NaN
            continue

        posterior_df.at[df_idx, "posterior_mean_vs30_km_per_s"] = mean_val
        posterior_df.at[df_idx, "posterior_standard_deviation_vs30_km_per_s"] = std_val

    return posterior_df

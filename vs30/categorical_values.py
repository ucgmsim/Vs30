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

# Constant for no-data category ID
ID_NODATA = 255

# Standard column name used for model IDs in DataFrames
STANDARD_ID_COLUMN = "model_id"

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
        - "posterior_n_observations"
        - "assumed_num_prior_observations"
        - "enforced_min_sigma"
    """
    # Enforce minimum sigma value
    categorical_model_df = categorical_model_df.copy()
    mask = categorical_model_df["standard_deviation_vs30_km_per_s"] < min_sigma
    categorical_model_df.loc[mask, "standard_deviation_vs30_km_per_s"] = min_sigma

    # Make a copy to update
    updated_categorical_model_df = categorical_model_df.copy()
    updated_categorical_model_df = updated_categorical_model_df.rename(
        columns={
            "mean_vs30_km_per_s": "prior_mean_vs30_km_per_s",
            "standard_deviation_vs30_km_per_s": "prior_standard_deviation_vs30_km_per_s",
        }
    )
    updated_categorical_model_df["assumed_num_prior_observations"] = n_prior
    updated_categorical_model_df["enforced_min_sigma"] = min_sigma
    updated_categorical_model_df["posterior_mean_vs30_km_per_s"] = (
        updated_categorical_model_df["prior_mean_vs30_km_per_s"]
    )
    updated_categorical_model_df["posterior_standard_deviation_vs30_km_per_s"] = (
        updated_categorical_model_df["prior_standard_deviation_vs30_km_per_s"]
    )
    updated_categorical_model_df["posterior_n_observations"] = n_prior

    for category_row_idx, category_row in updated_categorical_model_df.iterrows():
        observations_for_category_df = observations_df[
            observations_df[STANDARD_ID_COLUMN] == category_row[STANDARD_ID_COLUMN]
        ]

        for _, observation_row in observations_for_category_df.iterrows():
            new_variance = _new_var(
                category_row["posterior_standard_deviation_vs30_km_per_s"],
                category_row["posterior_n_observations"],
                observation_row["uncertainty"],
                category_row["posterior_mean_vs30_km_per_s"],
                observation_row["vs30"],
            )

            updated_categorical_model_df.at[
                category_row_idx, "posterior_mean_vs30_km_per_s"
            ] = _new_mean(
                category_row["posterior_mean_vs30_km_per_s"],
                category_row["posterior_n_observations"],
                new_variance,
                observation_row["vs30"],
            )

            updated_categorical_model_df.at[
                category_row_idx, "posterior_n_observations"
            ] += 1

    return updated_categorical_model_df


def cluster_update(prior, sites, letter):
    """
    Perform Bayesian update for clustered CPT data.

    Creates a model from the distribution of measured sites as clustered.
    This handles spatial clustering of measurements to avoid over-weighting
    dense measurement clusters.

    Parameters
    ----------
    prior : ndarray
        Prior model values (n_categories, 2) array of [vs30, stdv].
        Values only taken if no measurements available for ID.
    sites : DataFrame
        Observations with vs30, uncertainty, category ID, and cluster assignments.
        Must have columns: f"{letter}id", f"{letter}cluster", "vs30".
    letter : str
        Model letter prefix ("g" for geology, "t" for terrain).

    Returns
    -------
    ndarray
        Updated model values (n_categories, 2) array of [vs30, stdv].
    """
    # creates a model from the distribution of measured sites as clustered
    # prior: prior model, values only taken if no measurements available for ID
    posterior = np.copy(prior)
    # looping through model IDs
    for m in range(len(posterior)):
        vs_sum = 0
        # add 1 because IDs being used start at 1 in tiffs
        idtable = sites[sites[f"{letter}id"] == m + 1]
        clusters = idtable[f"{letter}cluster"].value_counts()
        # overall N is one per cluster, clusters labeled -1 are individual clusters
        n = len(clusters)
        if -1 in clusters.index:
            n += clusters[-1] - 1
        if n == 0:
            continue
        w = np.repeat(1 / n, len(idtable))
        for c in clusters.index:
            cidx = idtable[f"{letter}cluster"] == c
            ctable = idtable[cidx]
            if c == -1:
                # values not part of cluster, weight = 1 per value
                vs_sum += sum(np.log(ctable.vs30.values))
            else:
                # values in cluster, weight = 1 / cluster_size per value
                vs_sum += sum(np.log(ctable.vs30)) / len(ctable)
                w[cidx] /= len(ctable)
        posterior[m, 0] = exp(vs_sum / n)
        posterior[m, 1] = np.sqrt(
            sum(w * (np.log(idtable.vs30.values) - vs_sum / n) ** 2)
        )

    return posterior

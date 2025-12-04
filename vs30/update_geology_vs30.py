import time
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio import transform as rasterio_transform
from tqdm import tqdm


def grid_points_in_bbox(
    grid_locs,  # Mx2 array of grid point coordinates in NZTM
    obs_eastings_min,  # (n_obs, 1) precomputed obs_eastings - max_dist
    obs_eastings_max,  # (n_obs, 1) precomputed obs_eastings + max_dist
    obs_northings_min,  # (n_obs, 1) precomputed obs_northings - max_dist
    obs_northings_max,  # (n_obs, 1) precomputed obs_northings + max_dist
    start_grid_idx=0,  # Starting index of grid_locs in the full grid
):
    """
    Find grid points within bounding boxes of observations using fully vectorized NumPy.

    Uses broadcasting to compute all observation-grid pairs simultaneously.
    Returns collapsed mask and per-observation grid indices.

    Parameters
    ----------
    grid_locs : array_like, shape (M, 2)
        Grid point coordinates as (easting, northing) in NZTM.
    obs_eastings_min : ndarray, shape (N, 1)
        Precomputed obs_eastings - max_dist.
    obs_eastings_max : ndarray, shape (N, 1)
        Precomputed obs_eastings + max_dist.
    obs_northings_min : ndarray, shape (N, 1)
        Precomputed obs_northings - max_dist.
    obs_northings_max : ndarray, shape (N, 1)
        Precomputed obs_northings + max_dist.
    start_grid_idx : int, optional
        Starting index of grid_locs in the full grid (for offsetting indices).
        Default is 0.

    Returns
    -------
    chunk_mask : ndarray, shape (M,), dtype=bool
        Boolean array indicating which grid points in this chunk are affected
        by any observation (collapsed with np.any(axis=0)).
    obs_to_grid_indices : list of ndarray
        List of length n_obs. Each element is an array of grid point indices
        (in the full grid) that are within that observation's bounding box.
    """
    # Extract coordinates
    grid_eastings = grid_locs[:, 0]  # (n_grid,)
    grid_northings = grid_locs[:, 1]  # (n_grid,)

    # Vectorized bounding box check using broadcasting
    # (n_obs, 1) operation against (n_grid,) -> (n_obs, n_grid)
    # Checks: for each observation, which grid points are in its bounding box
    in_bbox = (
        (grid_eastings >= obs_eastings_min)
        & (grid_eastings <= obs_eastings_max)
        & (grid_northings >= obs_northings_min)
        & (grid_northings <= obs_northings_max)
    )

    # Collapse to single mask: which grid points are affected by any observation
    chunk_mask = np.any(in_bbox, axis=0)

    # For each observation, get the grid point indices within its bounding box
    # in_bbox shape: (n_obs, n_grid_chunk)
    n_obs = in_bbox.shape[0]
    obs_to_grid_indices = []

    for obs_idx in range(n_obs):
        # Get grid indices within this observation's bounding box
        grid_indices_in_chunk = np.where(in_bbox[obs_idx])[0]
        # Convert to full grid indices by adding start_grid_idx
        full_grid_indices = grid_indices_in_chunk + start_grid_idx
        obs_to_grid_indices.append(full_grid_indices)

    return chunk_mask, obs_to_grid_indices


def calculate_chunk_size(n_obs, total_memory_gb):
    """
    Calculate the maximum number of grid points per chunk based on total available memory.

    The memory-intensive operation is the boolean array of shape (n_obs, n_grid_chunk).
    NumPy boolean arrays use 1 byte per element (not 1 bit or 8 bytes).
    Memory usage per chunk: n_obs * n_grid_chunk * 1 byte (for boolean array)

    Parameters
    ----------
    n_obs : int
        Number of observations.
    total_memory_gb : float
        Total available memory in GB.

    Returns
    -------
    chunk_size : int
        Maximum number of grid points per chunk.
    """
    memory_per_chunk_bytes = total_memory_gb * 1024 * 1024 * 1024
    # Memory needed: n_obs * n_grid_chunk * 1 byte (NumPy boolean arrays are 1 byte/element)
    # Solve for n_grid_chunk: n_grid_chunk = memory_per_chunk_bytes / n_obs
    chunk_size = int(memory_per_chunk_bytes / n_obs)
    # Ensure at least 1 grid point per chunk
    return max(1, chunk_size)


if __name__ == "__main__":
    start_time = time.time()

    max_dist = 10000
    max_points = 500
    total_memory_gb = 10  # Total available memory in GB

    base_path = Path(__file__).parent

    # Open the geology.tif file
    with rasterio.open(base_path.parent / "vs30map" / "geology.tif") as src:
        median_vs30 = src.read(1)
        std_vs30 = src.read(2)

        # Get metadata
        transform = src.transform
        # crs = src.crs

    # Get NZTM coordinates for each grid point
    # Format: (N, 2) array where each row is [easting, northing]
    # Use rasterio's transform.xy to correctly convert row/col to coordinates
    rows = np.arange(median_vs30.shape[0]) + 0.5  # Add 0.5 to get pixel center
    cols = np.arange(median_vs30.shape[1]) + 0.5

    # Create meshgrid of all row/col combinations
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")

    # Use rasterio's transform.xy which correctly handles the transform
    # It accepts arrays and returns arrays
    xs, ys = rasterio_transform.xy(transform, row_grid, col_grid)

    # Flatten to get (N, 2) array: [[easting1, northing1], [easting2, northing2], ...]
    grid_locs = np.column_stack(
        (np.array(xs).flatten(), np.array(ys).flatten())
    ).astype(np.float64)

    # grid_locs = grid_locs[0:500000, :]

    # Load observed Vs30 data
    print()
    observed_vs30 = pd.read_csv(
        base_path / "resources/data/measured_vs30_original_filtered.csv"
    )

    # Extract observation locations (all observations, not just first 2)
    obs_locs = observed_vs30[["easting", "northing"]].values

    print(f"Loaded {len(observed_vs30)} observations")
    print(f"Grid has {len(grid_locs):,} points")

    # Diagnostic: Check coordinate ranges
    print("\nGrid coordinate ranges:")
    print(f"  Easting: {grid_locs[:, 0].min():.2f} to {grid_locs[:, 0].max():.2f}")
    print(f"  Northing: {grid_locs[:, 1].min():.2f} to {grid_locs[:, 1].max():.2f}")
    print("\nObservation coordinate ranges:")
    print(f"  Easting: {obs_locs[:, 0].min():.2f} to {obs_locs[:, 0].max():.2f}")
    print(f"  Northing: {obs_locs[:, 1].min():.2f} to {obs_locs[:, 1].max():.2f}")

    # Check if coordinates overlap
    grid_east_range = (grid_locs[:, 0].min(), grid_locs[:, 0].max())
    grid_north_range = (grid_locs[:, 1].min(), grid_locs[:, 1].max())
    obs_east_range = (obs_locs[:, 0].min(), obs_locs[:, 0].max())
    obs_north_range = (obs_locs[:, 1].min(), obs_locs[:, 1].max())

    east_overlap = not (
        grid_east_range[1] < obs_east_range[0] or obs_east_range[1] < grid_east_range[0]
    )
    north_overlap = not (
        grid_north_range[1] < obs_north_range[0]
        or obs_north_range[1] < grid_north_range[0]
    )

    print("\nCoordinate overlap check:")
    print(f"  Easting overlap: {east_overlap}")
    print(f"  Northing overlap: {north_overlap}")

    if not east_overlap or not north_overlap:
        print("\n⚠️  WARNING: Grid and observation coordinates do not overlap!")
        print("   This suggests a coordinate system mismatch.")
        print("   Expected NZTM (EPSG:2193) coordinates.")
        print("   NZTM eastings are typically 1,000,000 - 2,000,000")
        print("   NZTM northings are typically 4,000,000 - 7,000,000")

    # Get CRS as string for compatibility
    # NZTM is EPSG:2193
    crs_str = "EPSG:2193"

    # Calculate chunk size based on total memory
    chunk_size = calculate_chunk_size(len(obs_locs), total_memory_gb)
    n_chunks = int(np.ceil(len(grid_locs) / chunk_size))

    print(f"\nFinding active grid points within {max_dist}m of observations...")
    print(
        f"Processing in {n_chunks} chunks (chunk size: {chunk_size:,} grid points, "
        f"max memory: {total_memory_gb} GB)"
    )

    # Precompute observation bounds once (before chunking loop)
    # Shape: (n_obs, 1) for observations - keep dim for broadcasting
    obs_eastings = obs_locs[:, 0:1]  # (n_obs, 1)
    obs_northings = obs_locs[:, 1:2]  # (n_obs, 1)
    obs_eastings_min = obs_eastings - max_dist
    obs_eastings_max = obs_eastings + max_dist
    obs_northings_min = obs_northings - max_dist
    obs_northings_max = obs_northings + max_dist

    # Initialize collapsed boolean mask: (n_grid,)
    # True means grid point is affected by any observation
    grid_points_in_bbox_of_any_obs_mask = np.zeros(len(grid_locs), dtype=bool)

    # Initialize list to store grid indices for each observation
    # obs_to_grid_indices[i] will contain grid point indices within observation i's radius
    obs_to_grid_indices = [np.array([], dtype=np.int64) for _ in range(len(obs_locs))]

    # Process each chunk sequentially
    for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks"):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(grid_locs))
        grid_locs_chunk = grid_locs[start_idx:end_idx]

        # Process this chunk with precomputed bounds
        # Returns collapsed mask and per-observation grid indices
        chunk_mask, chunk_obs_to_grid = grid_points_in_bbox(
            grid_locs=grid_locs_chunk,
            obs_eastings_min=obs_eastings_min,
            obs_eastings_max=obs_eastings_max,
            obs_northings_min=obs_northings_min,
            obs_northings_max=obs_northings_max,
            start_grid_idx=start_idx,
        )

        # Store collapsed mask results
        grid_points_in_bbox_of_any_obs_mask[start_idx:end_idx] = chunk_mask

        # Accumulate grid indices for each observation
        for obs_idx, grid_indices in enumerate(chunk_obs_to_grid):
            if len(grid_indices) > 0:
                obs_to_grid_indices[obs_idx] = np.concatenate(
                    [obs_to_grid_indices[obs_idx], grid_indices]
                )

    n_affected = np.sum(grid_points_in_bbox_of_any_obs_mask)
    print(f"\nGrid points affected by any observation: {n_affected:,}")

    if n_affected == 0:
        print("\n⚠️  WARNING: No grid points found within bounding boxes!")
        print("   Checking a sample observation...")
        if len(obs_locs) > 0:
            sample_obs_idx = 0
            sample_obs = obs_locs[sample_obs_idx]
            print(
                f"   Sample observation {sample_obs_idx}: easting={sample_obs[0]:.2f}, northing={sample_obs[1]:.2f}"
            )
            print(
                f"   Bounding box: easting [{sample_obs[0] - max_dist:.2f}, {sample_obs[0] + max_dist:.2f}], "
                f"northing [{sample_obs[1] - max_dist:.2f}, {sample_obs[1] + max_dist:.2f}]"
            )

            # Check if any grid points are close
            distances = np.sqrt(
                (grid_locs[:, 0] - sample_obs[0]) ** 2
                + (grid_locs[:, 1] - sample_obs[1]) ** 2
            )
            min_dist = np.min(distances)
            closest_idx = np.argmin(distances)
            closest_point = grid_locs[closest_idx]
            print(
                f"   Closest grid point: easting={closest_point[0]:.2f}, northing={closest_point[1]:.2f}"
            )
            print(f"   Distance to closest grid point: {min_dist:.2f} m")

            if min_dist > max_dist:
                print(
                    f"   ⚠️  Closest grid point is {min_dist:.2f} m away, but max_dist is {max_dist} m"
                )
            else:
                print(
                    "   ✓ Closest grid point is within max_dist, but wasn't found by bounding box check"
                )
                print("   This suggests the bounding box logic may have an issue")

    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")

    print()

    # # Create reverse mapping: for each observation, which grid points are active
    # # Convert grid_to_obs (grid_idx -> [obs_indices]) to obs_to_grid (obs_idx -> [grid_indices])
    # obs_to_grid = {}
    # for grid_idx, obs_indices in grid_to_obs.items():
    #     for obs_idx in obs_indices:
    #         if obs_idx not in obs_to_grid:
    #             obs_to_grid[obs_idx] = []
    #         obs_to_grid[obs_idx].append(int(grid_idx))

    # # Add active grid point information to observed_vs30 DataFrame
    # # Store as list of grid point indices for each observation
    # observed_vs30["active_grid_indices"] = observed_vs30.index.map(
    #     lambda idx: obs_to_grid.get(idx, [])
    # )
    # observed_vs30["n_active_grid_points"] = observed_vs30["active_grid_indices"].map(
    #     len
    # )

    # print("\nObservations with active grid points:")
    # print(
    #     f"  {np.sum(observed_vs30['n_active_grid_points'] > 0)}/{len(observed_vs30)} observations affect grid points"
    # )
    # print(
    #     f"  Mean active grid points per observation: {observed_vs30['n_active_grid_points'].mean():.1f}"
    # )
    # print(
    #     f"  Max active grid points per observation: {observed_vs30['n_active_grid_points'].max()}"
    # )

    # print()

    # # Save updated DataFrame with active grid point information
    # output_file = "vs30/resources/data/measured_vs30_with_active_grid_points.csv"
    # print(f"\nSaving to {output_file}...")
    # # Convert list column to string for CSV compatibility
    # observed_vs30_export = observed_vs30.copy()
    # observed_vs30_export["active_grid_indices"] = observed_vs30_export[
    #     "active_grid_indices"
    # ].map(lambda x: ",".join(map(str, x)) if x else "")
    # observed_vs30_export.to_csv(output_file, index=False)
    # print(f"✓ Saved {len(observed_vs30)} observations with active grid point mappings")

    # print()

from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from tqdm import tqdm


def grid_points_in_bbox(
    grid_locs,  # Mx2 array of grid point coordinates in NZTM
    obs_eastings_min,  # (n_obs, 1) precomputed obs_eastings - max_dist
    obs_eastings_max,  # (n_obs, 1) precomputed obs_eastings + max_dist
    obs_northings_min,  # (n_obs, 1) precomputed obs_northings - max_dist
    obs_northings_max,  # (n_obs, 1) precomputed obs_northings + max_dist
):
    """
    Find active grid points within bounding boxes of observations using fully vectorized NumPy.

    Uses broadcasting to compute all observation-grid pairs simultaneously.
    This leverages BLAS/LAPACK multithreading for maximum performance.

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

    Returns
    -------
    active_mask : ndarray, shape (M,), dtype=bool
        Boolean array indicating which grid points are active (within bounding box
        of at least one observation).
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

    # The OR operation across observations can be done with any(axis=0)
    # because (n_obs, n_grid) -> (n_grid,)
    # A grid point is true if it's in the bounding box of ANY observation
    in_bbox_mask = np.any(in_bbox, axis=0)

    return in_bbox_mask


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
    # Compatible with precompute_active_grid_points_with_obs_mapping()

    # Direct calculation from transform
    # Transform: [a, b, c, d, e, f] where:
    #   x = a + b*col + c*row
    #   y = d + e*col + f*row
    # Add 0.5 to get pixel center coordinates
    rows = (np.arange(median_vs30.shape[0]) + 0.5)[:, None]  # Shape: (height, 1)
    cols = (np.arange(median_vs30.shape[1]) + 0.5)[None, :]  # Shape: (1, width)

    # Apply transform with broadcasting
    easting = transform[0] + transform[1] * cols + transform[2] * rows
    northing = transform[3] + transform[4] * cols + transform[5] * rows

    # Flatten to get (N, 2) array: [[easting1, northing1], [easting2, northing2], ...]
    grid_locs = np.column_stack((easting.flatten(), northing.flatten())).astype(
        np.float64
    )

    grid_locs = grid_locs[0:500000, :]

    # Load observed Vs30 data
    print()
    observed_vs30 = pd.read_csv(
        base_path / "resources/data/measured_vs30_original_filtered.csv"
    )

    # Extract observation locations (all observations, not just first 2)
    obs_locs = observed_vs30[["easting", "northing"]].values

    print(f"Loaded {len(observed_vs30)} observations")
    print(f"Grid has {len(grid_locs):,} points")

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

    # Initialize the result mask
    in_bbox_mask = np.zeros(len(grid_locs), dtype=bool)

    # Process each chunk sequentially
    for chunk_idx in tqdm(range(n_chunks), desc="Processing chunks"):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, len(grid_locs))
        grid_locs_chunk = grid_locs[start_idx:end_idx]

        # Process this chunk with precomputed bounds
        chunk_mask = grid_points_in_bbox(
            grid_locs=grid_locs_chunk,
            obs_eastings_min=obs_eastings_min,
            obs_eastings_max=obs_eastings_max,
            obs_northings_min=obs_northings_min,
            obs_northings_max=obs_northings_max,
        )

        # Store results in the appropriate positions
        in_bbox_mask[start_idx:end_idx] = chunk_mask

    print()

    # print()

    # n_active = np.sum(in_bbox_mask)
    # print(
    #     f"\n  Active grid points: {n_active:,}/{len(grid_locs):,} ({100 * n_active / len(grid_locs):.1f}%)"
    # )

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

#!/usr/bin/env python
"""
Wrapper to run the legacy Vs30 calculation with the expanded grid parameters
from the refactored config, and print execution time.
This version uses CLUSTERED observation data (--source cpt) with BAYESIAN POSTERIOR
updates that include spatial clustering for proper statistical weighting.
"""

import subprocess
import sys
import time

def main():
    grid_xmin = 1060050
    grid_xmax = 2120050
    grid_ymin = 4730050
    grid_ymax = 6250050
    grid_dx = 400
    grid_dy = 400


    # Output directory for legacy run
    output_dir = "/home/arr65/data/vs30/grid_models/viktor_cpt_clustered_400m"

    # Construct command
    cmd = [
        sys.executable, "run_vs30calc.py",
        "--xmin", str(grid_xmin),
        "--xmax", str(grid_xmax),
        "--ymin", str(grid_ymin),
        "--ymax", str(grid_ymax),
        "--dx", str(grid_dx),
        "--dy", str(grid_dy),
        "--out", output_dir,
        "--source", "cpt",  # Use clustered observation data
        "--gupdate", "posterior",  # Enable Bayesian posterior updates with clustering
        "--tupdate", "posterior",  # Enable Bayesian posterior updates with clustering
        "--nproc", "1",
        "--overwrite"
    ]

    print("Running legacy Vs30 calculation with expanded grid (CLUSTERED observation data + BAYESIAN CLUSTERING)...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Grid bounds: {grid_xmin:,}-{grid_xmax:,} E, {grid_ymin:,}-{grid_ymax:,} N")
    print(f"Grid spacing: {grid_dx}m x {grid_dy}m")
    print(f"Output directory: {output_dir}")
    print(f"Observation data source: CLUSTERED (cpt)")
    print(f"Model updates: BAYESIAN POSTERIOR with clustering enabled")
    print()

    # Record start time
    start_time = time.time()

    # Run the command (don't capture output so progress shows in real-time)
    print(f"Starting legacy calculation at {time.strftime('%H:%M:%S')}")
    result = subprocess.run(cmd, check=False)
    elapsed_time = time.time() - start_time

    if result.returncode == 0:
        print("Legacy calculation completed successfully!")
    else:
        print(f"Legacy calculation failed with exit code: {result.returncode}")

    print(f"  Total execution time: {elapsed_time:.1f} seconds")

if __name__ == "__main__":
    main()

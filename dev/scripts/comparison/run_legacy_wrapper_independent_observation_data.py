#!/usr/bin/env python
"""
Wrapper to run the legacy Vs30 calculation with the expanded grid parameters
from the refactored config, and print execution time.
This version uses INDEPENDENT observation data (--source original).
"""

import subprocess
import sys
import time

def main():
    # Grid parameters from current refactored config
    # grid_xmin = 1570050
    # grid_xmax = 1571050
    # grid_ymin = 5180050
    # grid_ymax = 5181050
    # grid_dx = 100
    # grid_dy = 100

    # Tiny domain grid parameters:

    grid_xmin = 1570050
    grid_xmax = 1571050
    grid_ymin = 5180050
    grid_ymax = 5181050
    grid_dx = 100
    grid_dy = 100


    # Output directory for legacy run
    output_dir = "/tmp/legacy_vs30_output_tiny_domain_independent"

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
        "--nproc", "1",
        "--overwrite"
    ]

    print("Running legacy Vs30 calculation with expanded grid (INDEPENDENT observation data)...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Grid bounds: {grid_xmin:,}-{grid_xmax:,} E, {grid_ymin:,}-{grid_ymax:,} N")
    print(f"Grid spacing: {grid_dx}m x {grid_dy}m")
    print(f"Output directory: {output_dir}")
    print(f"Observation data source: INDEPENDENT (original)")
    print()

    # Record start time
    start_time = time.time()

    # Run the command (don't capture output so progress shows in real-time)
    print(f"Starting legacy calculation at {time.strftime('%H:%M:%S')}")
    try:
        # Add timeout to prevent infinite hanging (1 hour = 3600 seconds)
        result = subprocess.run(cmd, timeout=3600, check=False)
        if result.returncode == 0:
            print("Legacy calculation completed successfully!")
        else:
            print(f"Legacy calculation failed with exit code: {result.returncode}")

        # Calculate execution time
        elapsed_time = time.time() - start_time
        print(f"  Total execution time: {elapsed_time:.1f} seconds")

    except subprocess.TimeoutExpired:
        elapsed_time = time.time() - start_time
        print(f"Legacy calculation timed out after {elapsed_time:.1f} seconds")
        print("This confirms there is an infinite loop or hang in the legacy code")
        sys.exit(1)
    except subprocess.CalledProcessError as e:
        print(f"Error running legacy calculation: {e}")
        elapsed_time = time.time() - start_time
        print(f"  Execution time before error: {elapsed_time:.1f} seconds")
        sys.exit(1)

if __name__ == "__main__":
    main()

"""
Fill nodata gaps in a benchmark VS30 raster using the gapfill module.

Usage:
    python dev/scripts/generators/gapfill_benchmark.py tests/benchmarks/jaehwi_v1p0.tif

Reads the 2-band benchmark raster (band 1 = vs30, band 2 = stdv), applies
the same gap-fill logic as the pipeline (nearest-neighbor fill for on-land
nodata pixels), and overwrites the file with the gap-filled result.
"""

import sys
from pathlib import Path

import numpy as np
import rasterio

# Ensure vs30 package is importable
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from vs30 import config, constants, gapfill, raster


def main(benchmark_path: Path) -> None:
    with rasterio.open(benchmark_path) as src:
        vs30 = src.read(1)
        stdv = src.read(2)
        profile = dict(src.profile)
        transform = src.transform
        nodata = src.nodata

    # Replace nodata sentinel with NaN for in-memory processing
    if nodata is not None:
        sentinel_mask = vs30 == nodata
        vs30 = vs30.astype(np.float64)
        stdv = stdv.astype(np.float64)
        vs30[sentinel_mask] = np.nan
        stdv[sentinel_mask] = np.nan

    n_nodata_before = np.count_nonzero(np.isnan(vs30))
    print(f"Nodata pixels before gap-fill: {n_nodata_before}")

    if n_nodata_before == 0:
        print("No nodata pixels to fill. Exiting.")
        return

    # Create geology ID array for the same grid
    nrows, ncols = vs30.shape
    xmin = transform.c
    ymax = transform.f
    dx = transform.a
    dy = -transform.e  # positive
    xmax = xmin + ncols * dx
    ymin = ymax - nrows * dy

    print(
        f"Grid: {ncols}x{nrows}, extent: ({xmin}, {ymin}) - ({xmax}, {ymax}), dx={dx}, dy={dy}"
    )
    print("Creating geology ID raster...")

    geol_ids, _ = raster.create_category_id_array(
        model_type=constants.ModelType.GEOLOGY,
        grid_config=config.GridConfig(
            grid_xmin=int(xmin),
            grid_xmax=int(xmax),
            grid_ymin=int(ymin),
            grid_ymax=int(ymax),
            grid_dx=int(dx),
            grid_dy=int(dy),
        ),
    )

    print("Running gap-fill...")
    filled_vs30, filled_stdv = gapfill.fill_nodata_grid(vs30, stdv, geol_ids, profile)

    n_nodata_after = np.count_nonzero(np.isnan(filled_vs30))
    n_filled = n_nodata_before - n_nodata_after
    print(f"Filled {n_filled} pixels")
    print(f"Nodata pixels after gap-fill: {n_nodata_after}")

    # Write back, converting NaN to the original float32 representation
    filled_vs30 = filled_vs30.astype(np.float32)
    filled_stdv = filled_stdv.astype(np.float32)

    # Update profile for writing
    write_profile = dict(profile)
    write_profile["count"] = 2
    write_profile["dtype"] = "float32"

    with rasterio.open(benchmark_path, "w", **write_profile) as dst:
        dst.write(filled_vs30, 1)
        dst.write(filled_stdv, 2)
        dst.set_band_description(1, "Vs30")
        dst.set_band_description(2, "Standard Deviation")

    print(f"Written to {benchmark_path}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(f"Usage: python {sys.argv[0]} <benchmark.tif>")
        sys.exit(1)
    main(Path(sys.argv[1]))

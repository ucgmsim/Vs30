"""Extract a subgrid from V1.0_26Mar.tif for fast experiment comparison."""
import rasterio
from rasterio.windows import from_bounds
from pathlib import Path

REF_TIF = Path("/home/arr65/data/vs30/grid_models/jaehwi_v1p0/V1.0_26Mar.tif")
XMIN, XMAX = 1555050, 1610050
YMIN, YMAX = 5145050, 5195050
OUT_DIR = Path("/home/arr65/src/vs30/dev/jaehwi_v1p0_reproduction")
OUT = OUT_DIR / "reference_subgrid.tif"

OUT_DIR.mkdir(parents=True, exist_ok=True)

with rasterio.open(REF_TIF) as src:
    window = from_bounds(XMIN, YMIN, XMAX, YMAX, src.transform)
    profile = src.profile.copy()
    profile.update(width=int(window.width), height=int(window.height),
                   transform=src.window_transform(window))
    with rasterio.open(OUT, "w", **profile) as dst:
        dst.write(src.read(window=window))

print(f"Wrote {OUT} ({int(window.width)}x{int(window.height)} pixels)")

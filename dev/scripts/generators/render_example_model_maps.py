"""Render example Vs30 maps for the wiki Home page.

Produces one PNG per supported model version and a 2x2 composite, sourced
from pre-computed grid outputs under /home/arr65/data/vs30/grid_models/.
Outputs are written to wiki/images/.

Run manually whenever the example outputs should be refreshed:

    python dev/scripts/generators/render_example_model_maps.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import rasterio

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "wiki" / "images"

GRID_MODELS = Path("/home/arr65/data/vs30/grid_models")

MODELS = {
    "foster_2019_approx": {
        "path": GRID_MODELS / "downloaded/foster_2019/foster_2019_vs30_mean.tif",
        "title": "foster_2019_approx (Foster et al. 2019 published map, 100 m)",
    },
    "modified_foster_2019": {
        "path": GRID_MODELS
        / "from_refactored_code/reproduced_modified_foster_2019_400m/combined_vs30.tif",
        "title": "modified_foster_2019 (refactored, 400 m)",
    },
    "jaehwi_v1p0": {
        "path": GRID_MODELS
        / "from_refactored_code/reproduced_jaehwi_v1p0_400m/combined_vs30.tif",
        "title": "jaehwi_v1p0 (refactored, 400 m)",
    },
    "viktor_cpt_clustering": {
        "path": GRID_MODELS / "viktor_cpt_clustered_400m/combined_mvn.tif",
        "title": "viktor_cpt_clustering (legacy, 400 m)",
    },
}

CMAP = "turbo"
MAX_DIM = 2000
PERCENTILE_CLIP = (1.0, 99.0)


def load_vs30(path: Path) -> tuple[np.ma.MaskedArray, tuple]:
    """Read band 1 from `path`, returning a masked array plus NZTM extent."""
    with rasterio.open(path) as src:
        scale = max(1, max(src.shape) // MAX_DIM)
        out_shape = (src.height // scale, src.width // scale)
        data = src.read(1, out_shape=out_shape).astype(np.float32)
        bounds = src.bounds
        nodata = src.nodata

    mask = np.isnan(data)
    if nodata is not None:
        mask |= data == nodata
    mask |= data <= 0
    return np.ma.array(data, mask=mask), (bounds.left, bounds.right, bounds.bottom, bounds.top)


def per_map_norm(data: np.ma.MaskedArray) -> mcolors.LogNorm:
    """Per-map log-norm clipped to central percentiles so each panel uses its own dynamic range."""
    valid = data.compressed()
    vmin, vmax = np.percentile(valid, PERCENTILE_CLIP)
    return mcolors.LogNorm(vmin=vmin, vmax=vmax)


def render_single(name: str, info: dict) -> Path:
    data, extent = load_vs30(info["path"])
    fig, ax = plt.subplots(figsize=(5.5, 7.5), dpi=130)
    im = ax.imshow(data, cmap=CMAP, norm=per_map_norm(data), extent=extent, origin="upper")
    ax.set_title(info["title"], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, pad=0.02)
    cbar.set_label("Vs30 (m/s)", fontsize=9)
    fig.tight_layout()
    out_path = OUTPUT_DIR / f"{name}.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_composite() -> Path:
    fig, axes = plt.subplots(2, 2, figsize=(12, 13), dpi=130)
    for ax, (name, info) in zip(axes.flat, MODELS.items()):
        data, extent = load_vs30(info["path"])
        im = ax.imshow(
            data, cmap=CMAP, norm=per_map_norm(data), extent=extent, origin="upper"
        )
        ax.set_title(info["title"], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
        cbar.set_label("Vs30 (m/s)", fontsize=9)
    fig.tight_layout()
    out_path = OUTPUT_DIR / "model_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, info in MODELS.items():
        path = render_single(name, info)
        print(f"Wrote {path.relative_to(REPO_ROOT)}")
    composite = render_composite()
    print(f"Wrote {composite.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

"""Render Vs30 comparison figure for the wiki Home page.

Produces one reference Vs30 map (Foster et al. 2019 published, resampled to
the 400 m benchmark grid) plus four log-ratio difference maps — one per
supported model version — sharing a diverging colourbar.

    ln(model / reference) > 0  ⇒ model predicts faster site than reference
    ln(model / reference) < 0  ⇒ model predicts softer site than reference

Individual PNGs per model are also written for deep-links from the wiki.
Inputs are pre-computed grid outputs under /home/arr65/data/vs30/grid_models/.

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
from rasterio.enums import Resampling
from rasterio.warp import reproject

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_DIR = REPO_ROOT / "wiki" / "images"

GRID_MODELS = Path("/home/arr65/data/vs30/grid_models")

REFERENCE = {
    "path": GRID_MODELS / "downloaded/foster_2019/foster_2019_vs30_mean.tif",
    "label": "foster_2019 (published)",
}

MODELS = {
    "foster_2019_approx": {
        "path": GRID_MODELS
        / "from_refactored_code/reproduced_foster_2019_approx_400m/combined_vs30.tif",
        "title": "foster_2019_approx",
    },
    "modified_foster_2019": {
        "path": GRID_MODELS
        / "from_refactored_code/reproduced_modified_foster_2019_400m/combined_vs30.tif",
        "title": "modified_foster_2019",
    },
    "jaehwi_v1p0": {
        "path": GRID_MODELS
        / "from_refactored_code/reproduced_jaehwi_v1p0_400m/combined_vs30.tif",
        "title": "jaehwi_v1p0",
    },
    "viktor_cpt_clustering": {
        "path": GRID_MODELS / "viktor_cpt_clustered_400m/combined_mvn.tif",
        "title": "viktor_cpt_clustering",
    },
}

REF_CMAP = "turbo"
DIFF_CMAP = "RdBu_r"
REF_PERCENTILE_CLIP = (1.0, 99.0)
DIFF_SYMMETRIC_LIMIT = 1.0  # ln-units; ≈ factor 2.72 each way


def _masked_from_array(data: np.ndarray, nodata: float | None) -> np.ma.MaskedArray:
    mask = np.isnan(data)
    if nodata is not None:
        mask |= data == nodata
    mask |= data <= 0
    return np.ma.array(data, mask=mask)


def load_on_target_grid(path: Path, target) -> np.ma.MaskedArray:
    """Read `path` band 1, reprojected onto the target raster grid."""
    with rasterio.open(path) as src:
        dst = np.full((target.height, target.width), np.nan, dtype=np.float32)
        src_data = src.read(1).astype(np.float32)
        if src.nodata is not None:
            src_data = np.where(src_data == src.nodata, np.nan, src_data)
        reproject(
            source=src_data,
            destination=dst,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=target.transform,
            dst_crs=target.crs,
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )
    return _masked_from_array(dst, None)


def load_native(path: Path) -> tuple[np.ma.MaskedArray, tuple]:
    """Read band 1 at native resolution, returning masked array and NZTM extent."""
    with rasterio.open(path) as src:
        data = src.read(1).astype(np.float32)
        bounds = src.bounds
        nodata = src.nodata
    return (
        _masked_from_array(data, nodata),
        (bounds.left, bounds.right, bounds.bottom, bounds.top),
    )


def reference_extent(target) -> tuple[float, float, float, float]:
    b = target.bounds
    return (b.left, b.right, b.bottom, b.top)


def reference_norm(ref: np.ma.MaskedArray) -> mcolors.LogNorm:
    valid = ref.compressed()
    vmin, vmax = np.percentile(valid, REF_PERCENTILE_CLIP)
    return mcolors.LogNorm(vmin=vmin, vmax=vmax)


def log_ratio(model: np.ma.MaskedArray, ref: np.ma.MaskedArray) -> np.ma.MaskedArray:
    combined_mask = np.ma.getmaskarray(model) | np.ma.getmaskarray(ref)
    ratio = np.log(model.filled(np.nan) / ref.filled(np.nan))
    return np.ma.array(ratio, mask=combined_mask | np.isnan(ratio))


def render_composite(
    ref: np.ma.MaskedArray,
    ref_extent: tuple,
    diffs: dict[str, np.ma.MaskedArray],
) -> Path:
    fig = plt.figure(figsize=(15, 11), dpi=130)
    gs = fig.add_gridspec(
        2, 3, width_ratios=[1.15, 1, 1], wspace=0.05, hspace=0.15
    )

    ax_ref = fig.add_subplot(gs[:, 0])
    ref_norm = reference_norm(ref)
    im_ref = ax_ref.imshow(
        ref, cmap=REF_CMAP, norm=ref_norm, extent=ref_extent, origin="upper"
    )
    ax_ref.set_title(f"Reference: {REFERENCE['label']}", fontsize=11)
    ax_ref.set_xticks([])
    ax_ref.set_yticks([])
    ax_ref.set_aspect("equal")
    cbar_ref = fig.colorbar(im_ref, ax=ax_ref, shrink=0.85, pad=0.02)
    cbar_ref.set_label("Vs30 (m/s)", fontsize=9)

    diff_norm = mcolors.Normalize(
        vmin=-DIFF_SYMMETRIC_LIMIT, vmax=DIFF_SYMMETRIC_LIMIT
    )
    diff_axes = []
    for idx, (name, diff) in enumerate(diffs.items()):
        row, col = divmod(idx, 2)
        ax = fig.add_subplot(gs[row, 1 + col])
        im = ax.imshow(
            diff, cmap=DIFF_CMAP, norm=diff_norm, extent=ref_extent, origin="upper"
        )
        ax.set_title(MODELS[name]["title"], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("equal")
        diff_axes.append((ax, im))

    cbar_diff = fig.colorbar(
        diff_axes[0][1],
        ax=[ax for ax, _ in diff_axes],
        orientation="vertical",
        shrink=0.85,
        pad=0.02,
    )
    cbar_diff.set_label("ln(model / reference)", fontsize=9)
    cbar_diff.ax.tick_params(labelsize=8)

    out_path = OUTPUT_DIR / "model_comparison.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_reference(ref: np.ma.MaskedArray, ref_extent: tuple) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 8.5), dpi=130)
    im = ax.imshow(
        ref,
        cmap=REF_CMAP,
        norm=reference_norm(ref),
        extent=ref_extent,
        origin="upper",
    )
    ax.set_title(REFERENCE["label"], fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("Vs30 (m/s)", fontsize=9)
    fig.tight_layout()
    out_path = OUTPUT_DIR / "reference_foster_2019.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def render_single_diff(
    name: str, diff: np.ma.MaskedArray, ref_extent: tuple
) -> Path:
    fig, ax = plt.subplots(figsize=(6.5, 8.5), dpi=130)
    norm = mcolors.Normalize(vmin=-DIFF_SYMMETRIC_LIMIT, vmax=DIFF_SYMMETRIC_LIMIT)
    im = ax.imshow(diff, cmap=DIFF_CMAP, norm=norm, extent=ref_extent, origin="upper")
    ax.set_title(f"{MODELS[name]['title']}  vs  {REFERENCE['label']}", fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect("equal")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8, pad=0.02)
    cbar.set_label("ln(model / reference)", fontsize=9)
    fig.tight_layout()
    out_path = OUTPUT_DIR / f"{name}_diff.png"
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Use one of the 400 m grids to define the target grid for the reference
    # (and hence the diffs). All refactored 400 m outputs share the same grid.
    target_path = MODELS["modified_foster_2019"]["path"]
    with rasterio.open(target_path) as target_src:
        target = target_src.profile
        target_transform = target_src.transform
        target_crs = target_src.crs
        target_bounds = target_src.bounds
        target_shape = (target_src.height, target_src.width)

    class _T:
        transform = target_transform
        crs = target_crs
        bounds = target_bounds
        height = target_shape[0]
        width = target_shape[1]

    ref = load_on_target_grid(REFERENCE["path"], _T)
    ref_extent = reference_extent(_T)

    diffs: dict[str, np.ma.MaskedArray] = {}
    for name, info in MODELS.items():
        model = load_on_target_grid(info["path"], _T)
        diffs[name] = log_ratio(model, ref)

    ref_png = render_reference(ref, ref_extent)
    print(f"Wrote {ref_png.relative_to(REPO_ROOT)}")

    for name, diff in diffs.items():
        single = render_single_diff(name, diff, ref_extent)
        print(f"Wrote {single.relative_to(REPO_ROOT)}")

    composite = render_composite(ref, ref_extent, diffs)
    print(f"Wrote {composite.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()

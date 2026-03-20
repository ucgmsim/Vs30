#!/usr/bin/env python3
"""Unified raster comparison tool for VS30 validation.

Replaces the individual comparison and diff-saving scripts with a single CLI
that can compare any two GeoTIFFs (printing statistics) or save pixel-wise
difference GeoTIFFs.
"""

import typing
from pathlib import Path
from typing import NamedTuple

import numpy as np
import rasterio
import typer
from qcore import cli
from rasterio.windows import from_bounds

app = typer.Typer(name="compare-rasters", help="Compare two GeoTIFF rasters.")

METRIC_SUFFIX = {"signed": "_diff", "absolute": "_abs_diff", "log": "_ln_diff"}


class OverlapData(NamedTuple):
    """Data from two rasters read over their overlapping region."""

    ref_data: np.ndarray
    repro_data: np.ndarray
    valid: np.ndarray
    transform: rasterio.transform.Affine
    crs: rasterio.crs.CRS
    ref_nodata: float | None
    repro_nodata: float | None
    rows: int
    cols: int


def load_overlap(
    ref_path: Path,
    ref_band: int,
    repro_path: Path,
    repro_band: int,
) -> OverlapData:
    """Open both rasters, compute their overlapping window, and read data."""
    with rasterio.open(ref_path) as ref_src, rasterio.open(repro_path) as repro_src:
        overlap_left = max(ref_src.bounds.left, repro_src.bounds.left)
        overlap_bottom = max(ref_src.bounds.bottom, repro_src.bounds.bottom)
        overlap_right = min(ref_src.bounds.right, repro_src.bounds.right)
        overlap_top = min(ref_src.bounds.top, repro_src.bounds.top)

        if overlap_left >= overlap_right or overlap_bottom >= overlap_top:
            raise typer.BadParameter("No spatial overlap between the two rasters.")

        ref_window = from_bounds(
            overlap_left, overlap_bottom, overlap_right, overlap_top,
            ref_src.transform,
        )
        repro_window = from_bounds(
            overlap_left, overlap_bottom, overlap_right, overlap_top,
            repro_src.transform,
        )

        ref_data = ref_src.read(ref_band, window=ref_window).astype(np.float64)
        repro_data = repro_src.read(repro_band, window=repro_window).astype(np.float64)

        # Trim to identical shape in case of 1-pixel rounding differences
        min_rows = min(ref_data.shape[0], repro_data.shape[0])
        min_cols = min(ref_data.shape[1], repro_data.shape[1])
        ref_data = ref_data[:min_rows, :min_cols]
        repro_data = repro_data[:min_rows, :min_cols]

        valid = np.ones((min_rows, min_cols), dtype=bool)
        if ref_src.nodata is not None:
            valid &= ref_data != ref_src.nodata
        if repro_src.nodata is not None:
            valid &= repro_data != repro_src.nodata
        valid &= ~np.isnan(ref_data) & ~np.isnan(repro_data)

        return OverlapData(
            ref_data=ref_data,
            repro_data=repro_data,
            valid=valid,
            transform=ref_src.window_transform(ref_window),
            crs=ref_src.crs,
            ref_nodata=ref_src.nodata,
            repro_nodata=repro_src.nodata,
            rows=min_rows,
            cols=min_cols,
        )


def compute_stats(ref_vals: np.ndarray, repro_vals: np.ndarray) -> dict:
    """Compute comparison statistics for 1-D arrays of valid pixel values."""
    diff = ref_vals - repro_vals
    abs_diff = np.abs(diff)

    result = {
        "ref_range": (float(ref_vals.min()), float(ref_vals.max())),
        "ref_mean": float(ref_vals.mean()),
        "repro_range": (float(repro_vals.min()), float(repro_vals.max())),
        "repro_mean": float(repro_vals.mean()),
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "std_abs_diff": float(np.std(abs_diff)),
        "median_abs_diff": float(np.median(abs_diff)),
    }

    nonzero = ref_vals != 0
    if np.any(nonzero):
        rel_diff = np.abs(diff[nonzero] / ref_vals[nonzero])
        result["max_rel_diff"] = float(np.max(rel_diff))
        result["mean_rel_diff"] = float(np.mean(rel_diff))

    for threshold in [1e-5, 1e-3, 1e-1, 1.0, 10.0]:
        pct = float(np.sum(abs_diff > threshold) / len(ref_vals) * 100)
        result[f"pct_diff_gt_{threshold}"] = pct

    flat_idx = np.argmax(abs_diff)
    result["max_diff_idx"] = int(flat_idx)
    result["max_diff_ref_val"] = float(ref_vals[flat_idx])
    result["max_diff_repro_val"] = float(repro_vals[flat_idx])

    return result


def print_stats(
    s: dict,
    n_valid: int,
    n_total: int,
    ref_shape: tuple[int, int],
    repro_shape: tuple[int, int],
    comparison_shape: tuple[int, int],
) -> None:
    """Print formatted comparison statistics."""
    print(f"  Reference shape:   {ref_shape}")
    print(f"  Reproduced shape:  {repro_shape}")
    print(f"  Comparison shape:  {comparison_shape}")
    print(f"  Valid pixels:      {n_valid:,} / {n_total:,}")
    print()
    print(f"  Reference range:   [{s['ref_range'][0]:.4f}, {s['ref_range'][1]:.4f}], mean={s['ref_mean']:.4f}")
    print(f"  Reproduced range:  [{s['repro_range'][0]:.4f}, {s['repro_range'][1]:.4f}], mean={s['repro_mean']:.4f}")
    print()
    print(f"  Max abs diff:      {s['max_abs_diff']:.6e}")
    print(f"  Mean abs diff:     {s['mean_abs_diff']:.6e}")
    print(f"  Median abs diff:   {s['median_abs_diff']:.6e}")
    print(f"  Std abs diff:      {s['std_abs_diff']:.6e}")

    if "max_rel_diff" in s:
        print(f"  Max rel diff:      {s['max_rel_diff']:.6e} ({s['max_rel_diff']*100:.4f}%)")
        print(f"  Mean rel diff:     {s['mean_rel_diff']:.6e} ({s['mean_rel_diff']*100:.4f}%)")

    print()
    for threshold in [1e-5, 1e-3, 1e-1, 1.0, 10.0]:
        key = f"pct_diff_gt_{threshold}"
        if key in s:
            print(f"  % pixels |diff| > {threshold:<8}: {s[key]:.2f}%")

    print()
    print(
        f"  Max diff at index {s['max_diff_idx']}: "
        f"reference={s['max_diff_ref_val']:.4f}, "
        f"reproduced={s['max_diff_repro_val']:.4f}"
    )


def suffixed_path(base: Path, metric: str) -> Path:
    """Return *base* with a metric-specific suffix inserted before the extension."""
    return base.with_stem(base.stem + METRIC_SUFFIX[metric])


def save_diff_tif(
    data: np.ndarray,
    valid: np.ndarray,
    output_path: Path,
    transform: rasterio.transform.Affine,
    crs: rasterio.crs.CRS,
    description: str,
    ref_path: Path,
    repro_path: Path,
) -> None:
    """Write a single-band float32 difference GeoTIFF."""
    nodata_out = np.float32(-9999)
    out = np.where(valid, data, nodata_out).astype(np.float32)

    profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "width": data.shape[1],
        "height": data.shape[0],
        "count": 1,
        "crs": crs,
        "transform": transform,
        "nodata": nodata_out,
        "compress": "deflate",
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(output_path, "w", **profile) as dst:
        dst.write(out, 1)
        dst.update_tags(
            description=description,
            reference_file=str(ref_path),
            reproduced_file=str(repro_path),
        )


def print_diff_summary(name: str, values: np.ndarray, n_valid: int) -> None:
    """Print summary stats for a saved difference raster."""
    print(f"\n  {name} — {n_valid:,} valid pixels:")
    print(f"    Min:    {values.min():.6f}")
    print(f"    Max:    {values.max():.6f}")
    print(f"    Mean:   {values.mean():.6f}")
    print(f"    Median: {np.median(values):.6f}")
    print(f"    Std:    {values.std():.6f}")


# ---------------------------------------------------------------------------
# CLI commands
# ---------------------------------------------------------------------------


@cli.from_docstring(app)
def stats(
    reference: typing.Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    reproduced: typing.Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    ref_band: typing.Annotated[int, typer.Option()] = 1,
    repro_band: typing.Annotated[int, typer.Option()] = 1,
) -> None:
    """Print comparison statistics for two rasters over their overlapping region.

    Examples
    --------
    Compare Foster 2019 median Vs30 against reproduced output::

        python compare_rasters.py stats foster_suppl_3.tif combined_vs30.tif

    Compare sigma (band 2 of reproduced) against Foster sigma file::

        python compare_rasters.py stats foster_suppl_5.tif combined_vs30.tif --repro-band 2

    Parameters
    ----------
    reference : Path
        Path to the reference GeoTIFF.
    reproduced : Path
        Path to the reproduced GeoTIFF.
    ref_band : int
        Band index to read from the reference file (1-based).
    repro_band : int
        Band index to read from the reproduced file (1-based).
    """
    overlap = load_overlap(reference, ref_band, reproduced, repro_band)
    n_valid = int(overlap.valid.sum())
    n_total = int(overlap.ref_data.size)

    if n_valid == 0:
        print("No valid overlapping pixels.")
        raise typer.Exit(code=1)

    ref_vals = overlap.ref_data[overlap.valid]
    repro_vals = overlap.repro_data[overlap.valid]
    s = compute_stats(ref_vals, repro_vals)

    # Read shapes from the source files (before trimming)
    with rasterio.open(reference) as src:
        ref_shape = src.shape
    with rasterio.open(reproduced) as src:
        repro_shape = src.shape

    print()
    print("=" * 80)
    print(f"  {reference.name} (band {ref_band})  vs  {reproduced.name} (band {repro_band})")
    print("=" * 80)

    print_stats(
        s,
        n_valid=n_valid,
        n_total=n_total,
        ref_shape=ref_shape,
        repro_shape=repro_shape,
        comparison_shape=(overlap.rows, overlap.cols),
    )


@cli.from_docstring(app)
def diff(
    reference: typing.Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    reproduced: typing.Annotated[Path, typer.Argument(exists=True, dir_okay=False)],
    output: typing.Annotated[Path, typer.Argument(dir_okay=False)],
    ref_band: typing.Annotated[int, typer.Option()] = 1,
    repro_band: typing.Annotated[int, typer.Option()] = 1,
) -> None:
    """Save pixel-wise difference GeoTIFFs and print summary statistics.

    Three files are always produced from the given output base path:
    ``<stem>_diff.tif`` (signed), ``<stem>_abs_diff.tif`` (absolute),
    and ``<stem>_ln_diff.tif`` (log-space).

    Examples
    --------
    Save all three difference rasters::

        python compare_rasters.py diff ref.tif repro.tif output_dir/vs30.tif

    Use band 2 of the reproduced file::

        python compare_rasters.py diff ref.tif repro.tif output_dir/vs30.tif --repro-band 2

    Parameters
    ----------
    reference : Path
        Path to the reference GeoTIFF.
    reproduced : Path
        Path to the reproduced GeoTIFF.
    output : Path
        Base output path. Metric suffixes are appended before the extension.
    ref_band : int
        Band index to read from the reference file (1-based).
    repro_band : int
        Band index to read from the reproduced file (1-based).
    """
    overlap = load_overlap(reference, ref_band, reproduced, repro_band)
    n_valid = int(overlap.valid.sum())

    if n_valid == 0:
        print("No valid overlapping pixels.")
        raise typer.Exit(code=1)

    print(f"Reference:  {reference}")
    print(f"Reproduced: {reproduced}")
    print(f"Valid pixels: {n_valid:,}")

    # Signed difference
    signed = overlap.ref_data - overlap.repro_data
    signed_path = suffixed_path(output, "signed")
    save_diff_tif(
        signed, overlap.valid, signed_path,
        overlap.transform, overlap.crs,
        "reference minus reproduced (signed)",
        reference, reproduced,
    )
    print_diff_summary(f"Signed diff  → {signed_path}", signed[overlap.valid], n_valid)

    # Absolute difference
    absolute = np.abs(signed)
    abs_path = suffixed_path(output, "absolute")
    save_diff_tif(
        absolute, overlap.valid, abs_path,
        overlap.transform, overlap.crs,
        "|reference - reproduced| (absolute)",
        reference, reproduced,
    )
    print_diff_summary(f"Abs diff     → {abs_path}", absolute[overlap.valid], n_valid)

    # Log-space difference (only where both values are positive)
    ln_valid = overlap.valid & (overlap.ref_data > 0) & (overlap.repro_data > 0)
    safe_ref = np.where(ln_valid, overlap.ref_data, 1.0)
    safe_repro = np.where(ln_valid, overlap.repro_data, 1.0)
    ln_diff = np.where(ln_valid, np.log(safe_ref) - np.log(safe_repro), 0.0)
    ln_path = suffixed_path(output, "log")
    save_diff_tif(
        ln_diff, ln_valid, ln_path,
        overlap.transform, overlap.crs,
        "ln(reference) - ln(reproduced)",
        reference, reproduced,
    )
    n_ln_valid = int(ln_valid.sum())
    print_diff_summary(f"Log diff     → {ln_path}", ln_diff[ln_valid], n_ln_valid)


if __name__ == "__main__":
    app()

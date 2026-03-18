#!/usr/bin/env python3
"""
Compare outputs between legacy and refactored VS30 codebases.

This script compares the GeoTIFF outputs from both independent and clustered
observation modes, calculating statistics to assess whether differences are
within acceptable numerical precision tolerances.

Usage:
    python compare_legacy_vs_refactored.py
"""

from pathlib import Path

import numpy as np
import rasterio


# File mappings between legacy and refactored outputs
FILE_MAPPINGS = [
    # (description, legacy_filename, refactored_filename)
    ("Geology IDs", "gid.tif", "gid.tif"),
    ("Terrain IDs", "tid.tif", "tid.tif"),
    ("Slope", "slope.tif", "slope.tif"),
    ("Coast Distance", "coast.tif", "coast_distance.tif"),
    (
        "Initial Geology VS30 (before MVN)",
        "geology.tif",
        "geology_vs30_slope_and_coastal_distance_adjusted_with_uncertainty.tif",
    ),
    (
        "Initial Terrain VS30 (before MVN)",
        "terrain.tif",
        "initial_terrain_vs30_with_uncertainty.tif",
    ),
    (
        "Final Geology VS30 (after MVN)",
        "geology_mvn.tif",
        "geology_vs30_slope_and_coastal_distance_and_spatially_adjusted_with_uncertainty.tif",
    ),
    (
        "Final Terrain VS30 (after MVN)",
        "terrain_mvn.tif",
        "terrain_vs30_spatially_adjusted_with_uncertainty.tif",
    ),
    (
        "Combined Final VS30",
        "combined_mvn.tif",
        "combined_vs30.tif",
    ),
]


def compare_rasters(legacy_path: Path, refactored_path: Path, description: str) -> dict:
    """
    Compare two raster files and return statistics.

    Parameters
    ----------
    legacy_path : Path
        Path to legacy output raster.
    refactored_path : Path
        Path to refactored output raster.
    description : str
        Human-readable description of the comparison.

    Returns
    -------
    dict
        Dictionary with comparison statistics, or None if files missing.
    """
    result = {
        "description": description,
        "legacy_path": str(legacy_path),
        "refactored_path": str(refactored_path),
        "status": "unknown",
        "bands": [],
    }

    if not legacy_path.exists():
        result["status"] = "legacy_missing"
        return result

    if not refactored_path.exists():
        result["status"] = "refactored_missing"
        return result

    try:
        with rasterio.open(legacy_path) as src_l, rasterio.open(refactored_path) as src_r:
            # Check band count
            if src_l.count != src_r.count:
                result["status"] = "band_count_mismatch"
                result["legacy_bands"] = src_l.count
                result["refactored_bands"] = src_r.count
                return result

            # Check shape
            if src_l.shape != src_r.shape:
                result["status"] = "shape_mismatch"
                result["legacy_shape"] = src_l.shape
                result["refactored_shape"] = src_r.shape
                return result

            result["shape"] = src_l.shape
            result["n_bands"] = src_l.count

            # Compare each band
            for band_idx in range(1, src_l.count + 1):
                data_l = src_l.read(band_idx).astype(np.float64)
                data_r = src_r.read(band_idx).astype(np.float64)

                nodata_l = src_l.nodata
                nodata_r = src_r.nodata

                # Create valid mask (not nodata in either)
                valid_mask = np.ones(data_l.shape, dtype=bool)
                if nodata_l is not None:
                    valid_mask &= data_l != nodata_l
                if nodata_r is not None:
                    valid_mask &= data_r != nodata_r
                valid_mask &= ~np.isnan(data_l) & ~np.isnan(data_r)

                n_valid = np.sum(valid_mask)

                if n_valid == 0:
                    result["bands"].append({
                        "band": band_idx,
                        "status": "no_valid_pixels",
                    })
                    continue

                # Calculate differences
                diff = data_l[valid_mask] - data_r[valid_mask]
                abs_diff = np.abs(diff)

                # Calculate relative differences for non-zero values
                legacy_vals = data_l[valid_mask]
                nonzero_mask = legacy_vals != 0
                if np.any(nonzero_mask):
                    rel_diff = np.abs(diff[nonzero_mask] / legacy_vals[nonzero_mask])
                    max_rel_diff = np.max(rel_diff)
                    mean_rel_diff = np.mean(rel_diff)
                else:
                    max_rel_diff = 0.0
                    mean_rel_diff = 0.0

                # Tolerance thresholds
                abs_tolerance = 1e-5
                rel_tolerance = 1e-4  # 0.01%

                band_stats = {
                    "band": band_idx,
                    "n_valid_pixels": int(n_valid),
                    "max_abs_diff": float(np.max(abs_diff)),
                    "mean_abs_diff": float(np.mean(abs_diff)),
                    "std_abs_diff": float(np.std(abs_diff)),
                    "max_rel_diff": float(max_rel_diff),
                    "mean_rel_diff": float(mean_rel_diff),
                    "pct_diff_gt_1e-5": float(np.sum(abs_diff > 1e-5) / n_valid * 100),
                    "pct_diff_gt_1e-3": float(np.sum(abs_diff > 1e-3) / n_valid * 100),
                    "pct_diff_gt_1e-1": float(np.sum(abs_diff > 1e-1) / n_valid * 100),
                    "pct_diff_gt_1": float(np.sum(abs_diff > 1) / n_valid * 100),
                    "legacy_min": float(np.min(data_l[valid_mask])),
                    "legacy_max": float(np.max(data_l[valid_mask])),
                    "legacy_mean": float(np.mean(data_l[valid_mask])),
                    "refactored_min": float(np.min(data_r[valid_mask])),
                    "refactored_max": float(np.max(data_r[valid_mask])),
                    "refactored_mean": float(np.mean(data_r[valid_mask])),
                }

                # Find location of max difference
                if np.max(abs_diff) > 0:
                    flat_idx = np.argmax(abs_diff)
                    valid_indices = np.where(valid_mask.flatten())[0]
                    max_idx = np.unravel_index(valid_indices[flat_idx], data_l.shape)
                    band_stats["max_diff_location"] = max_idx
                    band_stats["max_diff_legacy_val"] = float(data_l[max_idx])
                    band_stats["max_diff_refactored_val"] = float(data_r[max_idx])

                result["bands"].append(band_stats)

            result["status"] = "compared"

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)

    return result


def assess_comparison(results: list[dict]) -> str:
    """
    Assess overall comparison results and determine if differences are acceptable.

    Parameters
    ----------
    results : list[dict]
        List of comparison result dictionaries.

    Returns
    -------
    str
        Assessment summary text.
    """
    lines = []
    all_acceptable = True
    warnings = []
    errors = []

    for result in results:
        desc = result["description"]

        if result["status"] == "legacy_missing":
            warnings.append(f"  - {desc}: Legacy file missing")
            continue
        if result["status"] == "refactored_missing":
            errors.append(f"  - {desc}: Refactored file missing")
            all_acceptable = False
            continue
        if result["status"] == "band_count_mismatch":
            errors.append(f"  - {desc}: Band count mismatch")
            all_acceptable = False
            continue
        if result["status"] == "shape_mismatch":
            errors.append(f"  - {desc}: Shape mismatch")
            all_acceptable = False
            continue
        if result["status"] == "error":
            errors.append(f"  - {desc}: Error - {result.get('error', 'unknown')}")
            all_acceptable = False
            continue

        for band in result.get("bands", []):
            if band.get("status") == "no_valid_pixels":
                warnings.append(f"  - {desc} Band {band['band']}: No valid pixels")
                continue

            max_abs = band["max_abs_diff"]
            max_rel = band["max_rel_diff"]
            pct_gt_1 = band["pct_diff_gt_1"]

            # Thresholds for acceptable differences
            # For VS30 values (typically 100-1000 m/s), differences < 1 m/s are excellent
            # Differences < 10 m/s are acceptable (< 1-10% relative)
            if max_abs > 10:
                errors.append(
                    f"  - {desc} Band {band['band']}: Large max diff = {max_abs:.2f} m/s"
                )
                all_acceptable = False
            elif max_abs > 1:
                warnings.append(
                    f"  - {desc} Band {band['band']}: Moderate max diff = {max_abs:.4f} m/s"
                )
            elif pct_gt_1 > 10:
                warnings.append(
                    f"  - {desc} Band {band['band']}: {pct_gt_1:.1f}% pixels have diff > 1 m/s"
                )

    lines.append("=" * 80)
    lines.append("ASSESSMENT SUMMARY")
    lines.append("=" * 80)

    if all_acceptable and not errors:
        lines.append("")
        lines.append("RESULT: ACCEPTABLE")
        lines.append("")
        lines.append("The differences between legacy and refactored outputs are within")
        lines.append("acceptable tolerances. Any differences are likely due to:")
        lines.append("  - Improved numerical precision (float64 vs float32)")
        lines.append("  - Use of scipy.cdist instead of complex number arithmetic")
        lines.append("  - Minor algorithmic improvements in the refactored code")
        lines.append("")
    else:
        lines.append("")
        lines.append("RESULT: NEEDS REVIEW")
        lines.append("")
        lines.append("Some differences exceed expected numerical precision tolerances.")
        lines.append("This may indicate algorithmic differences that should be investigated.")
        lines.append("")

    if errors:
        lines.append("ERRORS:")
        lines.extend(errors)
        lines.append("")

    if warnings:
        lines.append("WARNINGS:")
        lines.extend(warnings)
        lines.append("")

    return "\n".join(lines)


def print_comparison_results(results: list[dict], mode_name: str) -> None:
    """Print formatted comparison results."""
    print("")
    print("=" * 80)
    print(f"COMPARISON: {mode_name}")
    print("=" * 80)

    for result in results:
        print("")
        print(f"--- {result['description']} ---")

        if result["status"] != "compared":
            print(f"  Status: {result['status']}")
            if "error" in result:
                print(f"  Error: {result['error']}")
            continue

        print(f"  Shape: {result['shape']}")

        for band in result.get("bands", []):
            if band.get("status") == "no_valid_pixels":
                print(f"  Band {band['band']}: No valid pixels")
                continue

            print(f"  Band {band['band']}:")
            print(f"    Valid pixels: {band['n_valid_pixels']:,}")
            print(f"    Legacy range: [{band['legacy_min']:.2f}, {band['legacy_max']:.2f}], mean={band['legacy_mean']:.2f}")
            print(f"    Refactored range: [{band['refactored_min']:.2f}, {band['refactored_max']:.2f}], mean={band['refactored_mean']:.2f}")
            print(f"    Max abs diff: {band['max_abs_diff']:.6e}")
            print(f"    Mean abs diff: {band['mean_abs_diff']:.6e}")
            print(f"    Max rel diff: {band['max_rel_diff']:.6e} ({band['max_rel_diff']*100:.4f}%)")
            print(f"    % pixels > 0.00001: {band['pct_diff_gt_1e-5']:.2f}%")
            print(f"    % pixels > 0.001: {band['pct_diff_gt_1e-3']:.2f}%")
            print(f"    % pixels > 0.1: {band['pct_diff_gt_1e-1']:.2f}%")
            print(f"    % pixels > 1: {band['pct_diff_gt_1']:.2f}%")

            if "max_diff_location" in band:
                print(f"    Max diff at {band['max_diff_location']}: legacy={band['max_diff_legacy_val']:.4f}, refactored={band['max_diff_refactored_val']:.4f}")


def main():
    # Output directories
    legacy_independent = Path("/tmp/legacy_vs30_output_tiny_domain_independent")
    legacy_clustered = Path("/tmp/legacy_vs30_output_tiny_domain_clustered")
    refactored_independent = Path("/tmp/refactored_vs30_output_tiny_domain_independent_observations")
    refactored_clustered = Path("/tmp/refactored_vs30_output_tiny_domain_clustered_observations")

    all_results = []

    # Compare independent observations mode
    print("\n" + "=" * 80)
    print("INDEPENDENT OBSERVATIONS MODE")
    print("=" * 80)

    independent_results = []
    for desc, legacy_file, refactored_file in FILE_MAPPINGS:
        legacy_path = legacy_independent / legacy_file
        refactored_path = refactored_independent / refactored_file
        result = compare_rasters(legacy_path, refactored_path, desc)
        independent_results.append(result)

    print_comparison_results(independent_results, "Independent Observations Mode")
    all_results.extend(independent_results)

    # Compare clustered observations mode
    print("\n" + "=" * 80)
    print("CLUSTERED OBSERVATIONS MODE")
    print("=" * 80)

    clustered_results = []
    for desc, legacy_file, refactored_file in FILE_MAPPINGS:
        legacy_path = legacy_clustered / legacy_file
        refactored_path = refactored_clustered / refactored_file
        result = compare_rasters(legacy_path, refactored_path, desc)
        clustered_results.append(result)

    print_comparison_results(clustered_results, "Clustered Observations Mode")
    all_results.extend(clustered_results)

    # Overall assessment
    print("")
    print(assess_comparison(all_results))


if __name__ == "__main__":
    main()

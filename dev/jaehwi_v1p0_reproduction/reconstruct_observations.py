"""
Reconstruct Jaehwi's exact observation dataset (671 observations) from his
source files, replicating the loading logic in sites_load_NSHM2022_0.py.

Sources:
  - McGann CPT Vs30 (NZMG coords, downsampled on 1km grid) -> 276 observations
  - Wotherspoon measured Vs30 (WGS84 lat/lon) -> 36 observations
  - GeoNet/Kaiser site metadata (WGS84 lat/lon, Q-based uncertainty) -> 359 observations

The current CSV files contain entries added after the v1p0 model was run:
  - Measured_Vs30data.csv now has 176 entries (140 SCPT/SDMT added later)
  - Geonet Metadata Summary_v1.4.csv now has 871 entries (512 Foster/Perrin-derived added later)

These post-v1p0 entries are filtered out to reconstruct the original 671.

Reference output: applied_Vs30_data.csv (671 rows)
"""

from math import sqrt
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from scipy.spatial import cKDTree, distance_matrix

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
CLEANED_UP = Path("/home/arr65/src/jaehwi_fork_vs30/Vs30/vs30/CleanedUp")
DATA_MCGANN = CLEANED_UP / "McGann_cptVs30data.csv"
DATA_WOTHERSPOON = CLEANED_UP / "Measured_Vs30data.csv"
DATA_KAISERETAL = CLEANED_UP / "Geonet Metadata Summary_v1.4.csv"

REFERENCE = Path(
    "/home/arr65/data/vs30/grid_models/jaehwi_v1p0/applied_Vs30_data.csv"
)
OUTPUT = Path(__file__).resolve().parent / "jaehwi_reconstructed_observations.csv"

# ---------------------------------------------------------------------------
# Coordinate transformers (matching Jaehwi's exact CRS setup)
# ---------------------------------------------------------------------------
wgs2nztm = Transformer.from_crs(4326, 2193, always_xy=True)
nzmg2nztm = Transformer.from_crs(27200, 2193, always_xy=True)


# ---------------------------------------------------------------------------
# Downsample McGann -- copied exactly from sites_load_NSHM2022_0.py lines 25-68
# ---------------------------------------------------------------------------
def downsample_mcg(sites_df: pd.DataFrame, res: int = 1000) -> pd.DataFrame:
    """
    Resample McGann points on 1km grid.
    res: grid resolution (m)
    """
    max_dist = sqrt(res**2 * 2) / 2
    x = sites_df["easting"].values
    y = sites_df["northing"].values

    # trying to copy R logic - extents
    # works for this dataset
    xmin = min(x)
    nx = round((max(x) - xmin) / res)
    if ((max(x) - xmin) / res) % 1 < 0.5:
        # this is run
        xmin += res / 2.0
        xmax = (nx - 1) * res + xmin
    else:
        xmax = nx * res + xmin
    ymax = max(y) - res
    ny = round((ymax - min(y)) / res) + 1
    if ((ymax - min(x)) / res) % 1 < 0.5:
        ymax -= res / 2.0
        ymin = ymax - (ny - 2) * res
    else:
        # this is run
        ymax += res / 2.0
        ymin = ymax - (ny - 1) * res

    # coarse grid
    grid_x = np.linspace(xmin, xmax, nx)
    grid_y = np.linspace(ymin, ymax, ny)[::-1]
    grid = np.dstack(np.meshgrid(grid_x, grid_y)).reshape(-1, 2)

    # distances from coarse grid, nearest neighbor
    dist = distance_matrix(grid, np.dstack((x, y))[0])
    nn = np.argmin(dist, axis=1)
    # cut out if no points within search area
    nn = nn[dist[np.arange(nn.size), nn] <= max_dist]

    # remove duplicate points from downsample algorithm
    mcg = sites_df.iloc[nn]
    return mcg[~mcg.duplicated()]


# ---------------------------------------------------------------------------
# Load each source
# ---------------------------------------------------------------------------
def load_mcgann() -> pd.DataFrame:
    """McGann CPT Vs30 -- NZMG coords, downsampled, then transformed to NZTM."""
    mcgann = pd.read_csv(
        DATA_MCGANN,
        usecols=[3, 4, 7],
        names=["easting", "northing", "vs30"],
        skiprows=1,
        engine="c",
        dtype=np.float32,
    )
    mcgann = downsample_mcg(mcgann)
    mcgann["easting"], mcgann["northing"] = nzmg2nztm.transform(
        mcgann["easting"].values, mcgann["northing"].values
    )
    mcgann["uncertainty"] = np.float32(0.2)
    mcgann["source"] = "mcgann"
    return mcgann


def load_wotherspoon() -> pd.DataFrame:
    """Wotherspoon measured Vs30 -- WGS84 lat/lon to NZTM.

    The current CSV has 176 entries, but only the original 36 non-SCPT/SDMT
    entries were present when Jaehwi ran v1p0. The 140 SCPT/SDMT entries
    (seismic CPT and seismic DMT data from the NZGD) were added later.
    """
    # Read with station names so we can filter
    wotherspoon = pd.read_csv(
        DATA_WOTHERSPOON,
        sep=",",
        usecols=[1, 2, 3, 4, 5],
        names=["station", "northing", "easting", "vs30", "q"],
        skiprows=1,
        engine="c",
        dtype={
            "easting": np.float32,
            "northing": np.float32,
            "vs30": np.float32,
            "q": np.float32,
        },
    )
    wotherspoon["station"] = wotherspoon["station"].str.strip()

    # Filter out SCPT/SDMT entries added after v1p0
    n_before = len(wotherspoon)
    is_scpt_sdmt = wotherspoon["station"].str.startswith("SCPT") | wotherspoon[
        "station"
    ].str.startswith("SDMT")
    wotherspoon = wotherspoon[~is_scpt_sdmt].copy()
    print(
        f"    Wotherspoon: {n_before} -> {len(wotherspoon)} "
        f"(filtered {is_scpt_sdmt.sum()} SCPT/SDMT entries added post-v1p0)"
    )

    wotherspoon["easting"], wotherspoon["northing"] = wgs2nztm.transform(
        wotherspoon["easting"].values, wotherspoon["northing"].values
    )
    wotherspoon["uncertainty"] = np.where(
        wotherspoon["q"] == 3, 0.5, wotherspoon["q"] / 10
    )
    wotherspoon["source"] = "wotherspoon"
    return wotherspoon


def load_kaiseretal() -> pd.DataFrame:
    """GeoNet/Kaiser site metadata -- WGS84 lat/lon to NZTM, Q-based uncertainty.

    The current CSV has 871 entries, but only the 359 entries with actual
    measurement-based Vs30 references were present when Jaehwi ran v1p0.
    The 512 entries with Foster et al. (2019) / Perrin et al. (2015) derived
    Vs30 values were added later.
    """
    # Read the full GeoNet file (all columns needed for filtering)
    geonet_full = pd.read_csv(DATA_KAISERETAL, encoding="utf-8-sig")

    # Filter out Foster/Perrin-derived entries added after v1p0
    n_before = len(geonet_full)
    is_foster_perrin = geonet_full["Vs30_Ref"].str.contains(
        "Foster|Perrin", na=False
    )
    geonet_filtered = geonet_full[~is_foster_perrin].copy()
    print(
        f"    Kaiser: {n_before} -> {len(geonet_filtered)} "
        f"(filtered {is_foster_perrin.sum()} Foster/Perrin-derived entries added post-v1p0)"
    )

    # Now load using the same usecols/dtype/converters as sites_load_NSHM2022_0.py
    # but from the filtered dataframe
    kaiseretal = pd.DataFrame(
        {
            "station": geonet_filtered.iloc[:, 0].astype(str).str.strip(),
            "northing": geonet_filtered.iloc[:, 1].astype(np.float32),
            "easting": geonet_filtered.iloc[:, 2].astype(np.float32),
            "vs30": geonet_filtered.iloc[:, 5].astype(np.float32),
            "q": geonet_filtered.iloc[:, 7].apply(
                lambda text: int(str(text).split("Q")[1])
            ),
        }
    )

    kaiseretal["easting"], kaiseretal["northing"] = wgs2nztm.transform(
        kaiseretal["easting"].values, kaiseretal["northing"].values
    )

    kaiseretal["uncertainty"] = np.where(
        kaiseretal["q"] == 3, 0.5, kaiseretal["q"] / 10
    )
    kaiseretal["source"] = "kaiser"
    return kaiseretal


# ---------------------------------------------------------------------------
# Main: load, concatenate, write, validate
# ---------------------------------------------------------------------------
def main() -> None:
    print("=== Loading sources ===")
    mcgann = load_mcgann()
    wotherspoon = load_wotherspoon()
    kaiseretal = load_kaiseretal()

    print("\n=== Source counts ===")
    print(f"  McGann (after downsample): {len(mcgann)}")
    print(f"  Wotherspoon:               {len(wotherspoon)}")
    print(f"  Kaiser/GeoNet:             {len(kaiseretal)}")

    combined = pd.concat([mcgann, wotherspoon, kaiseretal], ignore_index=True)
    print(f"  TOTAL:                     {len(combined)}  (target: 671)")

    # Write output CSV (pipeline columns only)
    combined[["easting", "northing", "vs30", "uncertainty"]].to_csv(
        OUTPUT, index=False
    )
    print(f"\nWrote: {OUTPUT}")

    # ------------------------------------------------------------------
    # Validate against applied_Vs30_data.csv
    # ------------------------------------------------------------------
    ref = pd.read_csv(REFERENCE)
    print(f"\n=== Reference file: {len(ref)} rows ===")

    tree_ref = cKDTree(ref[["easting", "northing"]].values)
    tree_new = cKDTree(combined[["easting", "northing"]].values)

    # For each reconstructed point, find nearest reference point
    dists_new_to_ref, idx_new_to_ref = tree_ref.query(
        combined[["easting", "northing"]].values
    )
    matched_mask = dists_new_to_ref < 1.0  # within 1 metre
    n_matched = matched_mask.sum()

    # For each reference point, find nearest reconstructed point
    dists_ref_to_new, _ = tree_new.query(ref[["easting", "northing"]].values)
    ref_matched_mask = dists_ref_to_new < 1.0
    n_ref_matched = ref_matched_mask.sum()

    print("\n=== Spatial matching (1m tolerance) ===")
    print(f"  Reconstructed matched in reference: {n_matched} / {len(combined)}")
    print(f"  Reference matched in reconstructed: {n_ref_matched} / {len(ref)}")
    print(f"  Extra in reconstructed:             {len(combined) - n_matched}")
    print(f"  Missing from reconstructed:         {len(ref) - n_ref_matched}")

    # Compare vs30 and uncertainty for matched points.
    # NOTE: Two pairs of Kaiser stations (465A/DAVS, 900B/941A) are
    # co-located at float32 precision, so 1-to-1 nearest-neighbor
    # matching may cross-assign them. We use set-based comparison at
    # each spatial cluster to handle this correctly.
    if n_matched > 0:
        matched_new = combined.loc[matched_mask].reset_index(drop=True)
        matched_ref_idx = idx_new_to_ref[matched_mask]
        matched_ref = ref.iloc[matched_ref_idx].reset_index(drop=True)

        # 1-to-1 comparison (simple, but noisy for co-located stations)
        vs30_diff = np.abs(
            matched_new["vs30"].values - matched_ref["vs30"].values
        )
        unc_diff = np.abs(
            matched_new["uncertainty"].values
            - matched_ref["uncertainty"].values
        )
        n_vs30_exact = int((vs30_diff < 1e-4).sum())
        n_vs30_close = int((vs30_diff < 0.01).sum())
        n_unc_exact = int((unc_diff < 1e-6).sum())

        print(f"\n=== Value comparison ({n_matched} matched points) ===")
        print(f"  Vs30 exact (<1e-4):         {n_vs30_exact} / {n_matched}")
        print(f"  Vs30 close (<0.01):         {n_vs30_close} / {n_matched}")
        print(f"  Uncertainty exact (<1e-6):   {n_unc_exact} / {n_matched}")

        # Set-based comparison at co-located clusters.
        # Group reconstructed points into spatial clusters (within 1m),
        # then compare the sorted Vs30/uncertainty multisets against the
        # corresponding reference rows.
        from scipy.cluster.hierarchy import fcluster, linkage

        coords = matched_new[["easting", "northing"]].values
        if len(coords) > 1:
            Z = linkage(coords, method="single", metric="euclidean")
            clusters = fcluster(Z, t=1.0, criterion="distance")
        else:
            clusters = np.array([1])

        vs30_cluster_mismatches = []
        unc_cluster_mismatches = []
        for cid in np.unique(clusters):
            mask_c = clusters == cid
            new_c = matched_new.loc[mask_c]
            ref_c = matched_ref.loc[mask_c]

            new_vs30 = np.sort(new_c["vs30"].values.astype(np.float64))
            ref_vs30 = np.sort(ref_c["vs30"].values.astype(np.float64))
            if not np.allclose(new_vs30, ref_vs30, atol=0.01):
                vs30_cluster_mismatches.append(
                    (new_c, ref_c, new_vs30, ref_vs30)
                )

            new_unc = np.sort(new_c["uncertainty"].values.astype(np.float64))
            ref_unc = np.sort(ref_c["uncertainty"].values.astype(np.float64))
            if not np.allclose(new_unc, ref_unc, atol=1e-4):
                unc_cluster_mismatches.append(
                    (new_c, ref_c, new_unc, ref_unc)
                )

        n_clusters = len(np.unique(clusters))
        print(f"\n=== Set-based comparison ({n_clusters} spatial clusters) ===")
        print(f"  Clusters with Vs30 mismatch:        {len(vs30_cluster_mismatches)}")
        print(f"  Clusters with uncertainty mismatch:  {len(unc_cluster_mismatches)}")

        # Known discrepancy: MHEZ was Q3 when v1p0 was run, now Q4 in CSV.
        # This causes uncertainty 0.5 (Q3) vs 0.4 (Q4) for 1 station.
        # Co-located stations (465A/DAVS, 900B/941A) have identical
        # float32 coords, so the pipeline treats them as one location.
        for new_c, ref_c, new_v, ref_v in vs30_cluster_mismatches:
            stations = ref_c["station"].values if "station" in ref_c else ["?"]
            print(
                f"    Vs30: stations={list(stations)} "
                f"new={new_v} ref={ref_v}"
            )
        for new_c, ref_c, new_u, ref_u in unc_cluster_mismatches:
            stations = ref_c["station"].values if "station" in ref_c else ["?"]
            print(
                f"    Unc:  stations={list(stations)} "
                f"new={new_u} ref={ref_u}"
            )

    # Breakdown matched by source
    if "source" in combined.columns:
        print("\n=== Match rate by source ===")
        for src in ["mcgann", "wotherspoon", "kaiser"]:
            src_mask = combined["source"] == src
            src_matched = (src_mask & matched_mask).sum()
            src_total = src_mask.sum()
            status = "PASS" if src_matched == src_total else "FAIL"
            print(f"  {src:12s}: {src_matched}/{src_total} matched  [{status}]")

    # Overall result
    all_spatial = n_matched == len(combined) == len(ref) == n_ref_matched
    if all_spatial:
        print(f"\n=== RESULT: PASS -- {len(combined)} observations, all 671 spatially matched ===")
    else:
        print("\n=== RESULT: PARTIAL MATCH ===")
        print(f"  Count match: {len(combined) == len(ref)}")
        print(f"  All spatially matched: {n_matched == len(combined)}")


if __name__ == "__main__":
    main()

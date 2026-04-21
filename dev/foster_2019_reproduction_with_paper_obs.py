#!/usr/bin/env python3
"""
Reproduce the foster_2019_approx benchmark using a paper-aligned observation set.

The refactored foster_2019_approx pipeline uses a 412-row observation CSV derived
from modified_foster_2019. The paper itself published 393 observations in
supplement file 2:
  /home/arr65/data/vs30/grid_models/downloaded/foster_2019/15_eeri_35_4_suppl_2_es1_online.txt

The supplement's Lat/Lon are rounded to 6 decimals and Vs30 to 1 decimal.
Feeding those lossy values into the pipeline actually makes things worse,
so instead we construct a "paper-aligned" subset of the current 412-row
CSV by matching each supplement row to the nearest current-CSV row and
keeping only those current rows. This preserves full coordinate/Vs30
precision while removing the ~19 current rows the paper doesn't have
(presumably obs the paper deduped via the "within 2 m" rule).

Steps:
  1. Parse 393 supplement observations.
  2. Match each to the nearest row in the current 412-row CSV (<10 m).
  3. Write the matched subset as a temporary obs CSV (full precision).
  4. Run points_pipeline for two cohorts (prior-dominated, near-obs)
     using this paper-aligned CSV.
  5. Compare to the benchmark raster at the same pixel coordinates.

If cohort-B agreement improves vs. the 412-row CSV, the refactored
pipeline reproduces the paper faithfully; the remaining ~1.6 % divergence
is just the obs-set difference. If it doesn't improve, the divergence is
intrinsic to the MVN numerical implementation.
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from pyproj import Transformer
from scipy.spatial import cKDTree

from vs30 import constants, pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))
from conftest import load_fixed_model_config  # noqa: E402

SUPPLEMENT_PATH = Path(
    "/home/arr65/data/vs30/grid_models/downloaded/foster_2019/15_eeri_35_4_suppl_2_es1_online.txt"
)
BENCHMARK_PATH = Path(
    "/home/arr65/src/Vs30/tests/benchmarks/foster_2019_approx.tif"
)

N_PER_COHORT = 30
NEAR_OBS_M = 500.0
SEED = 42

NZTM_TO_WGS = Transformer.from_crs(2193, 4326, always_xy=True)
WGS_TO_NZTM = Transformer.from_crs(4326, 2193, always_xy=True)


def build_paper_aligned_obs_csv(
    current_csv: Path, output_path: Path, max_match_dist_m: float = 10.0
) -> pd.DataFrame:
    """Build a 393-row obs CSV faithful to the paper's published observation set.

    The paper's supplement has 393 rows, but 11 of them are "colocated pairs"
    that the current 412-row CSV only represents once:
      - 4 pairs are byte-identical duplicates in the supplement (probably
        accidental) — e.g. the same McGann observation written twice.
      - 7 pairs are the same physical station measured by two independent
        sources (Kaiser et al. 2017 AND Wotherspoon et al. 2013), each
        reporting its own Vs30 — the current CSV keeps only the Kaiser row.

    For each supplement row we find its nearest current-CSV row. The first
    supplement row to claim a current row gets that row's full-precision
    data (easting/northing/vs30/uncertainty/etc.). Any subsequent supplement
    row landing on the same current row becomes an "extra" row whose
    easting/northing are re-projected from the supplement's lat/lon and
    whose Vs30/uncertainty come from the supplement directly. This gives
    exactly 393 rows: 382 at full precision + 11 extras reconstructed from
    the supplement.
    """
    supp = pd.read_csv(
        SUPPLEMENT_PATH,
        sep=r"\s+",
        skiprows=1,
        header=None,
        usecols=[0, 1, 2, 3],
        names=["lat", "lon", "vs30", "sigma_meas"],
    )
    supp_e, supp_n = WGS_TO_NZTM.transform(supp["lon"].values, supp["lat"].values)
    supp_xy = np.column_stack([supp_e, supp_n])

    current_df = pd.read_csv(current_csv, comment="#")
    current_xy = current_df[["easting", "northing"]].to_numpy()

    tree = cKDTree(current_xy)
    dists, idx = tree.query(supp_xy, k=1)

    if np.any(dists >= max_match_dist_m):
        bad = int((dists >= max_match_dist_m).sum())
        raise ValueError(
            f"{bad} supplement obs have no current-CSV match within "
            f"{max_match_dist_m} m (max dist: {dists.max():.2f} m)"
        )

    claimed: set[int] = set()
    out_rows: list[dict] = []
    n_from_current = 0
    n_from_supplement = 0
    for i in range(len(supp)):
        cur_row_idx = int(idx[i])
        if cur_row_idx not in claimed:
            claimed.add(cur_row_idx)
            out_rows.append(current_df.iloc[cur_row_idx].to_dict())
            n_from_current += 1
        else:
            out_rows.append(
                {
                    "easting": float(supp_e[i]),
                    "northing": float(supp_n[i]),
                    "vs30": float(supp["vs30"].iloc[i]),
                    "uncertainty": float(supp["sigma_meas"].iloc[i]),
                    "source": "supplement_extra",
                    "station": np.nan,
                    "q": np.nan,
                }
            )
            n_from_supplement += 1

    aligned_df = pd.DataFrame(out_rows, columns=current_df.columns)

    header = (
        "# Foster (2019) paper-aligned observations\n"
        f"# Supplement: {SUPPLEMENT_PATH.name} ({len(supp)} rows)\n"
        f"# Matched against: {current_csv.name} ({len(current_df)} rows)\n"
        f"# Total output rows: {len(aligned_df)}\n"
        f"#   - {n_from_current} from current CSV (full precision)\n"
        f"#   - {n_from_supplement} reconstructed from supplement (lossy)\n"
    )
    with open(output_path, "w") as f:
        f.write(header)
        aligned_df.to_csv(f, index=False)
    return aligned_df


def sample_pixel_centers(rng: np.random.Generator, n_candidates: int) -> np.ndarray:
    """Return n_candidates random valid pixel centres (NZTM) from the benchmark raster."""
    with rasterio.open(BENCHMARK_PATH) as src:
        band = src.read(1)
        nodata = src.nodata
        valid = ~np.isnan(band)
        if nodata is not None:
            valid &= band != nodata
        rows, cols = np.where(valid)
        idx = rng.choice(len(rows), size=min(n_candidates, len(rows)), replace=False)
        xs, ys = rasterio.transform.xy(src.transform, rows[idx], cols[idx], offset="center")
    return np.column_stack([xs, ys])


def split_cohorts(pixel_xy: np.ndarray, obs_xy: np.ndarray):
    tree = cKDTree(obs_xy)
    dists, _ = tree.query(pixel_xy, k=1)
    prior_mask = dists > constants.MAX_DIST_M
    near_mask = dists < NEAR_OBS_M
    return pixel_xy[prior_mask], pixel_xy[near_mask]


def run_points_pipeline(cfg: dict, pixel_xy: np.ndarray, obs_csv: Path) -> pd.DataFrame:
    lons, lats = NZTM_TO_WGS.transform(pixel_xy[:, 0], pixel_xy[:, 1])
    return pipeline.points_pipeline(
        longitudes=np.asarray(lons),
        latitudes=np.asarray(lats),
        geology_categorical_csv=cfg["geology_categorical_csv"],
        terrain_categorical_csv=cfg["terrain_categorical_csv"],
        clustered_observations_csv=cfg.get("clustered_observations_csv"),
        independent_observations_csv=obs_csv,
        combination_method=constants.CombinationMethod(cfg["combination_method"]),
        combine_ratio=cfg.get("combine_ratio"),
        noisy=cfg["noisy"],
        do_bayesian_update=cfg["do_bayesian_update"],
        n_proc=1,
        geology_corr_fn=cfg.get("geology_corr_fn"),
        terrain_corr_fn=cfg.get("terrain_corr_fn"),
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        apply_coastal_distance_mod=cfg["apply_coastal_distance_mod"],
        fill_gaps=cfg.get("fill_gaps", False),
    )


def sample_benchmark(pixel_xy: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    with rasterio.open(BENCHMARK_PATH) as src:
        samples = np.array(list(src.sample(pixel_xy, indexes=[1, 2])))
    return samples[:, 0], samples[:, 1]


def summarise(label: str, ref_vs30, ref_stdv, bench_vs30, bench_stdv) -> None:
    vs30_diff = ref_vs30 - bench_vs30
    stdv_diff = ref_stdv - bench_stdv
    vs30_rel = np.abs(vs30_diff) / np.abs(bench_vs30)
    stdv_rel = np.abs(stdv_diff) / np.abs(bench_stdv)
    print(f"\n=== {label} ===")
    print(f"N = {len(bench_vs30)}")
    print(
        f"Vs30 abs diff: min={np.min(np.abs(vs30_diff)):.4g}, "
        f"median={np.median(np.abs(vs30_diff)):.4g}, "
        f"mean={np.mean(np.abs(vs30_diff)):.4g}, "
        f"max={np.max(np.abs(vs30_diff)):.4g}"
    )
    print(
        f"Vs30 rel diff: median={np.median(vs30_rel):.4g}, "
        f"max={np.max(vs30_rel):.4g}"
    )
    print(f"  % within 0.1%:  {100 * np.mean(vs30_rel < 1e-3):.1f}")
    print(f"  % within 1%:    {100 * np.mean(vs30_rel < 1e-2):.1f}")
    print(
        f"Stdv rel diff: median={np.median(stdv_rel):.4g}, "
        f"max={np.max(stdv_rel):.4g}"
    )
    print(f"  % within 1%:    {100 * np.mean(stdv_rel < 1e-2):.1f}")


def main() -> None:
    cfg = load_fixed_model_config(constants.FixedModelVersion.FOSTER_2019_APPROX)

    with tempfile.TemporaryDirectory() as tmpdir:
        paper_obs_csv = Path(tmpdir) / "foster_2019_paper_aligned_obs.csv"
        paper_obs_df = build_paper_aligned_obs_csv(
            cfg["independent_observations_csv"], paper_obs_csv
        )
        print(f"Built paper-aligned obs CSV with {len(paper_obs_df)} rows: {paper_obs_csv}")

        paper_obs_xy = paper_obs_df[["easting", "northing"]].to_numpy()

        rng = np.random.default_rng(SEED)
        candidate_xy = sample_pixel_centers(rng, n_candidates=200_000)
        prior_xy_all, near_xy_all = split_cohorts(candidate_xy, paper_obs_xy)
        print(f"Of {len(candidate_xy)} candidate pixels (relative to paper obs):")
        print(f"  prior-dominated (>{constants.MAX_DIST_M} m from obs): {len(prior_xy_all)}")
        print(f"  near-obs        (<{NEAR_OBS_M} m from obs):         {len(near_xy_all)}")

        rng.shuffle(prior_xy_all)
        rng.shuffle(near_xy_all)
        prior_xy = prior_xy_all[:N_PER_COHORT]
        near_xy = near_xy_all[:N_PER_COHORT]
        combined_xy = np.concatenate([prior_xy, near_xy])

        print(f"\nRunning points_pipeline for {len(combined_xy)} points with paper obs…")
        result = run_points_pipeline(cfg, combined_xy, paper_obs_csv)

    bench_vs30, bench_stdv = sample_benchmark(combined_xy)
    ref_vs30 = result[constants.ObservationColumn.VS30].to_numpy()
    ref_stdv = result[constants.COL_COMBINED_STDV].to_numpy()

    prior_slice = slice(0, N_PER_COHORT)
    near_slice = slice(N_PER_COHORT, 2 * N_PER_COHORT)
    summarise(
        "COHORT A: prior-dominated (paper obs)",
        ref_vs30[prior_slice], ref_stdv[prior_slice],
        bench_vs30[prior_slice], bench_stdv[prior_slice],
    )
    summarise(
        "COHORT B: observation-influenced (paper obs)",
        ref_vs30[near_slice], ref_stdv[near_slice],
        bench_vs30[near_slice], bench_stdv[near_slice],
    )


if __name__ == "__main__":
    main()

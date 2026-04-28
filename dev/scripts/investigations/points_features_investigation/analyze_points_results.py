"""Analyse points-mode performance sweep results.

Reads results_points.csv (clean, nproc=1 only) and, if present,
results_points_partial_with_buggy_nproc8.csv (partial sweep that captured a
per-chunk obs-prep redundancy bug in vs30/parallel.py::run_parallel_locations
causing nproc=8 cells to run ~150x slower than nproc=1 at large N_query).
Both CSVs are combined, deduplicating on (N_query, N_obs, nproc, rep) in
favour of the clean CSV.

Produces:
- results_points_medians.csv  -- per-cell medians over (N_query, N_obs, nproc)
- results_points_speedup.csv  -- nproc=8/nproc=1 speedup pivot
- figures/absolute_time_nproc1_points.png  -- headline wall-time heatmap
- figures/multiproc_speedup_points_buggy.png  -- bug-manifestation speedup heatmap

Run::

    python -m dev.scripts.investigations.points_features_investigation.analyze_points_results
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — runs headless

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).parent
RESULTS_CSV = HERE / "results_points.csv"
PARTIAL_CSV = HERE / "results_points_partial_with_buggy_nproc8.csv"
FIGURES_DIR = HERE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)

# Keys that uniquely identify a single timing observation
_DEDUP_KEYS = ["N_query", "N_obs", "nproc", "rep"]


def load_results() -> pd.DataFrame:
    """Load and combine both CSVs.

    Strategy:
    - Read the clean CSV (all nproc=1, full N_query range).  Tag source.
    - If the partial CSV exists, append it (nproc=1 + nproc=8, small N_query).
      Tag source.
    - Where a row appears in both (small-N_query nproc=1 cells), keep the
      clean-CSV version by sorting so clean rows sort last then calling
      drop_duplicates(keep="last").
    - Drop rows with NaN t_total_s.
    """
    df_clean = pd.read_csv(RESULTS_CSV)
    df_clean["source"] = "trimmed_clean"

    frames = []
    if PARTIAL_CSV.exists():
        df_partial = pd.read_csv(PARTIAL_CSV)
        df_partial["source"] = "partial_with_buggy_nproc8"
        frames.append(df_partial)

    frames.append(df_clean)

    df = pd.concat(frames, ignore_index=True)
    df = df.dropna(subset=["t_total_s"])

    # When the same (N_query, N_obs, nproc, rep) appears in both CSVs, keep
    # the clean-CSV row (which sorts last because we appended it last).
    df = df.drop_duplicates(subset=_DEDUP_KEYS, keep="last")
    df = df.sort_values(_DEDUP_KEYS).reset_index(drop=True)
    return df


def cell_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Median t_total_s per (N_query, N_obs, nproc)."""
    return df.groupby(["N_query", "N_obs", "nproc"])["t_total_s"].median().reset_index()


def speedup_table(med: pd.DataFrame) -> pd.DataFrame:
    """Pivot total time, then divide nproc=1 by nproc=8 to get speedup.

    A speedup > 1 means nproc=8 wins. < 1 means nproc=1 wins.
    """
    piv = med.pivot_table(
        index=["N_query", "N_obs"],
        columns="nproc",
        values="t_total_s",
        aggfunc="median",
    )
    if {1, 8}.issubset(piv.columns):
        piv["speedup_nproc8_vs_1"] = piv[1] / piv[8]
    return piv


def _fmt_seconds(s: float) -> str:
    """Format seconds compactly: 1.6s / 37s / 3.7m."""
    if s < 60:
        if s < 10:
            return f"{s:.1f}s"
        return f"{s:.0f}s"
    return f"{s / 60:.1f}m"


def write_absolute_time_heatmap(med: pd.DataFrame, out_path: Path) -> None:
    """Log-coloured heatmap of nproc=1 wall time over N_query x N_obs."""
    nproc1 = med[med["nproc"] == 1].copy()
    if nproc1.empty:
        return

    n_query_vals = sorted(nproc1["N_query"].unique())
    n_obs_vals = sorted(nproc1["N_obs"].unique())
    matrix = np.full((len(n_query_vals), len(n_obs_vals)), np.nan)

    for _, row in nproc1.iterrows():
        i = n_query_vals.index(row["N_query"])
        j = n_obs_vals.index(row["N_obs"])
        matrix[i, j] = row["t_total_s"]

    log_matrix = np.log10(matrix)

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(log_matrix, cmap="YlOrRd", aspect="auto")
    ax.set_xticks(range(len(n_obs_vals)))
    ax.set_xticklabels([f"{v:,}" for v in n_obs_vals])
    ax.set_yticks(range(len(n_query_vals)))
    ax.set_yticklabels([f"{v:,}" for v in n_query_vals])
    ax.set_xlabel("N_obs")
    ax.set_ylabel("N_query")
    ax.set_title("points_pipeline wall time (nproc=1)")

    finite = log_matrix[np.isfinite(log_matrix)]
    log_mid = float(np.nanmedian(finite)) if finite.size else 0.0

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=9)
                continue
            lv = log_matrix[i, j]
            color = "white" if lv > log_mid else "black"
            ax.text(
                j,
                i,
                _fmt_seconds(v),
                ha="center",
                va="center",
                color=color,
                fontsize=9,
                fontweight="bold",
            )

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log10(wall time [s])")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def write_speedup_heatmap(piv: pd.DataFrame, out_path: Path) -> None:
    """Log2-coloured heatmap of nproc=8/nproc=1 speedup over N_query x N_obs.

    The data comes from the partial buggy sweep, so the speedup reflects the
    per-chunk obs-prep redundancy bug rather than the inherent multiproc trade-
    off — hence the 'buggy' suffix in the output filename.
    """
    if "speedup_nproc8_vs_1" not in piv.columns:
        return
    speedup = piv["speedup_nproc8_vs_1"].dropna()
    n_query_vals = sorted(speedup.index.get_level_values("N_query").unique())
    n_obs_vals = sorted(speedup.index.get_level_values("N_obs").unique())
    matrix = np.full((len(n_query_vals), len(n_obs_vals)), np.nan)
    for (n_query, n_obs), val in speedup.items():
        i = n_query_vals.index(n_query)
        j = n_obs_vals.index(n_obs)
        matrix[i, j] = val

    fig, ax = plt.subplots(figsize=(7, 6))
    log_matrix = np.log2(matrix)
    finite = log_matrix[np.isfinite(log_matrix)]
    cap = max(2.0, float(np.nanmax(np.abs(finite))) if finite.size else 2.0)
    im = ax.imshow(log_matrix, cmap="RdBu", vmin=-cap, vmax=cap, aspect="auto")
    ax.set_xticks(range(len(n_obs_vals)))
    ax.set_xticklabels([f"{v:,}" for v in n_obs_vals])
    ax.set_yticks(range(len(n_query_vals)))
    ax.set_yticklabels([f"{v:,}" for v in n_query_vals])
    ax.set_xlabel("N_obs")
    ax.set_ylabel("N_query")
    ax.set_title(
        "Multiproc speedup (nproc=8/nproc=1) — buggy implementation\n"
        "see findings: per-chunk obs-prep redundancy in run_parallel_locations"
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            v = matrix[i, j]
            if np.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", fontsize=8)
                continue
            ax.text(
                j,
                i,
                f"{v:.2f}",
                ha="center",
                va="center",
                color="white" if abs(log_matrix[i, j]) > cap / 2 else "black",
                fontsize=8,
            )
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log2(speedup)  [+ = nproc=8 wins]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    if not RESULTS_CSV.exists():
        raise SystemExit(
            f"No results_points.csv at {RESULTS_CSV}; run run_points_sweep first."
        )
    df = load_results()

    nproc1_count = (df["nproc"] == 1).sum()
    nproc8_count = (df["nproc"] == 8).sum()
    partial_loaded = PARTIAL_CSV.exists()
    print(
        f"Loaded {len(df)} rows total  "
        f"(nproc=1: {nproc1_count}, nproc=8: {nproc8_count}, "
        f"partial CSV {'found' if partial_loaded else 'not found'})"
    )

    med = cell_medians(df)
    medians_csv = HERE / "results_points_medians.csv"
    med.to_csv(medians_csv, index=False)

    piv = speedup_table(med)
    speedup_csv = HERE / "results_points_speedup.csv"
    piv.to_csv(speedup_csv)

    abs_out = FIGURES_DIR / "absolute_time_nproc1_points.png"
    write_absolute_time_heatmap(med, abs_out)

    speedup_out = FIGURES_DIR / "multiproc_speedup_points_buggy.png"
    write_speedup_heatmap(piv, speedup_out)

    print("\nWrote:")
    print(f"  {medians_csv}")
    print(f"  {speedup_csv}")
    print(f"  {abs_out}")
    print(f"  {speedup_out}")
    print()
    print("Speedup table (nproc=8 / nproc=1)  [NaN = nproc=8 not measured]:")
    print(piv.to_string())


if __name__ == "__main__":
    main()

"""Analyse the perf-features-investigation results.

Reads results_isolated.csv (and results_full_pipeline.csv if present),
computes per-cell median timings, multiproc and ffap speedups, and writes
heatmaps to figures/.

Run::

    python -m dev.scripts.investigations.perf_features_investigation.analyze_results
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
ISOLATED_CSV = HERE / "results_isolated.csv"
FULL_CSV = HERE / "results_full_pipeline.csv"
FIGURES_DIR = HERE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_isolated() -> pd.DataFrame:
    df = pd.read_csv(ISOLATED_CSV)
    df = df.dropna(subset=["t_total_s"])
    return df


def cell_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Median t_total_s, t_bbox_s, t_spatial_s per (N_obs, N_grid_target, nproc, ffap)."""
    return (
        df.groupby(["N_obs", "N_grid_target", "nproc", "ffap"])[
            ["t_total_s", "t_bbox_s", "t_spatial_s", "N_grid_actual", "N_affected"]
        ]
        .median()
        .reset_index()
    )


def speedup_pivot(
    med: pd.DataFrame, fixed_col: str, fixed_val, contrast_col: str
) -> pd.DataFrame:
    """Pivot total time over (N_obs, N_grid_target) for two values of contrast_col.

    Returns a DataFrame with one column per contrast value.
    """
    sub = med[med[fixed_col] == fixed_val]
    piv = sub.pivot_table(
        index=["N_obs", "N_grid_target"],
        columns=contrast_col,
        values="t_total_s",
        aggfunc="median",
    )
    return piv


def write_heatmap(piv: pd.Series, label: str, out_path: Path) -> None:
    """Write a log-scale heatmap of piv over (N_obs x N_grid_target).

    piv is a Series-shaped pivot whose values are speedups (>1 = win).
    """
    n_obs_vals = sorted(piv.index.get_level_values("N_obs").unique())
    n_grid_vals = sorted(piv.index.get_level_values("N_grid_target").unique())
    matrix = np.full((len(n_obs_vals), len(n_grid_vals)), np.nan)
    for (n_obs, n_grid), val in piv.items():
        i = n_obs_vals.index(n_obs)
        j = n_grid_vals.index(n_grid)
        matrix[i, j] = val
    fig, ax = plt.subplots(figsize=(7, 5))
    log_matrix = np.log2(matrix)
    finite = log_matrix[np.isfinite(log_matrix)]
    cap = max(2.0, float(np.nanmax(np.abs(finite))) if finite.size else 2.0)
    im = ax.imshow(log_matrix, cmap="RdBu", vmin=-cap, vmax=cap, aspect="auto")
    ax.set_xticks(range(len(n_grid_vals)))
    ax.set_xticklabels([f"{v:,}" for v in n_grid_vals])
    ax.set_yticks(range(len(n_obs_vals)))
    ax.set_yticklabels([f"{v:,}" for v in n_obs_vals])
    ax.set_xlabel("N_grid_target")
    ax.set_ylabel("N_obs")
    ax.set_title(label)
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
    cbar.set_label("log2(speedup)  [+ = numerator wins]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    if not ISOLATED_CSV.exists():
        raise SystemExit(
            f"No results_isolated.csv found at {ISOLATED_CSV}; "
            "run run_isolated_sweep first."
        )
    df = load_isolated()
    med = cell_medians(df)
    med.to_csv(HERE / "results_isolated_medians.csv", index=False)

    # ---- Multiproc speedup (nproc=1 / nproc=8) ----------------------------
    for ffap_fixed in (True, False):
        piv = speedup_pivot(med, "ffap", ffap_fixed, "nproc")
        if {1, 8}.issubset(piv.columns):
            piv["speedup"] = piv[1] / piv[8]
            label = (
                f"Multiproc speedup (nproc=8 vs 1), "
                f"ffap={'ON' if ffap_fixed else 'OFF'}"
            )
            out = (
                FIGURES_DIR
                / f"multiproc_speedup_ffap_{'on' if ffap_fixed else 'off'}.png"
            )
            write_heatmap(piv["speedup"], label, out)

    # ---- ffap speedup (ffap=OFF / ffap=ON) --------------------------------
    for nproc_fixed in (1, 8):
        piv = speedup_pivot(med, "nproc", nproc_fixed, "ffap")
        if {True, False}.issubset(piv.columns):
            piv["speedup"] = piv[False] / piv[True]
            label = f"ffap speedup (ON vs OFF), nproc={nproc_fixed}"
            out = FIGURES_DIR / f"ffap_speedup_nproc{nproc_fixed}.png"
            write_heatmap(piv["speedup"], label, out)

    # ---- Best-strategy table ----------------------------------------------
    best = med.loc[
        med.groupby(["N_obs", "N_grid_target"])["t_total_s"].idxmin()
    ].reset_index(drop=True)
    best.to_csv(HERE / "results_isolated_best_strategy.csv", index=False)

    print("Wrote:")
    print(f"  {HERE / 'results_isolated_medians.csv'}")
    print(f"  {HERE / 'results_isolated_best_strategy.csv'}")
    print(f"  {FIGURES_DIR}/*.png")


if __name__ == "__main__":
    main()

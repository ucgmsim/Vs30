"""Analyse the post-fix points-pipeline performance sweep.

Reads results_points_post_fix.csv (252 cells: 7 N_query x 3 N_obs x 4 nproc
x 3 reps) and produces:

- results_points_post_fix_medians.csv  -- per-cell medians per nproc
- results_points_post_fix_speedup.csv  -- pivot with t1/t2/t4/t8 columns plus
    speedup_2_vs_1, speedup_4_vs_1, speedup_8_vs_1
- figures/speedup_nproc8_vs_1_post_fix.png  -- endpoint speedup heatmap
- figures/best_nproc_per_cell.png  -- categorical heatmap labelling each cell
    with the nproc value that minimises wall time

Run::

    python -m dev.scripts.investigations.points_features_investigation.analyze_points_post_fix_results
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # non-interactive backend — runs headless

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

HERE = Path(__file__).parent
RESULTS_CSV = HERE / "results_points_post_fix.csv"
FIGURES_DIR = HERE / "figures"

NPROC_VALUES = [1, 2, 4, 8]
NPROC_COLORS = {
    1: "#1f77b4",  # blue
    2: "#2ca02c",  # green
    4: "#ff7f0e",  # orange
    8: "#d62728",  # red
}


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    return df.dropna(subset=["t_total_s"])


def cell_medians(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["N_query", "N_obs", "nproc"])["t_total_s"].median().reset_index()


def speedup_table(med: pd.DataFrame) -> pd.DataFrame:
    """Pivot per-cell medians and compute speedup vs nproc=1 for K in {2, 4, 8}."""
    piv = med.pivot_table(
        index=["N_query", "N_obs"],
        columns="nproc",
        values="t_total_s",
        aggfunc="median",
    )
    if 1 in piv.columns:
        for k in (2, 4, 8):
            if k in piv.columns:
                piv[f"speedup_{k}_vs_1"] = piv[1] / piv[k]
    return piv


def write_endpoint_speedup_heatmap(piv: pd.DataFrame, out_path: Path) -> None:
    """Log2-coloured heatmap of the nproc=8 vs nproc=1 speedup."""
    if "speedup_8_vs_1" not in piv.columns:
        return
    speedup = piv["speedup_8_vs_1"].dropna()
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
    ax.set_title("Multiproc speedup (nproc=8 / nproc=1) — post-fix")
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


def write_best_nproc_heatmap(piv: pd.DataFrame, out_path: Path) -> None:
    """Categorical heatmap labelling each cell with the nproc value that wins."""
    nproc_cols = [n for n in NPROC_VALUES if n in piv.columns]
    if not nproc_cols:
        return
    times = piv[nproc_cols]

    n_query_vals = sorted(piv.index.get_level_values("N_query").unique())
    n_obs_vals = sorted(piv.index.get_level_values("N_obs").unique())
    best_nproc = np.full((len(n_query_vals), len(n_obs_vals)), -1, dtype=int)
    speedups = np.full((len(n_query_vals), len(n_obs_vals)), np.nan)

    for (n_query, n_obs), row in times.iterrows():
        i = n_query_vals.index(n_query)
        j = n_obs_vals.index(n_obs)
        valid = row.dropna()
        if valid.empty:
            continue
        # Ties resolve to the lowest nproc (idxmin returns first-positional
        # match; valid is constructed in NPROC_VALUES order, ascending). That
        # is the operationally preferred tie-break: less CPU for equal time.
        winning_nproc = int(valid.idxmin())
        best_nproc[i, j] = winning_nproc
        if 1 in valid.index:
            speedups[i, j] = valid.loc[1] / valid.loc[winning_nproc]

    fig, ax = plt.subplots(figsize=(8, 6))
    color_matrix = np.zeros((len(n_query_vals), len(n_obs_vals), 3))
    for i in range(len(n_query_vals)):
        for j in range(len(n_obs_vals)):
            n = int(best_nproc[i, j])
            if n in NPROC_COLORS:
                color_matrix[i, j] = matplotlib.colors.to_rgb(NPROC_COLORS[n])
            else:
                color_matrix[i, j] = (0.9, 0.9, 0.9)
    ax.imshow(color_matrix, aspect="auto")
    ax.set_xticks(range(len(n_obs_vals)))
    ax.set_xticklabels([f"{v:,}" for v in n_obs_vals])
    ax.set_yticks(range(len(n_query_vals)))
    ax.set_yticklabels([f"{v:,}" for v in n_query_vals])
    ax.set_xlabel("N_obs")
    ax.set_ylabel("N_query")
    ax.set_title("Best nproc per cell (cell colour = winning nproc)")
    for i in range(len(n_query_vals)):
        for j in range(len(n_obs_vals)):
            n = int(best_nproc[i, j])
            sp = speedups[i, j]
            if n < 0:
                ax.text(j, i, "—", ha="center", va="center", fontsize=10)
                continue
            label = f"nproc={n}\n{sp:.2f}x" if not np.isnan(sp) else f"nproc={n}"
            # nproc=1 (blue) and nproc=8 (red) are dark; nproc=2 (green) and
            # nproc=4 (orange) are mid-luminance — black text reads better on
            # those.
            text_color = "white" if n in (1, 8) else "black"
            ax.text(j, i, label, ha="center", va="center", fontsize=9, color=text_color)

    legend_handles = [
        plt.matplotlib.patches.Patch(color=NPROC_COLORS[n], label=f"nproc={n}")
        for n in nproc_cols
    ]
    ax.legend(handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    if not RESULTS_CSV.exists():
        raise SystemExit(
            f"No results_points_post_fix.csv at {RESULTS_CSV}; run run_points_sweep first."
        )
    FIGURES_DIR.mkdir(exist_ok=True)

    df = load_results()
    print(f"Loaded {len(df)} rows ({df['nproc'].nunique()} nproc values)")

    med = cell_medians(df)
    medians_csv = HERE / "results_points_post_fix_medians.csv"
    med.to_csv(medians_csv, index=False)

    piv = speedup_table(med)
    speedup_csv = HERE / "results_points_post_fix_speedup.csv"
    piv.to_csv(speedup_csv)

    endpoint_out = FIGURES_DIR / "speedup_nproc8_vs_1_post_fix.png"
    write_endpoint_speedup_heatmap(piv, endpoint_out)

    best_out = FIGURES_DIR / "best_nproc_per_cell.png"
    write_best_nproc_heatmap(piv, best_out)

    print("\nWrote:")
    print(f"  {medians_csv}")
    print(f"  {speedup_csv}")
    print(f"  {endpoint_out}")
    print(f"  {best_out}")
    print()
    print("Speedup table:")
    print(piv.to_string())


if __name__ == "__main__":
    main()

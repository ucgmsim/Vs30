"""Analyse the post-fix points-pipeline performance sweep.

Reads results_points_post_fix.csv (249 rows: 4 nproc × 21 cells × 3 reps,
minus the 3 missing nproc=8 reps) and produces:

- results_points_post_fix_medians.csv  -- per-cell medians per nproc
- results_points_post_fix_speedup.csv  -- pivot with t1/t2/t4/t8 columns plus
    speedup_2_vs_1, speedup_4_vs_1, speedup_8_vs_1
- figures/speedup_nproc8_vs_1_post_fix.png  -- endpoint speedup heatmap
- figures/best_nproc_per_cell.png  -- categorical heatmap labelling each cell
    with the nproc value that minimises wall time

Also reads results_balanced_blas_supplement.csv (63 rows: 3 nproc × 21 cells
× 1 rep) and combines it with the post-fix sweep to produce:

- results_points_post_fix_combined_medians.csv  -- long-format medians with a
    ``config`` column disambiguating the 7 distinct configurations across both
    sweeps (21 cells × 7 configs = 147 rows)
- results_points_post_fix_combined_wide.csv  -- wide-format with one row per
    (N_query, N_obs) and one column per config (21 rows)
- figures/best_config_per_cell.png  -- categorical heatmap labelling each cell
    with the configuration that wins across all 7 configs
- figures/best_multiproc_config_per_cell.png  -- same but excluding nproc=1,
    surfacing the cell-size-dependent multiproc pattern

Config labels used in the combined outputs:

- ``nproc=1``                -- sequential path, default 8-thread BLAS
- ``nproc=2 (handicapped)`` -- post_fix sweep, 1 BLAS thread/worker (2 active cores)
- ``nproc=2 (balanced)``    -- supplement sweep, 4 BLAS threads/worker (8 cores)
- ``nproc=4 (handicapped)`` -- post_fix sweep, 1 BLAS thread/worker (4 active cores)
- ``nproc=4 (balanced)``    -- supplement sweep, 2 BLAS threads/worker (8 cores)
- ``nproc=6 (override)``    -- supplement sweep, 2 BLAS threads/worker (12 on 8 cores)
- ``nproc=8``               -- post_fix sweep, 1 BLAS thread/worker = balanced

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
SUPPLEMENT_CSV = HERE / "results_balanced_blas_supplement.csv"
FIGURES_DIR = HERE / "figures"

NPROC_VALUES = [1, 2, 4, 8]
NPROC_COLORS = {
    1: "#1f77b4",  # blue
    2: "#2ca02c",  # green
    4: "#ff7f0e",  # orange
    8: "#d62728",  # red
}

# Colours for the seven distinct configurations used in the combined analysis.
CONFIG_COLORS = {
    "nproc=1": "#1f77b4",  # blue
    "nproc=2 (handicapped)": "#cccccc",  # light gray
    "nproc=2 (balanced)": "#2ca02c",  # green
    "nproc=4 (handicapped)": "#ffbb78",  # light orange
    "nproc=4 (balanced)": "#ff7f0e",  # orange
    "nproc=6 (override)": "#9467bd",  # purple
    "nproc=8": "#d62728",  # red
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


def load_combined_results() -> pd.DataFrame:
    """Read both CSVs and return a combined DataFrame with a ``config`` column.

    Config labels:
    - post_fix CSV: ``nproc=1`` and ``nproc=8`` have no qualifier; nproc=2 and
      nproc=4 are labelled ``(handicapped)`` because they only used 1 BLAS
      thread per worker, leaving cores idle.
    - supplement CSV: nproc=2 and nproc=4 are labelled ``(balanced)`` (full
      CPU); nproc=6 is ``(override)`` (mild oversubscription).
    """

    def _derive_post_fix_config(nproc: int) -> str:
        if nproc in (1, 8):
            return f"nproc={nproc}"
        return f"nproc={nproc} (handicapped)"

    def _derive_supplement_config(nproc: int) -> str:
        if nproc == 6:
            return "nproc=6 (override)"
        return f"nproc={nproc} (balanced)"

    parts = []

    df_pf = pd.read_csv(RESULTS_CSV).dropna(subset=["t_total_s"])
    df_pf = df_pf.copy()
    df_pf["config"] = df_pf["nproc"].apply(_derive_post_fix_config)
    parts.append(df_pf)

    if SUPPLEMENT_CSV.exists():
        df_sup = pd.read_csv(SUPPLEMENT_CSV).dropna(subset=["t_total_s"])
        df_sup = df_sup.copy()
        df_sup["config"] = df_sup["nproc"].apply(_derive_supplement_config)
        parts.append(df_sup)

    return pd.concat(parts, ignore_index=True)


def combined_cell_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Return long-format medians grouped by (N_query, N_obs, config)."""
    return (
        df.groupby(["N_query", "N_obs", "config"])["t_total_s"].median().reset_index()
    )


def combined_wide_table(med_long: pd.DataFrame) -> pd.DataFrame:
    """Pivot long-format combined medians into wide format (one column per config)."""
    return med_long.pivot_table(
        index=["N_query", "N_obs"],
        columns="config",
        values="t_total_s",
        aggfunc="median",
    ).reset_index()


def write_best_config_heatmap(
    med_long: pd.DataFrame,
    out_path: Path,
    exclude: list[str] | None = None,
) -> None:
    """Categorical heatmap labelling each cell with the winning configuration.

    Parameters
    ----------
    med_long:
        Long-format DataFrame from ``combined_cell_medians``.
    out_path:
        Destination PNG path.
    exclude:
        Optional list of config labels to exclude before finding the winner.
        Pass ``["nproc=1"]`` to produce the multiproc-only view.
    """
    data = med_long.copy()
    if exclude:
        data = data[~data["config"].isin(exclude)]

    n_query_vals = sorted(data["N_query"].unique())
    n_obs_vals = sorted(data["N_obs"].unique())

    best_config = {}
    best_time = {}
    for (n_query, n_obs), grp in data.groupby(["N_query", "N_obs"]):
        idx = grp["t_total_s"].idxmin()
        best_config[(n_query, n_obs)] = grp.loc[idx, "config"]
        best_time[(n_query, n_obs)] = grp.loc[idx, "t_total_s"]

    fig, ax = plt.subplots(figsize=(10, 7))
    color_matrix = np.zeros((len(n_query_vals), len(n_obs_vals), 3))
    for i, nq in enumerate(n_query_vals):
        for j, no in enumerate(n_obs_vals):
            cfg = best_config.get((nq, no))
            if cfg is not None and cfg in CONFIG_COLORS:
                color_matrix[i, j] = matplotlib.colors.to_rgb(CONFIG_COLORS[cfg])
            else:
                color_matrix[i, j] = (0.9, 0.9, 0.9)

    ax.imshow(color_matrix, aspect="auto")
    ax.set_xticks(range(len(n_obs_vals)))
    ax.set_xticklabels([f"{v:,}" for v in n_obs_vals])
    ax.set_yticks(range(len(n_query_vals)))
    ax.set_yticklabels([f"{v:,}" for v in n_query_vals])
    ax.set_xlabel("N_obs")
    ax.set_ylabel("N_query")

    excl_note = "" if not exclude else f" (excl. {', '.join(exclude)})"
    ax.set_title(f"Best config per cell{excl_note}")

    for i, nq in enumerate(n_query_vals):
        for j, no in enumerate(n_obs_vals):
            cfg = best_config.get((nq, no))
            t = best_time.get((nq, no))
            if cfg is None:
                ax.text(j, i, "—", ha="center", va="center", fontsize=8)
                continue
            t_str = f"{t:.1f}s" if t is not None and not np.isnan(t) else ""
            label = f"{cfg}\n{t_str}" if t_str else cfg
            # Light-background configs (light gray, light orange) need dark text.
            light_configs = {"nproc=2 (handicapped)", "nproc=4 (handicapped)"}
            text_color = "black" if cfg in light_configs else "white"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                fontsize=7,
                color=text_color,
            )

    # Legend: only show configs that actually appear in the data.
    present_configs = set(best_config.values())
    legend_handles = [
        plt.matplotlib.patches.Patch(color=CONFIG_COLORS[cfg], label=cfg)
        for cfg in CONFIG_COLORS
        if cfg in present_configs
    ]
    ax.legend(
        handles=legend_handles, bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8
    )
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

    # --- Combined analysis (post-fix + balanced-BLAS supplement) ---
    if not SUPPLEMENT_CSV.exists():
        print(f"\n[skip] {SUPPLEMENT_CSV} not found; skipping combined analysis.")
        return

    df_combined = load_combined_results()
    print(
        f"\nCombined dataset: {len(df_combined)} rows, "
        f"{df_combined['config'].nunique()} configs"
    )

    med_long = combined_cell_medians(df_combined)
    combined_medians_csv = HERE / "results_points_post_fix_combined_medians.csv"
    med_long.to_csv(combined_medians_csv, index=False)

    wide = combined_wide_table(med_long)
    combined_wide_csv = HERE / "results_points_post_fix_combined_wide.csv"
    wide.to_csv(combined_wide_csv, index=False)

    best_config_out = FIGURES_DIR / "best_config_per_cell.png"
    write_best_config_heatmap(med_long, best_config_out)

    best_multiproc_out = FIGURES_DIR / "best_multiproc_config_per_cell.png"
    write_best_config_heatmap(med_long, best_multiproc_out, exclude=["nproc=1"])

    print("\nWrote (combined):")
    print(f"  {combined_medians_csv}")
    print(f"  {combined_wide_csv}")
    print(f"  {best_config_out}")
    print(f"  {best_multiproc_out}")

    # Summary: does nproc=1 win every cell?
    nproc1_wins_all = True
    if "nproc=1" in med_long["config"].values:
        pivot_check = med_long.pivot_table(
            index=["N_query", "N_obs"], columns="config", values="t_total_s"
        )
        if "nproc=1" in pivot_check.columns:
            other_cols = [c for c in pivot_check.columns if c != "nproc=1"]
            for col in other_cols:
                diff = pivot_check[col] - pivot_check["nproc=1"]
                if (diff <= 0).any():
                    nproc1_wins_all = False
                    break
    if nproc1_wins_all:
        print("\nSummary: nproc=1 wins in every cell across all 7 configs.")
    else:
        print("\nSummary: nproc=1 does NOT win in every cell — check data.")


if __name__ == "__main__":
    main()

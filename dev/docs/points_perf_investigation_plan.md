# Points-Mode Performance Investigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a small benchmarking harness that times `pipeline.points_pipeline` end-to-end across a 7×3×2 (N_query × N_obs × strategy) matrix, run the sweep, and write a findings document with a recommendation on the points-mode multiproc question.

**Architecture:** Self-contained harness directory mirroring the grid investigation's structure but smaller. The harness loads the `modified_foster_2019` config, pre-generates a deterministic pool of NZ-land random query points, materialises subsampled observation CSVs once per N_obs value, then loops the matrix and times `pipeline.points_pipeline` end-to-end. Reuses `subsample_observations` from the grid harness via direct import.

**Tech Stack:** Python 3.13, pytest, ruff, mamba `vs30_venv`. Activate with:

```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv
```

(Henceforth abbreviated as `<activate>`.)

**Reference docs:**
- `dev/docs/points_perf_investigation_design.md` — methodology, scope, risks.
- `dev/docs/perf_features_investigation_findings.md` — grid-mode findings, especially §7.4 (the `MIN_DIST_ENFORCED` clamp side-finding).

---

## File map

| Path | Responsibility |
|---|---|
| `dev/scripts/investigations/points_features_investigation/.gitignore` | Excludes `results_*.csv`, `figures/`, Python caches, the temp obs-CSV directory. |
| `dev/scripts/investigations/points_features_investigation/README.md` | Brief reproduction instructions. |
| `dev/scripts/investigations/points_features_investigation/bench_utils.py` | Helpers: `generate_nz_land_points`, `materialize_obs_csvs`, `time_one_run`. Reuses `subsample_observations` from the grid harness. |
| `dev/scripts/investigations/points_features_investigation/test_bench_utils.py` | Unit tests for the helpers. |
| `dev/scripts/investigations/points_features_investigation/run_points_sweep.py` | Driver: loops the test matrix, writes `results_points.csv` incrementally. |
| `dev/scripts/investigations/points_features_investigation/analyze_points_results.py` | Loads CSV, computes medians + speedup, writes 1–2 heatmaps. |
| `dev/docs/points_perf_investigation_findings.md` | Final deliverable: short findings doc with the recommendation. |

Production code (`vs30/`) is **not modified** by this investigation.

---

## Task 1: Scaffold the harness directory

**Files:**
- Create: `dev/scripts/investigations/points_features_investigation/.gitignore`
- Create: `dev/scripts/investigations/points_features_investigation/README.md`

- [ ] **Step 1: Create the directory and `.gitignore`**

```bash
mkdir -p /home/arr65/src/Vs30/dev/scripts/investigations/points_features_investigation
```

Write `dev/scripts/investigations/points_features_investigation/.gitignore`:

```
results_*.csv
figures/
__pycache__/
*.pyc
.pytest_cache/
obs_csvs/
```

- [ ] **Step 2: Create README.md skeleton**

Write `dev/scripts/investigations/points_features_investigation/README.md`:

```markdown
# Points-Mode Performance Investigation Harness

Code to measure the multiprocessing / BLAS-MT tradeoff in
`pipeline.points_pipeline` end-to-end.

See:
- `dev/docs/points_perf_investigation_design.md` — methodology and rationale
- `dev/docs/points_perf_investigation_findings.md` — results and recommendations

## Run order

```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv

# Run the sweep (~60-115 minutes)
python -m dev.scripts.investigations.points_features_investigation.run_points_sweep

# Analyse
python -m dev.scripts.investigations.points_features_investigation.analyze_points_results
```

Outputs (gitignored, regenerable):
- `results_points.csv` — raw per-cell timings
- `results_points_medians.csv` — per-cell medians
- `obs_csvs/` — materialised subsampled observation CSVs
- `figures/*.png` — heatmaps
```

- [ ] **Step 3: Commit**

```bash
cd /home/arr65/src/Vs30 && \
git add dev/scripts/investigations/points_features_investigation/.gitignore \
        dev/scripts/investigations/points_features_investigation/README.md && \
git commit -m "$(cat <<'EOF'
investigations(points-perf): scaffold harness directory

Create dev/scripts/investigations/points_features_investigation/ with
.gitignore and README in preparation for the points-mode multiproc
investigation harness.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: bench_utils — generate_nz_land_points

The query-point generator. Rejection-sample WGS84 lon/lats within the NZ
bounding box, keep only those landing on a valid IwahashiPike pixel (i.e.,
land). Deterministic with a seed.

**Files:**
- Create: `dev/scripts/investigations/points_features_investigation/test_bench_utils.py`
- Create: `dev/scripts/investigations/points_features_investigation/bench_utils.py`

- [ ] **Step 1: Write the failing tests**

Create `test_bench_utils.py`:

```python
"""Unit tests for the points-perf-investigation harness."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

import bench_utils


def test_generate_nz_land_points_count() -> None:
    lons, lats = bench_utils.generate_nz_land_points(100, seed=42)
    assert len(lons) == 100
    assert len(lats) == 100


def test_generate_nz_land_points_determinism() -> None:
    lons1, lats1 = bench_utils.generate_nz_land_points(50, seed=42)
    lons2, lats2 = bench_utils.generate_nz_land_points(50, seed=42)
    np.testing.assert_array_equal(lons1, lons2)
    np.testing.assert_array_equal(lats1, lats2)


def test_generate_nz_land_points_within_nz_bbox() -> None:
    lons, lats = bench_utils.generate_nz_land_points(100, seed=42)
    # Approximate NZ bounding box (WGS84)
    assert np.all(lons >= 165.0) and np.all(lons <= 180.0)
    assert np.all(lats >= -48.0) and np.all(lats <= -34.0)


def test_generate_nz_land_points_all_on_land() -> None:
    """Every returned point should land on a valid IwahashiPike pixel."""
    from qcore import coordinates

    from vs30 import category, constants

    lons, lats = bench_utils.generate_nz_land_points(100, seed=42)
    nztm = coordinates.wgs_depth_to_nztm(np.column_stack([lats, lons]))
    eastings = nztm[:, 1]
    northings = nztm[:, 0]
    points = np.column_stack([eastings, northings])
    terrain_ids = category.assign_to_category_terrain(points)
    assert np.all(terrain_ids != constants.RASTER_ID_NODATA_VALUE), (
        "Some sampled points fall on terrain nodata (i.e., off NZ land)"
    )
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
pytest dev/scripts/investigations/points_features_investigation/test_bench_utils.py -v
```

Expected: collection error / `ModuleNotFoundError: bench_utils`.

- [ ] **Step 3: Implement `generate_nz_land_points`**

Create `bench_utils.py`:

```python
"""Helpers for the points-perf-investigation harness."""

import datetime as _dt
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from qcore import coordinates

from vs30 import category, constants, pipeline

# Reuse subsample_observations from the grid harness via path import.
# This avoids duplicating the canonical subsampling routine.
_GRID_HARNESS_DIR = (
    Path(__file__).resolve().parents[1] / "perf_features_investigation"
)
sys.path.insert(0, str(_GRID_HARNESS_DIR))
from bench_utils import subsample_observations  # noqa: E402, F401  -- re-exported


# WGS84 bounding box that comfortably covers all NZ land. Slightly looser than
# tight to allow for the rejection sampler's land-mask filtering to do its job.
_NZ_LON_MIN, _NZ_LON_MAX = 165.0, 180.0
_NZ_LAT_MIN, _NZ_LAT_MAX = -48.0, -34.0


def generate_nz_land_points(n: int, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    """Return ``n`` random (lon, lat) points uniformly distributed over NZ land.

    Rejection samples WGS84 lon/lat within the NZ bounding box and keeps only
    points that land on a valid (non-nodata) IwahashiPike terrain raster pixel.

    Parameters
    ----------
    n
        Number of points to return.
    seed
        Seed for the numpy random generator.

    Returns
    -------
    lons, lats : np.ndarray
        Two ``(n,)`` arrays of WGS84 longitudes and latitudes.
    """
    rng = np.random.default_rng(seed)
    kept_lons: list[float] = []
    kept_lats: list[float] = []
    while len(kept_lons) < n:
        # Over-sample by ~3x; about 30-40% of the NZ bbox is land.
        batch_size = max(3 * n, 1000)
        lons = rng.uniform(_NZ_LON_MIN, _NZ_LON_MAX, batch_size)
        lats = rng.uniform(_NZ_LAT_MIN, _NZ_LAT_MAX, batch_size)
        nztm = coordinates.wgs_depth_to_nztm(np.column_stack([lats, lons]))
        eastings = nztm[:, 1]
        northings = nztm[:, 0]
        points = np.column_stack([eastings, northings])
        terrain_ids = category.assign_to_category_terrain(points)
        land_mask = terrain_ids != constants.RASTER_ID_NODATA_VALUE
        kept_lons.extend(lons[land_mask].tolist())
        kept_lats.extend(lats[land_mask].tolist())
    return np.array(kept_lons[:n]), np.array(kept_lats[:n])
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
pytest dev/scripts/investigations/points_features_investigation/test_bench_utils.py -v
```

Expected: 4 passed. Each test invokes the terrain raster sampler — the run takes ~10–20 s.

- [ ] **Step 5: Verify ruff is clean**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
ruff check dev/scripts/investigations/points_features_investigation/bench_utils.py \
           dev/scripts/investigations/points_features_investigation/test_bench_utils.py && \
ruff format --check dev/scripts/investigations/points_features_investigation/bench_utils.py \
           dev/scripts/investigations/points_features_investigation/test_bench_utils.py
```

Both should exit 0. If either fails, run `ruff format <files>` and re-check.

- [ ] **Step 6: Commit**

```bash
cd /home/arr65/src/Vs30 && \
git add dev/scripts/investigations/points_features_investigation/bench_utils.py \
        dev/scripts/investigations/points_features_investigation/test_bench_utils.py && \
git commit -m "$(cat <<'EOF'
investigations(points-perf): add generate_nz_land_points helper

Rejection-samples WGS84 points uniformly over the NZ bounding box, keeps
only those landing on a valid IwahashiPike terrain pixel. Deterministic
with a seed. Reuses subsample_observations from the grid harness via a
path-based import.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: bench_utils — materialize_obs_csvs and time_one_run

Two helpers:

- `materialize_obs_csvs(out_dir, n_obs_values, seed)`: pre-generates one
  subsampled CSV per `N_obs` value, returns a dict of `N_obs → Path`.
- `time_one_run(...)`: timing wrapper around `pipeline.points_pipeline`.

**Files:**
- Modify: `dev/scripts/investigations/points_features_investigation/test_bench_utils.py`
- Modify: `dev/scripts/investigations/points_features_investigation/bench_utils.py`

- [ ] **Step 1: Append failing tests**

Append to `test_bench_utils.py`:

```python
def test_materialize_obs_csvs_creates_files(tmp_path) -> None:
    paths = bench_utils.materialize_obs_csvs(
        out_dir=tmp_path, n_obs_values=[50, 100], seed=42
    )
    assert set(paths.keys()) == {50, 100}
    for n_obs, path in paths.items():
        assert path.exists()
        df = pd.read_csv(path)
        # Required columns must survive the round-trip
        required = {"easting", "northing", "vs30", "uncertainty"}
        assert required.issubset(df.columns)
        assert len(df) == n_obs


def test_time_one_run_returns_expected_keys(tmp_path) -> None:
    # Materialise a small obs CSV
    paths = bench_utils.materialize_obs_csvs(
        out_dir=tmp_path, n_obs_values=[100], seed=42
    )
    # 5 query points, just enough to exercise the pipeline end-to-end
    lons, lats = bench_utils.generate_nz_land_points(5, seed=42)
    cfg = bench_utils.load_modified_foster_2019_config()
    row = bench_utils.time_one_run(
        lons=lons,
        lats=lats,
        obs_csv_path=paths[100],
        nproc=1,
        rep=0,
        cfg=cfg,
    )
    expected_keys = {
        "N_query", "N_obs", "nproc", "rep",
        "t_total_s", "peak_rss_mb", "timestamp_iso",
    }
    assert expected_keys.issubset(row.keys())
    assert row["t_total_s"] > 0
    assert row["N_query"] == 5
    assert row["N_obs"] == 100
```

Add `import pandas as pd` at the top of `test_bench_utils.py` if not already present.

- [ ] **Step 2: Run tests to verify they fail**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
pytest dev/scripts/investigations/points_features_investigation/test_bench_utils.py::test_materialize_obs_csvs_creates_files \
       dev/scripts/investigations/points_features_investigation/test_bench_utils.py::test_time_one_run_returns_expected_keys -v
```

Expected: FAIL — `materialize_obs_csvs`, `time_one_run`, and `load_modified_foster_2019_config` not defined.

- [ ] **Step 3: Implement the helpers**

Append to `bench_utils.py`:

```python
def materialize_obs_csvs(
    out_dir: Path, n_obs_values: list[int], seed: int = 42
) -> dict[int, Path]:
    """Subsample viktor_cpt observations and write one CSV per N_obs value.

    Pre-generating these once-per-sweep avoids the cost of a temp-file write
    per cell. The returned paths are usable as ``independent_observations_csv``
    arguments to ``pipeline.points_pipeline``.

    Parameters
    ----------
    out_dir
        Directory to write the CSVs into. Created if it does not exist.
    n_obs_values
        Distinct N_obs values to materialise.
    seed
        Seed forwarded to ``subsample_observations``.

    Returns
    -------
    dict[int, Path]
        Map of N_obs → path of the CSV containing that many observations.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[int, Path] = {}
    for n_obs in n_obs_values:
        df = subsample_observations(n_obs, seed=seed)
        path = out_dir / f"obs_subsampled_n{n_obs}.csv"
        df.to_csv(path, index=False)
        paths[n_obs] = path
    return paths


def load_modified_foster_2019_config() -> dict:
    """Return the resolved modified_foster_2019 model config.

    Wraps ``cli.load_model_config`` (which builds correlation-function partials
    and resolves resource paths) for the harness.
    """
    from vs30 import cli  # local import — cli has heavy transitive imports

    return cli.load_model_config(constants.FixedModelVersion.MODIFIED_FOSTER_2019)


def _peak_rss_mb() -> float:
    """Peak resident-set size of the current process in MB.

    Linux ``ru_maxrss`` is in kibibytes, so divide by 1024 for MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def time_one_run(
    lons: np.ndarray,
    lats: np.ndarray,
    obs_csv_path: Path,
    nproc: int,
    rep: int,
    cfg: dict,
) -> dict:
    """Time one ``pipeline.points_pipeline`` call end-to-end.

    The harness uses the modified_foster_2019 config but injects the
    sweep's subsampled observations as ``independent_observations_csv``.
    With ``do_bayesian_update=False`` (the modified_foster_2019 default),
    the points pipeline does not invoke DBSCAN clustering, so the
    "independent vs clustered" distinction is moot for the spatial
    adjustment — both go through the same ``compute_spatial_adjustment_at_points``
    code path. Forced to ``False`` here to remove ambiguity.

    Parameters
    ----------
    lons, lats
        WGS84 query coordinates.
    obs_csv_path
        Path to the materialised observation CSV for this cell.
    nproc
        ``nproc`` passed through to ``points_pipeline``. Strategy endpoints
        for the sweep are 1 (BLAS multi-threaded) and 8 (BLAS single-threaded
        inside workers).
    rep
        Repetition index, recorded for downstream median computation.
    cfg
        Resolved model config from ``load_modified_foster_2019_config``.

    Returns
    -------
    dict
        CSV-row-shaped fields recording the timing.
    """
    n_obs = len(pd.read_csv(obs_csv_path))
    t0 = time.perf_counter()
    pipeline.points_pipeline(
        longitudes=lons,
        latitudes=lats,
        geology_categorical_csv=cfg["geology_categorical_csv"],
        terrain_categorical_csv=cfg["terrain_categorical_csv"],
        clustered_observations_csv=None,
        independent_observations_csv=obs_csv_path,
        combination_method=constants.CombinationMethod(cfg["combination_method"]),
        combine_ratio=cfg["combine_ratio"],
        noisy=cfg["noisy"],
        mvn=cfg["mvn"],
        do_bayesian_update=False,
        include_intermediate=False,
        nproc=nproc,
        geology_corr_fn=cfg["geology_corr_fn"],
        terrain_corr_fn=cfg["terrain_corr_fn"],
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        apply_coastal_distance_mod=cfg["apply_coastal_distance_mod"],
        fill_gaps=False,
    )
    t_total = time.perf_counter() - t0
    return {
        "N_query": int(len(lons)),
        "N_obs": n_obs,
        "nproc": nproc,
        "rep": rep,
        "t_total_s": t_total,
        "peak_rss_mb": _peak_rss_mb(),
        "timestamp_iso": _dt.datetime.now().isoformat(timespec="seconds"),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
pytest dev/scripts/investigations/points_features_investigation/test_bench_utils.py -v
```

Expected: 6 tests pass. The `time_one_run` test exercises the full points pipeline end-to-end at 5 query points × 100 obs — runs in ~30 s.

- [ ] **Step 5: Verify ruff is clean**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
ruff check dev/scripts/investigations/points_features_investigation/bench_utils.py \
           dev/scripts/investigations/points_features_investigation/test_bench_utils.py && \
ruff format --check dev/scripts/investigations/points_features_investigation/bench_utils.py \
           dev/scripts/investigations/points_features_investigation/test_bench_utils.py
```

If `ruff format --check` fails, run `ruff format <files>` and re-verify.

- [ ] **Step 6: Commit**

```bash
cd /home/arr65/src/Vs30 && \
git add dev/scripts/investigations/points_features_investigation/bench_utils.py \
        dev/scripts/investigations/points_features_investigation/test_bench_utils.py && \
git commit -m "$(cat <<'EOF'
investigations(points-perf): add materialize_obs_csvs + time_one_run

materialize_obs_csvs pre-generates one subsampled observation CSV per
N_obs value once per sweep; time_one_run wraps a single end-to-end
pipeline.points_pipeline call and returns a CSV-row dict. The harness
forces do_bayesian_update=False and fill_gaps=False so the timing
captures only the pipeline's per-point work.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: run_points_sweep.py driver + smoke verification

The Phase-1 driver. Loops the test matrix and writes results incrementally.
First version uses a `--smoke` flag with a tiny matrix to verify end-to-end
before launching the full sweep.

**Files:**
- Create: `dev/scripts/investigations/points_features_investigation/run_points_sweep.py`

- [ ] **Step 1: Write the driver**

Create `run_points_sweep.py`:

```python
"""Points-mode performance sweep driver.

Loops the (N_query x N_obs x nproc x rep) matrix and writes one CSV row per
cell. CSV is written incrementally so partial results survive an interrupt.

Run with::

    python -m dev.scripts.investigations.points_features_investigation.run_points_sweep
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import bench_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("points_sweep")

HERE = Path(__file__).parent
OUT_CSV = HERE / "results_points.csv"
OBS_DIR = HERE / "obs_csvs"

CSV_FIELDS = [
    "N_query",
    "N_obs",
    "nproc",
    "rep",
    "t_total_s",
    "peak_rss_mb",
    "timestamp_iso",
]

# Full sweep — overridden by --smoke.
N_QUERY_VALUES = [1, 10, 100, 1_000, 10_000, 50_000, 100_000]
N_OBS_VALUES = [100, 1_000, 35_706]
NPROC_VALUES = [1, 8]
N_REPS = 3


def _append_row(row: dict) -> None:
    new_file = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Run a tiny matrix (3 N_query x 1 N_obs x 2 nproc x 1 rep) for verification.",
    )
    args = parser.parse_args()

    if args.smoke:
        n_query_values = [1, 10, 100]
        n_obs_values = [100]
        n_reps = 1
    else:
        n_query_values = N_QUERY_VALUES
        n_obs_values = N_OBS_VALUES
        n_reps = N_REPS

    logger.info("Loading modified_foster_2019 config...")
    cfg = bench_utils.load_modified_foster_2019_config()

    logger.info(f"Materialising obs CSVs for N_obs in {n_obs_values}")
    obs_paths = bench_utils.materialize_obs_csvs(OBS_DIR, n_obs_values)

    largest_n_query = max(n_query_values)
    logger.info(f"Pre-generating {largest_n_query:,} NZ-land query points...")
    lons_pool, lats_pool = bench_utils.generate_nz_land_points(largest_n_query, seed=42)
    logger.info(f"Pool ready ({len(lons_pool):,} points).")

    for n_query in n_query_values:
        # Sub-sample the pre-generated pool deterministically (first n_query points).
        lons = lons_pool[:n_query]
        lats = lats_pool[:n_query]
        for n_obs in n_obs_values:
            obs_csv_path = obs_paths[n_obs]
            for nproc in NPROC_VALUES:
                for rep in range(n_reps):
                    logger.info(
                        f"  cell N_query={n_query:>6} N_obs={n_obs:>6} "
                        f"nproc={nproc} rep={rep}"
                    )
                    try:
                        row = bench_utils.time_one_run(
                            lons=lons,
                            lats=lats,
                            obs_csv_path=obs_csv_path,
                            nproc=nproc,
                            rep=rep,
                            cfg=cfg,
                        )
                    except Exception:
                        logger.exception(
                            "    cell failed — recording empty row and moving on"
                        )
                        row = {k: None for k in CSV_FIELDS}
                        row.update(
                            {
                                "N_query": n_query,
                                "N_obs": n_obs,
                                "nproc": nproc,
                                "rep": rep,
                            }
                        )
                    _append_row(row)

    logger.info(f"Sweep complete - results at {OUT_CSV}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify ruff is clean**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
ruff check dev/scripts/investigations/points_features_investigation/run_points_sweep.py && \
ruff format --check dev/scripts/investigations/points_features_investigation/run_points_sweep.py
```

If `ruff format --check` fails, run `ruff format <file>` and re-verify.

- [ ] **Step 3: Run the smoke variant**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
rm -f dev/scripts/investigations/points_features_investigation/results_points.csv && \
python -m dev.scripts.investigations.points_features_investigation.run_points_sweep --smoke 2>&1 | tail -40
```

Expected: completes in ~3–10 minutes; produces `results_points.csv` with `3 × 1 × 2 × 1 = 6` rows + 1 header = 7 lines. Final log line: `Sweep complete - results at ...`.

- [ ] **Step 4: Sanity-check the smoke output**

```bash
<activate> && cd /home/arr65/src/Vs30 && python -c "
import pandas as pd
df = pd.read_csv('dev/scripts/investigations/points_features_investigation/results_points.csv')
print(df)
print('---')
print(f'rows: {len(df)}')
print(f'null t_total_s cells: {int(df[\"t_total_s\"].isna().sum())}')
piv = df.pivot_table(index=['N_query','N_obs'], columns='nproc', values='t_total_s')
print(piv)
"
```

Expected: 6 rows, no nulls, sane numbers (probably nproc=1 is faster across the board at these tiny sizes due to spawn overhead).

- [ ] **Step 5: Commit (the CSV is gitignored)**

```bash
cd /home/arr65/src/Vs30 && \
git add dev/scripts/investigations/points_features_investigation/run_points_sweep.py && \
git commit -m "$(cat <<'EOF'
investigations(points-perf): add run_points_sweep driver

Driver loops the (N_query x N_obs x nproc x rep) matrix and writes
results_points.csv incrementally. Pre-generates the largest-N_query
pool of NZ-land points once and slices smaller cells from it for
deterministic re-use. Materialises subsampled observation CSVs once
per N_obs value into obs_csvs/. --smoke flag runs a tiny 6-row
matrix for verification.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: Execute the full sweep

Pure execution — no code changes. Expected wall time 60–115 minutes; run in
the background so the session can continue with other work.

- [ ] **Step 1: Launch the full sweep**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
rm -f dev/scripts/investigations/points_features_investigation/results_points.csv && \
nohup python -m dev.scripts.investigations.points_features_investigation.run_points_sweep \
    > dev/scripts/investigations/points_features_investigation/sweep.log 2>&1 &
echo "PID: $!"
```

(Or use the Bash tool with `run_in_background=True` so the agent can poll
without blocking.)

- [ ] **Step 2: Monitor periodically**

```bash
tail -n 30 dev/scripts/investigations/points_features_investigation/sweep.log
wc -l dev/scripts/investigations/points_features_investigation/results_points.csv
```

Expected at completion: a "Sweep complete" log line and 7 × 3 × 2 × 3 = 126 data rows + 1 header = 127 lines in the CSV.

- [ ] **Step 3: Check for failed cells**

```bash
<activate> && cd /home/arr65/src/Vs30 && python -c "
import pandas as pd
df = pd.read_csv('dev/scripts/investigations/points_features_investigation/results_points.csv')
fails = df[df['t_total_s'].isna()]
print(f'{len(fails)} failed cells')
print(fails[['N_query','N_obs','nproc','rep']].to_string() if len(fails) else 'none')
"
```

If any failed cells: inspect `sweep.log` for the traceback. Decide whether to
re-run those cells or accept the gap. **No commit** — this step is execution.

---

## Task 6: analyze_points_results script

Loads the CSV, computes medians, and writes a small heatmap of the multiproc
speedup over the (N_query, N_obs) plane.

**Files:**
- Create: `dev/scripts/investigations/points_features_investigation/analyze_points_results.py`

- [ ] **Step 1: Write the script**

Create `analyze_points_results.py`:

```python
"""Analyse points-mode performance sweep results.

Reads results_points.csv, computes per-cell median wall times, builds the
multiproc speedup pivot (nproc=1 / nproc=8), and writes a heatmap to
figures/.

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
FIGURES_DIR = HERE / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def load_results() -> pd.DataFrame:
    df = pd.read_csv(RESULTS_CSV)
    df = df.dropna(subset=["t_total_s"])
    return df


def cell_medians(df: pd.DataFrame) -> pd.DataFrame:
    """Median t_total_s per (N_query, N_obs, nproc)."""
    return (
        df.groupby(["N_query", "N_obs", "nproc"])["t_total_s"]
        .median()
        .reset_index()
    )


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


def write_speedup_heatmap(piv: pd.DataFrame, out_path: Path) -> None:
    """Log2-coloured heatmap of nproc=8/nproc=1 speedup over N_query x N_obs."""
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
    ax.set_title("Multiproc speedup (nproc=8 vs 1) for points_pipeline")
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
    med = cell_medians(df)
    med.to_csv(HERE / "results_points_medians.csv", index=False)

    piv = speedup_table(med)
    piv.to_csv(HERE / "results_points_speedup.csv")

    out = FIGURES_DIR / "multiproc_speedup_points.png"
    write_speedup_heatmap(piv, out)

    print("Wrote:")
    print(f"  {HERE / 'results_points_medians.csv'}")
    print(f"  {HERE / 'results_points_speedup.csv'}")
    print(f"  {out}")
    print()
    print("Speedup table (nproc=8 / nproc=1):")
    print(piv.to_string())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify ruff is clean**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
ruff check dev/scripts/investigations/points_features_investigation/analyze_points_results.py && \
ruff format --check dev/scripts/investigations/points_features_investigation/analyze_points_results.py
```

If `ruff format --check` fails, run `ruff format <file>` and re-verify.

- [ ] **Step 3: Run the analysis**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
python -m dev.scripts.investigations.points_features_investigation.analyze_points_results 2>&1 | tail -30
```

Expected: prints "Wrote: …" and prints the speedup table. Produces 1 PNG in `figures/`.

- [ ] **Step 4: Commit**

```bash
cd /home/arr65/src/Vs30 && \
git add dev/scripts/investigations/points_features_investigation/analyze_points_results.py && \
git commit -m "$(cat <<'EOF'
investigations(points-perf): add analyze_points_results script

Loads results_points.csv, computes per-cell medians and the
nproc=8/nproc=1 speedup pivot, writes a log-scale heatmap and two
auxiliary CSVs (medians, speedup table).

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: Write findings doc

Synthesise the sweep results into the deliverable Markdown file.

**Files:**
- Create: `dev/docs/points_perf_investigation_findings.md`
- Optional: Copy heatmap into `dev/docs/figures/points_perf/` for committed reference.

- [ ] **Step 1: Read the results**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
cat dev/scripts/investigations/points_features_investigation/results_points_speedup.csv && \
cat dev/scripts/investigations/points_features_investigation/results_points_medians.csv
```

Note the patterns: where (if anywhere) does the speedup cross 1.0? Is the
"1 query point" cell dominated by spawn overhead as expected? Does the
speedup grow monotonically with N_query?

Note also the **absolute** time at the largest cell (N_query=100k, N_obs=35 706,
nproc=1). This is the input to the §7 decision rule in the design doc:

- < 30 s → ffap follow-up not justified.
- > 5 min → ffap follow-up may be justified.
- In between → judgement call.

- [ ] **Step 2: Copy the heatmap into a tracked location**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
mkdir -p dev/docs/figures/points_perf && \
cp dev/scripts/investigations/points_features_investigation/figures/multiproc_speedup_points.png \
   dev/docs/figures/points_perf/
```

- [ ] **Step 3: Write the findings doc**

Create `dev/docs/points_perf_investigation_findings.md` with this skeleton; fill in
numbers and recommendation from the actual results:

```markdown
# Points-Mode Performance Investigation — Findings

**Date:** [today]
**Branch:** `vs30_refactor`
**Status:** Complete — sweep done, analysis written.
**Predecessors:**
- [Design](points_perf_investigation_design.md)
- [Grid investigation findings](perf_features_investigation_findings.md)

## 1. Summary

[One-line recommendation per axis investigated.]

| Question | Answer | Evidence |
|---|---|---|
| Does multiproc ever win for `points_pipeline` in realistic regimes? | [yes / no / yes-above-N_query=X] | [headline numbers] |
| Should `vs30 points --nproc` default change? | [yes/no, recommend value] | [as above] |
| Is a `find_affected_points` follow-up worth pursuing? | [yes/no, per the §7 decision rule] | [N_query=100k × N_obs=35706 took T s sequentially] |

## 2. Methodology

Brief — points to the design doc for full detail.

## 3. Hardware and software

- CPU: Intel Core i7-9700, 8 cores @ 3.00 GHz, no hyperthreading.
- Memory: 32 GiB.
- Vs30 commit at sweep time: [git rev-parse HEAD]

## 4. Results

| N_query | N_obs=100 t1 / t8 (s) | N_obs=1000 t1 / t8 (s) | N_obs=35706 t1 / t8 (s) |
|---|---|---|---|
| 1 | … | … | … |
| 10 | … | … | … |
| ... | ... | ... | ... |
| 100 000 | … | … | … |

(Fill from `results_points_speedup.csv`. `t1` is nproc=1 with multi-threaded
BLAS; `t8` is nproc=8 with single-threaded BLAS. Speedup = `t1 / t8`.)

![Multiproc speedup heatmap](figures/points_perf/multiproc_speedup_points.png)

## 5. Conclusions and recommendations

### 5.1 Multiproc

[One of:
- "Remove the multiproc path from points_pipeline — nproc=1 wins everywhere tested."
- "Keep multiproc, change CLI default from -1 to 1; engage multiproc only when N_query >= X."
- "Keep multiproc as-is; the existing default is correct for typical workloads."
]

### 5.2 ffap follow-up

[One of:
- "Skip — sequential is fast enough at all measured cells (max T s)."
- "Worth a follow-up — at the upper-bound cell (N_query=100k × N_obs=35706), sequential takes T s. A targeted profiling investigation would be the natural next step."
]

## 6. Reproducibility

The harness is at `dev/scripts/investigations/points_features_investigation/`.
To reproduce:

```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv && \
cd /home/arr65/src/Vs30 && \
python -m dev.scripts.investigations.points_features_investigation.run_points_sweep && \
python -m dev.scripts.investigations.points_features_investigation.analyze_points_results
```
```

Fill in the blanks from the actual results, paying attention to:
- The headline summary in §1 must be backed by the table in §4.
- Recommendations in §5 must be concrete (which file/line to change, if any).
- The §5.2 ffap framing follows the decision rule in the design doc §7.

- [ ] **Step 4: Final validation**

Re-run the harness tests to confirm nothing has been broken by analysis-side
changes:

```bash
<activate> && cd /home/arr65/src/Vs30 && \
pytest dev/scripts/investigations/points_features_investigation/test_bench_utils.py -v 2>&1 | tail -10
```

Expected: 6 passed.

Re-confirm production code is untouched:

```bash
cd /home/arr65/src/Vs30 && git log --stat HEAD~10..HEAD -- vs30/ tests/ 2>&1 | head -10
```

Expected: empty (or, if nonempty, you should be able to recognise every commit
in the list as something you intended). The investigation should not have
modified `vs30/` or `tests/`.

- [ ] **Step 5: Commit**

```bash
cd /home/arr65/src/Vs30 && \
git add dev/docs/points_perf_investigation_findings.md dev/docs/figures/points_perf/ && \
git commit -m "$(cat <<'EOF'
docs: add points-mode perf investigation findings

Findings from the points-mode multiproc investigation. Recommendation:
[one-line summary]. Investigation also gives a [yes/no] answer to the
"is find_affected_points worth investigating" follow-up question per
the design doc's §7 decision rule.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

(Edit the commit message to match the actual recommendation.)

---

## Self-review

**Spec coverage** (from `dev/docs/points_perf_investigation_design.md`):

- §1 Purpose → Task 7 (findings doc)
- §2 Scope → enforced throughout (multiproc only, end-to-end timing only)
- §3 Methodology
  - 3.1 Two-strategy comparison → Task 4 (`NPROC_VALUES = [1, 8]`)
  - 3.2 Test matrix → Task 4 + Task 5
  - 3.3 Query-point sampling → Task 2
  - 3.4 Categorical-model config → Task 3 (`load_modified_foster_2019_config`)
  - 3.5 Per-cell timing → Task 3 (`time_one_run`)
- §4 Hardware → Task 7 records
- §5 Deliverable artefacts → all tasks together
- §6 Risks → mitigations folded into the relevant tasks (variance via 3 reps; OOM analysed in design)
- §7 ffap follow-up framing → Task 7 Step 1 + Step 3 §5.2
- §8 Branch / commit strategy → each task commits; production code untouched

**Placeholder scan:** every code block contains real, runnable code. The
findings-doc skeleton has clearly-marked blanks (`…` and bracketed prompts) to
be filled in from actual results — this is intentional template scaffolding,
not a TODO.

**Type consistency:** the helper signatures used in later tasks
(`generate_nz_land_points` → `(np.ndarray, np.ndarray)`,
`materialize_obs_csvs` → `dict[int, Path]`,
`time_one_run(..., obs_csv_path: Path, nproc: int, rep: int, cfg: dict) → dict`)
match the call sites in `run_points_sweep.py` and the test cases in
`test_bench_utils.py`.

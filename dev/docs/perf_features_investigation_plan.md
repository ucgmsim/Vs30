# Performance Features Investigation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a benchmarking harness that measures the effect of the multiprocessing path and the `find_affected_pixels` bbox pre-filter on `vs30` grid-mode performance, run a parameter sweep, and write a findings doc that recommends keeping or removing each feature.

**Architecture:** A self-contained directory `dev/scripts/investigations/perf_features_investigation/` holds the benchmarking harness (`bench_utils.py`), Phase 1 isolated-MVN driver (`run_isolated_sweep.py`), Phase 2 full-pipeline driver (`run_full_pipeline_confirmation.py`), and analysis script (`analyze_results.py`). The harness imports the production `vs30` package and times only `find_affected_pixels` and `compute_spatial_adjustments` / `run_parallel_spatial_fit` in isolation; it does not modify any production code.

**Tech Stack:** Python 3, pandas, numpy, matplotlib, scipy. Existing `vs30_venv` mamba environment activated via:

```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv
```

(Henceforth abbreviated as `<activate>`.)

---

## File map

Files created by this plan:

| Path | Responsibility |
|---|---|
| `dev/scripts/investigations/perf_features_investigation/bench_utils.py` | Reusable helpers: subsample_observations, make_raster_data, make_full_bbox_result, bypass_observation_threshold, time_one_run, run_numerical_equivalence_check |
| `dev/scripts/investigations/perf_features_investigation/run_isolated_sweep.py` | Phase 1 driver — loops the parameter matrix, writes results_isolated.csv |
| `dev/scripts/investigations/perf_features_investigation/run_full_pipeline_confirmation.py` | Phase 2 driver — runs `pipeline.grid_pipeline` end-to-end on 4 cohorts |
| `dev/scripts/investigations/perf_features_investigation/analyze_results.py` | Loads CSVs, computes medians + speedup tables, writes heatmaps to figures/ |
| `dev/scripts/investigations/perf_features_investigation/test_bench_utils.py` | Unit tests for bench_utils |
| `dev/scripts/investigations/perf_features_investigation/README.md` | How to reproduce the investigation |
| `dev/scripts/investigations/perf_features_investigation/.gitignore` | Excludes results_*.csv and figures/ |
| `dev/docs/perf_features_investigation_findings.md` | Final deliverable — findings, conclusions, recommendations |

Production code (`vs30/`) is **not** modified.

---

## Task 1: Set up directory and gitignore

**Files:**
- Create: `dev/scripts/investigations/perf_features_investigation/.gitignore`
- Create: `dev/scripts/investigations/perf_features_investigation/README.md`

- [ ] **Step 1: Create the directory and `.gitignore`**

```bash
mkdir -p /home/arr65/src/Vs30/dev/scripts/investigations/perf_features_investigation
```

Write `.gitignore`:

```
results_*.csv
figures/
__pycache__/
*.pyc
.pytest_cache/
```

- [ ] **Step 2: Create README.md skeleton**

```markdown
# Performance Features Investigation Harness

Code to measure the effect of multiprocessing and `find_affected_pixels`
on `vs30` grid-mode performance.

See:
- `dev/docs/perf_features_investigation_design.md` — methodology and rationale
- `dev/docs/perf_features_investigation_findings.md` — results and recommendations

## Run order

```bash
# Activate env (assumes mamba/vs30_venv already configured)
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv

# Phase 1: isolated MVN sweep
python -m dev.scripts.investigations.perf_features_investigation.run_isolated_sweep

# Phase 2: full-pipeline confirmation
python -m dev.scripts.investigations.perf_features_investigation.run_full_pipeline_confirmation

# Analyse
python -m dev.scripts.investigations.perf_features_investigation.analyze_results
```

Outputs:
- `results_isolated.csv`, `results_full_pipeline.csv` — raw timings (gitignored)
- `figures/*.png` — speedup heatmaps (gitignored)
```

- [ ] **Step 3: Commit**

```bash
cd /home/arr65/src/Vs30 && git add dev/scripts/investigations/perf_features_investigation/.gitignore dev/scripts/investigations/perf_features_investigation/README.md && git commit -m "$(cat <<'EOF'
investigations(perf): scaffold harness directory

Create dev/scripts/investigations/perf_features_investigation/ with
.gitignore and README skeleton in preparation for the perf features
investigation harness.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: bench_utils — subsample_observations

**Files:**
- Create: `dev/scripts/investigations/perf_features_investigation/test_bench_utils.py`
- Create: `dev/scripts/investigations/perf_features_investigation/bench_utils.py`

- [ ] **Step 1: Write the failing test**

`test_bench_utils.py`:

```python
"""Unit tests for the perf-features-investigation benchmarking harness."""

import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent))

import bench_utils


def test_subsample_observations_count() -> None:
    df = bench_utils.subsample_observations(100, seed=42)
    assert len(df) == 100


def test_subsample_observations_determinism() -> None:
    df1 = bench_utils.subsample_observations(100, seed=42)
    df2 = bench_utils.subsample_observations(100, seed=42)
    pd.testing.assert_frame_equal(df1, df2)


def test_subsample_observations_required_columns() -> None:
    df = bench_utils.subsample_observations(50, seed=42)
    required = {"easting", "northing", "vs30", "uncertainty"}
    assert required.issubset(df.columns)


def test_subsample_observations_too_many_raises() -> None:
    with pytest.raises(ValueError, match="exceeds available"):
        bench_utils.subsample_observations(10**9, seed=42)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
<activate> && cd /home/arr65/src/Vs30 && pytest dev/scripts/investigations/perf_features_investigation/test_bench_utils.py -v
```

Expected: ERROR — `bench_utils` does not exist yet.

- [ ] **Step 3: Implement `subsample_observations`**

Create `bench_utils.py`:

```python
"""Helpers for the perf-features-investigation benchmarking harness."""

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[4]
VIKTOR_OBS_PATH = (
    REPO_ROOT / "vs30/resources/observations/viktor_inferred_vs30_from_cpt.csv"
)


def subsample_observations(n: int, seed: int = 42) -> pd.DataFrame:
    """Return a deterministic subsample of viktor_cpt observations.

    Parameters
    ----------
    n
        Number of observations to return.
    seed
        Seed for the numpy random generator.

    Returns
    -------
    pd.DataFrame
        Subsampled observations with the standard required columns.

    Raises
    ------
    ValueError
        If ``n`` exceeds the number of available observations.
    """
    df = pd.read_csv(VIKTOR_OBS_PATH, comment="#", skipinitialspace=True)
    if n > len(df):
        raise ValueError(
            f"n ({n}) exceeds available observations ({len(df)})"
        )
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(df), size=n, replace=False)
    return df.iloc[idx].reset_index(drop=True)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
<activate> && cd /home/arr65/src/Vs30 && pytest dev/scripts/investigations/perf_features_investigation/test_bench_utils.py -v
```

Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
cd /home/arr65/src/Vs30 && git add dev/scripts/investigations/perf_features_investigation/bench_utils.py dev/scripts/investigations/perf_features_investigation/test_bench_utils.py && git commit -m "$(cat <<'EOF'
investigations(perf): add subsample_observations helper

Deterministic subsample of viktor_cpt observations for the perf-features
parameter sweep.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: bench_utils — make_raster_data

`make_raster_data(n_target)` builds a real `RasterData` of approximately the requested valid-pixel count. Strategy: use a fixed central-NZ subdomain at varying resolutions to hit each target.

**Files:**
- Modify: `dev/scripts/investigations/perf_features_investigation/test_bench_utils.py`
- Modify: `dev/scripts/investigations/perf_features_investigation/bench_utils.py`

- [ ] **Step 1: Add failing tests**

Append to `test_bench_utils.py`:

```python
def test_make_raster_data_returns_valid_raster_data() -> None:
    raster_data, profile = bench_utils.make_raster_data(n_target=1000)
    # Real RasterData with a non-empty valid mask
    assert raster_data.valid_flat_indices.size > 0
    # Profile carries transform and crs
    assert "transform" in profile
    assert profile["transform"] is not None


def test_make_raster_data_n_target_scales() -> None:
    # Larger n_target should produce more valid pixels
    rd_small, _ = bench_utils.make_raster_data(n_target=1000)
    rd_large, _ = bench_utils.make_raster_data(n_target=100_000)
    assert rd_large.valid_flat_indices.size > rd_small.valid_flat_indices.size
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
<activate> && cd /home/arr65/src/Vs30 && pytest dev/scripts/investigations/perf_features_investigation/test_bench_utils.py::test_make_raster_data_returns_valid_raster_data -v
```

Expected: FAIL — `make_raster_data` not defined.

- [ ] **Step 3: Implement `make_raster_data`**

Append to `bench_utils.py`:

```python
from vs30 import config, constants, pipeline, spatial


# (target_label, dx, dy, x_extent_m, y_extent_m) — chosen empirically to hit
# the targets within ~2× tolerance. Actual N_valid is logged at runtime so
# analysis can use the true value instead of the target.
_GRID_PRESETS: dict[int, tuple[int, int, int, int]] = {
    1_000:    (2000, 2000,  90_000,  90_000),
    10_000:   (1000, 1000, 150_000, 150_000),
    100_000:  ( 500,  500, 250_000, 250_000),
    1_000_000:( 200,  200, 350_000, 350_000),
}

# Centre of the subdomain — chosen near central NZ so the box always lands
# on land. NZTM (easting, northing).
_DOMAIN_CENTRE = (1_580_000, 5_180_000)


def make_raster_data(n_target: int):
    """Build a real RasterData of approximately ``n_target`` valid pixels.

    Uses the production ``pipeline.create_initial_vs30_arrays`` with
    ``model_type=TERRAIN`` to populate a sub-region of NZ. The exact
    valid-pixel count varies with the underlying terrain raster; callers
    should log ``raster_data.valid_flat_indices.size`` rather than rely on
    ``n_target`` exactly.

    Parameters
    ----------
    n_target
        Approximate number of valid pixels to return.

    Returns
    -------
    raster_data : spatial.RasterData
        Real raster data backed by the IwahashiPike terrain raster.
    profile : dict
        Rasterio profile (transform, crs, nodata).
    """
    if n_target not in _GRID_PRESETS:
        raise ValueError(
            f"n_target must be one of {sorted(_GRID_PRESETS)}, got {n_target}"
        )
    dx, dy, x_extent, y_extent = _GRID_PRESETS[n_target]
    cx, cy = _DOMAIN_CENTRE
    grid_config = config.GridConfig(
        grid_xmin=cx - x_extent // 2,
        grid_xmax=cx + x_extent // 2,
        grid_ymin=cy - y_extent // 2,
        grid_ymax=cy + y_extent // 2,
        grid_dx=dx,
        grid_dy=dy,
    )
    vs30_array, stdv_array, _, profile = pipeline.create_initial_vs30_arrays(
        grid_config,
        constants.ModelType.TERRAIN,
        # Read the canonical terrain categorical CSV, which is bundled.
        pipeline.read_categorical_csv(
            constants.RESOURCE_PATH
            / constants.RESOURCE_SUBDIRS["terrain_categorical_csv"]
            / "terrain_model_prior_mean_and_standard_deviation.csv"
        ),
    )
    raster_data = spatial.RasterData.from_arrays(
        vs30=vs30_array,
        stdv=stdv_array,
        transform=profile["transform"],
        crs=profile.get("crs", constants.NZTM_CRS),
        nodata=constants.NODATA_VALUE,
    )
    return raster_data, profile
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
<activate> && cd /home/arr65/src/Vs30 && pytest dev/scripts/investigations/perf_features_investigation/test_bench_utils.py -v
```

Expected: 6 passed. If `test_make_raster_data_n_target_scales` fails because the actual valid-pixel counts don't increase monotonically with `n_target`, tweak `_GRID_PRESETS` (resolution / extent) until they do.

- [ ] **Step 5: Verify the actual N_valid values**

Run a small ad-hoc check:

```bash
<activate> && cd /home/arr65/src/Vs30 && python -c "
import sys
sys.path.insert(0, 'dev/scripts/investigations/perf_features_investigation')
from bench_utils import make_raster_data
for n in [1_000, 10_000, 100_000, 1_000_000]:
    rd, _ = make_raster_data(n)
    print(f'target={n:>9,}  actual={rd.valid_flat_indices.size:>9,}')
"
```

Expected output: actual values within 2× of targets, monotonically increasing. If they aren't, adjust `_GRID_PRESETS` and re-run until satisfied. Note the actual values for use in the findings doc.

- [ ] **Step 6: Commit**

```bash
cd /home/arr65/src/Vs30 && git add dev/scripts/investigations/perf_features_investigation/bench_utils.py dev/scripts/investigations/perf_features_investigation/test_bench_utils.py && git commit -m "$(cat <<'EOF'
investigations(perf): add make_raster_data helper

Build real RasterData of approximately n_target valid pixels by clipping
the NZ domain at varying resolutions. Uses the bundled IwahashiPike
terrain raster so model_type=TERRAIN does not require slope/coast arrays.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: bench_utils — make_full_bbox_result

A trivial helper: build a `BoundingBoxResult` whose `mask` is the raster's full `valid_mask`, used to disable the bbox pre-filter for the OFF condition.

**Files:**
- Modify: `dev/scripts/investigations/perf_features_investigation/test_bench_utils.py`
- Modify: `dev/scripts/investigations/perf_features_investigation/bench_utils.py`

- [ ] **Step 1: Add failing test**

Append to `test_bench_utils.py`:

```python
import numpy as np


def test_make_full_bbox_result_marks_all_valid_pixels() -> None:
    raster_data, _ = bench_utils.make_raster_data(n_target=1000)
    bbox = bench_utils.make_full_bbox_result(raster_data, n_obs=10)
    assert bbox.mask.shape == (raster_data.vs30.size,)
    # Every valid pixel should be marked affected.
    assert bool(np.all(bbox.mask[raster_data.valid_flat_indices]))
    assert bbox.n_affected_pixels == raster_data.valid_flat_indices.size
```

- [ ] **Step 2: Run test to verify it fails**

```bash
<activate> && cd /home/arr65/src/Vs30 && pytest dev/scripts/investigations/perf_features_investigation/test_bench_utils.py::test_make_full_bbox_result_marks_all_valid_pixels -v
```

Expected: FAIL — `make_full_bbox_result` not defined.

- [ ] **Step 3: Implement `make_full_bbox_result`**

Append to `bench_utils.py`:

```python
def make_full_bbox_result(
    raster_data: spatial.RasterData, n_obs: int
) -> spatial.BoundingBoxResult:
    """Build a BoundingBoxResult that marks every valid pixel as affected.

    Used to disable the ``find_affected_pixels`` pre-filter for the OFF
    condition. Each observation's index list is set to the full valid-pixel
    set so the parallel path (which uses obs_to_grid_indices) still works.

    Parameters
    ----------
    raster_data
        Raster whose valid pixels become the affected set.
    n_obs
        Number of observations — needed to size obs_to_grid_indices.

    Returns
    -------
    spatial.BoundingBoxResult
        Mask covers every valid pixel; per-observation index lists each
        contain the full valid_flat_indices array.
    """
    mask = np.zeros(raster_data.vs30.size, dtype=bool)
    mask[raster_data.valid_flat_indices] = True
    obs_to_grid_indices = [
        raster_data.valid_flat_indices.copy() for _ in range(n_obs)
    ]
    return spatial.BoundingBoxResult(
        mask=mask,
        obs_to_grid_indices=obs_to_grid_indices,
        n_affected_pixels=int(mask.sum()),
    )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
<activate> && cd /home/arr65/src/Vs30 && pytest dev/scripts/investigations/perf_features_investigation/test_bench_utils.py::test_make_full_bbox_result_marks_all_valid_pixels -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
cd /home/arr65/src/Vs30 && git add dev/scripts/investigations/perf_features_investigation/bench_utils.py dev/scripts/investigations/perf_features_investigation/test_bench_utils.py && git commit -m "$(cat <<'EOF'
investigations(perf): add make_full_bbox_result helper

Builds a BoundingBoxResult that marks every valid pixel as affected,
used to disable the find_affected_pixels pre-filter for the OFF
benchmark condition.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: bench_utils — bypass_observation_threshold

Context manager that monkey-patches `MULTIPROCESS_OBSERVATION_THRESHOLD` so the production guard does not pre-empt the multiproc condition we want to measure when N_obs > 1000.

**Files:**
- Modify: `dev/scripts/investigations/perf_features_investigation/test_bench_utils.py`
- Modify: `dev/scripts/investigations/perf_features_investigation/bench_utils.py`

- [ ] **Step 1: Add failing test**

Append to `test_bench_utils.py`:

```python
def test_bypass_observation_threshold_restores_original() -> None:
    original = constants.MULTIPROCESS_OBSERVATION_THRESHOLD
    with bench_utils.bypass_observation_threshold():
        assert constants.MULTIPROCESS_OBSERVATION_THRESHOLD == 10**12
    assert constants.MULTIPROCESS_OBSERVATION_THRESHOLD == original


def test_bypass_observation_threshold_restores_on_exception() -> None:
    original = constants.MULTIPROCESS_OBSERVATION_THRESHOLD
    with pytest.raises(RuntimeError):
        with bench_utils.bypass_observation_threshold():
            raise RuntimeError("boom")
    assert constants.MULTIPROCESS_OBSERVATION_THRESHOLD == original
```

Add to imports at top of `test_bench_utils.py`:

```python
from vs30 import constants
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
<activate> && cd /home/arr65/src/Vs30 && pytest dev/scripts/investigations/perf_features_investigation/test_bench_utils.py::test_bypass_observation_threshold_restores_original -v
```

Expected: FAIL — `bypass_observation_threshold` not defined.

- [ ] **Step 3: Implement `bypass_observation_threshold`**

Append to `bench_utils.py`:

```python
import contextlib


@contextlib.contextmanager
def bypass_observation_threshold():
    """Disable the n_obs > 1000 fallback for one measurement.

    The production guard at ``pipeline.compute_spatial_adjustment_on_grid``
    forces ``nproc=1`` whenever the observation count exceeds
    ``MULTIPROCESS_OBSERVATION_THRESHOLD``. To measure the multiproc path
    in that regime we temporarily raise the threshold to a value larger
    than any conceivable observation count, then restore it.
    """
    original = constants.MULTIPROCESS_OBSERVATION_THRESHOLD
    constants.MULTIPROCESS_OBSERVATION_THRESHOLD = 10**12
    try:
        yield
    finally:
        constants.MULTIPROCESS_OBSERVATION_THRESHOLD = original
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
<activate> && cd /home/arr65/src/Vs30 && pytest dev/scripts/investigations/perf_features_investigation/test_bench_utils.py -v
```

Expected: all tests pass (4 + 2 + 1 + 2 = 9).

- [ ] **Step 5: Commit**

```bash
cd /home/arr65/src/Vs30 && git add dev/scripts/investigations/perf_features_investigation/bench_utils.py dev/scripts/investigations/perf_features_investigation/test_bench_utils.py && git commit -m "$(cat <<'EOF'
investigations(perf): add bypass_observation_threshold context manager

Temporarily raises MULTIPROCESS_OBSERVATION_THRESHOLD so the production
guard does not pre-empt the multiproc condition during measurement.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 6: bench_utils — time_one_run

The workhorse that runs a single (N_obs, N_grid, nproc, ffap) cell and returns a timing dict.

**Files:**
- Modify: `dev/scripts/investigations/perf_features_investigation/test_bench_utils.py`
- Modify: `dev/scripts/investigations/perf_features_investigation/bench_utils.py`

- [ ] **Step 1: Add failing test**

Append to `test_bench_utils.py`:

```python
def test_time_one_run_returns_expected_keys() -> None:
    raster_data, _ = bench_utils.make_raster_data(n_target=1000)
    obs_df = bench_utils.subsample_observations(50, seed=42)
    obs_data = bench_utils.prepare_terrain_obs_data(obs_df, raster_data)
    row = bench_utils.time_one_run(
        raster_data=raster_data,
        obs_data=obs_data,
        nproc=1,
        ffap=True,
        rep=0,
    )
    expected_keys = {
        "N_obs", "N_grid_actual", "N_affected", "nproc", "ffap", "rep",
        "t_bbox_s", "t_spatial_s", "t_total_s", "peak_rss_mb",
        "timestamp_iso",
    }
    assert expected_keys.issubset(row.keys())
    assert row["t_total_s"] >= row["t_bbox_s"]
    assert row["t_spatial_s"] > 0
    assert row["t_bbox_s"] >= 0
```

- [ ] **Step 2: Run test to verify it fails**

```bash
<activate> && cd /home/arr65/src/Vs30 && pytest dev/scripts/investigations/perf_features_investigation/test_bench_utils.py::test_time_one_run_returns_expected_keys -v
```

Expected: FAIL — `time_one_run` not defined.

- [ ] **Step 3: Implement `prepare_terrain_obs_data` and `time_one_run`**

Append to `bench_utils.py`:

```python
import datetime as _dt
import functools
import resource
import time

from vs30 import parallel, raster, utils

DEFAULT_CORR_FN = functools.partial(
    utils.exponential_correlation_function, phi=constants.DEFAULT_TERRAIN_PHI
)


def prepare_terrain_obs_data(
    obs_df: pd.DataFrame, raster_data: spatial.RasterData
) -> spatial.ObservationData:
    """Build ObservationData for the terrain model from an observation DataFrame.

    Wraps ``spatial.prepare_observation_data`` for the TERRAIN branch (no
    slope/coast arrays). Reads the bundled posterior terrain CSV used by
    the modified_foster_2019 model so the categorical lookups exercise
    realistic Vs30 / stdv values.
    """
    posterior_csv = (
        constants.RESOURCE_PATH
        / constants.RESOURCE_SUBDIRS["terrain_categorical_csv"]
        / "terrain_model_posterior_from_foster_2019_mean_and_standard_deviation.csv"
    )
    model_df = pipeline.read_categorical_csv(posterior_csv)
    mean_col, std_col = raster.select_vs30_columns_by_priority(list(model_df.columns))
    max_id = int(model_df[constants.STANDARD_ID_COLUMN].max())
    updated_model_table = np.full((max_id, 2), np.nan)
    ids = model_df[constants.STANDARD_ID_COLUMN].values.astype(int) - 1
    valid = (ids >= 0) & (ids < max_id)
    updated_model_table[ids[valid], 0] = model_df[mean_col].values[valid]
    updated_model_table[ids[valid], 1] = model_df[std_col].values[valid]

    return spatial.prepare_observation_data(
        observations=obs_df,
        raster_data=raster_data,
        updated_model_table=updated_model_table,
        model_type=constants.ModelType.TERRAIN,
        apply_alluvium_slope_mod=False,
        apply_coastal_distance_mod=False,
        noisy=True,
    )


def _peak_rss_mb() -> float:
    """Peak resident-set size of the current process in MB.

    Linux ``ru_maxrss`` is in kibibytes, so divide by 1024 for MB.
    """
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def time_one_run(
    raster_data: spatial.RasterData,
    obs_data: spatial.ObservationData,
    nproc: int,
    ffap: bool,
    rep: int,
    corr_fn=DEFAULT_CORR_FN,
    max_dist_m: int = constants.MAX_DIST_M,
    max_points: int = constants.MAX_POINTS,
    cov_reduc: float = constants.COV_REDUC,
    noisy: bool = True,
    max_spatial_boolean_array_memory_gb: float = 1.0,
) -> dict:
    """Measure one (N_obs, N_grid, nproc, ffap, rep) cell.

    Returns a dict suitable for a CSV row.
    """
    # ---- Bounding-box phase -----------------------------------------------
    if ffap:
        t0 = time.perf_counter()
        bbox = spatial.find_affected_pixels(
            raster_data,
            obs_data,
            max_spatial_boolean_array_memory_gb=max_spatial_boolean_array_memory_gb,
            model_type=constants.ModelType.TERRAIN,
            max_dist_m=max_dist_m,
            nproc=nproc,
        )
        t_bbox = time.perf_counter() - t0
    else:
        bbox = make_full_bbox_result(raster_data, n_obs=len(obs_data.locations))
        t_bbox = 0.0

    # ---- Spatial-adjustment phase -----------------------------------------
    # nproc=1 => let BLAS use all cores (do NOT wrap in single_threaded_blas).
    # nproc>1 => parallel.run_parallel_spatial_fit handles single_threaded_blas
    #            internally; we only bypass the production observation-threshold
    #            guard so the multiproc path is actually exercised when N_obs > 1000.
    t0 = time.perf_counter()
    if nproc == 1:
        spatial.compute_spatial_adjustments(
            raster_data,
            obs_data,
            bbox,
            corr_fn,
            max_dist_m=max_dist_m,
            max_points=max_points,
            noisy=noisy,
            cov_reduc=cov_reduc,
        )
    else:
        with bypass_observation_threshold():
            affected_flat_indices = np.where(bbox.mask)[0]
            parallel.run_parallel_spatial_fit(
                affected_flat_indices=affected_flat_indices,
                raster_data=raster_data,
                obs_data=obs_data,
                corr_fn=corr_fn,
                model_type=constants.ModelType.TERRAIN,
                max_dist_m=max_dist_m,
                max_points=max_points,
                noisy=noisy,
                cov_reduc=cov_reduc,
                nproc=nproc,
            )
    t_spatial = time.perf_counter() - t0

    return {
        "N_obs": len(obs_data.locations),
        "N_grid_actual": int(raster_data.valid_flat_indices.size),
        "N_affected": int(bbox.n_affected_pixels),
        "nproc": nproc,
        "ffap": ffap,
        "rep": rep,
        "t_bbox_s": t_bbox,
        "t_spatial_s": t_spatial,
        "t_total_s": t_bbox + t_spatial,
        "peak_rss_mb": _peak_rss_mb(),
        "timestamp_iso": _dt.datetime.now().isoformat(timespec="seconds"),
    }
```

- [ ] **Step 4: Run test to verify it passes**

```bash
<activate> && cd /home/arr65/src/Vs30 && pytest dev/scripts/investigations/perf_features_investigation/test_bench_utils.py::test_time_one_run_returns_expected_keys -v
```

Expected: PASS. (May take ~10-30 seconds due to the actual computation.)

- [ ] **Step 5: Commit**

```bash
cd /home/arr65/src/Vs30 && git add dev/scripts/investigations/perf_features_investigation/bench_utils.py dev/scripts/investigations/perf_features_investigation/test_bench_utils.py && git commit -m "$(cat <<'EOF'
investigations(perf): add time_one_run + prepare_terrain_obs_data

time_one_run measures a single (N_obs, N_grid, nproc, ffap) cell and
returns a CSV-row-shaped dict. prepare_terrain_obs_data wraps
spatial.prepare_observation_data for the TERRAIN model so the harness
does not need slope/coast arrays.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 7: bench_utils — run_numerical_equivalence_check

Asserts that all four (nproc × ffap) combinations produce identical results on a small case before the sweep starts.

**Files:**
- Modify: `dev/scripts/investigations/perf_features_investigation/test_bench_utils.py`
- Modify: `dev/scripts/investigations/perf_features_investigation/bench_utils.py`

- [ ] **Step 1: Add failing test**

Append to `test_bench_utils.py`:

```python
def test_numerical_equivalence_check_passes_on_small_case() -> None:
    """All four (nproc × ffap) combinations should produce identical output."""
    bench_utils.run_numerical_equivalence_check(
        n_obs=200,
        n_target=1000,
        nproc_options=(1, 2),  # use 2 instead of 8 to keep test fast
    )
```

- [ ] **Step 2: Run test to verify it fails**

```bash
<activate> && cd /home/arr65/src/Vs30 && pytest dev/scripts/investigations/perf_features_investigation/test_bench_utils.py::test_numerical_equivalence_check_passes_on_small_case -v
```

Expected: FAIL — `run_numerical_equivalence_check` not defined.

- [ ] **Step 3: Implement `run_numerical_equivalence_check`**

Append to `bench_utils.py`:

```python
def _compute_one(
    raster_data: spatial.RasterData,
    obs_data: spatial.ObservationData,
    nproc: int,
    ffap: bool,
    corr_fn=DEFAULT_CORR_FN,
    max_dist_m: int = constants.MAX_DIST_M,
    max_points: int = constants.MAX_POINTS,
    cov_reduc: float = constants.COV_REDUC,
    noisy: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Run one variant and return the (vs30, stdv) arrays."""
    if ffap:
        bbox = spatial.find_affected_pixels(
            raster_data,
            obs_data,
            max_spatial_boolean_array_memory_gb=1.0,
            model_type=constants.ModelType.TERRAIN,
            max_dist_m=max_dist_m,
            nproc=nproc,
        )
    else:
        bbox = make_full_bbox_result(raster_data, n_obs=len(obs_data.locations))
    if nproc == 1:
        return spatial.compute_spatial_adjustments(
            raster_data,
            obs_data,
            bbox,
            corr_fn,
            max_dist_m=max_dist_m,
            max_points=max_points,
            noisy=noisy,
            cov_reduc=cov_reduc,
        )
    with bypass_observation_threshold():
        affected_flat_indices = np.where(bbox.mask)[0]
        return parallel.run_parallel_spatial_fit(
            affected_flat_indices=affected_flat_indices,
            raster_data=raster_data,
            obs_data=obs_data,
            corr_fn=corr_fn,
            model_type=constants.ModelType.TERRAIN,
            max_dist_m=max_dist_m,
            max_points=max_points,
            noisy=noisy,
            cov_reduc=cov_reduc,
            nproc=nproc,
        )


def run_numerical_equivalence_check(
    n_obs: int = 200,
    n_target: int = 1_000,
    nproc_options: tuple[int, ...] = (1, 8),
    atol: float = 1e-9,
    rtol: float = 1e-7,
) -> None:
    """Confirm all (nproc × ffap) variants produce identical output.

    Run before the sweep so we know any timing differences reflect the
    feature's effect, not a logic divergence between variants.

    Raises
    ------
    AssertionError
        If any variant differs from the reference (nproc=1, ffap=True).
    """
    raster_data, _ = make_raster_data(n_target=n_target)
    obs_df = subsample_observations(n_obs, seed=42)
    obs_data = prepare_terrain_obs_data(obs_df, raster_data)

    ref_vs30, ref_stdv = _compute_one(raster_data, obs_data, nproc=1, ffap=True)

    for nproc in nproc_options:
        for ffap in (True, False):
            if (nproc, ffap) == (1, True):
                continue
            vs30, stdv = _compute_one(raster_data, obs_data, nproc=nproc, ffap=ffap)
            np.testing.assert_allclose(
                vs30, ref_vs30, atol=atol, rtol=rtol,
                err_msg=f"vs30 mismatch at nproc={nproc}, ffap={ffap}",
            )
            np.testing.assert_allclose(
                stdv, ref_stdv, atol=atol, rtol=rtol,
                err_msg=f"stdv mismatch at nproc={nproc}, ffap={ffap}",
            )
```

- [ ] **Step 4: Run test to verify it passes**

```bash
<activate> && cd /home/arr65/src/Vs30 && pytest dev/scripts/investigations/perf_features_investigation/test_bench_utils.py::test_numerical_equivalence_check_passes_on_small_case -v
```

Expected: PASS (may take 30–60 s).

- [ ] **Step 5: Commit**

```bash
cd /home/arr65/src/Vs30 && git add dev/scripts/investigations/perf_features_investigation/bench_utils.py dev/scripts/investigations/perf_features_investigation/test_bench_utils.py && git commit -m "$(cat <<'EOF'
investigations(perf): add run_numerical_equivalence_check

Asserts every (nproc x ffap) variant produces identical (vs30, stdv)
output on a small case, so any subsequent timing differences reflect the
feature's effect rather than a logic divergence.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 8: run_isolated_sweep.py — Phase 1 driver (smoke test)

Build the driver that loops the parameter matrix. First version uses a tiny matrix to verify end-to-end before committing to the full sweep.

**Files:**
- Create: `dev/scripts/investigations/perf_features_investigation/run_isolated_sweep.py`

- [ ] **Step 1: Write the driver**

```python
"""Phase 1 driver — isolated MVN parameter sweep.

Loops the (N_obs x N_grid x nproc x ffap x rep) matrix and writes one CSV
row per cell. CSV is written incrementally so partial results survive an
interrupt or crash.

Run with::

    python -m dev.scripts.investigations.perf_features_investigation.run_isolated_sweep
"""

import argparse
import csv
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import bench_utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("perf_sweep")

OUT_CSV = Path(__file__).parent / "results_isolated.csv"

CSV_FIELDS = [
    "N_obs", "N_grid_target", "N_grid_actual", "N_affected",
    "nproc", "ffap", "rep",
    "t_bbox_s", "t_spatial_s", "t_total_s",
    "peak_rss_mb", "timestamp_iso",
]

# Full sweep — overridden when --smoke is passed.
N_OBS_VALUES = [50, 100, 250, 500, 1000, 2500, 5000, 10000, 35709]
N_GRID_VALUES = [1_000, 10_000, 100_000, 1_000_000]
NPROC_VALUES = [1, 8]
FFAP_VALUES = [True, False]
N_REPS = 3

# Skip cells whose previous-rep total time exceeds this — the sweep budget
# is finite and very long cells contribute little additional information.
PER_CELL_TIME_BUDGET_S = 1800.0


def _append_row(row: dict) -> None:
    new_file = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _run_cell(
    raster_data, obs_data, n_obs, n_grid_target, nproc, ffap, n_reps
) -> None:
    for rep in range(n_reps):
        logger.info(
            f"  cell N_obs={n_obs:>6} N_grid_target={n_grid_target:>9,} "
            f"nproc={nproc} ffap={int(ffap)} rep={rep}"
        )
        try:
            row = bench_utils.time_one_run(
                raster_data=raster_data, obs_data=obs_data,
                nproc=nproc, ffap=ffap, rep=rep,
            )
        except Exception:
            logger.exception("    cell failed — recording empty row and moving on")
            row = {k: None for k in CSV_FIELDS}
            row.update({
                "N_obs": n_obs, "N_grid_target": n_grid_target,
                "nproc": nproc, "ffap": ffap, "rep": rep,
            })
            _append_row(row)
            return
        row["N_grid_target"] = n_grid_target
        _append_row(row)
        if row["t_total_s"] > PER_CELL_TIME_BUDGET_S and rep < n_reps - 1:
            logger.warning(
                f"    cell exceeded budget ({row['t_total_s']:.1f}s) — skipping remaining reps"
            )
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Run a tiny matrix (3 N_obs x 2 N_grid x 2 nproc x 2 ffap x 1 rep)")
    args = parser.parse_args()

    if args.smoke:
        n_obs_values = [50, 250, 1000]
        n_grid_values = [1_000, 10_000]
        n_reps = 1
    else:
        n_obs_values = N_OBS_VALUES
        n_grid_values = N_GRID_VALUES
        n_reps = N_REPS

    logger.info("Numerical equivalence guardrail starting…")
    bench_utils.run_numerical_equivalence_check(
        n_obs=200, n_target=1_000, nproc_options=(1, 8)
    )
    logger.info("Numerical equivalence guardrail passed.")

    for n_obs in n_obs_values:
        obs_df = bench_utils.subsample_observations(n_obs, seed=42)
        for n_grid in n_grid_values:
            logger.info(
                f"Building raster data: N_obs={n_obs}, N_grid_target={n_grid:,}"
            )
            raster_data, _ = bench_utils.make_raster_data(n_grid)
            obs_data = bench_utils.prepare_terrain_obs_data(obs_df, raster_data)
            for nproc in NPROC_VALUES:
                for ffap in FFAP_VALUES:
                    _run_cell(
                        raster_data, obs_data, n_obs, n_grid, nproc, ffap, n_reps
                    )

    logger.info(f"Sweep complete — results at {OUT_CSV}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke variant**

```bash
<activate> && cd /home/arr65/src/Vs30 && rm -f dev/scripts/investigations/perf_features_investigation/results_isolated.csv && python -m dev.scripts.investigations.perf_features_investigation.run_isolated_sweep --smoke
```

Expected: Completes in ~5–15 minutes; produces `results_isolated.csv` with `3 × 2 × 2 × 2 × 1 = 24` rows. Skim the CSV — every cell should have non-null timings, and `t_total_s` should generally be ≤ 60 s for the smoke configurations.

- [ ] **Step 3: Sanity-check the smoke output**

```bash
<activate> && cd /home/arr65/src/Vs30 && python -c "
import pandas as pd
df = pd.read_csv('dev/scripts/investigations/perf_features_investigation/results_isolated.csv')
print(df)
print('---')
print('null cells:', int(df['t_total_s'].isna().sum()))
print('any nproc=8 with ffap=ON faster than nproc=1 with ffap=ON?')
piv = df.pivot_table(index=['N_obs','N_grid_target'], columns=['nproc','ffap'], values='t_total_s')
print(piv)
"
```

Expected: 24 rows, no nulls, sane numbers. Note any unexpected patterns — they're not failures, but worth flagging in the findings doc.

- [ ] **Step 4: Commit the driver and the smoke results**

The CSV is gitignored, so only the `.py` is committed.

```bash
cd /home/arr65/src/Vs30 && git add dev/scripts/investigations/perf_features_investigation/run_isolated_sweep.py && git commit -m "$(cat <<'EOF'
investigations(perf): add run_isolated_sweep driver

Phase 1 driver loops (N_obs x N_grid x nproc x ffap x rep) and writes
results_isolated.csv incrementally. --smoke flag runs a tiny matrix
(24 rows) for verification.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 9: Run the full Phase 1 sweep

Pure execution — no code changes. Expected wall time 3–6 hours; run in a screen / tmux / nohup session so a disconnected SSH won't kill it.

- [ ] **Step 1: Launch the full sweep**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
rm -f dev/scripts/investigations/perf_features_investigation/results_isolated.csv && \
nohup python -m dev.scripts.investigations.perf_features_investigation.run_isolated_sweep \
    > dev/scripts/investigations/perf_features_investigation/sweep_phase1.log 2>&1 &
echo "PID: $!"
```

(Alternatively use `Bash` with `run_in_background=True` so the agent can poll and continue work.)

- [ ] **Step 2: Monitor periodically**

```bash
tail -n 40 dev/scripts/investigations/perf_features_investigation/sweep_phase1.log
wc -l dev/scripts/investigations/perf_features_investigation/results_isolated.csv
```

Expected at completion: a "Sweep complete" log line; up to 9 × 4 × 2 × 2 × 3 = 432 rows (some may be skipped/errored — those are also recorded). Note any cells with `null` timings.

- [ ] **Step 3: Investigate any failed cells**

```bash
<activate> && cd /home/arr65/src/Vs30 && python -c "
import pandas as pd
df = pd.read_csv('dev/scripts/investigations/perf_features_investigation/results_isolated.csv')
fails = df[df['t_total_s'].isna()]
print(f'{len(fails)} failed cells:')
print(fails[['N_obs','N_grid_target','nproc','ffap','rep']].to_string())
"
```

If any fails are present, inspect the log for the corresponding traceback. Decide whether to re-run individual cells or accept the gap. (No commit — this is investigation.)

---

## Task 10: analyze_results.py — tables and medians

Phase 1 done; build the analysis. Loads the CSV, computes median timings per cell, builds speedup tables.

**Files:**
- Create: `dev/scripts/investigations/perf_features_investigation/analyze_results.py`

- [ ] **Step 1: Write the script**

```python
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


def speedup_pivot(med: pd.DataFrame, fixed_col: str, fixed_val, contrast_col: str
                  ) -> pd.DataFrame:
    """Pivot total time over (N_obs, N_grid_target) for two values of contrast_col.

    Returns a DataFrame with one column per contrast value plus a 'speedup'
    column = (slower / faster strategy time). speedup_col_label is which
    strategy is in the numerator.
    """
    sub = med[med[fixed_col] == fixed_val]
    piv = sub.pivot_table(
        index=["N_obs", "N_grid_target"],
        columns=contrast_col,
        values="t_total_s",
        aggfunc="median",
    )
    return piv


def write_heatmap(piv: pd.DataFrame, label: str, out_path: Path) -> None:
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
    cap = max(2.0, np.nanmax(np.abs(log_matrix)) if matrix.size else 2.0)
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
                continue
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color="white" if abs(log_matrix[i, j]) > cap / 2 else "black",
                    fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("log2(speedup)  [+ = numerator wins]")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def main() -> None:
    df = load_isolated()
    med = cell_medians(df)
    med.to_csv(HERE / "results_isolated_medians.csv", index=False)

    # ---- Multiproc speedup (nproc=1 / nproc=8) ----------------------------
    for ffap_fixed in (True, False):
        piv = speedup_pivot(med, "ffap", ffap_fixed, "nproc")
        if {1, 8}.issubset(piv.columns):
            piv["speedup"] = piv[1] / piv[8]
            label = f"Multiproc speedup (nproc=8 vs 1), ffap={'ON' if ffap_fixed else 'OFF'}"
            out = FIGURES_DIR / f"multiproc_speedup_ffap_{'on' if ffap_fixed else 'off'}.png"
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
    best = (
        med.loc[med.groupby(["N_obs", "N_grid_target"])["t_total_s"].idxmin()]
        .reset_index(drop=True)
    )
    best.to_csv(HERE / "results_isolated_best_strategy.csv", index=False)

    print("Wrote:")
    print(f"  {HERE / 'results_isolated_medians.csv'}")
    print(f"  {HERE / 'results_isolated_best_strategy.csv'}")
    print(f"  {FIGURES_DIR}/*.png")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the analysis**

```bash
<activate> && cd /home/arr65/src/Vs30 && python -m dev.scripts.investigations.perf_features_investigation.analyze_results
```

Expected: prints "Wrote: …" and produces 4 PNGs in `figures/`, plus 2 CSVs (medians, best_strategy). Open one heatmap to confirm it's readable:

```bash
xdg-open dev/scripts/investigations/perf_features_investigation/figures/multiproc_speedup_ffap_on.png 2>/dev/null || echo "open the file manually"
```

- [ ] **Step 3: Commit**

```bash
cd /home/arr65/src/Vs30 && git add dev/scripts/investigations/perf_features_investigation/analyze_results.py && git commit -m "$(cat <<'EOF'
investigations(perf): add analyze_results script

Loads results_isolated.csv, computes per-cell medians, multiproc and
ffap speedup pivots, writes log-scale heatmaps and a best-strategy
table.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 11: run_full_pipeline_confirmation.py — Phase 2 driver

End-to-end runs of `pipeline.grid_pipeline` for the four cohorts in §5.1 of the design doc.

**Files:**
- Create: `dev/scripts/investigations/perf_features_investigation/run_full_pipeline_confirmation.py`

- [ ] **Step 1: Write the driver**

```python
"""Phase 2 driver — full-pipeline confirmation runs.

Runs pipeline.grid_pipeline end-to-end for four (obs density, resolution)
cohorts and records wall-time. Used to confirm that the Phase 1 ordering
of strategies survives the surrounding pipeline overhead.

Run::

    python -m dev.scripts.investigations.perf_features_investigation.run_full_pipeline_confirmation
"""

import argparse
import csv
import datetime as dt
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import bench_utils  # noqa: F401  -- ensures repo root is on sys.path

from vs30 import config, constants, pipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("perf_full_pipeline")

HERE = Path(__file__).parent
OUT_CSV = HERE / "results_full_pipeline.csv"

CSV_FIELDS = [
    "cohort_label", "N_obs_label", "resolution_m", "nproc",
    "t_total_s", "timestamp_iso",
]

# (cohort_label, observations_csv, model_version_for_other_resources, dx_dy)
COHORTS = [
    ("sparse_coarse",
     constants.RESOURCE_PATH / constants.RESOURCE_SUBDIRS["independent_observations_csv"]
       / "modified_foster_2019_measured_vs30_independent_observations.csv",
     constants.FixedModelVersion.MODIFIED_FOSTER_2019,
     5000),
    ("sparse_fine",
     constants.RESOURCE_PATH / constants.RESOURCE_SUBDIRS["independent_observations_csv"]
       / "modified_foster_2019_measured_vs30_independent_observations.csv",
     constants.FixedModelVersion.MODIFIED_FOSTER_2019,
     500),
    ("dense_coarse",
     constants.RESOURCE_PATH / constants.RESOURCE_SUBDIRS["clustered_observations_csv"]
       / "viktor_inferred_vs30_from_cpt.csv",
     constants.FixedModelVersion.VIKTOR_CPT_CLUSTERING,
     5000),
    ("dense_fine",
     constants.RESOURCE_PATH / constants.RESOURCE_SUBDIRS["clustered_observations_csv"]
       / "viktor_inferred_vs30_from_cpt.csv",
     constants.FixedModelVersion.VIKTOR_CPT_CLUSTERING,
     500),
]


def _load_cli_config(version: constants.FixedModelVersion) -> dict:
    """Reuse cli.load_model_config to get fully-resolved config + corr fns."""
    from vs30 import cli
    return cli.load_model_config(version)


def _grid_for(resolution: int) -> config.GridConfig:
    base = constants.FULL_NZ_GRID_CONFIG
    return config.GridConfig(
        grid_xmin=base.grid_xmin,
        grid_xmax=base.grid_xmax,
        grid_ymin=base.grid_ymin,
        grid_ymax=base.grid_ymax,
        grid_dx=resolution,
        grid_dy=resolution,
    )


def _append_row(row: dict) -> None:
    new_file = not OUT_CSV.exists()
    with OUT_CSV.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def run_cohort(cohort_label: str, version: constants.FixedModelVersion,
               resolution: int, nproc: int) -> None:
    cfg = _load_cli_config(version)
    grid = _grid_for(resolution)
    logger.info(f"=== {cohort_label} resolution={resolution}m nproc={nproc} ===")
    t0 = time.perf_counter()
    pipeline.grid_pipeline(
        grid_config=grid,
        output_dir=None,
        geology_categorical_csv=cfg["geology_categorical_csv"],
        terrain_categorical_csv=cfg["terrain_categorical_csv"],
        clustered_observations_csv=cfg["clustered_observations_csv"],
        independent_observations_csv=cfg["independent_observations_csv"],
        combination_method=constants.CombinationMethod(cfg["combination_method"]),
        combine_ratio=cfg["combine_ratio"],
        noisy=cfg["noisy"],
        do_bayesian_update=cfg["do_bayesian_update"],
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        apply_coastal_distance_mod=cfg["apply_coastal_distance_mod"],
        fill_gaps=cfg["fill_gaps"],
        geology_corr_fn=cfg["geology_corr_fn"],
        terrain_corr_fn=cfg["terrain_corr_fn"],
        nproc=nproc,
    )
    t_total = time.perf_counter() - t0
    _append_row({
        "cohort_label": cohort_label,
        "N_obs_label": str(version),
        "resolution_m": resolution,
        "nproc": nproc,
        "t_total_s": t_total,
        "timestamp_iso": dt.datetime.now().isoformat(timespec="seconds"),
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true",
                        help="Only the cheapest two cohorts at one nproc")
    args = parser.parse_args()

    if args.smoke:
        cohorts = COHORTS[:2]
        nproc_options = (1,)
    else:
        cohorts = COHORTS
        nproc_options = (1, 8)

    for label, _, version, resolution in cohorts:
        for nproc in nproc_options:
            run_cohort(label, version, resolution, nproc)

    logger.info(f"Phase 2 complete — results at {OUT_CSV}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the smoke variant**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
rm -f dev/scripts/investigations/perf_features_investigation/results_full_pipeline.csv && \
python -m dev.scripts.investigations.perf_features_investigation.run_full_pipeline_confirmation --smoke
```

Expected: ~10–30 minutes; produces 2 rows in `results_full_pipeline.csv`. Sanity-check that both rows have positive `t_total_s` and the values are plausible (sparse-coarse should be < 5 minutes).

- [ ] **Step 3: Commit**

```bash
cd /home/arr65/src/Vs30 && git add dev/scripts/investigations/perf_features_investigation/run_full_pipeline_confirmation.py && git commit -m "$(cat <<'EOF'
investigations(perf): add full-pipeline confirmation driver

Phase 2 driver runs pipeline.grid_pipeline end-to-end for four
(obs density x resolution) cohorts at both nproc=1 and nproc=8 to verify
the Phase 1 ordering survives end-to-end pipeline overhead.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 12: Run Phase 2 confirmation

Pure execution — wall budget 2–4 hours.

- [ ] **Step 1: Launch the full Phase 2**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
rm -f dev/scripts/investigations/perf_features_investigation/results_full_pipeline.csv && \
nohup python -m dev.scripts.investigations.perf_features_investigation.run_full_pipeline_confirmation \
    > dev/scripts/investigations/perf_features_investigation/sweep_phase2.log 2>&1 &
echo "PID: $!"
```

- [ ] **Step 2: Monitor**

```bash
tail -n 40 dev/scripts/investigations/perf_features_investigation/sweep_phase2.log
wc -l dev/scripts/investigations/perf_features_investigation/results_full_pipeline.csv
```

Expected at completion: 4 cohorts × 2 nproc = 8 rows in the CSV; "Phase 2 complete" log line.

---

## Task 13: Write findings doc

Synthesise Phase 1 and Phase 2 results into the deliverable Markdown file.

**Files:**
- Create: `dev/docs/perf_features_investigation_findings.md`

- [ ] **Step 1: Read the analysis outputs**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
python -m dev.scripts.investigations.perf_features_investigation.analyze_results && \
cat dev/scripts/investigations/perf_features_investigation/results_isolated_medians.csv && \
cat dev/scripts/investigations/perf_features_investigation/results_isolated_best_strategy.csv && \
cat dev/scripts/investigations/perf_features_investigation/results_full_pipeline.csv
```

Note the patterns: where do the speedup heatmaps cross 1.0? Where does each strategy win?

- [ ] **Step 2: Draft the findings doc**

Create `dev/docs/perf_features_investigation_findings.md` with the structure from §8 of the design doc:

1. **Summary** — one sentence per feature: keep / remove / keep-with-revised-threshold, with the headline number (e.g., "ffap is 3× slower in every regime tested → remove").
2. **Methodology** — 3–5 sentences linking back to the design doc.
3. **Hardware and software** — exact CPU model, BLAS impl, NumPy/SciPy versions, environment hash.
4. **Phase 1 results** — for each feature: heatmap embedded as image, narrative interpretation, table of crossover points (or "no crossover — feature is X-erous everywhere").
5. **Phase 2 results** — table of the 8 cohort × nproc cells; comparison to Phase 1's prediction. If any cell inverts the prediction, investigate before concluding.
6. **Conclusions and recommendations** — per feature:
   - Beneficial / neutral / detrimental, with quantified evidence.
   - If keeping: any threshold or heuristic that should be revised, with a specific recommended value.
   - If removing: a brief sketch of which functions / lines / files would go away.
7. **Reproducibility** — pointer to `dev/scripts/investigations/perf_features_investigation/README.md`.

The figures referenced are in `dev/scripts/investigations/perf_features_investigation/figures/` (gitignored). Either copy them into a committed `dev/docs/figures/perf_features/` directory, or reference them from the source location with a note that they need to be regenerated. **Decision rule: copy the four heatmaps into `dev/docs/figures/perf_features/` and reference those committed copies, so the findings doc renders correctly on GitHub.**

```bash
<activate> && cd /home/arr65/src/Vs30 && \
mkdir -p dev/docs/figures/perf_features && \
cp dev/scripts/investigations/perf_features_investigation/figures/*.png dev/docs/figures/perf_features/
```

The findings doc itself: write the actual narrative based on the data. Example structure for one feature:

```markdown
## Multiprocessing of the per-pixel MVN loop

### Phase 1 results

![Multiproc speedup, ffap=ON](figures/perf_features/multiproc_speedup_ffap_on.png)

**Heatmap reading:** values > 1.0 indicate `nproc=8` (multiproc + single-threaded BLAS) is faster than `nproc=1` (BLAS multi-threaded).

**Observed pattern:** [actual pattern from data — fill in].

**Crossover:** [where does speedup cross 1.0? quote actual numbers].

### Phase 2 confirmation

| Cohort | nproc=1 (s) | nproc=8 (s) | Phase 1 winner | Phase 2 winner | Match? |
|---|---|---|---|---|---|
| sparse_coarse | … | … | … | … | … |
| …             | … | … | … | … | … |

### Recommendation

[One of:
- "**Remove the multiproc path.** It is slower than `nproc=1` in every regime tested. The complexity in `parallel.run_parallel_spatial_fit`, the `MULTIPROCESS_OBSERVATION_THRESHOLD` guard, and the `single_threaded_blas` plumbing can all go."
- "**Keep multiproc, revise the threshold to N.** It wins for N_obs < N and N_grid > M. Update `MULTIPROCESS_OBSERVATION_THRESHOLD` from 1000 to N; consider also gating on N_grid."
- "**Keep multiproc as-is.** The current threshold is well-calibrated."
]
```

- [ ] **Step 3: Commit**

```bash
cd /home/arr65/src/Vs30 && git add dev/docs/perf_features_investigation_findings.md dev/docs/figures/perf_features/ && git commit -m "$(cat <<'EOF'
docs: add perf features investigation findings

Findings from the empirical investigation of multiprocessing and
find_affected_pixels in the Vs30 grid pipeline. Includes per-feature
recommendations backed by Phase 1 isolated-MVN sweep data and Phase 2
end-to-end confirmation runs.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 14: Final review

- [ ] **Step 1: Re-read the findings doc end to end**

Confirm:
- Every claim in the Summary is supported by data in the body.
- Every recommendation is concrete (a code change one could implement) and points to specific files / lines.
- The doc renders correctly on GitHub (images load).

- [ ] **Step 2: Re-run all unit tests**

```bash
<activate> && cd /home/arr65/src/Vs30 && pytest dev/scripts/investigations/perf_features_investigation/test_bench_utils.py -v
```

Expected: all tests pass.

- [ ] **Step 3: Confirm no production code was modified**

```bash
cd /home/arr65/src/Vs30 && git log --stat HEAD~10..HEAD -- vs30/ tests/
```

Expected: empty (or no commits in the investigation branch touch `vs30/` or `tests/`). If something was changed in production code, evaluate whether it was intentional and either revert it or note it explicitly in the findings doc.

- [ ] **Step 4: Notify the user**

Report: investigation complete, recommendations listed, all artefacts committed.

---

## Self-review

**Spec coverage:** every section of `perf_features_investigation_design.md` maps to at least one task:
- §1 Purpose → Task 13 (findings doc)
- §2 Scope → enforced throughout (grid mode only; production code untouched per Task 14 Step 3)
- §3 Background → referenced in findings doc
- §4 Methodology → Tasks 8/9 (Phase 1), Tasks 11/12 (Phase 2)
- §5 Test matrix → Task 8 (driver) + Task 9 (execution)
- §6 Harness structure → Tasks 1–8
- §7 Numerical equivalence guardrail → Task 7 (impl), called inside Task 8
- §8 Outputs → Tasks 10 (analysis), 13 (findings doc)
- §9 Risks → mitigations are baked into the relevant tasks (PER_CELL_TIME_BUDGET_S in Task 8; bypass_observation_threshold in Tasks 6 & 7)
- §10 Branch and commit strategy → each task has its own commit; production code untouched

**Placeholder scan:** every code block contains real, runnable code. Findings-doc narrative is templated with the structure spelled out and a decision rule for what "fill in" means in context (paste actual numbers from CSV).

**Type consistency:** `BoundingBoxResult`, `RasterData`, `ObservationData` all use the production types from `vs30.spatial`; helper signatures in tasks 4–7 line up with their callers in tasks 8 and 11.

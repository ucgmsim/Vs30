# Grid Bounds Alignment Fix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Eliminate the 50 m systematic misalignment between the refactored
Vs30 grid pipeline output and the bundled `IwahashiPike.tif` source raster
by changing `FULL_NZ_GRID_CONFIG` and `BENCHMARK_NZ_GRID` to bounds whose
pixel centres land exactly on IwahashiPike pixel centres, and fix related
incorrect docstrings/comments.

**Architecture:** The refactored code uses pixel-edge convention (proven by
`vs30/raster.py:238-240`'s `nx = round((xmax-xmin)/dx)` formula and use of
`rasterio.transform.from_bounds`). Currently `FULL_NZ_GRID_CONFIG` has
`grid_xmin=1060050` etc., which under pixel-edge convention puts every
output pixel centre exactly between two IwahashiPike pixels — forcing GDAL
nearest-neighbour to break a tie at every pixel. This change picks bounds
that produce centres at IwahashiPike centre coordinates (`x mod 100 == 50`,
`y mod 100 == 50`), removing the tie. For `dx=100` the correct `xmin` ends
in `..00` (e.g. `1060100`); for `dx=5000` the correct `xmin` ends in `..50`
(e.g. `1060050`). The two grids therefore land on different `xmin` values
by design — both correct, for different reasons. Benchmark TIFFs must be
regenerated from each legacy codebase at the new bounds so the
`test_benchmarks.py` comparison is meaningful.

**Tech Stack:** Python (vs30 package), pytest, rasterio, GDAL, conda
(`vs30_venv` for refactored, `oldvs30_venv` for legacy).

**Pre-reading (the executing agent MUST read these before starting):**
- `dev/docs/grid_bounds_semantics_investigation.md` — full evidence and
  rationale. Especially §3 (proof of pixel-edge semantics), §4 (proof of
  IwahashiPike misalignment), and §6 (cross-codebase comparison table).
- `dev/docs/generating_benchmarks_from_legacy_code.md` — exact commands
  for legacy benchmark regeneration. **The bounds documented in that file
  are about to change as part of this plan; the agent must update the
  document AND use the new bounds when regenerating.**
- `dev/CLAUDE.md` — environment activation pattern. Always
  `source .../conda.sh && source .../mamba.sh && mamba activate vs30_venv`.
  For legacy code, use the full path to `oldvs30_venv`'s Python binary
  (`/home/arr65/miniforge-pypy3/envs/oldvs30_venv/bin/python3`); do NOT
  `mamba activate oldvs30_venv` (it doesn't work reliably in non-interactive
  shells).
- Auto-memory facts that apply: legacy code deadlocks with `--nproc>1` —
  always pass `--nproc 1` to legacy CLIs.

**Branch:** `grid-bounds-alignment-fix` (off `vs30_refactor`). Work in a
git worktree.

---

## File Structure

| File                                                            | Responsibility                                                  | Action     |
|-----------------------------------------------------------------|-----------------------------------------------------------------|------------|
| `tests/test_grid_alignment.py`                                  | NEW: parametrised test asserting both reference grids' centres land on IwahashiPike centres. | Create     |
| `vs30/constants.py`                                             | `FULL_NZ_GRID_CONFIG` definition + comment                       | Modify     |
| `tests/test_benchmarks.py`                                      | `BENCHMARK_NZ_GRID` definition                                   | Modify     |
| `vs30/gapfill.py`                                               | `create_local_grid_config` snap formula + docstring              | Modify     |
| `dev/docs/generating_benchmarks_from_legacy_code.md`            | Legacy regen commands (bounds + table)                           | Modify     |
| `tests/benchmarks/jaehwi_v1p0.tif`                              | Regenerated via legacy jaehwi_fork code at new bounds            | Regenerate |
| `tests/benchmarks/modified_foster_2019.tif`                     | Regenerated via legacy pre-refactor code at new bounds           | Regenerate |
| `tests/benchmarks/viktor_cpt_clustering.tif`                    | Regenerated via legacy pre-refactor code at new bounds           | Regenerate |

Files NOT touched (intentionally): `tests/benchmarks/foster_2019_approx_points.csv`
(points pipeline samples at exact coordinates, unaffected by grid alignment).

---

## Task 1: Add alignment regression test (initially failing)

**Files:**
- Create: `tests/test_grid_alignment.py`

- [ ] **Step 1.1: Write the test file**

```python
"""
Test that the canonical grid configurations used in production and
benchmarks produce pixel centres aligned with IwahashiPike pixel centres.

If a grid's pixel centres land between IwahashiPike centres, GDAL's
nearest-neighbour resampling has to break a tie at every pixel — which is
non-deterministic across GDAL versions and produces ~22 % terrain ID
differences vs the IwahashiPike-aligned grid (see
dev/docs/grid_bounds_semantics_investigation.md §4.1).

These tests pin the alignment as a contract.
"""

import pytest
import rasterio

from vs30 import config, constants


def _assert_grid_aligned_with_iwahashipike(grid: config.GridConfig) -> None:
    """Assert every pixel centre of the grid lands on an IwahashiPike pixel centre."""
    iw_path = constants.GEOSPATIAL_DIR / constants.TERRAIN_RASTER_FILENAME
    with rasterio.open(iw_path) as src:
        iw_t = src.transform
        iw_dx = iw_t.a
        iw_dy = abs(iw_t.e)

    nx = round((grid.grid_xmax - grid.grid_xmin) / grid.grid_dx)
    ny = round((grid.grid_ymax - grid.grid_ymin) / grid.grid_dy)
    grid_t = rasterio.transform.from_bounds(
        grid.grid_xmin, grid.grid_ymin, grid.grid_xmax, grid.grid_ymax, nx, ny
    )

    # Pixel (0,0) centre in real-world coords.
    ul_centre_x = grid_t.c + grid_t.a / 2
    ul_centre_y = grid_t.f + grid_t.e / 2

    # Express that centre as a fractional pixel coordinate within IwahashiPike.
    # An IwahashiPike pixel centre has fractional coord = integer + 0.5.
    iw_col_f = (ul_centre_x - iw_t.c) / iw_dx
    iw_row_f = (iw_t.f - ul_centre_y) / iw_dy

    col_offset_pixels = abs(iw_col_f - (round(iw_col_f - 0.5) + 0.5))
    row_offset_pixels = abs(iw_row_f - (round(iw_row_f - 0.5) + 0.5))

    assert col_offset_pixels < 1e-6, (
        f"Grid pixel (0,0) CENTRE x = {ul_centre_x} is offset by "
        f"{col_offset_pixels * iw_dx:.1f} m from the nearest IwahashiPike pixel CENTRE. "
        f"This causes GDAL's nearest-neighbour resampling to tie at every pixel."
    )
    assert row_offset_pixels < 1e-6, (
        f"Grid pixel (0,0) CENTRE y = {ul_centre_y} is offset by "
        f"{row_offset_pixels * iw_dy:.1f} m from the nearest IwahashiPike pixel CENTRE."
    )


def test_full_nz_grid_config_aligned_with_iwahashipike():
    """The production NZ-wide grid must be IwahashiPike-aligned."""
    _assert_grid_aligned_with_iwahashipike(constants.FULL_NZ_GRID_CONFIG)


def test_benchmark_nz_grid_aligned_with_iwahashipike():
    """The 5 km benchmark grid must also be IwahashiPike-aligned."""
    # pytest adds tests/ to sys.path, matching the existing convention used
    # in tests/test_grid_points_consistency.py:20 and tests/test_benchmarks.py:30.
    from test_benchmarks import BENCHMARK_NZ_GRID
    _assert_grid_aligned_with_iwahashipike(BENCHMARK_NZ_GRID)
```

- [ ] **Step 1.2: Run the test — both should FAIL**

Run:
```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv && \
pytest tests/test_grid_alignment.py -v
```

Expected: BOTH tests fail with messages like
`Grid pixel (0,0) CENTRE x = 1060100.0 is offset by 50.0 m from the nearest IwahashiPike pixel CENTRE.`

If they pass, something is wrong — re-read `dev/docs/grid_bounds_semantics_investigation.md` §4 to confirm the misalignment exists, and inspect the test logic. Do not proceed.

- [ ] **Step 1.3: Commit**

```bash
git add tests/test_grid_alignment.py
git commit -m "test: pin grid alignment to IwahashiPike (currently failing)

The two canonical grid configs (FULL_NZ_GRID_CONFIG and BENCHMARK_NZ_GRID)
both produce pixel centres 50 m offset from IwahashiPike pixel centres.
This adds the test that asserts the alignment we want; the next two
commits fix the grid bounds to make it pass.

See dev/docs/grid_bounds_semantics_investigation.md."
```

---

## Task 2: Fix `FULL_NZ_GRID_CONFIG`

**Files:**
- Modify: `vs30/constants.py:255-269`

- [ ] **Step 2.1: Update the bounds and the comment**

Replace the existing block

```python
# Full New Zealand land extent at standard 100m resolution.
# IMPORTANT: These bounds define the canonical NZ domain and MUST NOT be changed.
# Used for coastal distance calculations, gap-fill grid alignment, and CLI defaults.
# The xmin/xmax/ymin/ymax values are pixel centres (not pixel edges) on the
# 100m NZTM grid, so they are offset by 50m (half a cell) from round-number
# corners. This convention keeps pixel-centre arithmetic clean and avoids
# sub-pixel shifts when resampling.
FULL_NZ_GRID_CONFIG: config.GridConfig = config.GridConfig(
    grid_xmin=1060050,
    grid_xmax=2120050,
    grid_ymin=4730050,
    grid_ymax=6250050,
    grid_dx=100,
    grid_dy=100,
)
```

with

```python
# Full New Zealand land extent at standard 100m resolution.
# Used for coastal distance calculations, gap-fill grid alignment, and CLI defaults.
#
# Convention: xmin/xmax/ymin/ymax are PIXEL EDGES (outer bounds), per
# rasterio.transform.from_bounds() and GDAL outputBounds. The number of
# pixels is (xmax-xmin)/dx, and pixel CENTRES are at xmin + dx/2 + n*dx.
#
# These specific bounds are chosen so that pixel CENTRES (1060150, 1060250,
# ..., 2120050 in x) coincide exactly with the bundled IwahashiPike.tif
# pixel centres (which end in ..50 in both axes — see
# dev/docs/grid_bounds_semantics_investigation.md). This avoids GDAL's
# nearest-neighbour tie-break at every pixel during terrain resampling.
FULL_NZ_GRID_CONFIG: config.GridConfig = config.GridConfig(
    grid_xmin=1060100,
    grid_xmax=2120100,
    grid_ymin=4730100,
    grid_ymax=6250100,
    grid_dx=100,
    grid_dy=100,
)
```

- [ ] **Step 2.2: Run the FULL_NZ alignment test — should PASS now**

Run:
```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv && \
pytest tests/test_grid_alignment.py::test_full_nz_grid_config_aligned_with_iwahashipike -v
```

Expected: PASS.
The benchmark test should still FAIL (we fix it in the next task).

- [ ] **Step 2.3: Commit**

```bash
git add vs30/constants.py
git commit -m "fix(constants): align FULL_NZ_GRID_CONFIG with IwahashiPike pixel centres

Change xmin/xmax/ymin/ymax from ..050 to ..100 so that pixel centres
(at xmin + dx/2 + n*dx with dx=100) land exactly on IwahashiPike pixel
centres (which end in ..50). Eliminates the per-pixel tie in GDAL's
nearest-neighbour terrain resampling.

The recently-added comment claiming the bounds were 'pixel centres'
was incorrect — see dev/docs/grid_bounds_semantics_investigation.md
for the full evidence.

This shifts the canonical NZ domain by 50 m east and 50 m north
relative to the previous (legacy-inherited) misaligned grid. The
shape of the output (10600 x 15200 pixels at 100 m) is unchanged."
```

---

## Task 3: Fix `BENCHMARK_NZ_GRID`

**Files:**
- Modify: `tests/test_benchmarks.py:38-45`

- [ ] **Step 3.1: Update the bounds**

Replace

```python
# Shared grid for modified_foster_2019, jaehwi_v1p0, and viktor_cpt_clustering.
BENCHMARK_NZ_GRID = config.GridConfig(
    grid_xmin=1060100,
    grid_xmax=2120100,
    grid_ymin=4730100,
    grid_ymax=6250100,
    grid_dx=5000,
    grid_dy=5000,
)
```

with

```python
# Shared grid for modified_foster_2019, jaehwi_v1p0, and viktor_cpt_clustering.
#
# At dx=dy=5000, pixel CENTRES sit at xmin + 2500 + n*5000. To land each
# centre exactly on an IwahashiPike pixel centre (which is at coordinates
# ending in ..50 in both axes), xmin/ymin must end in ..50 — different
# from FULL_NZ_GRID_CONFIG's ..100 (which is correct for dx=100). See
# dev/docs/grid_bounds_semantics_investigation.md.
BENCHMARK_NZ_GRID = config.GridConfig(
    grid_xmin=1060050,
    grid_xmax=2120050,
    grid_ymin=4730050,
    grid_ymax=6250050,
    grid_dx=5000,
    grid_dy=5000,
)
```

- [ ] **Step 3.2: Run both alignment tests — both should PASS**

Run:
```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv && \
pytest tests/test_grid_alignment.py -v
```

Expected: both PASS.

- [ ] **Step 3.3: Commit**

```bash
git add tests/test_benchmarks.py
git commit -m "fix(tests): align BENCHMARK_NZ_GRID with IwahashiPike pixel centres

Change xmin/xmax/ymin/ymax from ..100 to ..050 so that 5 km pixel
centres (at xmin + 2500 + n*5000) land exactly on IwahashiPike pixel
centres (which end in ..50). The xmin convention differs from
FULL_NZ_GRID_CONFIG's ..100 because at dx=5000 the xmin offset that
gives IwahashiPike-aligned centres is different — both grids are now
aligned, just via different xmin values.

The benchmark TIFFs themselves still need to be regenerated from
legacy code at the new bounds; that happens in subsequent commits.
Tests will fail until benchmarks are regenerated."
```

---

## Task 4: Fix `gapfill.create_local_grid_config` snap formula and docstring

**Background:** The `snap_e` formula
`grid_xmin + round((easting - grid_xmin) / dx) * dx` snaps to grid pixel
*edges* (because `grid_xmin` is a pixel edge under the codebase's
convention). The docstring claims it snaps to pixel centres. After the
Task 2 change, `grid_xmin=1060100` and the snap produces values ending in
`..00` — but FULL_NZ pixel centres end in `..50`, so the local grid
created from `snap_e ± half_width` is offset by `dx/2` from the FULL_NZ
grid. The fix is to snap to true pixel centres.

**Files:**
- Modify: `vs30/gapfill.py:207-254`

- [ ] **Step 4.1: Update the function**

Replace the body of `create_local_grid_config` (and its docstring) with:

```python
def create_local_grid_config(
    easting: float,
    northing: float,
    gapfill_grid_config: config.GridConfig,
    half_width: int,
) -> config.GridConfig:
    """
    Create a local grid config for gap-filling a single point.

    Snaps the point to the nearest pixel CENTRE in the reference grid
    (where pixel centres are at ``grid_xmin + grid_dx/2 + n*grid_dx``,
    per the codebase's pixel-edge bounds convention), then creates a
    local grid of size (2 * half_width) on each side. The local grid's
    centre pixel CENTRE coincides with that snapped point, so it shares
    the same pixel-centre lattice as the reference grid.

    Parameters
    ----------
    easting : float
        Query point easting (NZTM).
    northing : float
        Query point northing (NZTM).
    gapfill_grid_config : GridConfig
        Reference grid config defining the pixel alignment.
    half_width : int
        Half-width of the local grid in meters. Must be a multiple of
        ``grid_dx`` plus ``grid_dx / 2`` (e.g. 150 m for dx=100 m, 250 m
        for dx=100 m, …) so the local grid's outer bounds remain pixel
        edges.

    Returns
    -------
    GridConfig
        Local grid config aligned to the reference grid.
    """
    # Pixel centres of the reference grid lie at grid_xmin + dx/2 + n*dx.
    # Snap the query point to the nearest such centre.
    half_dx = gapfill_grid_config.grid_dx / 2
    half_dy = gapfill_grid_config.grid_dy / 2
    first_centre_x = gapfill_grid_config.grid_xmin + half_dx
    first_centre_y = gapfill_grid_config.grid_ymin + half_dy

    snap_e = first_centre_x + round((easting - first_centre_x) / gapfill_grid_config.grid_dx) * gapfill_grid_config.grid_dx
    snap_n = first_centre_y + round((northing - first_centre_y) / gapfill_grid_config.grid_dy) * gapfill_grid_config.grid_dy

    # Build the local grid. Bounds are pixel EDGES, so the local grid's
    # centre pixel has its centre at snap_e/snap_n exactly.
    return config.GridConfig(
        grid_xmin=snap_e - half_width,
        grid_xmax=snap_e + half_width,
        grid_ymin=snap_n - half_width,
        grid_ymax=snap_n + half_width,
        grid_dx=gapfill_grid_config.grid_dx,
        grid_dy=gapfill_grid_config.grid_dy,
    )
```

- [ ] **Step 4.2: Run the fast grid/points consistency tests**

Run:
```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv && \
pytest tests/test_grid_points_consistency.py -v
```

Expected: PASS for both fast-tier model versions.

If the tests fail, the most likely cause is that the `LOCAL_GRID_HALF_WIDTH = 150` in `tests/test_grid_points_consistency.py:39` is no longer compatible with the new snap behaviour. Check whether the local grid's bounds still produce a 3×3 grid containing the snapped centre. If the half-width math no longer works (it should still work — `snap_e ± 150` still has a 3-pixel-wide pixel-edge span), debug from there. Do not proceed until tests pass.

- [ ] **Step 4.3: Run the unit/raster tests as a regression guard**

Run:
```bash
pytest tests/test_gapfill.py tests/test_raster.py tests/test_spatial.py -v
```

Expected: all PASS.

- [ ] **Step 4.4: Commit**

```bash
git add vs30/gapfill.py
git commit -m "fix(gapfill): snap to true pixel centres, not pixel edges

create_local_grid_config previously computed snap_e as
grid_xmin + n*grid_dx, which under pixel-edge bounds convention is a
pixel EDGE, not a pixel centre. The docstring claimed pixel-centre
behaviour; the implementation didn't deliver it. The local 3x3 grid
this builds (used by test_grid_points_consistency) was therefore
offset by dx/2 from the reference FULL_NZ grid, which previously
masked the IwahashiPike misalignment by accidentally aligning local
grids with IwahashiPike via this offset.

After the FULL_NZ_GRID_CONFIG fix, both FULL_NZ and the local grids
need to share a centre lattice — this commit makes snap_e land on
true FULL_NZ pixel centres."
```

---

## Task 5: Update legacy benchmark regeneration doc

**Files:**
- Modify: `dev/docs/generating_benchmarks_from_legacy_code.md`

- [ ] **Step 5.1: Update the bounds table and command snippets**

In the "Grid Bounds" section, replace the BENCHMARK_NZ_GRID row of the
table with:

```
| `BENCHMARK_NZ_GRID` | 1060050 | 2120050 | 4730050 | 6250050 | modified_foster_2019, jaehwi_v1p0, viktor_cpt_clustering |
```

Then in each of the three model sections, replace the
`--xmin 1060100 --xmax 2120100 --ymin 4730100 --ymax 6250100` flags with:

```
--xmin 1060050 --xmax 2120050 --ymin 4730050 --ymax 6250050
```

Also delete the now-stale "Coordinate note" paragraph at the bottom of
the jaehwi_v1p0 section (the one that says "Jaehwi's fork has different
default grid bounds…"), since the explicit bounds make it moot.

Add a sentence to the "Grid Bounds" intro paragraph noting the change:

> These bounds were updated 2026-05-02 from `..100` to `..050` so that
> the 5 km pixel centres land on IwahashiPike pixel centres (centres at
> coordinates ending in `..50`), eliminating GDAL nearest-neighbour
> tie-break ambiguity. See
> `dev/docs/grid_bounds_semantics_investigation.md`.

- [ ] **Step 5.2: Commit**

```bash
git add dev/docs/generating_benchmarks_from_legacy_code.md
git commit -m "docs: update legacy benchmark regen doc with new bounds

Reflects the BENCHMARK_NZ_GRID change from ..100 to ..050. The legacy
codes accept any bounds via CLI flags; using the new aligned bounds
produces benchmarks that the refactored pipeline can reproduce
without GDAL tie-break drift."
```

---

## Task 6: Regenerate `tests/benchmarks/modified_foster_2019.tif`

**Files:**
- Regenerate: `tests/benchmarks/modified_foster_2019.tif`

- [ ] **Step 6.1: Run legacy code with the new bounds**

```bash
cd /home/arr65/src/pre-refactor-Vs30-for-comparison && \
PATH="/home/arr65/miniforge-pypy3/envs/oldvs30_venv/bin:$PATH" \
/home/arr65/miniforge-pypy3/envs/oldvs30_venv/bin/python3 run_vs30calc.py \
    --source original \
    --gupdate posterior_paper --tupdate posterior_paper \
    --xmin 1060050 --xmax 2120050 --ymin 4730050 --ymax 6250050 \
    --dx 5000 --dy 5000 \
    --out /tmp/bench_modified_foster_2019 \
    --nproc 1 --overwrite
```

Expected: completes in ~30-60 s. Final stdout shows the path to
`combined_mvn.tif` or similar.

- [ ] **Step 6.2: Verify the output alignment**

Run:
```bash
gdalinfo /tmp/bench_modified_foster_2019/combined_mvn.tif | grep -E "^(Origin|Pixel Size|Size)\b"
```

Expected:
```
Size is 212, 304
Origin = (1060050.000000000000000,6250050.000000000000000)
Pixel Size = (5000.000000000000000,-5000.000000000000000)
```

If the origin is anything else, do not copy — re-check the legacy command flags.

- [ ] **Step 6.3: Copy into place**

```bash
cp /tmp/bench_modified_foster_2019/combined_mvn.tif \
   /home/arr65/src/Vs30/tests/benchmarks/modified_foster_2019.tif
```

(Adjust the source filename if `combined_mvn.tif` is not what the legacy
code wrote — check the directory listing first with `ls /tmp/bench_modified_foster_2019/`.)

- [ ] **Step 6.4: Run the modified_foster_2019 benchmark test**

```bash
cd /home/arr65/src/Vs30 && \
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv && \
pytest tests/test_benchmarks.py::test_modified_foster_2019 -v
```

Expected: PASS.

If the test fails, capture the failure mode:
- "Valid data masks differ" → mask mismatch; the legacy and refactored
  pipelines disagree about which pixels are nodata. Investigate before
  proceeding.
- "Data values differ beyond tolerance" → numerical drift exceeding
  `TEST_RTOL = 1e-3`. Compare a few specific pixels via `rasterio` to
  understand the magnitude. May indicate that the algorithm reacts to
  alignment changes more than expected; flag for human review.

Do not proceed to the next benchmark until this one passes.

- [ ] **Step 6.5: Commit**

```bash
git add tests/benchmarks/modified_foster_2019.tif
git commit -m "test(benchmarks): regen modified_foster_2019 at new bounds

Generated by pre-refactor legacy code at xmin=1060050 xmax=2120050
ymin=4730050 ymax=6250050 dx=dy=5000 — the new IwahashiPike-aligned
BENCHMARK_NZ_GRID. test_modified_foster_2019 passes against this
benchmark."
```

---

## Task 7: Regenerate `tests/benchmarks/viktor_cpt_clustering.tif`

**Files:**
- Regenerate: `tests/benchmarks/viktor_cpt_clustering.tif`

- [ ] **Step 7.1: Run legacy code with the new bounds**

```bash
cd /home/arr65/src/pre-refactor-Vs30-for-comparison && \
PATH="/home/arr65/miniforge-pypy3/envs/oldvs30_venv/bin:$PATH" \
/home/arr65/miniforge-pypy3/envs/oldvs30_venv/bin/python3 run_vs30calc.py \
    --source cpt \
    --gupdate posterior --tupdate posterior \
    --xmin 1060050 --xmax 2120050 --ymin 4730050 --ymax 6250050 \
    --dx 5000 --dy 5000 \
    --out /tmp/bench_viktor_cpt_clustering \
    --nproc 1 --overwrite
```

- [ ] **Step 7.2: Verify alignment**

```bash
gdalinfo /tmp/bench_viktor_cpt_clustering/combined_mvn.tif | grep -E "^(Origin|Pixel Size|Size)\b"
```

Expected `Origin = (1060050, 6250050)`, `Size is 212, 304`.

- [ ] **Step 7.3: Copy into place**

```bash
cp /tmp/bench_viktor_cpt_clustering/combined_mvn.tif \
   /home/arr65/src/Vs30/tests/benchmarks/viktor_cpt_clustering.tif
```

- [ ] **Step 7.4: Run the viktor_cpt_clustering benchmark test**

```bash
cd /home/arr65/src/Vs30 && \
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv && \
pytest tests/test_benchmarks.py::test_viktor_cpt_clustering -v
```

Expected: PASS.

- [ ] **Step 7.5: Commit**

```bash
git add tests/benchmarks/viktor_cpt_clustering.tif
git commit -m "test(benchmarks): regen viktor_cpt_clustering at new bounds

Generated by pre-refactor legacy code with --source cpt --gupdate posterior
--tupdate posterior at the new IwahashiPike-aligned bounds."
```

---

## Task 8: Regenerate `tests/benchmarks/jaehwi_v1p0.tif`

**Files:**
- Regenerate: `tests/benchmarks/jaehwi_v1p0.tif`

- [ ] **Step 8.1: Run legacy jaehwi_fork code**

```bash
cd /home/arr65/src/jaehwi_fork_vs30/Vs30_2026 && \
PATH="/home/arr65/miniforge-pypy3/envs/oldvs30_venv/bin:$PATH" \
/home/arr65/miniforge-pypy3/envs/oldvs30_venv/bin/python3 run_vs30calc_V1.py \
    --gupdate posterior --tupdate posterior \
    --xmin 1060050 --xmax 2120050 --ymin 4730050 --ymax 6250050 \
    --dx 5000 --dy 5000 \
    --out /tmp/bench_jaehwi_v1p0 \
    --nproc 1 --overwrite
```

- [ ] **Step 8.2: Verify alignment**

```bash
gdalinfo /tmp/bench_jaehwi_v1p0/combined_mvn.tif | grep -E "^(Origin|Pixel Size|Size)\b"
```

Expected `Origin = (1060050, 6250050)`, `Size is 212, 304`.

- [ ] **Step 8.3: Copy into place and apply gap-fill**

```bash
cp /tmp/bench_jaehwi_v1p0/combined_mvn.tif \
   /home/arr65/src/Vs30/tests/benchmarks/jaehwi_v1p0.tif

cd /home/arr65/src/Vs30 && \
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv && \
python dev/scripts/generators/gapfill_benchmark.py \
   tests/benchmarks/jaehwi_v1p0.tif
```

(Per `dev/docs/generating_benchmarks_from_legacy_code.md`, the jaehwi
fork doesn't fill nodata gaps but the refactored pipeline does, so the
benchmark must be gap-filled to match.)

- [ ] **Step 8.4: Run the jaehwi_v1p0 benchmark test**

```bash
pytest tests/test_benchmarks.py::test_jaehwi_v1p0 -v
```

Expected: PASS.

- [ ] **Step 8.5: Commit**

```bash
git add tests/benchmarks/jaehwi_v1p0.tif
git commit -m "test(benchmarks): regen jaehwi_v1p0 at new bounds

Generated by jaehwi_fork legacy code (run_vs30calc_V1.py with --gupdate
posterior --tupdate posterior) at the new IwahashiPike-aligned bounds,
then post-processed with dev/scripts/generators/gapfill_benchmark.py
to match the refactored pipeline's gap-fill behaviour."
```

---

## Task 9: Run the full default-tier test suite

- [ ] **Step 9.1: Run pytest tests/**

```bash
cd /home/arr65/src/Vs30 && \
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv && \
pytest tests/ -v
```

Expected: ALL pass (3 alignment tests + 4 benchmarks + unit tests + the
fast-tier `test_grid_points_consistency_fast` parametrize).

If any tests fail, do NOT mark this task complete. Investigate. Likely
suspects:
- `test_foster_2019_approx_points_benchmark` — should be unaffected by
  the alignment change (it uses points pipeline, no grid). If it fails,
  the alignment fix has a side-effect we didn't anticipate; report.
- `test_grid_points_consistency_fast` — covered in Task 4 already, but
  re-check.

- [ ] **Step 9.2: No commit needed**

This task is verification-only; no files change.

---

## Task 10: Run the slow-tier test suite

Required. **Takes ~40 minutes on this workstation** — well over the
Bash tool's 10 min max foreground timeout, so it MUST be launched with
`run_in_background=true`. The PR cannot open until this passes;
slow-tier failures are the most likely place a subtle alignment
regression would show up, since they exercise points across the full
domain.

- [ ] **Step 10.1: Launch --runslow in the background**

Bash tool call:
- command:
  ```bash
  cd /home/arr65/src/Vs30 && \
  source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
  source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
  mamba activate vs30_venv && \
  pytest tests/ --runslow -v 2>&1 | tee /tmp/runslow_output.log
  ```
- `run_in_background`: **`true`** (mandatory — foreground will time out)

Expected: command starts immediately and you receive a notification
when it completes (~40 min).

While the slow run is in flight: do NOT poll, sleep, or busy-wait.
You will be auto-notified on completion. If you have unrelated work
to do, do it; otherwise just wait for the notification.

- [ ] **Step 10.2: Inspect the result on completion**

When the background task notifies completion, read the tail of the log:

```bash
tail -100 /tmp/runslow_output.log
```

Expected to find: a final pytest summary line like
`====== N passed in MMM.MMs ======` with no failures.

If failures occur, do NOT proceed to Task 11. Investigate likely
suspects:
- `test_grid_points_consistency_slow[<version>]` — a point in the
  parametrize set lands somewhere where the new alignment changes
  the categorical lookup vs the points pipeline. Inspect the failing
  point's coordinates against `tests/fixtures/consistency_test_points.csv`
  and the QMAP/IwahashiPike rasters at that location.
- A benchmark test failure here that didn't fail in Task 9 implies
  flakiness — re-run once to confirm before debugging.

- [ ] **Step 10.3: No commit needed.**

---

## Task 11: Open PR

- [ ] **Step 11.1: Push branch**

```bash
git push -u origin grid-bounds-alignment-fix
```

- [ ] **Step 11.2: Create the PR**

Title: `fix: align grid configs with IwahashiPike pixel centres`

Body:

```markdown
## Summary

- Eliminates the 50 m systematic misalignment between `FULL_NZ_GRID_CONFIG`
  / `BENCHMARK_NZ_GRID` and the bundled `IwahashiPike.tif`. Every
  refactored output pixel previously sat on an IwahashiPike pixel boundary,
  forcing GDAL nearest-neighbour to break a tie at every pixel.
- Adds a regression test (`tests/test_grid_alignment.py`) that pins the
  alignment as a contract.
- Fixes the `gapfill.create_local_grid_config` snap formula, which was
  snapping to pixel edges despite the docstring claiming pixel centres.
- Regenerates the three 5 km benchmark TIFFs from the legacy codebases
  at the new aligned bounds.

## Context

See `dev/docs/grid_bounds_semantics_investigation.md` for the full
evidence (multi-codebase comparison, empirical impact: ~22 % of
Christchurch terrain IDs flip between aligned and misaligned grids).

## Test plan

- [x] `pytest tests/test_grid_alignment.py -v`
- [x] `pytest tests/test_benchmarks.py -v`
- [x] `pytest tests/test_grid_points_consistency.py -v` (fast tier)
- [x] `pytest tests/` (default tier full)
- [x] `pytest tests/ --runslow` (slow tier — full 38-point grid/points
  consistency across all 4 model versions)

## Out of scope (for follow-up)

- NZ_Vs30 HDF5 regeneration in `/home/arr65/src/nzcvm_data/` — the
  HDF5 files there bake in the old `..050` grid; they will need to be
  regenerated, but that is a separate PR in a separate repo.
- `foster_2019_approx` 100 m grid benchmark — currently benchmarked at
  points, not as a grid; alignment fix only improves it.
```

```bash
gh pr create --base vs30_refactor --title "fix: align grid configs with IwahashiPike pixel centres" --body "$(cat <<'EOF'
[paste body above]
EOF
)"
```

---

## Self-review — done after writing this plan

1. **Spec coverage:**
   - User wants `..100` bounds for the production grid → Task 2 ✓
   - User wants benchmarks regenerated to be valid → Tasks 6–8 ✓
   - User wants robust fix → tests pin contract; investigation doc
     stays as record ✓
   - User said NZ_Vs30 HDF5 is out of scope → no HDF5 task ✓
2. **Placeholders:** no TBD / TODO / "similar to Task N" / "implement
   later" found.
3. **Type consistency:** `BENCHMARK_NZ_GRID` is referenced in Task 1's
   test via `from tests.test_benchmarks import BENCHMARK_NZ_GRID`, and
   defined as `config.GridConfig(...)` consistent with how the existing
   `tests/test_benchmarks.py:38-45` defines it. `FULL_NZ_GRID_CONFIG` is
   referenced consistently throughout. The new `gapfill` function
   signature is unchanged from the existing one (only the body changes),
   so no callers break.

---

## Execution Handoff

Plan complete and saved to `dev/docs/grid_bounds_alignment_fix_plan.md`.

Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per
task, review between tasks, fast iteration with worktree isolation.

**2. Inline Execution** — Execute tasks in this session using
`superpowers:executing-plans`, with checkpoints for review.

Which approach?

# Grid-Points Consistency Test Expansion — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the limited 3-point consistency test with broad-coverage testing across the full NZ domain and all four model versions.

**Architecture:** A pre-computed CSV fixture defines ~40 test points (deliberate + random, all on-land). The test loads each model version config, runs `points_pipeline` once with all points batched, then runs `grid_pipeline` on a tiny 3x3 grid per point and compares the center pixel. Two pytest tiers (fast / slow) split by whether the model uses coastal distance computation.

**Tech Stack:** pytest (parametrize, markers), numpy, pandas, geopandas, shapely, rasterio, qcore.coordinates

---

## File Structure

```
tests/
  fixtures/
    consistency_test_points.csv           # Pre-computed test point coordinates (new)
  test_grid_points_consistency.py         # Rewritten test (replace existing)
dev/
  generate_consistency_test_points.py     # Script to generate/regenerate the fixture CSV (new)
pyproject.toml                            # Add [tool.pytest.ini_options] slow marker (modify)
```

---

### Task 1: Create the test point generation script

**Files:**
- Create: `dev/generate_consistency_test_points.py`

This script generates `tests/fixtures/consistency_test_points.csv`. It is run manually (not by pytest) and committed alongside its output. It selects deliberate points by looking up specific geology categories and geographic regions, then adds random on-land points.

- [ ] **Step 1: Write the generation script**

```python
"""
Generate the consistency test points CSV fixture.

Produces tests/fixtures/consistency_test_points.csv containing ~40 on-land
points used by test_grid_points_consistency.py. Run manually whenever the
point set needs updating:

    python dev/generate_consistency_test_points.py

Deliberate points target specific geology categories, coastal zones, cities,
observation-sparse areas, and geology boundaries. Random points fill the
remaining coverage uniformly across the NZ domain.
"""

from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import shapely
from qcore import coordinates

from vs30 import category, constants, raster

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "tests" / "fixtures"
OUTPUT_CSV = FIXTURES_DIR / "consistency_test_points.csv"

# Number of random points to generate (after land filtering)
N_RANDOM = 25


def _snap_to_pixel_center(easting: float, northing: float) -> tuple[float, float]:
    """Snap an NZTM coordinate to the nearest FULL_NZ_GRID_CONFIG pixel center.

    Pixel centers sit at grid_xmin + (j + 0.5) * dx. The snap logic in
    gapfill.create_local_grid_config snaps to grid_xmin + k * dx (grid edges).
    We use the same convention here so that when we later call
    create_local_grid_config, the center pixel of the 3x3 grid lands exactly
    on our snapped coordinate.
    """
    nz = constants.FULL_NZ_GRID_CONFIG
    snap_e = nz.grid_xmin + round((easting - nz.grid_xmin) / nz.grid_dx) * nz.grid_dx
    snap_n = nz.grid_ymin + round((northing - nz.grid_ymin) / nz.grid_dy) * nz.grid_dy
    return snap_e, snap_n


def _nztm_to_wgs84(easting: float, northing: float) -> tuple[float, float]:
    """Convert a single NZTM point to WGS84 (lat, lon)."""
    result = coordinates.nztm_to_wgs_depth(np.array([[northing, easting]]))
    return float(result[0, 0]), float(result[0, 1])


def _wgs84_to_nztm(lat: float, lon: float) -> tuple[float, float]:
    """Convert a single WGS84 point to NZTM (easting, northing)."""
    result = coordinates.wgs_depth_to_nztm(np.array([[lat, lon]]))
    return float(result[0, 1]), float(result[0, 0])


def _load_coastline() -> shapely.Geometry:
    """Load and union the NZ coastline polygon."""
    coastline_path = constants.GEOSPATIAL_DIR / constants.COASTLINE_SHAPEFILE_PATH
    raster.ensure_shapefile_extracted(coastline_path, "coast")
    coast_gdf = gpd.read_file(coastline_path)
    return coast_gdf.geometry.union_all()


def _geology_id_at(easting: float, northing: float) -> int:
    """Return the geology category ID at a single NZTM point."""
    ids = category.assign_to_category_geology(np.array([[easting, northing]]))
    return int(ids[0])


def _find_point_with_geology_id(
    target_gid: int,
    coast_union: shapely.Geometry,
    rng: np.random.Generator,
    max_attempts: int = 5000,
) -> tuple[float, float] | None:
    """Random-sample until we find an on-land point with the given geology ID."""
    nz = constants.FULL_NZ_GRID_CONFIG
    for _ in range(max_attempts):
        e = rng.uniform(nz.grid_xmin, nz.grid_xmax)
        n = rng.uniform(nz.grid_ymin, nz.grid_ymax)
        e, n = _snap_to_pixel_center(e, n)
        pt = shapely.Point(e, n)
        if not shapely.within(pt, coast_union):
            continue
        if _geology_id_at(e, n) == target_gid:
            return e, n
    return None


def generate_deliberate_points(
    coast_union: shapely.Geometry,
    rng: np.random.Generator,
) -> list[dict]:
    """Generate deliberately chosen test points."""
    points = []

    # --- Major cities (WGS84 lat/lon → NZTM → snap) ---
    cities = {
        "Auckland": (-36.8485, 174.7633),
        "Wellington": (-41.2865, 174.7762),
        "Christchurch": (-43.5321, 172.6362),
        "Dunedin": (-45.8788, 170.5028),
        "Hamilton": (-37.7870, 175.2793),
    }
    for name, (lat, lon) in cities.items():
        e, n = _wgs84_to_nztm(lat, lon)
        e, n = _snap_to_pixel_center(e, n)
        lat_s, lon_s = _nztm_to_wgs84(e, n)
        points.append(
            {"name": name.lower(), "longitude": lon_s, "latitude": lat_s, "category": "city"}
        )

    # --- Rare geology categories ---
    rare_gids = {
        1: "peat",
        5: "lacustrine",
        9: "outwash",
        14: "volcanic",
    }
    for gid, description in rare_gids.items():
        result = _find_point_with_geology_id(gid, coast_union, rng)
        if result is not None:
            e, n = result
            lat, lon = _nztm_to_wgs84(e, n)
            points.append(
                {
                    "name": f"geology_gid{gid}_{description}",
                    "longitude": lon,
                    "latitude": lat,
                    "category": "rare_geology",
                }
            )
        else:
            print(f"WARNING: Could not find point for GID {gid} ({description})")

    # --- Coastal-sensitive geology (GID 4 alluvium, GID 10 flood plain near coast) ---
    coastal_gids = {4: "alluvium_coastal", 10: "flood_plain_coastal"}
    for gid, description in coastal_gids.items():
        # Find a point with this GID that is within 5 km of coast
        nz = constants.FULL_NZ_GRID_CONFIG
        for _ in range(5000):
            e_r = rng.uniform(nz.grid_xmin, nz.grid_xmax)
            n_r = rng.uniform(nz.grid_ymin, nz.grid_ymax)
            e_r, n_r = _snap_to_pixel_center(e_r, n_r)
            pt = shapely.Point(e_r, n_r)
            if not shapely.within(pt, coast_union):
                continue
            coast_boundary = coast_union.boundary
            if shapely.distance(pt, coast_boundary) > 5000:
                continue
            if _geology_id_at(e_r, n_r) == gid:
                lat, lon = _nztm_to_wgs84(e_r, n_r)
                points.append(
                    {
                        "name": f"geology_gid{gid}_{description}",
                        "longitude": lon,
                        "latitude": lat,
                        "category": "coastal_sensitive",
                    }
                )
                break
        else:
            print(f"WARNING: Could not find coastal point for GID {gid}")

    # --- Observation-sparse area (Fiordland / remote West Coast) ---
    # Pick a point in the southwest corner of the South Island
    fiordland_lat, fiordland_lon = -45.5, 167.0
    e, n = _wgs84_to_nztm(fiordland_lat, fiordland_lon)
    e, n = _snap_to_pixel_center(e, n)
    lat, lon = _nztm_to_wgs84(e, n)
    points.append(
        {
            "name": "fiordland_sparse",
            "longitude": lon,
            "latitude": lat,
            "category": "observation_sparse",
        }
    )

    # --- Near geology boundary ---
    # Find two adjacent pixels with different geology IDs
    nz = constants.FULL_NZ_GRID_CONFIG
    for _ in range(5000):
        e_r = rng.uniform(nz.grid_xmin + 200, nz.grid_xmax - 200)
        n_r = rng.uniform(nz.grid_ymin + 200, nz.grid_ymax - 200)
        e_r, n_r = _snap_to_pixel_center(e_r, n_r)
        pt = shapely.Point(e_r, n_r)
        if not shapely.within(pt, coast_union):
            continue
        gid_center = _geology_id_at(e_r, n_r)
        if gid_center == 0 or gid_center == constants.RASTER_ID_NODATA_VALUE:
            continue
        # Check the pixel 100m to the east
        gid_east = _geology_id_at(e_r + 100, n_r)
        if gid_east != gid_center and gid_east != 0 and gid_east != constants.RASTER_ID_NODATA_VALUE:
            lat, lon = _nztm_to_wgs84(e_r, n_r)
            points.append(
                {
                    "name": f"geology_boundary_gid{gid_center}_gid{gid_east}",
                    "longitude": lon,
                    "latitude": lat,
                    "category": "geology_boundary",
                }
            )
            break
    else:
        print("WARNING: Could not find geology boundary point")

    return points


def generate_random_points(
    n: int,
    coast_union: shapely.Geometry,
    rng: np.random.Generator,
    existing_eastings_northings: list[tuple[float, float]],
) -> list[dict]:
    """Generate random on-land points, avoiding duplicates with existing points."""
    nz = constants.FULL_NZ_GRID_CONFIG
    existing_set = set(existing_eastings_northings)
    points = []

    # Generate more candidates than needed, filter to on-land
    while len(points) < n:
        batch_size = n * 10
        eastings = rng.uniform(nz.grid_xmin, nz.grid_xmax, size=batch_size)
        northings = rng.uniform(nz.grid_ymin, nz.grid_ymax, size=batch_size)

        for e, n_coord in zip(eastings, northings):
            e, n_coord = _snap_to_pixel_center(e, n_coord)
            if (e, n_coord) in existing_set:
                continue
            pt = shapely.Point(e, n_coord)
            if not shapely.within(pt, coast_union):
                continue
            # Exclude water pixels (GID 0)
            gid = _geology_id_at(e, n_coord)
            if gid == 0 or gid == constants.RASTER_ID_NODATA_VALUE:
                continue
            lat, lon = _nztm_to_wgs84(e, n_coord)
            points.append(
                {
                    "name": f"random_{len(points):02d}",
                    "longitude": lon,
                    "latitude": lat,
                    "category": "random",
                }
            )
            existing_set.add((e, n_coord))
            if len(points) >= n:
                break

    return points


def main():
    print("Loading coastline...")
    coast_union = _load_coastline()

    rng = np.random.default_rng(42)

    print("Generating deliberate points...")
    deliberate = generate_deliberate_points(coast_union, rng)
    print(f"  Generated {len(deliberate)} deliberate points")

    # Collect NZTM coordinates of deliberate points to avoid duplicates
    existing_en = []
    for p in deliberate:
        e, n = _wgs84_to_nztm(p["latitude"], p["longitude"])
        existing_en.append(_snap_to_pixel_center(e, n))

    print(f"Generating {N_RANDOM} random points...")
    random_pts = generate_random_points(N_RANDOM, coast_union, rng, existing_en)
    print(f"  Generated {len(random_pts)} random points")

    all_points = deliberate + random_pts
    df = pd.DataFrame(all_points)
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {len(df)} points to {OUTPUT_CSV}")
    print(f"Categories: {df['category'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the generation script to produce the fixture CSV**

Run:
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vs30_venv && python dev/generate_consistency_test_points.py
```

Expected: Script prints the number of points generated per category and writes `tests/fixtures/consistency_test_points.csv`.

- [ ] **Step 3: Verify the generated CSV**

Run:
```bash
head -5 tests/fixtures/consistency_test_points.csv && wc -l tests/fixtures/consistency_test_points.csv
```

Expected: CSV has columns `name,longitude,latitude,category` and ~35-45 data rows (plus header).

- [ ] **Step 4: Commit**

```bash
git add dev/generate_consistency_test_points.py tests/fixtures/consistency_test_points.csv
git commit -m "add consistency test point generation script and fixture CSV"
```

---

### Task 2: Register the `slow` pytest marker

**Files:**
- Modify: `pyproject.toml`

- [ ] **Step 1: Add pytest marker configuration**

Add the following section to the end of `pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "slow: marks tests as slow (deselect with '-m \"not slow\"')",
]
```

- [ ] **Step 2: Verify the marker is registered**

Run:
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vs30_venv && pytest --markers | grep slow
```

Expected: Output includes `@pytest.mark.slow: marks tests as slow`.

- [ ] **Step 3: Commit**

```bash
git add pyproject.toml
git commit -m "register slow pytest marker in pyproject.toml"
```

---

### Task 3: Write the config-loading helper in conftest

**Files:**
- Modify: `tests/conftest.py` (add a helper that loads a `FixedModelVersion` config for the consistency test, without importing typer from `cli.py`)

The existing `cli.load_model_config` (`vs30/cli.py:59`) raises `typer.BadParameter` on validation errors — a CLI concern. The test needs the same config resolution (CSV path resolution and correlation function building) without pulling in typer. We add a thin helper to `conftest.py`.

- [ ] **Step 1: Add the helper to conftest.py**

Append the following to the end of `tests/conftest.py`:

```python
def load_fixed_model_config(version: constants.FixedModelVersion) -> dict:
    """
    Load and resolve a fixed model version's YAML config for testing.

    Mirrors cli.load_model_config but without the typer dependency.
    Resolves CSV paths relative to the resources directory and builds
    correlation function callables.

    Parameters
    ----------
    version : FixedModelVersion
        Model version to load.

    Returns
    -------
    dict
        Resolved config dict ready to pass to pipeline functions.
    """
    from vs30.cli import resolve_correlation_function

    config_path = constants.MODEL_VERSION_TO_CONFIG[version]
    with open(config_path) as f:
        config_data = yaml.safe_load(f)

    # Resolve CSV paths
    for key, subdir in constants.RESOURCE_SUBDIRS.items():
        if config_data.get(key):
            config_data[key] = constants.RESOURCE_PATH / subdir / config_data[key]

    # Build correlation functions
    config_data["geology_corr_fn"] = resolve_correlation_function(
        config_data["geology_correlation"]
    )
    config_data["terrain_corr_fn"] = resolve_correlation_function(
        config_data["terrain_correlation"]
    )

    return config_data
```

- [ ] **Step 2: Verify it loads without error**

Run:
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vs30_venv && python -c "
from conftest import load_fixed_model_config
from vs30.constants import FixedModelVersion
cfg = load_fixed_model_config(FixedModelVersion.FOSTER_2019)
print('Keys:', sorted(cfg.keys()))
print('geology_csv:', cfg['geology_categorical_csv'])
" 2>&1 | head -5
```

Run from the `tests/` directory. Expected: prints config keys and resolved CSV path.

- [ ] **Step 3: Commit**

```bash
git add tests/conftest.py
git commit -m "add load_fixed_model_config helper to test conftest"
```

---

### Task 4: Rewrite the consistency test

**Files:**
- Modify: `tests/test_grid_points_consistency.py` (full rewrite)

This is the core task. The test loads each model version, runs `points_pipeline` once with all ~40 points, then loops over each point running `grid_pipeline` on a 3x3 grid and comparing.

- [ ] **Step 1: Write the new test file**

Replace the entire contents of `tests/test_grid_points_consistency.py` with:

```python
"""
Test that grid and points pipelines produce consistent Vs30 values
across the full NZ domain for all fixed model versions.

For each test point, a tiny 3x3 grid (300m x 300m at 100m resolution) is
generated and run through grid_pipeline.  The center pixel is compared
against the batched points_pipeline result at the same coordinates.

The test is split into two tiers:
- Fast tier (foster_2019, jaehwi_v1p0): no coastal distance computation,
  runs in ~4-5 minutes.
- Slow tier (modified_foster_2019, viktor_cpt_clustering): coastal distance
  extends to full NZ domain per point, runs in ~25-30 minutes.
"""

import numpy as np
import pandas as pd
import pytest
from qcore import coordinates

from conftest import FIXTURES_DIR, load_fixed_model_config
from vs30 import constants, gapfill, pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

POINTS_CSV = FIXTURES_DIR / "consistency_test_points.csv"

# Tolerances for approximate grid/points agreement.
# The 3x3 grid approach minimises resampling discrepancy (most observations
# fall outside the tiny grid and use direct source sampling in both paths).
VS30_RTOL = 0.03
STDV_RTOL = 0.30

# Half-width for the 3x3 local grid (150m each side of center → 300m / 100m = 3 pixels).
LOCAL_GRID_HALF_WIDTH = 150

# Model versions that do NOT use coastal distance (fast tier).
FAST_VERSIONS = [
    constants.FixedModelVersion.FOSTER_2019,
    constants.FixedModelVersion.JAEHWI_V1P0,
]

# Model versions that DO use coastal distance (slow tier).
SLOW_VERSIONS = [
    constants.FixedModelVersion.MODIFIED_FOSTER_2019,
    constants.FixedModelVersion.VIKTOR_CPT_CLUSTERING,
]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def load_test_points() -> pd.DataFrame:
    """Load the pre-computed test points CSV."""
    df = pd.read_csv(POINTS_CSV)
    assert {"name", "longitude", "latitude", "category"}.issubset(df.columns)
    return df


def run_points_pipeline_for_version(
    cfg: dict, points_df: pd.DataFrame
) -> pd.DataFrame:
    """Run points_pipeline once with all test points for a given model config."""
    return pipeline.points_pipeline(
        longitudes=points_df["longitude"].values,
        latitudes=points_df["latitude"].values,
        geology_categorical_csv=cfg["geology_categorical_csv"],
        terrain_categorical_csv=cfg["terrain_categorical_csv"],
        clustered_observations_csv=cfg.get("clustered_observations_csv"),
        independent_observations_csv=cfg.get("independent_observations_csv"),
        combination_method=constants.CombinationMethod(cfg["combination_method"]),
        combine_ratio=cfg.get("combine_ratio"),
        noisy=cfg["noisy"],
        do_bayesian_update=cfg["do_bayesian_update"],
        nproc=1,
        geology_corr_fn=cfg.get("geology_corr_fn"),
        terrain_corr_fn=cfg.get("terrain_corr_fn"),
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        apply_coastal_distance_mod=cfg["apply_coastal_distance_mod"],
    )


def run_grid_pipeline_at_point(
    cfg: dict, easting: float, northing: float
) -> tuple[float, float]:
    """Run grid_pipeline on a 3x3 grid centered on (easting, northing).

    Returns the center pixel (row=1, col=1) Vs30 and stdv.
    """
    local_config = gapfill.create_local_grid_config(
        easting, northing, constants.FULL_NZ_GRID_CONFIG, LOCAL_GRID_HALF_WIDTH
    )

    result = pipeline.grid_pipeline(
        grid_config=local_config,
        output_dir=None,
        geology_categorical_csv=cfg["geology_categorical_csv"],
        terrain_categorical_csv=cfg["terrain_categorical_csv"],
        clustered_observations_csv=cfg.get("clustered_observations_csv"),
        independent_observations_csv=cfg.get("independent_observations_csv"),
        combination_method=constants.CombinationMethod(cfg["combination_method"]),
        combine_ratio=cfg.get("combine_ratio"),
        noisy=cfg["noisy"],
        do_bayesian_update=cfg["do_bayesian_update"],
        nproc=1,
        geology_corr_fn=cfg.get("geology_corr_fn"),
        terrain_corr_fn=cfg.get("terrain_corr_fn"),
        apply_alluvium_slope_mod=cfg["apply_alluvium_slope_mod"],
        apply_coastal_distance_mod=cfg["apply_coastal_distance_mod"],
    )

    grid_vs30 = result["combined_vs30"]
    grid_stdv = result["combined_stdv"]

    # Center pixel of the 3x3 grid
    return float(grid_vs30[1, 1]), float(grid_stdv[1, 1])


def _check_consistency_for_version(version: constants.FixedModelVersion):
    """Core comparison logic shared by fast and slow tiers."""
    cfg = load_fixed_model_config(version)
    points_df = load_test_points()

    # Batch points pipeline call
    points_result = run_points_pipeline_for_version(cfg, points_df)

    # Convert lon/lat to NZTM for grid pipeline calls
    lats = points_df["latitude"].values
    lons = points_df["longitude"].values
    nztm = coordinates.wgs_depth_to_nztm(np.column_stack([lats, lons]))
    eastings = nztm[:, 1]
    northings = nztm[:, 0]

    failures = []
    for i in range(len(points_df)):
        name = points_df["name"].iloc[i]
        e, n = eastings[i], northings[i]

        grid_vs30, grid_stdv = run_grid_pipeline_at_point(cfg, e, n)

        pts_vs30 = points_result[constants.ObservationColumn.VS30].iloc[i]
        pts_stdv = points_result[constants.COL_COMBINED_STDV].iloc[i]

        # Skip nodata points (both pipelines should agree on nodata)
        if np.isnan(grid_vs30) and np.isnan(pts_vs30):
            continue

        # Check Vs30
        if not np.isclose(pts_vs30, grid_vs30, rtol=VS30_RTOL):
            failures.append(
                f"  {name}: Vs30 mismatch — "
                f"grid={grid_vs30:.2f}, points={pts_vs30:.2f}, "
                f"rdiff={abs(pts_vs30 - grid_vs30) / grid_vs30:.4f}"
            )

        # Check Stdv
        if not np.isclose(pts_stdv, grid_stdv, rtol=STDV_RTOL):
            failures.append(
                f"  {name}: Stdv mismatch — "
                f"grid={grid_stdv:.2f}, points={pts_stdv:.2f}, "
                f"rdiff={abs(pts_stdv - grid_stdv) / grid_stdv:.4f}"
            )

    if failures:
        msg = f"\n{version.value}: {len(failures)} failure(s):\n" + "\n".join(failures)
        pytest.fail(msg)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("version", FAST_VERSIONS, ids=lambda v: v.value)
def test_grid_points_consistency_fast(version):
    """Grid/points consistency for models without coastal distance (~2-4 min)."""
    _check_consistency_for_version(version)


@pytest.mark.slow
@pytest.mark.parametrize("version", SLOW_VERSIONS, ids=lambda v: v.value)
def test_grid_points_consistency_slow(version):
    """Grid/points consistency for models with coastal distance (~12-15 min each)."""
    _check_consistency_for_version(version)
```

- [ ] **Step 2: Run the fast tier to verify it works**

Run:
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vs30_venv && pytest tests/test_grid_points_consistency.py::test_grid_points_consistency_fast -v --tb=short 2>&1 | tail -20
```

Expected: Two tests pass (`foster_2019` and `jaehwi_v1p0`). Each takes ~2-5 minutes.

- [ ] **Step 3: Run a single slow-tier test to spot-check**

Run:
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vs30_venv && pytest tests/test_grid_points_consistency.py::test_grid_points_consistency_slow[modified_foster_2019] -v --tb=short 2>&1 | tail -20
```

Expected: Passes but takes ~12-15 minutes.

- [ ] **Step 4: Verify slow tests are excluded by default marker filtering**

Run:
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vs30_venv && pytest tests/test_grid_points_consistency.py -m "not slow" -v --collect-only 2>&1 | tail -10
```

Expected: Only the two `test_grid_points_consistency_fast` items are collected. The two `test_grid_points_consistency_slow` items are deselected.

- [ ] **Step 5: Commit**

```bash
git add tests/test_grid_points_consistency.py
git commit -m "rewrite grid/points consistency test with full NZ coverage and all model versions"
```

---

### Task 5: Verify existing tests still pass

**Files:** (none modified — verification only)

- [ ] **Step 1: Run the full non-slow test suite**

Run:
```bash
source ~/miniforge3/etc/profile.d/conda.sh && conda activate vs30_venv && pytest -m "not slow" -v --tb=short 2>&1 | tail -30
```

Expected: All tests pass. The rewritten `test_grid_points_consistency.py` replaces the old test; no other test files are affected.

- [ ] **Step 2: Commit (if any tolerance adjustments were needed)**

If tolerances needed adjusting based on empirical results from Task 4, commit those changes:

```bash
git add tests/test_grid_points_consistency.py
git commit -m "adjust consistency test tolerances based on empirical results"
```

If no adjustments needed, skip this step.

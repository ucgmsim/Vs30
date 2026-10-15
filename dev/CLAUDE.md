# CLAUDE.md

> Describes `master` as of the October 2026 wrap-up. Copy this file to the root of a `master` checkout to use it with Claude Code. The code on this branch is a 13 May 2026 snapshot.

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VS30 is a Python package for calculating and mapping Vs30 (shear-wave velocity at 30m depth) values across New Zealand for seismic hazard analysis. It uses a multi-stage Bayesian framework combining categorical geology/terrain models with spatial observations and statistical adjustments.

**Repository**: https://github.com/ucgmsim/Vs30

**Scientific Reference**: Foster et al. (2019), https://doi.org/10.1193/121118EQS281M, provides the scientific context and methodology for this codebase.

## Refactoring Context

This codebase was refactored from legacy code to improve readability and understandability, especially for researchers and non-software engineers, with some performance improvements. The refactored code should produce the same output as the legacy code (small differences due to numerical precision or algorithm changes are acceptable).

**Priority**: Readability is the most important priority and should not be sacrificed for small performance improvements.

**Status**: The refactoring project is nearing completion, but additional features and testing still need to be implemented.

**Validation**: When making large changes, compare `vs30 grid` output to that produced by the legacy wrapper scripts. Expect intended differences within 10 km of observations: since f539ab8, observations take slope and coast distance from their exact location (legacy used pixel values), and the grid benchmarks in `tests/benchmarks/` are refactored-pipeline output.

**Developer Resources**: Maintainer scripts (legacy comparison, observation and benchmark generators) and developer notes live on the `dev_scripts_archive` branch under `dev/`. Its code is a 13 May 2026 snapshot; use it as a reference, not as the current package.

## Coding Conventions

These conventions complement the readability priority above. Apply them to all code changes in this repo unless a task explicitly overrides them.

- **Inline expressions; avoid intermediate variables.** Only name a value when the name documents a derived concept that isn't obvious from the expression itself. Don't introduce a variable just to give an expression a name.
- **Comments only for the non-obvious.** Skip narration of *what* code does — well-named identifiers handle that. Add a comment only when an experienced Python reader could be confused by *why*: a hidden invariant, a subtle correctness point, a workaround for a specific bug.
- **No section-header banner comments** (e.g. `# ---\n# Helpers\n# ---`). Use module layout and function names to communicate structure.
- **Imports at the top of the module.** No deferred imports inside functions unless there's a real reason (circular import, optional dependency).
- **No leading underscores on private helpers.** Module-local helpers use plain names.
- **No bare `*` keyword-only markers** in function signatures. Reorder parameters (non-defaults first) instead.
- **Helpers must earn their existence.** Only extract a helper when there's real reuse (≥2 call sites) or when the inline math is genuinely hard to read. A single-use helper that just wraps a one-line expression is an alias and should be inlined.
- **Docstrings: prefer concise.** One-line summary for trivial functions; expand only when the function's contract isn't obvious from its signature and name.
- **Notes rule.** Every note, README or `.md` file on `master` needs a clear production purpose; developer notes go on `dev_scripts_archive`.

## Environment Setup

Create the project environment with `uv sync --all-extras --dev` and run commands through it, e.g. `uv run pytest tests/` or `uv run vs30 points …`.

## Legacy Codebase Reference

The pre-refactor code is this repository's `legacy-final` tag. It needs its own environment, since it predates this package's dependencies, and must run with `--nproc 1` (it deadlocks otherwise). Jaehwi's fork, the reference for the `jaehwi_v1p0` benchmark, is https://github.com/elshawy/Vs30_2026 (`run_vs30calc_V1.py`). Benchmark-generation commands are in `dev/docs/generating_benchmarks_from_legacy_code.md` on this branch.

The comparison wrapper scripts (`run_legacy_wrapper_*.py`) are in `dev/scripts/comparison/` on this branch. Each hardcodes its grid bounds, spacing and output directory, and runs from a checkout of the legacy code.

## Build and Installation

```bash
# Install in development mode (extracts shapefiles during install)
pip install -e /path/to/vs30

# The setup.py custom build step extracts vs30/resources/geospatial/shapefiles.tar.xz
```

No system GDAL is needed: rasterio's wheels bundle it, and the package doesn't import `osgeo`.

## CLI Commands

The package provides a `vs30` CLI entry point via Typer with two commands, `grid` and `points`. `MODEL` is a bundled model version (`foster_2019_approx`, `modified_foster_2019`, `viktor_cpt_clustering`, `jaehwi_v1p0`; configs in `vs30/configs/`) or the path to a custom YAML config. The user guide is `wiki/Usage.md`.

```bash
# Vs30 grid; all of NZ at 100 m unless bounds or spacing are given
vs30 grid foster_2019_approx output_dir

# A region (bounds are pixel edges, NZTM2000 metres) at a coarser spacing
vs30 grid foster_2019_approx output_dir \
    --grid-xmin 1740100 --grid-xmax 1760100 \
    --grid-ymin 5410100 --grid-ymax 5430100 \
    --grid-dx 200 --grid-dy 200

# Vs30 at specific lat/lon sites
vs30 points foster_2019_approx sites.csv results.csv

# A custom model config, and input columns not named longitude/latitude
vs30 points my_model.yaml sites.csv results.csv --lon-column lon --lat-column lat
```

Both commands take `--include-intermediate` (also write per-model and intermediate results), `--nproc N` (cores for DBSCAN and the MVN's BLAS threads; default -1, all cores) and `--verbose` (step-by-step progress messages). `grid` also takes `--mvn-chunk-memory-gb`.

### Location-Based Queries

The `points` command computes Vs30 at specific latitude/longitude points without generating full raster grids. This is efficient for querying a small number of sites.

The input CSV needs WGS84 longitude and latitude columns. The output keeps the input columns and adds `easting`, `northing`, `vs30` (m/s) and `stdv` (standard deviation of ln Vs30); `--include-intermediate` adds geology/terrain IDs and per-model Vs30 values.

## Running Tests

The suite has two tiers:

```bash
# Default — unit tests, benchmark tests, 3-city grid/points smoke. ~1 min.
pytest tests/

# Default + full 38-point grid/points consistency across all 4 model versions. ~15 min.
pytest tests/ --runslow
```

Tests decorated with `@pytest.mark.slow` are skipped unless `--runslow` is
passed (see `conftest.py::pytest_collection_modifyitems`).

CI (`.github/workflows/ci.yml`) runs ucgmsim/meta-ci-action's shared checks (ruff, deptry, npdlint, ty, pytest with coverage) on pull requests and pushes to master; `extra-checks.yml` runs the slow tier and a wheel smoke test on pushes to master. Run the same checks locally with `uv sync --all-extras --dev` and `uvx lefthook run pre-push --all-files --no-tty`.

## Architecture

### Pipeline Stages

1. **Categorical Update** (`category.py`): Bayesian updating of Vs30 values per geology/terrain category using observations. Uses DBSCAN clustering to prevent over-weighting of spatially clustered measurements.

2. **Raster Creation** (`raster.py`): Creates ID rasters from QMAP geology or IwahashiPike terrain data, then maps category IDs to Vs30 values.

3. **Hybrid Modifications** (`raster.py`): Adjusts geology Vs30 for specific groups based on slope and coastal distance (NZ-empirical calibration).

4. **Spatial Adjustment** (`spatial.py`): MVN (Multivariate Normal) conditioning to update pixels using nearby observations with exponential or Matérn spatial correlation (`correlations.py`).

5. **Model Combination** (`utils.combine_vs30_models`, called from `pipeline.py`): Combines geology and terrain models into final prediction.

6. **Gap-filling** (`gapfill.py`, models with `fill_gaps: true`): On-land pixels or sites with no Vs30 take the value of the nearest valid pixel.

### Core Modules

- `cli.py` - Typer commands (`grid`, `points`), input checks, logging setup
- `pipeline.py` - Grid and points pipeline orchestration
- `config.py` - Model YAML config loading and validation; `GridConfig`
- `category.py` - Bayesian updates, observation assignment, DBSCAN clustering
- `raster.py` - Raster creation, hybrid geology modifications, coast distance (rasterio + scipy)
- `spatial.py` - MVN spatial interpolation (main algorithm complexity)
- `points.py` - Per-point geology/terrain processing for points mode
- `correlations.py` - Spatial correlation functions (exponential, Matérn)
- `gapfill.py` - Gap-filling of on-land nodata pixels
- `constants.py` - Scientific/algorithmic constants, file and column names
- `utils.py` - Model combination and data helpers

### Configuration

Each model version is a YAML config in `vs30/configs/`, loaded and validated by `config.py`:
- Categorical model and observation CSVs
- Pipeline switches (`do_bayesian_update`, `mvn`, `noisy`, hybrid modifications, `fill_gaps`)
- Combination method and ratio
- Geology and terrain correlation models

Other parameters are constants in `constants.py` (MAX_DIST_M, MAX_POINTS, N_PRIOR, MIN_SIGMA, DBSCAN EPS/MIN_GROUP, hybrid model thresholds, output file and column names). Grid bounds and spacing come from the `grid` command's options (default `config.FULL_NZ_GRID_CONFIG`).

### Key Implementation Details

- **Coordinate System**: NZTM2000 (EPSG:2193), all coordinates in meters
- **Log-space**: Bayesian updates and MVN conditioning work in log-space (log-normal Vs30 distribution)
- **Observation Bias**: DBSCAN clustering groups nearby measurements; each cluster = one pseudo-observation
- **Correlation**: Set per model in its YAML config: exponential `corr = exp(-d/phi)` with phi 1407 m (geology) and 993 m (terrain) in three bundled models; Matérn in `foster_2019_approx`

### Data Flow

```
Observations → Bayesian Update → Category CSV
                                     ↓
                              Create ID Raster → Map to Vs30 Raster
                                                       ↓
                              [Hybrid Mods for geology only]
                                                       ↓
                              MVN Spatial Adjustment
                                                       ↓
                              Combine Geology + Terrain
                                                       ↓
                              [Gap-fill, if the model enables it] → Final Map
```

### Key Data Structures

- `ObservationData` (spatial.py): Bundle of observation locations, vs30, residuals, uncertainties
- `RasterData` (spatial.py): Input raster with vs30, stddev, and validity masks
- `PixelData` (spatial.py): Per-pixel inputs to the MVN update (location, vs30, stdv)

## VsViewer

`VsViewer/vs_calc` estimates Vs30 from CPT and SPT data. It installs separately from `VsViewer/`, and its tests are in `VsViewer/tests`. Its Vs30 uncertainty is provisional: the standard deviation of ln(Vs30) is `sqrt(0.2**2 + (ln(10) * Boore residual)**2)`; see `dev/docs/vs_calc/vs30_uncertainty_followup.md` on `dev_scripts_archive`.

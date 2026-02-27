# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VS30 is a Python package for calculating and mapping Vs30 (shear-wave velocity at 30m depth) values across New Zealand for seismic hazard analysis. It uses a multi-stage Bayesian framework combining categorical geology/terrain models with spatial observations and statistical adjustments.

**Repository**: https://github.com/ucgmsim/Vs30

**Scientific Reference**: `reference_papers/foster_2019_nz_vs30_map.pdf` provides the scientific context and methodology for this codebase.

## Refactoring Context

This codebase was refactored from legacy code to improve readability and understandability, especially for researchers and non-software engineers, with some performance improvements. The refactored code should produce the same output as the legacy code (small differences due to numerical precision or algorithm changes are acceptable).

**Priority**: Readability is the most important priority and should not be sacrificed for small performance improvements.

**Status**: The refactoring project is nearing completion, but additional features and testing still need to be implemented.

**Validation**: When making large changes, compare the `full-pipeline` output to that produced by the legacy wrapper scripts. Comparison scripts are available in the `temp/` directory (note: `temp/` contains temporary development code that will not be included in the final release).

## Environment Setup

Activate the Python environment before running any commands:
```bash
mamba activate vs30_venv
```

All Python/pytest/vs30 commands must be run with the environment activated, e.g.:
```bash
mamba activate vs30_venv && pytest tests/
mamba activate vs30_venv && vs30 grid
```

## Legacy Codebase Reference

This code was refactored from a legacy codebase available at:
```
/home/arr65/src/pre-refactor-Vs30-for-comparison
```

To run the legacy code, use a separate environment:
```bash
mamba activate oldvs30_venv && python <script>
```

Helper wrapper scripts in the legacy directory can run the legacy code with the same settings as defined in the refactored code's `config.yaml` file, useful for comparison testing.

## Build and Installation

```bash
# Install in development mode (extracts shapefiles during install)
pip install -e /path/to/vs30

# The setup.py custom build step extracts vs30/data/shapefiles.tar.xz
```

## CLI Commands

The package provides a `vs30` CLI entry point via Typer. Main commands:

```bash
# Run the full pipeline (most common) - generates raster grids
vs30 grid

# Compute Vs30 at specific lat/lon locations (without generating rasters)
vs30 points \
    --locations-csv sites.csv \
    --output-csv results.csv

# Individual pipeline stages:
vs30 update-priors                     # Bayesian update of category means/stdev
vs30 make-initial-vs30-raster          # Convert categorical models to raster
vs30 adjust-geology-vs30-by-slope-and-coastal-distance  # Apply hybrid modifications
vs30 spatial-fit                       # MVN-based spatial adjustments
vs30 combine                           # Merge geology and terrain models
vs30 plot-posterior-values             # Visualize posterior distributions
```

### Location-Based Queries

The `points` command computes Vs30 at specific latitude/longitude points without generating full raster grids. This is efficient for querying a small number of sites.

```bash
# Basic usage
vs30 points \
    --locations-csv sites.csv \
    --output-csv results.csv

# With custom column names
vs30 points \
    --locations-csv sites.csv \
    --lon-column lon \
    --lat-column lat \
    --output-csv results.csv

# With coastal distance raster for hybrid geology modifications
vs30 points \
    --locations-csv sites.csv \
    --output-csv results.csv \
    --coast-distance-raster /path/to/coast_distance.tif
```

Input CSV must have longitude and latitude columns (WGS84). Output includes geology/terrain IDs, intermediate Vs30 values, and final combined Vs30.

## Running Tests

```bash
# Run regression tests (parameterized by processor count)
pytest tests/test_regression.py

# Run a specific test
pytest tests/test_regression.py::test_vs30calc_regression -v
```

## Architecture

### Pipeline Stages

1. **Categorical Update** (`category.py`): Bayesian updating of Vs30 values per geology/terrain category using observations. Uses DBSCAN clustering to prevent over-weighting of spatially clustered measurements.

2. **Raster Creation** (`raster.py`): Creates ID rasters from QMAP geology or IwahashiPike terrain data, then maps category IDs to Vs30 values.

3. **Hybrid Modifications** (`raster.py`): Adjusts geology Vs30 for specific groups based on slope and coastal distance (NZ-empirical calibration).

4. **Spatial Adjustment** (`spatial.py`): MVN (Multivariate Normal) conditioning to update pixels using nearby observations with exponential spatial correlation.

5. **Model Combination** (`cli.py`): Combines geology and terrain models into final prediction.

### Core Modules

- `cli.py` - CLI orchestration, pipeline commands
- `category.py` - Bayesian updates, observation assignment, DBSCAN clustering
- `spatial.py` - MVN spatial interpolation (main algorithm complexity)
- `raster.py` - Raster creation, hybrid geology modifications
- `constants.py` - Configuration loading from config.yaml
- `utils.py` - Shared utilities (correlation function, distance calculations)

### Configuration

All parameters centralized in `config.yaml` (loaded at import via `constants.py`):
- Grid parameters (domain bounds, resolution)
- Spatial parameters (phi correlation lengths, max_dist_m, max_points)
- Bayesian parameters (n_prior, min_sigma, DBSCAN eps/min_group)
- Hybrid model parameters (slope/distance thresholds)
- Input/output paths

### Key Implementation Details

- **Coordinate System**: NZTM2000 (EPSG:2193), all coordinates in meters
- **Log-space**: Bayesian updates and MVN conditioning work in log-space (log-normal Vs30 distribution)
- **Observation Bias**: DBSCAN clustering groups nearby measurements; each cluster = one pseudo-observation
- **Correlation**: Exponential model `corr = exp(-d/phi)` with separate phi for geology (1407m) and terrain (993m)

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
                              Combine Geology + Terrain → Final Map
```

### Key Data Structures

- `ObservationData` (spatial.py): Bundle of observation locations, vs30, residuals, uncertainties
- `RasterData` (spatial.py): Input raster with vs30, stddev, and validity masks
- `MVNUpdateResult` (spatial.py): Output of updating one pixel

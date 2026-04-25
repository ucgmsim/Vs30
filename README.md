# VS30 - Shear-Wave Velocity Mapping for New Zealand

A Python package estimating Vs30 (time-averaged shear-wave velocity in the upper 30 meters) values across New Zealand.

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/ucgmsim/Vs30.git
```

### Install from Source (Development Mode)

```bash
git clone https://github.com/ucgmsim/Vs30.git
cd Vs30
pip install -e .
```

## CLI Commands

The package provides a `vs30` command-line interface with two main commands:

### Compute VS30 at Specific Locations

Calculate Vs30 at lat/lon points listed in a CSV using a predefined model version:

```bash
vs30 points foster_2019_approx locations.csv results.csv
```

### Generate VS30 Grid Maps

Run the complete Vs30 mapping workflow on a regular grid (writes GeoTIFFs):

```bash
vs30 grid \
    --version foster_2019_approx \
    --grid-xmin 1060050 --grid-xmax 2120050 \
    --grid-ymin 4730050 --grid-ymax 6250050 \
    --grid-dx 100 --grid-dy 100 \
    --output-dir ./vs30_out
```

Available model versions: `foster_2019_approx`, `modified_foster_2019`, `jaehwi_v1p0`, `viktor_cpt_clustering`.

Each command also has a `-custom` variant (`points-custom`, `grid-custom`) that exposes every scientific parameter individually for ablation experiments. Run `vs30 <command> --help` for the full option list.

See the [Usage page](wiki/Usage.md) for input formats, parameter overrides, and grid sizing guidance.

## How It Works

The Vs30 mapping pipeline combines categorical geology and terrain models with observational data through several stages:

1. **Bayesian Update**: Refines categorical model values using spatially-clustered observational data
2. **Raster Creation**: Creates VS30 maps from categorical models  
3. **Hybrid Modifications**: Applies slope and coastal distance adjustments to geology models
4. **Spatial Adjustment**: Uses multivariate normal conditioning to incorporate nearby observations
5. **Model Combination**: Combines geology and terrain models using weighted averaging

The methodology is based on Foster et al. (2019) "A Vs30 Map for New Zealand Based on Geologic and Terrain Proxy Variables and Field Measurements".

## Coordinate System

All coordinates use NZTM2000 (EPSG:2193) in meters. The `points` command accepts WGS84 lat/lon input and converts internally.

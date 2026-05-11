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

The package provides a `vs30` command-line interface with `points` and `grid` subcommands. See the [Usage page](wiki/Usage.md) for command syntax, input formats, supported model versions, custom YAML configs, and grid sizing guidance.

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

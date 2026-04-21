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

The package provides a `vs30` command-line interface with the following commands:

### Generate VS30 Grid Maps

Run the complete Vs30 mapping workflow using a predefined model version:

```bash
vs30 grid output_directory foster_2019_approx
```

Available model versions: `foster_2019_approx`, `modified_foster_2019`, `jaehwi_v1p0`, `viktor_cpt_clustering`

Options:
- `--nproc`: Number of parallel processes (-1 for all cores)
- `--noisy/--no-noisy`: Apply noise weighting in spatial adjustment (default: True)
- `--max-spatial-boolean-array-memory-gb`: Memory limit for spatial arrays (default: 1.0)

### Compute VS30 at Specific Locations

Calculate Vs30 at specific lat/lon points using a predefined model version:

```bash
vs30 points locations.csv results.csv foster_2019_approx
```

Options:
- `--lon-column`: Name of longitude column (default: "longitude") 
- `--lat-column`: Name of latitude column (default: "latitude")
- `--include-intermediate/--final-only`: Include individual model outputs (default: include)
- `--nproc`: Number of parallel processes

### Update Categorical Model Values

Update categorical model values using Bayesian updates:

```bash
vs30 update-priors model.csv output_dir --model-type geology
```

## Configuration

The package uses predefined model configurations (`foster_2019_approx`, `modified_foster_2019`, `jaehwi_v1p0`, `viktor_cpt_clustering`) that include all necessary parameters and file paths. For custom configurations, see the advanced commands (`grid-custom`, `points-custom`).

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

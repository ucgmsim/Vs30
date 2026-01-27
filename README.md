# VS30 - Shear-Wave Velocity Mapping for New Zealand

A Python package for calculating and mapping Vs30 (time-averaged shear-wave velocity in the upper 30 meters) values across New Zealand for seismic hazard analysis. The package uses a multi-stage Bayesian framework combining categorical geology/terrain models with spatial observations and statistical adjustments.

## Installation

### Install from GitHub

```bash
pip install git+https://github.com/ucgmsim/Vs30.git
```

### Install from Source (Development Mode)

Clone the repository and install in editable mode:

```bash
git clone https://github.com/ucgmsim/Vs30.git
cd Vs30
pip install -e .
```

The installation automatically extracts required shapefiles from the bundled archive.

### Dependencies

The package requires Python 3.10+ and will automatically install dependencies including:
- numpy, pandas, scipy
- rasterio, geopandas, shapely
- scikit-learn (for DBSCAN clustering)
- typer (for CLI)
- pydantic (for configuration)
- tqdm (for progress bars)

## CLI Commands

The package provides a `vs30` command-line interface with the following commands:

### Full Pipeline

Run the complete Vs30 mapping workflow:

```bash
vs30 full-pipeline
```

This runs both geology and terrain pipelines, then combines them into a final map. Output is saved to the directory specified in `config.yaml`.

Options:
- `--output-dir`, `-d`: Output directory (default from config)
- `--n-proc`: Number of parallel processes (-1 for all cores)
- `--geology-csv`: Custom geology categorical model CSV
- `--terrain-csv`: Custom terrain categorical model CSV

### Compute at Specific Locations

Calculate Vs30 at specific lat/lon points without generating full raster grids:

```bash
vs30 compute-at-locations \
    --locations-csv sites.csv \
    --output-csv results.csv
```

Input CSV must have `longitude` and `latitude` columns in WGS84 (EPSG:4326). Coordinates are converted internally to NZTM2000 (EPSG:2193) for processing. Use `--lon-column` and `--lat-column` to specify different column names.

Options:
- `--coast-distance-raster`: Required for hybrid geology modifications
- `--include-intermediate/--final-only`: Include intermediate values in output
- `--n-proc`: Number of parallel processes

### Individual Pipeline Stages

Run specific stages of the pipeline:

```bash
# Update categorical model values with Bayesian updates
vs30 update-categorical-vs30-models \
    --categorical-model-csv model.csv \
    --clustered-observations-csv cpt.csv \
    --output-dir output/ \
    --model-type geology

# Create initial Vs30 rasters from categorical models
vs30 make-initial-vs30-raster --geology --terrain --output-dir output/

# Apply slope and coastal distance adjustments to geology
vs30 adjust-geology-vs30-by-slope-and-coastal-distance \
    --input-raster geology_initial.tif \
    --id-raster gid.tif \
    --output-dir output/

# Apply spatial adjustment using observations
vs30 spatial-fit \
    --input-raster geology_hybrid.tif \
    --observations-csv observations.csv \
    --model-values-csv updated_model.csv \
    --output-dir output/ \
    --model-type geology

# Combine geology and terrain models
vs30 combine \
    --geology-tif geology_final.tif \
    --terrain-tif terrain_final.tif \
    --output-path combined.tif
```

### Visualization

Plot prior vs posterior categorical values:

```bash
vs30 plot-posterior-values \
    --csv-path updated_model.csv \
    --output-dir plots/
```

### Using a Custom Configuration

Specify a custom config file for any command:

```bash
vs30 --config /path/to/config.yaml full-pipeline
```

## How It Works

The Vs30 mapping pipeline consists of several stages that combine categorical models with observational data to produce spatially-adjusted Vs30 maps.

### Overview

```
Observations ──► Bayesian Update ──► Category CSV
                                          │
                               Create ID Raster ──► Map to Vs30 Raster
                                                          │
                               [Hybrid Modifications - geology only]
                                                          │
                               Spatial Adjustment
                                                          │
                               Combine Geology + Terrain ──► Final Map
```

### Stage 1: Bayesian Update of Categorical Models

**Module**: `category.py`

Each geology and terrain category has a prior Vs30 mean and standard deviation based on published studies. The Bayesian update stage refines these values using measured observations.

**Key features**:
- **DBSCAN clustering**: Observations are spatially clustered to prevent over-weighting of dense measurement clusters (e.g., from geotechnical investigations in urban areas). Each cluster is treated as a single pseudo-observation.
- **Sequential updates**: Clustered observations (e.g., CPT data) are processed first to correct for spatial sampling bias, then independent observations (direct Vs30 measurements) refine the model.
- **Log-normal assumption**: All Bayesian updates operate in log-space, reflecting the log-normal distribution of Vs30 values.

### Stage 2: Raster Creation

**Module**: `raster.py`

Creates geospatial rasters from the categorical models:

1. **ID Rasters**:
   - Geology IDs from QMAP shapefile polygons
   - Terrain IDs from IwahashiPike classification raster

2. **Vs30 Rasters**: Maps category IDs to Vs30 mean and standard deviation values from the updated categorical model CSVs.

### Stage 3: Hybrid Geology Modifications

**Module**: `raster.py`

Applies empirical adjustments to geology-based Vs30 values based on New Zealand-specific calibration:

1. **Slope-based adjustments**: For certain geology categories, steeper terrain indicates more consolidated material with higher Vs30. Uses log-linear interpolation between slope thresholds.

2. **Coastal distance adjustments**: Near-coast sediments (alluvium, floodplain deposits) tend to be unconsolidated with lower Vs30. Values increase with distance from the coast.

### Stage 4: Spatial Adjustment

**Module**: `spatial.py`

The spatial adjustment stage uses Multivariate Normal (MVN) conditioning to update raster pixels based on nearby observations. This is the most computationally intensive stage.

**Algorithm**:

1. **Identify affected pixels**: Before computing updates, the algorithm first identifies which pixels are within `max_dist_m` of any observation. This uses a bounding-box search to efficiently find candidate pixels.

   For clustered observations (e.g., CPT data), an optimization is applied: only every Nth observation within each cluster is checked when determining affected pixels. Since clustered points are spatially close together, they largely affect the same pixels, so checking every Nth point still covers most of the cluster's spatial footprint. This subsampling may miss a small number of pixels at the outer edges of a cluster's influence zone—pixels that would have received only a weak update anyway since they are near the `max_dist_m` boundary. The parameter `obs_subsample_step_for_clustered` in `config.yaml` controls this behavior (default: 100). Setting it to 1 will check every observation in each cluster for complete coverage, but this significantly increases computation time for large clustered datasets.

2. **Build covariance matrices**: For each affected pixel, a covariance matrix is constructed relating the pixel to nearby observations using an exponential correlation function:
   ```
   correlation = exp(-distance / phi)
   ```
   where `phi` is the correlation length (different for geology and terrain models).

3. **MVN conditioning**: The posterior Vs30 and uncertainty are computed using standard MVN conditioning formulas, which adjust the model prediction toward observed values based on spatial correlation.

4. **Apply updates**: Updated values are written to the output raster.

**Key parameters** (from `config.yaml`):
- `phi_geology`, `phi_terrain`: Correlation lengths (meters)
- `max_dist_m`: Maximum distance to consider observations
- `max_points`: Maximum observations per pixel update
- `noisy`: Whether to apply uncertainty-based noise weighting
- `cov_reduc`: Covariance reduction for dissimilar Vs30 values

### Stage 5: Model Combination

**Module**: `cli.py`

The final stage combines geology and terrain model outputs using weighted averaging in log-space. This produces a geometric mean of the two models, which is appropriate for log-normally distributed Vs30 values.

**Combination methods** (configured via `combination_method` in `config.yaml`):

1. **Ratio-based weighting** (default): Set `combination_method` to a float value representing the ratio of geology weight to terrain weight:
   - `1.0` = equal weighting (both models contribute equally)
   - `2.0` = geology has twice the weight of terrain
   - `0.5` = terrain has twice the weight of geology

   The weights are computed as: `w_geology = ratio / (ratio + 1)`, `w_terrain = 1 / (ratio + 1)`

2. **Standard deviation weighting**: Set `combination_method: "standard_deviation_weighting"` to weight models inversely by their uncertainty (lower uncertainty = more weight). The weighting uses the formula: `weight ~ (sigma^2)^-k`, where `k` is controlled by the `k_value` parameter.
   - Higher `k_value` (e.g., 5.0) gives more weight to the model with lower uncertainty
   - Lower `k_value` (e.g., 1.0) makes the weighting less sensitive to uncertainty differences
   - Default is `k_value: 3.0`

**Log-space combination**: The combination is performed in log-space (geometric averaging):
```
combined_vs30 = exp(w_g * log(geology_vs30) + w_t * log(terrain_vs30))
```

For example, with equal weighting (`ratio=1.0`) and inputs of 200 and 400 m/s, the result is approximately 283 m/s (geometric mean), not 300 m/s (arithmetic mean).

**Combined standard deviation**: The output uncertainty accounts for both the individual model uncertainties and the difference between model predictions using the mixture of log-normals formula.

## Configuration

All parameters are centralized in `config.yaml`. Key sections:

- **Grid parameters**: Domain bounds and resolution (NZTM2000 coordinates)
- **Spatial parameters**: Correlation lengths, max distance, max observations
- **Bayesian parameters**: Prior weight, minimum sigma, DBSCAN settings
- **Hybrid parameters**: Slope and coastal distance thresholds
- **File paths**: Input/output locations

## Coordinate System

All coordinates use NZTM2000 (EPSG:2193) in meters. The `compute-at-locations` command accepts WGS84 lat/lon input and converts internally.

## Scientific Reference

The methodology is based on:

> Foster, K.M., et al. (2019). "A Vs30 Map for New Zealand Based on Geologic and Terrain Proxy Variables and Field Measurements"

See `reference_papers/foster_2019_nz_vs30_map.pdf` for detailed scientific background.

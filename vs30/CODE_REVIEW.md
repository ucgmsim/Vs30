# Code Review: vs30 Codebase

This document contains code review findings for the vs30 seismic hazard Vs30 modelling codebase.
The review focuses on improving readability for scientific researchers, removing unnecessary code,
and identifying potential improvements.

---

## Table of Contents

1. [General Observations](#general-observations)
2. [config.yaml](#configyaml)
3. [constants.py](#constantspy)
4. [utils.py](#utilspy)
5. [category.py](#categorypy)
6. [raster.py](#rasterpy)
7. [spatial.py](#spatialpy)
8. [cli.py](#clipy)
9. [Summary of Priority Items](#summary-of-priority-items)

---

## General Observations

### Strengths
- Good use of dataclasses for structured data (ObservationData, PixelData, etc.)
- Consistent use of type hints throughout
- Well-structured config.yaml with helpful comments
- Good separation of concerns between modules

### Areas for Improvement
- Several functions are quite long and could benefit from refactoring
- Some unnecessary variable assignments that obscure the code
- Inconsistent documentation depth across modules
- Some module-level docstrings could be more descriptive for scientific readers

---

## config.yaml

### Issues

**1. Missing Units in Comments (Lines 62-77)**
- Location: Lines 62-77
- Issue: Some parameters have units specified, others don't
- Recommendation: Add units consistently to all parameters

```yaml
# Current (inconsistent):
max_dist_m: 10000
phi_geology: 1407

# Recommended:
max_dist_m: 10000  # Maximum distance in meters for considering observations
phi_geology: 1407  # Correlation length in meters for geology model
```

**2. Cryptic Parameter Names (Lines 88, 168)**
- Location: `eps: 15000.0` (line 88), `min_dist_enforced: 0.1` (line 168)
- Issue: `eps` is not immediately clear to non-ML researchers; `min_dist_enforced` unclear what it enforces
- Recommendation: Add more descriptive comments or rename:

```yaml
# eps is the DBSCAN epsilon parameter - maximum distance between two samples
# to be considered in the same neighborhood (in meters)
eps: 15000.0

# Minimum distance enforced in correlation calculations to prevent
# division by zero when points are at the same location (in meters)
min_dist_enforced: 0.1
```

**3. Missing Scientific Context for Hybrid Parameters (Lines 128-165)**
- Location: Lines 128-165
- Issue: The hybrid model parameters lack scientific context explaining why these specific geology groups have modifications
- Recommendation: Add comments explaining the geophysical reasoning:

```yaml
# --- Hybrid Geology Model Parameters ---
# These modifications adjust Vs30 values for specific geology categories
# based on empirical relationships between Vs30, slope, and coastal distance.
#
# mod6 (Alluvium, GID 4): Near-coast alluvial deposits typically have lower
# Vs30 due to unconsolidated sediments. Values increase with distance from coast.
#
# mod13 (Floodplain, GID 10): Similar coastal proximity effect for floodplain deposits.
```

---

## constants.py

### Issues

**1. Missing Module-Level Documentation**
- Location: Lines 1-4
- Issue: The module docstring is minimal and doesn't explain the design pattern
- Recommendation: Expand the docstring:

```python
"""
Centralized constants and configuration loading for the vs30 package.

This module serves as a single source of truth for all configuration parameters.
Values are loaded from config.yaml at import time and exported as module-level
constants. This pattern allows:
1. Easy access via `from vs30 import constants` then `constants.MAX_DIST_M`
2. Validation that all required config keys exist at startup
3. Type-safe access to configuration values

Usage:
    from vs30 import constants

    max_distance = constants.MAX_DIST_M
    phi = constants.PHI["geology"]
"""
```

**2. Missing Type Annotations for Constants**
- Location: Throughout file
- Issue: Constants don't have explicit type annotations, making it harder to understand expected types
- Recommendation: While Python doesn't require this, adding comments or type hints improves clarity:

```python
# General Configuration
MAX_DIST_M: int = _config["max_dist_m"]  # Maximum observation distance in meters
MAX_POINTS: int = _config["max_points"]  # Maximum observations per pixel
```

**3. Unused Import**
- Location: Line 4
- Issue: `yaml` is imported but `_load_config()` could use a more robust pattern
- Recommendation: Consider adding error handling for malformed YAML

---

## utils.py

### Issues

**1. Missing Module Docstring**
- Location: Top of file
- Issue: No module-level documentation explaining the purpose
- Recommendation: Add a descriptive docstring:

```python
"""
Utility functions for the vs30 package.

This module provides:
- Configuration file loading
- Distance and correlation calculations used in MVN spatial processing

The correlation function implements an exponential decay model commonly used
in geostatistics for spatial interpolation of soil properties.
"""
```

**2. Incomplete Docstring for correlation_function (Line 73)**
- Location: Lines 73-93
- Issue: The Notes section mentions the formula but doesn't explain the scientific meaning
- Recommendation: Expand the scientific context:

```python
"""
Calculate correlation function from distances.

This implements an exponential correlation model used in geostatistics:
    correlation = exp(-distance / phi)

where phi (correlation length) represents the distance at which correlation
decays to approximately 37% (1/e). This model assumes spatial stationarity
and isotropy - the correlation depends only on distance, not direction.

Parameters
----------
...

Scientific Background
--------------------
In Vs30 modelling, spatial correlation captures the tendency for nearby
soil properties to be similar. The phi parameter is typically calibrated
from empirical data - larger phi values indicate more spatially continuous
geology, while smaller values indicate more heterogeneous conditions.
"""
```

**3. Unused Variable Assignment Potential**
- Location: Line 93
- Issue: `np.maximum(constants.MIN_DIST_ENFORCED, distances)` is computed inline, which is fine, but a comment explaining why would help
- Recommendation: Add inline comment:

```python
# Enforce minimum distance to prevent correlation = 1/exp(0) = 1 for co-located points
return 1 / np.exp(np.maximum(constants.MIN_DIST_ENFORCED, distances) / phi)
```

---

## category.py

### Issues

**1. Incomplete Module Docstring**
- Location: Lines 1-9
- Issue: Missing explanation of the Bayesian update methodology
- Recommendation: Add scientific context at the module level:

```python
"""
Functions for relating measurement vs30 values to geology/terrain categories
and performing Bayesian updates of categorical mean and standard deviation values.

Scientific Background
--------------------
This module implements a two-stage Bayesian updating process:

1. Clustered observations (e.g., CPT data) are processed with DBSCAN spatial
   clustering to account for sampling bias in geotechnical investigations,
   which tend to be concentrated in urban/infrastructure areas.

2. Independent observations (e.g., direct Vs30 measurements) then refine the
   bias-corrected model.

The Bayesian update assumes log-normal distributions for Vs30 values, which is
standard practice in seismic hazard analysis.
"""
```

**2. Cryptic Function Names with Leading Underscores**
- Location: `_new_mean`, `_new_var` (lines 114, 138)
- Issue: These are mathematical functions but their names don't convey meaning
- Recommendation: Either rename or add clearer docstrings:

```python
def _bayesian_posterior_mean(mu_0, n0, var, y):
    """
    Compute the posterior mean for a log-normal Bayesian update.

    This implements the conjugate prior update for a normal distribution
    in log-space, which is equivalent to assuming Vs30 follows a log-normal
    distribution.
    ...
    """
```

**3. Long Function: update_with_independent_data (Lines 169-312)**
- Location: Lines 169-312 (143 lines)
- Issue: Function is too long and handles multiple responsibilities
- Recommendation: Extract helper functions:
  - `_identify_prior_columns()` - handle column name resolution
  - `_initialize_posterior_columns()` - setup output DataFrame
  - `_update_single_category()` - perform update for one category

**4. Long Function: update_with_clustered_data (Lines 376-482)**
- Location: Lines 376-482 (106 lines)
- Issue: Function is moderately long with complex logic
- Recommendation: Add more inline comments explaining the cluster weighting logic:

```python
# Overall N is one per cluster, clusters labeled -1 are individual clusters
# This ensures spatially clustered measurements don't over-weight the posterior
n = len(clusters)
if -1 in clusters.index:
    # Points not in any cluster (-1 label) each count as individual observations
    n += clusters[-1] - 1
```

**5. Unused Parameter: model_type in update_with_clustered_data**
- Location: Line 377
- Issue: `model_type` parameter is documented as deprecated but still required
- Recommendation: Either remove it or add a deprecation warning:

```python
def update_with_clustered_data(
    prior_df: pd.DataFrame,
    sites_df: pd.DataFrame,
    model_type: str = "geology"  # Deprecated, kept for API compatibility
) -> pd.DataFrame:
```

**6. Inefficient DataFrame Iteration (Lines 268-310)**
- Location: Lines 268-310
- Issue: Nested iteration over DataFrames is slow for large datasets
- Note: Readability should be prioritized over performance here, but consider adding a comment:

```python
# Note: This nested iteration is intentional for clarity. For very large
# observation datasets (>10,000 observations), consider vectorizing.
for category_row_idx, category_row in updated_categorical_model_df.iterrows():
```

---

## raster.py

### Issues

**1. Incomplete Module Docstring**
- Location: Lines 1-6
- Issue: Doesn't explain the hybrid geology model concept
- Recommendation: Expand with scientific context:

```python
"""
Common functions for creating categorical VS30 rasters from terrain and geology data.

This module provides:
- Category ID raster creation (terrain from IwahashiPike, geology from QMAP)
- VS30 value mapping from categorical models
- Hybrid geology modifications based on slope and coastal distance

Hybrid Geology Model
-------------------
The geology-based Vs30 model includes empirical modifications for certain
geology categories where Vs30 is known to vary with:
1. Slope - steeper terrain generally indicates more consolidated material
2. Coastal distance - near-coast sediments are often unconsolidated

These modifications are based on New Zealand-specific calibration studies.
"""
```

**2. Unnecessary Module-Level Variable Assignments (Lines 25-38)**
- Location: Lines 25-38
- Issue: Variables like `XMIN`, `XMAX`, etc. are assigned from constants but only used in function default parameters
- Recommendation: Use constants directly in function signatures:

```python
# Current:
XMIN = constants.GRID_XMIN
...
def create_category_id_raster(..., xmin: float = XMIN, ...):

# Recommended:
def create_category_id_raster(..., xmin: float = constants.GRID_XMIN, ...):
```

This removes 8 unnecessary module-level variables and makes the code more explicit about where values come from.

**3. Long Function: create_category_id_raster (Lines 165-280)**
- Location: Lines 165-280 (115 lines)
- Issue: Function handles both terrain and geology cases with significantly different logic
- Recommendation: Split into two functions:
  - `_create_terrain_id_raster()` - handle terrain case
  - `_create_geology_id_raster()` - handle geology case
  - Keep `create_category_id_raster()` as a dispatcher

**4. Long Function: create_vs30_raster_from_ids (Lines 347-462)**
- Location: Lines 347-462 (115 lines)
- Issue: Function is moderately long
- Recommendation: Extract CSV loading into a helper:

```python
def _load_vs30_mapping_from_csv(csv_path: str) -> dict[int, tuple[float, float]]:
    """Load ID-to-VS30 mapping from CSV file."""
    ...
```

**5. Long Function: create_coast_distance_raster (Lines 465-571)**
- Location: Lines 465-571 (106 lines)
- Issue: Complex function with GDAL operations
- Recommendation: Add more section comments explaining the algorithm:

```python
# Step 1: Rasterize land polygons
# We rasterize the coastline shapefile to create a binary raster where
# 1 = land, 0 = water (or outside bounds)

# Step 2: Compute proximity distances
# GDAL's ComputeProximity calculates distance from each land pixel to
# the nearest water pixel (value=0), giving us coastal distance

# Step 3: Crop to study domain
# If we extended beyond the study domain for accuracy, crop back
```

**6. Long Function: apply_hybrid_geology_modifications (Lines 627-773)**
- Location: Lines 627-773 (146 lines)
- Issue: Function is too long with multiple modification types
- Recommendation: Split into:
  - `_apply_sigma_reduction()` - handle standard deviation modifications
  - `_apply_slope_based_vs30()` - handle hybrid slope interpolation
  - `_apply_coastal_distance_vs30()` - handle mod6/mod13

**7. Unnecessary Variable Assignment (Lines 419-421)**
- Location: Lines 419-421
- Issue: `transform` and `crs` are extracted but could be used inline:

```python
# Current:
profile = src.profile.copy()
transform = src.transform
crs = src.crs
...
"crs": crs,
"transform": transform,

# Recommended (if only used once):
output_profile = {
    ...
    "crs": profile["crs"],
    "transform": profile["transform"],
}
```

**8. Magic Numbers in Comments (Lines 737, 756-760)**
- Location: Lines 737, 756-760
- Issue: Comments contain magic numbers that should reference config values:

```python
# Current:
# Formula: 240 + (500-240) * (dist - 8000) / (20000 - 8000)

# Recommended:
# Formula: vs30_min + (vs30_max - vs30_min) * (dist - dist_min) / (dist_max - dist_min)
# Where vs30_min/max and dist_min/max come from config (hybrid_mod6_*)
```

---

## spatial.py

### Issues

**1. Confusing Self-Import (Line 18)**
- Location: Line 18
- Issue: `from vs30 import spatial as mvn` - the module imports itself as `mvn`
- Recommendation: This is confusing. Either:
  - Remove the alias and use `spatial.` prefix
  - Or rename internal uses to avoid the self-import

**2. Module Docstring Too Brief**
- Location: Lines 1-5
- Issue: Doesn't explain what MVN means or the scientific approach
- Recommendation:

```python
"""
Multivariate Normal (MVN) distribution-based spatial adjustment for Vs30 values.

This module implements spatial interpolation using MVN conditioning, which
adjusts model Vs30 predictions based on nearby measurements. The approach:

1. For each pixel near observations, builds a covariance matrix relating
   the pixel to nearby measurements
2. Uses MVN conditioning to compute the posterior (updated) Vs30 and
   uncertainty given the observations
3. Applies updates to create spatially-adjusted Vs30 maps

The covariance structure uses an exponential correlation function, with
separate correlation lengths (phi) for geology and terrain models.

Key Parameters
--------------
- phi: Correlation length controlling spatial smoothness
- max_dist_m: Maximum distance to consider observations
- max_points: Maximum number of observations per pixel update
- noisy: Whether to apply noise weighting based on observation uncertainty
- cov_reduc: Covariance reduction factor for dissimilar Vs30 values
"""
```

**3. Dataclass Documentation Incomplete**
- Location: Lines 28-38, 41-48, 51-59, 62-72
- Issue: Dataclass docstrings don't explain the scientific meaning of fields
- Recommendation: Add field descriptions:

```python
@dataclass
class ObservationData:
    """
    Bundled observation data for spatial processing.

    Attributes
    ----------
    locations : ndarray, shape (n_obs, 2)
        Observation coordinates as [easting, northing] in NZTM (meters).
    vs30 : ndarray, shape (n_obs,)
        Measured Vs30 values in m/s.
    model_vs30 : ndarray, shape (n_obs,)
        Model-predicted Vs30 at observation locations (for computing residuals).
    model_stdv : ndarray, shape (n_obs,)
        Model standard deviation at observation locations (log-space).
    residuals : ndarray, shape (n_obs,)
        Log residuals: log(measured_vs30 / model_vs30). Positive values indicate
        the model under-predicts Vs30 at this location.
    omega : ndarray, shape (n_obs,)
        Noise weights accounting for observation uncertainty. When noisy=True,
        omega < 1 down-weights observations with high uncertainty.
    uncertainty : ndarray, shape (n_obs,)
        Observation uncertainty (standard deviation in log-space).
    """
```

**4. Long Function: prepare_observation_data (Lines 227-369)**
- Location: Lines 227-369 (142 lines)
- Issue: Function is too long with geology-specific logic embedded
- Recommendation: Extract geology modifications:

```python
def _apply_hybrid_modifications_at_observations(
    obs_locs, model_vs30, model_stdv, model_ids, output_dir, raster_data, hybrid_params
) -> tuple[np.ndarray, np.ndarray]:
    """Apply hybrid geology modifications at observation locations."""
    ...
```

**5. Long Function: compute_mvn_updates (Lines 818-952)**
- Location: Lines 818-952 (134 lines)
- Issue: Function is long with progress tracking logic mixed with computation
- Recommendation: The function structure is reasonable but could use more section comments

**6. Repeated Empty ObservationData Creation (Lines 582-604)**
- Location: Lines 582-590 and 596-604
- Issue: Identical empty ObservationData creation repeated twice
- Recommendation: Extract to helper:

```python
def _empty_observation_data() -> ObservationData:
    """Create an empty ObservationData object for no-observation cases."""
    return ObservationData(
        locations=np.empty((0, 2)),
        vs30=np.empty(0),
        model_vs30=np.empty(0),
        model_stdv=np.empty(0),
        residuals=np.empty(0),
        omega=np.empty(0),
        uncertainty=np.empty(0),
    )
```

**7. Validation Functions Use Assertions (Lines 183-224)**
- Location: Lines 183-224
- Issue: Using `assert` for validation that should raise proper exceptions
- Note: For scientific code prioritizing readability, assertions are acceptable but consider:

```python
# Current:
assert raster_data.vs30.shape == raster_data.stdv.shape, "Band shapes must match"

# Alternative for production code:
if raster_data.vs30.shape != raster_data.stdv.shape:
    raise ValueError(
        f"Band shapes must match: vs30 {raster_data.vs30.shape} != stdv {raster_data.stdv.shape}"
    )
```

**8. Docstring Incomplete for compute_mvn_updates (Lines 829-848)**
- Location: Lines 829-848
- Issue: Docstring mentions legacy behavior but parameters don't include a legacy flag
- Recommendation: Remove or clarify the legacy comment:

```python
"""
Compute MVN updates for all affected pixels.

Parameters
----------
...
model_name : str
    Model name ("geology" or "terrain").

Returns
-------
...
"""
```

---

## cli.py

### Issues

**1. Module Docstring Incorrect**
- Location: Lines 1-6
- Issue: Says "MVN updates for geology and terrain models" but this is the CLI module
- Recommendation:

```python
"""
Command-line interface for the vs30 package.

This module provides CLI commands for the full Vs30 modelling pipeline:
- update-categorical-vs30-models: Bayesian update of categorical Vs30 values
- make-initial-vs30-raster: Create initial Vs30 rasters from categories
- adjust-geology-vs30-by-slope-and-coastal-distance: Apply hybrid modifications
- spatial-fit: MVN-based spatial adjustment
- combine: Combine geology and terrain models
- full-pipeline: Run the complete workflow

Usage:
    vs30 full-pipeline  # Run everything with default config
    vs30 --help         # See all available commands
"""
```

**2. Very Long Functions**
- `update_categorical_vs30_models`: Lines 33-311 (278 lines)
- `spatial_fit`: Lines 541-714 (173 lines)
- `full_pipeline_for_geology_or_terrain`: Lines 841-1032 (191 lines)
- `full_pipeline`: Lines 1156-1302 (146 lines)
- Recommendation: These are CLI command functions which tend to be procedural, but consider extracting validation logic and repeated patterns.

**3. Duplicate Config Loading (Lines 641-643)**
- Location: Lines 641-643 in spatial_fit
- Issue: Config is loaded twice:

```python
# Line 595-596:
cfg_path = Path(__file__).parent / "config.yaml"
cfg = load_config(cfg_path)

# Lines 641-643:
cfg_path = Path(__file__).parent / "config.yaml"
cfg = load_config(cfg_path)
```

- Recommendation: Remove the duplicate load

**4. Unnecessary Variable Assignments (Lines 598-603)**
- Location: Lines 598-603
- Issue: Variables are extracted from config but each is only used once:

```python
# Current:
max_dist_m = cfg["max_dist_m"]
max_points = cfg["max_points"]
phi = cfg[f"phi_{model_type}"]
noisy = cfg["noisy"]
cov_reduc = cfg["cov_reduc"]

# Then used as:
bbox_result = spatial.find_affected_pixels(raster_data, obs_data, max_dist_m=max_dist_m)
```

- Recommendation: For clarity, either:
  - Use directly: `max_dist_m=cfg["max_dist_m"]`
  - Or keep variables but add comments explaining the parameters

**5. Inconsistent Error Handling Pattern**
- Location: Throughout CLI functions
- Issue: Some functions use try/except/raise typer.Exit(1), others don't
- Recommendation: Standardize error handling across all commands

**6. Hardcoded Placeholder Value (Lines 161-163)**
- Location: Lines 161-163
- Issue: `-9999` is hardcoded as a placeholder value filter:

```python
categorical_model_df = categorical_model_df[
    categorical_model_df["mean_vs30_km_per_s"] != -9999
]
```

- Recommendation: Move to config or constants:

```python
categorical_model_df = categorical_model_df[
    categorical_model_df["mean_vs30_km_per_s"] != constants.MODEL_NODATA
]
```

Note: Check if -9999 matches the config `model_nodata: -32767`

**7. Plot Formatting Constants Could Be in Config (Lines 782-817)**
- Location: Lines 782-817 in plot_posterior_values
- Issue: Plot formatting (offset=0.2, capsize=5, fontsize values) are hardcoded
- Note: These are probably fine as hardcoded values for plot aesthetics, but could be moved to config if users need customization

**8. Unused Variable (Lines 982-987)**
- Location: Lines 982-987
- Issue: `id_raster_name` is computed but then overwritten:

```python
id_raster_name = (
    constants.GEOLOGY_ID_FILENAME
    if model_type == "geology"
    else constants.TERRAIN_ID_FILENAME
)
id_raster = output_dir / id_raster_name

# ...then immediately:
id_raster = output_dir / "gid.tif"  # Overwrites!
```

- Recommendation: Remove the dead code or fix the logic

---

## Summary of Priority Items

### High Priority (Affects Readability/Correctness)

1. **cli.py:982-994** - Dead code where `id_raster` is computed then overwritten
2. **cli.py:641-643** - Duplicate config loading
3. **spatial.py:18** - Confusing self-import as `mvn`
4. **cli.py:1-6** - Incorrect module docstring
5. **cli.py:161-163** - Hardcoded -9999 doesn't match config's model_nodata

### Medium Priority (Improves Maintainability)

6. **raster.py:25-38** - Unnecessary module-level variable assignments
7. **category.py:169-312** - Long function should be split
8. **raster.py:627-773** - Long function should be split
9. **spatial.py:582-604** - Repeated empty ObservationData creation
10. **category.py:377** - Unused deprecated parameter

### Lower Priority (Documentation Improvements)

11. Add scientific context to module docstrings (all modules)
12. Add units consistently to config.yaml comments
13. Expand dataclass field documentation in spatial.py
14. Add inline comments explaining geophysical reasoning in hybrid model code

### Style/Consistency

15. Standardize error handling pattern in CLI functions
16. Consider type annotations for constants
17. Add more section comments in long functions

---

## Notes for Implementation

When addressing these items:

1. **Prioritize readability over minor performance gains** - The primary users are scientific researchers
2. **Add comments explaining "why" not just "what"** - Scientific context is crucial
3. **Test after each change** - Run `vs30 full-pipeline` to verify functionality
4. **Keep functions focused** - Extract helpers when a function does multiple things
5. **Use descriptive names** - Variable names should reflect scientific meaning

---

*Review conducted: January 2026*
*Reviewer: Code Review Assistant*

# Design: Original Foster (2019) Model Version

**Date**: 2026-03-23
**Status**: Approved
**Context**: [dev/differences_between_foster_2019_and_modified_foster_2019.md](/dev/differences_between_foster_2019_and_modified_foster_2019.md)

## Goal

Enable the original Foster (2019) Vs30 model as a selectable model version
(`foster_2019`) via a new YAML config file, faithfully reproducing the published
model from Foster et al. (2019, *Earthquake Spectra* 35(4)).

The modified_foster_2019 model will continue to work as before, with its config
updated to use the same new config schema.

## Differences to Implement

Four differences between the original and modified models (documented in
`dev/differences_between_foster_2019_and_modified_foster_2019.md`):

1. **No coastal distance modifications** — original uses slope only, no distance-to-coast
   adjustments for alluvium (G06) or floodplain (G13).
2. **Matérn spatial correlation for geology** — original uses Matérn (κ=0.9, range=20 km,
   sill=0.15, nugget=0.05) instead of exponential (φ=1407 m). Terrain uses exponential
   in both versions.
3. **Strict Q3 exclusion** — original excludes all Kaiser Q3 stations. Modified retains
   Q3 stations with 3-character names.
4. **Same categorical posteriors** — both versions use the same pre-computed posteriors
   from the original R code.


## Config YAML Schema

All config files must specify `geology_correlation` and `terrain_correlation` sections
(no defaults, no backward compatibility — all configs updated).

### foster_2019.yaml

```yaml
# Original Foster (2019) Vs30 Model Configuration
# Reproduces the published model from Foster et al. (2019, Earthquake Spectra).

# --- Categorical model files ---
# Same posteriors as modified_foster_2019 (computed by the original R code).
geology_categorical_csv: geology_model_posterior_from_modified_foster_2019_mean_and_standard_deviation.csv
terrain_categorical_csv: terrain_model_posterior_from_modified_foster_2019_mean_and_standard_deviation.csv

# --- Observation files ---
# Original excludes ALL Kaiser Q3 stations.
independent_observations_csv: foster_2019_measured_vs30_independent_observations.csv
clustered_observations_csv:

# --- Pipeline settings ---
do_bayesian_update: false
noisy: true
combination_method: ratio
combine_ratio: 1.0

# --- Hybrid geology modifications ---
# Original: slope only, no coastal distance adjustments.
apply_coastal_distance_mod: false

# --- Spatial correlation ---
# Geology: Matérn (κ=0.9) as fitted in the original R code.
# Note: paper suggests MVN replaces nugget with omega weighting,
# but R code uses both. We match the R code for fidelity.
geology_correlation:
  model: matern
  range: 20000
  sill: 0.15
  nugget: 0.05
  kappa: 0.9

# Terrain: Exponential. The original R code uses Matérn with κ=0.5,
# which is mathematically equivalent to exponential.
terrain_correlation:
  model: exponential
  phi: 993
```

### modified_foster_2019.yaml (updated)

```yaml
# Modified Foster (2019) Vs30 Model Configuration

geology_categorical_csv: geology_model_posterior_from_modified_foster_2019_mean_and_standard_deviation.csv
terrain_categorical_csv: terrain_model_posterior_from_modified_foster_2019_mean_and_standard_deviation.csv

independent_observations_csv: modified_foster_2019_measured_vs30_independent_observations.csv
clustered_observations_csv:

do_bayesian_update: false
noisy: true
combination_method: ratio
combine_ratio: 1.0

apply_coastal_distance_mod: true

geology_correlation:
  model: exponential
  phi: 1407

terrain_correlation:
  model: exponential
  phi: 993
```

### Other configs (jaehwi_v1p0, viktor_cpt_clustering)

Add the required fields with their current effective values:

```yaml
apply_coastal_distance_mod: true

geology_correlation:
  model: exponential
  phi: 1407

terrain_correlation:
  model: exponential
  phi: 993
```


## Code Changes

### A. Correlation Functions (`utils.py`)

Rename `correlation_function()` to `exponential_correlation_function()`. Add
`matern_correlation_function()`.

**Exponential** (rename of existing):
```python
def exponential_correlation_function(
    distances: np.ndarray,
    phi: float,
    min_dist: float = constants.MIN_DIST_ENFORCED,
) -> np.ndarray:
    return np.exp(-np.maximum(min_dist, distances) / phi)
```

**Matérn** (new):
```python
def matern_correlation_function(
    distances: np.ndarray,
    range_m: float,
    sill: float,
    nugget: float,
    kappa: float,
    min_dist: float = constants.MIN_DIST_ENFORCED,
) -> np.ndarray:
    """
    Matérn correlation with nugget.

    Uses scipy.special.kv (modified Bessel function of the second kind)
    and scipy.special.gamma.

    The range parameter follows gstat's convention where it is a scale
    parameter (NOT the practical range). The sqrt(2*kappa) scaling is
    NOT applied because R's gstat package already absorbs it into the
    range parameter — i.e., gstat's vgm(range=r) uses r directly as
    the denominator in exp(-d/r) * polynomial, not sqrt(2*kappa)*d/r.
    This must be verified against R's gstat output during implementation.

    At d near 0 (d=min_dist), returns approximately sill/(sill+nugget),
    reflecting the nugget discontinuity.
    """
    d = np.maximum(min_dist, distances)
    scaled = d / range_m
    rho = (2**(1-kappa) / gamma(kappa)) * (scaled**kappa) * kv(kappa, scaled)
    # Clamp any NaN from numerical edge cases (kv can overflow for very small d)
    rho = np.where(np.isfinite(rho), rho, 1.0)
    return rho * sill / (sill + nugget)
```

**Implementation note on Matérn parameterization**: The standard textbook Matérn uses
`sqrt(2*kappa) * d / range` as the scaled argument. However, R's gstat package uses
a different parameterization where the range parameter is a scale parameter without
the `sqrt(2*kappa)` factor. Since the original R code's fitted parameters
(range=20,000 m) come from gstat's `vgm()` function, we must match gstat's convention.
**This must be verified during implementation** by computing correlation values from
the R code and comparing against the Python implementation.

Both functions have the same interface: `(distances: ndarray) → ndarray` of correlations.

**Nugget interaction with existing code**: The `compute_spatial_adjustment_for_pixel()`
function in `spatial.py` already applies a "variance shrinkage" using `corr_zero`
(the correlation at d=0, which is ~1.0 for exponential). For Matérn with nugget,
`corr_zero` will be ~0.75, meaning the prior variance is reduced to 75% — this is the
intended behavior from the nugget. This operates *independently* from the omega noise
weighting (`noisy=true`), which handles per-observation measurement uncertainty. The
paper suggests these capture different phenomena (nugget = micro-scale spatial
variation; omega = measurement quality), and the R code applies both simultaneously.

### B. Config Resolution

Add a helper (in `cli.py` or a small module) that resolves a correlation config
section into a callable. Use `functools.partial` (not lambdas) because the callable
must be picklable for multiprocessing workers in `parallel.py`.

```python
def resolve_correlation_function(config_section: dict) -> Callable[[np.ndarray], np.ndarray]:
    model = config_section["model"]
    if model == "exponential":
        return functools.partial(
            exponential_correlation_function, phi=config_section["phi"]
        )
    elif model == "matern":
        return functools.partial(
            matern_correlation_function,
            range_m=config_section["range"],
            sill=config_section["sill"],
            nugget=config_section["nugget"],
            kappa=config_section["kappa"],
        )
    else:
        raise ValueError(f"Unknown correlation model: {model}")
```

This produces two picklable callables at config load time — one for geology, one
for terrain.

Config loading should validate that required keys are present (e.g., `geology_correlation`
and `terrain_correlation` must exist, and each must contain a `model` key with the
appropriate sub-keys for that model type). Raise a clear error early if missing.

### C. Pipeline Plumbing

Replace `model_type`-based phi lookup with the resolved correlation callable.
**`model_type` is NOT removed** — it continues to serve its other purposes (raster
creation dispatch, output filename selection, progress labels, etc.). Only its role
as a proxy for looking up `constants.PHI[model_type]` is replaced.

**Current call chain** (model_type threaded through to select phi):
```
cli.py → pipeline.compute_grid(..., model_type)
  → spatial.build_covariance_matrix(..., model_type)
    → utils.correlation_function(distances, constants.PHI[model_type])
```

**New call chain** (correlation callable threaded through):
```
cli.py → resolve geology_corr_fn, terrain_corr_fn from config
  → pipeline.compute_grid(..., geology_corr_fn, terrain_corr_fn)
    → pipeline.compute_spatial_adjustment_on_grid(..., corr_fn)
      → spatial.build_covariance_matrix(..., corr_fn)
        → corr_fn(distance_matrix)
```

Functions that need signature changes to accept `corr_fn` (grid pipeline):
- `pipeline.compute_spatial_adjustment_on_grid()`
- `pipeline.compute_grid()`
- `spatial.build_covariance_matrix()`
- `spatial.compute_spatial_adjustment_for_pixel()`
- `spatial.find_affected_pixels()`

Functions that need signature changes (points pipeline):
- `spatial.compute_spatial_adjustment_at_points()`
- `parallel.process_geology_at_points()`
- `parallel.process_terrain_at_points()`
- `parallel.LocationsChunkConfig` dataclass
- `parallel.run_parallel_locations()`
- `parallel.process_pixels_chunk()` — passes `corr_fn` via its dict-based config
  (picklable because we use `functools.partial`)

### D. Coastal Distance Flag

Add `apply_coastal_distance_mod` as a config key. Pass through
`pipeline.compute_hybrid_geology_arrays()` (which must also gain this parameter)
down to `raster.apply_hybrid_geology_modifications()`:

```python
# In pipeline.compute_hybrid_geology_arrays():
if apply_coastal_distance_mod:
    coast_dist_array = raster.compute_coast_distance_array(profile)
else:
    coast_dist_array = None  # skip expensive computation

raster.apply_hybrid_geology_modifications(
    vs30_array, stdv_array, id_array, slope_array, coast_dist_array,
    mod6=apply_coastal_distance_mod,
    mod13=apply_coastal_distance_mod,
)
```

The same flag must also be threaded through the points pipeline:
- `parallel.process_geology_at_points()` — conditionally skip
  `raster.compute_coastal_distance_at_points()`

When `false`:
- Alluvium (G06, internal GID 4) gets slope-based modification (already in code,
  currently skipped when mod6=True)
- Floodplain (G13, internal GID 10) gets no special treatment
- Coastal distance computation is skipped entirely (performance benefit)

### E. Model Registration (`constants.py`)

Add `FOSTER_2019` to `FixedModelVersion`:

```python
class FixedModelVersion(StrEnum):
    FOSTER_2019 = "foster_2019"
    MODIFIED_FOSTER_2019 = "modified_foster_2019"
    JAEHWI_V1P0 = "jaehwi_v1p0"
    VIKTOR_CPT_CLUSTERING = "viktor_cpt_clustering"
```

Add to `MODEL_VERSION_TO_CONFIG`:
```python
FixedModelVersion.FOSTER_2019: CONFIGS_DIR / "foster_2019.yaml",
```

### F. Remove Dead Constants (`constants.py`)

Remove (replaced by per-config correlation sections):
- `PHI_GEOLOGY`
- `PHI_TERRAIN`
- `PHI` dictionary

Keep (used only when `apply_coastal_distance_mod: true`):
- `HYBRID_MOD6_*` and `HYBRID_MOD13_*` constants


## New Observation CSV

**File**: `vs30/resources/observations/foster_2019_measured_vs30_independent_observations.csv`

**Derivation from existing CSV**: The modified_foster_2019 CSV retains some Kaiser Q3
stations (those with 3-character station names). To create the foster_2019 CSV, filter
the modified CSV by dropping the rows that were additionally retained:
drop rows where `source == "kaiseretal"` AND `q == 3.0` AND `len(station) == 3`.
This removes the ~60 Q3 broadband stations that the modified version retains but the
original excluded.

Equivalently, the original filter applied to the raw Kaiser data is simply: drop ALL
rows where `q == 3` (no station name exception).

**Creation script**: `dev/observations/foster_2019/create_foster_2019_observations.py`
- Documents the filtering logic for reproducibility
- Also includes the legacy-environment version for generating from raw sources


## Files Changed

| File | Change |
|------|--------|
| `vs30/utils.py` | Rename `correlation_function` → `exponential_correlation_function`; add `matern_correlation_function` |
| `vs30/spatial.py` | Accept `corr_fn` callable; update `build_covariance_matrix`, `compute_spatial_adjustment_for_pixel`, `find_affected_pixels`, `compute_spatial_adjustment_at_points` |
| `vs30/pipeline.py` | Thread `corr_fn` and `apply_coastal_distance_mod` through pipeline stages; update `compute_hybrid_geology_arrays` to conditionally skip coast distance computation |
| `vs30/parallel.py` | Thread `corr_fn` and `apply_coastal_distance_mod`; update `process_geology_at_points`, `process_terrain_at_points`, `LocationsChunkConfig`, `run_parallel_locations`, `process_pixels_chunk` |
| `vs30/cli.py` | Resolve correlation config to callables via `functools.partial`; pass `apply_coastal_distance_mod`; add config validation for required fields |
| `vs30/constants.py` | Add `FOSTER_2019` to StrEnum; remove `PHI_GEOLOGY`, `PHI_TERRAIN`, `PHI` dict |
| `vs30/configs/foster_2019.yaml` | New config file |
| `vs30/configs/modified_foster_2019.yaml` | Add correlation and coastal distance fields |
| `vs30/configs/jaehwi_v1p0.yaml` | Add correlation and coastal distance fields |
| `vs30/configs/viktor_cpt_clustering.yaml` | Add correlation and coastal distance fields |
| `vs30/resources/observations/foster_2019_measured_vs30_independent_observations.csv` | New observation CSV |
| `dev/observations/foster_2019/create_foster_2019_observations.py` | New creation script |
| `tests/` | Update test configs with new required fields |


## Testing

- Regression tests with `modified_foster_2019` must continue to pass (no behavioral change)
- New `foster_2019` model should run without errors
- Matérn correlation function should be unit-tested against known values (e.g., κ=0.5
  should match exponential; verify against scipy or R's gstat output)

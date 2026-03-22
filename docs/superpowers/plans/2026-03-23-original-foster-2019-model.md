# Original Foster (2019) Model Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Enable the original Foster (2019) Vs30 model as a selectable model version with config-driven correlation functions (Matern vs exponential), a coastal distance modification flag, and strict Q3 observation filtering.

**Architecture:** Replace hard-coded PHI constants with config-driven correlation function callables (`functools.partial`) threaded through the spatial→pipeline→parallel→CLI call chain. Each config YAML explicitly declares its correlation model (exponential or Matern) and coastal distance behavior. The existing `model_type` parameter remains for non-correlation purposes (raster dispatch, filenames, labels).

**Tech Stack:** Python, NumPy, SciPy (`scipy.special.kv`, `scipy.special.gamma`), `functools.partial`, YAML configs, pandas

**Spec:** `docs/superpowers/specs/2026-03-23-original-foster-2019-model-design.md`

---

## File Structure

| File | Role |
|------|------|
| `vs30/utils.py` | Rename `correlation_function` → `exponential_correlation_function`; add `matern_correlation_function` |
| `vs30/cli.py` | Add `resolve_correlation_function()` helper; wire config → corr_fn in `points` and `grid` commands |
| `vs30/constants.py` | Add `FOSTER_2019` to `FixedModelVersion`; remove `PHI_GEOLOGY`, `PHI_TERRAIN`, `PHI` dict |
| `vs30/spatial.py` | Replace `model_type` PHI lookup with `corr_fn` callable in all MVN functions |
| `vs30/pipeline.py` | Thread `corr_fn` and `apply_coastal_distance_mod` through grid + points pipelines |
| `vs30/parallel.py` | Thread `corr_fn` and `apply_coastal_distance_mod` through parallel workers |
| `vs30/configs/foster_2019.yaml` | New config for original model |
| `vs30/configs/modified_foster_2019.yaml` | Add correlation and coastal distance fields |
| `vs30/configs/jaehwi_v1p0.yaml` | Add correlation and coastal distance fields |
| `vs30/configs/viktor_cpt_clustering.yaml` | Add correlation and coastal distance fields |
| `vs30/resources/observations/foster_2019_measured_vs30_independent_observations.csv` | New filtered observation CSV |
| `dev/observations/foster_2019/create_foster_2019_observations.py` | Reproducible creation script |
| `tests/test_utils.py` | Add correlation function tests |
| `tests/fixtures/test_config_*.yaml` | Add required correlation fields |

---

### Task 1: Rename Exponential Correlation Function and Add Unit Test

**Files:**
- Modify: `vs30/utils.py:9-32`
- Test: `tests/test_utils.py`

- [ ] **Step 1: Write the failing test for the renamed function**

Add to `tests/test_utils.py`:

```python
class TestExponentialCorrelationFunction:
    """Tests for the exponential correlation function."""

    def test_zero_distance_returns_near_one(self):
        """Correlation at zero distance ≈ 1.0 (limited by MIN_DIST_ENFORCED)."""
        distances = np.array([0.0])
        result = utils.exponential_correlation_function(distances, phi=1407)
        assert result[0] > 0.999

    def test_correlation_decays_with_distance(self):
        """Correlation decays as distance increases."""
        distances = np.array([0.0, 100.0, 500.0, 1407.0, 5000.0])
        result = utils.exponential_correlation_function(distances, phi=1407)
        assert np.all(np.diff(result) < 0)  # Monotonically decreasing

    def test_at_phi_correlation_is_1_over_e(self):
        """At distance=phi, correlation ≈ 1/e ≈ 0.368."""
        distances = np.array([1407.0])
        result = utils.exponential_correlation_function(distances, phi=1407)
        assert np.isclose(result[0], np.exp(-1), rtol=0.01)

    def test_practical_range_three_phi(self):
        """At 3*phi, correlation ≈ 0.05 (5% practical range)."""
        phi = 1407
        distances = np.array([3 * phi])
        result = utils.exponential_correlation_function(distances, phi=phi)
        assert np.isclose(result[0], np.exp(-3), rtol=0.01)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && mamba activate vs30_venv && pytest tests/test_utils.py::TestExponentialCorrelationFunction -v`
Expected: FAIL with `AttributeError: module 'vs30.utils' has no attribute 'exponential_correlation_function'`

- [ ] **Step 3: Rename the function in utils.py**

In `vs30/utils.py`, rename `correlation_function` to `exponential_correlation_function` (line 9). This is a simple rename of the function name only — the body is unchanged.

```python
def exponential_correlation_function(
    distances: np.ndarray,
    phi: float,
    min_dist: float = constants.MIN_DIST_ENFORCED,
) -> np.ndarray:
    """
    Calculate exponential correlation from distances.

    Parameters
    ----------
    distances : ndarray
        Array of distances in meters. Can be scalar, 1D, or 2D (distance matrix).
    phi : float
        Correlation length parameter in meters.
    min_dist : float, optional
        Minimum distance enforced to prevent division issues. Default is MIN_DIST_ENFORCED.

    Returns
    -------
    ndarray
        Correlation values between 0 and 1. Same shape as distances.

    """
    return np.exp(-np.maximum(min_dist, distances) / phi)
```

- [ ] **Step 4: Update all references to the old name**

Use grep to find all callers of `correlation_function` and update them to `exponential_correlation_function`. The callers are:

- `vs30/spatial.py:770` — `utils.correlation_function(distance_matrix, constants.PHI[model_type])` (will be replaced by corr_fn in Task 4, but must compile now)
- `vs30/spatial.py:1167-1168` — `utils.correlation_function(np.array([0.0]), constants.PHI[model_type])`
- `vs30/spatial.py:1386-1387` — same pattern
- `vs30/spatial.py:900-902` — same pattern
- `vs30/parallel.py:587-588` — same pattern

Replace all `utils.correlation_function(` with `utils.exponential_correlation_function(`.

- [ ] **Step 5: Run test to verify it passes**

Run: `source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && mamba activate vs30_venv && pytest tests/test_utils.py::TestExponentialCorrelationFunction -v`
Expected: PASS

- [ ] **Step 6: Run the full test suite to verify nothing is broken by the rename**

Run: `source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && mamba activate vs30_venv && pytest tests/ -v`
Expected: All existing tests PASS

- [ ] **Step 7: Commit**

```bash
git add vs30/utils.py vs30/spatial.py vs30/parallel.py tests/test_utils.py
git commit -m "rename correlation_function to exponential_correlation_function"
```

---

### Task 2: Add Matern Correlation Function

**Files:**
- Modify: `vs30/utils.py`
- Test: `tests/test_utils.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_utils.py`:

```python
class TestMaternCorrelationFunction:
    """Tests for the Matérn correlation function.

    Uses the original Foster (2019) parameters: range=20000, sill=0.15,
    nugget=0.05, kappa=0.9.
    """

    def test_near_zero_distance_returns_sill_over_total(self):
        """At d≈0, correlation ≈ sill/(sill+nugget) = 0.15/0.20 = 0.75."""
        distances = np.array([0.0])
        result = utils.matern_correlation_function(
            distances, range_m=20000, sill=0.15, nugget=0.05, kappa=0.9,
        )
        assert np.isclose(result[0], 0.75, atol=0.02)

    def test_correlation_decays_with_distance(self):
        """Correlation decays as distance increases."""
        distances = np.array([100.0, 1000.0, 5000.0, 20000.0, 50000.0])
        result = utils.matern_correlation_function(
            distances, range_m=20000, sill=0.15, nugget=0.05, kappa=0.9,
        )
        assert np.all(np.diff(result) < 0)

    def test_kappa_half_matches_exponential_shape(self):
        """Matérn with kappa=0.5 and no nugget should match exponential shape.

        This is a mathematical identity: Matérn(κ=0.5) ∝ exp(-d/range).
        With nugget=0, the correlation at d should equal exp(-d/range).
        """
        distances = np.array([100.0, 500.0, 1000.0, 5000.0])
        range_m = 993.0
        result = utils.matern_correlation_function(
            distances, range_m=range_m, sill=1.0, nugget=0.0, kappa=0.5,
        )
        expected = np.exp(-distances / range_m)
        np.testing.assert_allclose(result, expected, rtol=0.05)

    def test_large_distance_approaches_zero(self):
        """At very large distances, correlation → 0."""
        distances = np.array([200000.0])
        result = utils.matern_correlation_function(
            distances, range_m=20000, sill=0.15, nugget=0.05, kappa=0.9,
        )
        assert result[0] < 0.01

    def test_returns_correct_shape(self):
        """Output shape matches input shape."""
        distances = np.array([[100, 200], [300, 400]], dtype=float)
        result = utils.matern_correlation_function(
            distances, range_m=20000, sill=0.15, nugget=0.05, kappa=0.9,
        )
        assert result.shape == (2, 2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && mamba activate vs30_venv && pytest tests/test_utils.py::TestMaternCorrelationFunction -v`
Expected: FAIL with `AttributeError: module 'vs30.utils' has no attribute 'matern_correlation_function'`

- [ ] **Step 3: Implement the Matern correlation function**

Add to `vs30/utils.py` after the exponential function:

```python
from scipy.special import gamma, kv


def matern_correlation_function(
    distances: np.ndarray,
    range_m: float,
    sill: float,
    nugget: float,
    kappa: float,
    min_dist: float = constants.MIN_DIST_ENFORCED,
) -> np.ndarray:
    """
    Calculate Matérn correlation with nugget from distances.

    Uses the gstat parameterization where the range parameter is used
    directly as the scale (NOT multiplied by sqrt(2*kappa)). This matches
    R's gstat::vgm() which was used to fit the original model parameters.

    Parameters
    ----------
    distances : ndarray
        Array of distances in meters.
    range_m : float
        Matérn range (scale) parameter in meters (gstat convention).
    sill : float
        Partial sill (variance contribution from spatial correlation).
    nugget : float
        Nugget variance (micro-scale variation / measurement noise).
    kappa : float
        Matérn smoothness parameter.
    min_dist : float, optional
        Minimum distance enforced to prevent numerical issues.

    Returns
    -------
    ndarray
        Correlation values. Same shape as distances.
        Near zero distance, returns approximately sill/(sill+nugget).
    """
    d = np.maximum(min_dist, distances)
    scaled = d / range_m
    rho = (2 ** (1 - kappa) / gamma(kappa)) * (scaled ** kappa) * kv(kappa, scaled)
    # Clamp NaN from numerical edge cases (kv can overflow for very small d)
    rho = np.where(np.isfinite(rho), rho, 1.0)
    return rho * sill / (sill + nugget)
```

Also add the import at the top of `vs30/utils.py`:

```python
from scipy.special import gamma, kv
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && mamba activate vs30_venv && pytest tests/test_utils.py::TestMaternCorrelationFunction -v`
Expected: PASS

- [ ] **Step 5: Run full test suite**

Run: `source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && mamba activate vs30_venv && pytest tests/ -v`
Expected: All PASS

- [ ] **Step 6: Commit**

```bash
git add vs30/utils.py tests/test_utils.py
git commit -m "add matern_correlation_function for original Foster 2019 model"
```

---

### Task 3: Add Config Resolution Helper

**Files:**
- Modify: `vs30/cli.py`
- Test: `tests/test_utils.py`

- [ ] **Step 1: Write the failing test**

Add to `tests/test_utils.py`:

```python
import functools
from vs30.cli import resolve_correlation_function


class TestResolveCorrelationFunction:
    """Tests for config → correlation callable resolution."""

    def test_exponential_resolution(self):
        """Exponential config resolves to callable that matches direct call."""
        config = {"model": "exponential", "phi": 1407}
        fn = resolve_correlation_function(config)
        distances = np.array([0.0, 100.0, 1407.0])
        expected = utils.exponential_correlation_function(distances, phi=1407)
        np.testing.assert_array_equal(fn(distances), expected)

    def test_matern_resolution(self):
        """Matern config resolves to callable that matches direct call."""
        config = {
            "model": "matern",
            "range": 20000,
            "sill": 0.15,
            "nugget": 0.05,
            "kappa": 0.9,
        }
        fn = resolve_correlation_function(config)
        distances = np.array([100.0, 5000.0, 20000.0])
        expected = utils.matern_correlation_function(
            distances, range_m=20000, sill=0.15, nugget=0.05, kappa=0.9,
        )
        np.testing.assert_array_equal(fn(distances), expected)

    def test_unknown_model_raises_error(self):
        """Unknown model type raises ValueError."""
        config = {"model": "unknown"}
        with pytest.raises(ValueError, match="Unknown correlation model"):
            resolve_correlation_function(config)

    def test_result_is_picklable(self):
        """Resolved callable must be picklable for multiprocessing."""
        import pickle
        config = {"model": "exponential", "phi": 1407}
        fn = resolve_correlation_function(config)
        pickled = pickle.dumps(fn)
        fn2 = pickle.loads(pickled)
        distances = np.array([100.0, 1000.0])
        np.testing.assert_array_equal(fn(distances), fn2(distances))

    def test_matern_is_picklable(self):
        """Matern callable must also be picklable."""
        import pickle
        config = {
            "model": "matern",
            "range": 20000,
            "sill": 0.15,
            "nugget": 0.05,
            "kappa": 0.9,
        }
        fn = resolve_correlation_function(config)
        pickled = pickle.dumps(fn)
        fn2 = pickle.loads(pickled)
        distances = np.array([100.0, 5000.0])
        np.testing.assert_array_equal(fn(distances), fn2(distances))
```

Also add `import pytest` at the top of the test file if not already present.

- [ ] **Step 2: Run test to verify it fails**

Run: `source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && mamba activate vs30_venv && pytest tests/test_utils.py::TestResolveCorrelationFunction -v`
Expected: FAIL with `ImportError: cannot import name 'resolve_correlation_function'`

- [ ] **Step 3: Implement the resolver**

Add to `vs30/cli.py` after the existing imports:

```python
import functools

from vs30 import utils


def resolve_correlation_function(
    config_section: dict,
) -> Callable[[np.ndarray], np.ndarray]:
    """
    Resolve a correlation config section into a picklable callable.

    Parameters
    ----------
    config_section : dict
        Must contain a "model" key ("exponential" or "matern") plus the
        model-specific parameters.

    Returns
    -------
    callable
        Function with signature (distances: ndarray) -> ndarray.
        Uses functools.partial for picklability in multiprocessing.
    """
    model = config_section["model"]
    if model == "exponential":
        return functools.partial(
            utils.exponential_correlation_function,
            phi=config_section["phi"],
        )
    elif model == "matern":
        return functools.partial(
            utils.matern_correlation_function,
            range_m=config_section["range"],
            sill=config_section["sill"],
            nugget=config_section["nugget"],
            kappa=config_section["kappa"],
        )
    else:
        raise ValueError(f"Unknown correlation model: {model}")
```

Also add the necessary imports to the top of `vs30/cli.py`:

```python
import functools
from collections.abc import Callable

import numpy as np
```

- [ ] **Step 4: Run test to verify it passes**

Run: `source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && mamba activate vs30_venv && pytest tests/test_utils.py::TestResolveCorrelationFunction -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add vs30/cli.py tests/test_utils.py
git commit -m "add resolve_correlation_function for config-driven correlation models"
```

---

### Task 4: Create Config Files and Register Model Version

**Files:**
- Create: `vs30/configs/foster_2019.yaml`
- Modify: `vs30/configs/modified_foster_2019.yaml`
- Modify: `vs30/configs/jaehwi_v1p0.yaml`
- Modify: `vs30/configs/viktor_cpt_clustering.yaml`
- Modify: `vs30/constants.py:15-29`
- Modify: `tests/fixtures/test_config_*.yaml` (6 files)

> **Note:** The new YAML fields (`geology_correlation`, `terrain_correlation`, `apply_coastal_distance_mod`) are not yet consumed by code at this point. They are added now so configs are complete before the plumbing tasks begin.

- [ ] **Step 1: Create foster_2019.yaml**

Create `vs30/configs/foster_2019.yaml`:

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

- [ ] **Step 2: Update modified_foster_2019.yaml**

Add to `vs30/configs/modified_foster_2019.yaml` (append after existing content):

```yaml
apply_coastal_distance_mod: true

geology_correlation:
  model: exponential
  phi: 1407

terrain_correlation:
  model: exponential
  phi: 993
```

- [ ] **Step 3: Update jaehwi_v1p0.yaml**

Add the same correlation and coastal distance fields to `vs30/configs/jaehwi_v1p0.yaml`:

```yaml
apply_coastal_distance_mod: true

geology_correlation:
  model: exponential
  phi: 1407

terrain_correlation:
  model: exponential
  phi: 993
```

- [ ] **Step 4: Update viktor_cpt_clustering.yaml**

Add the same fields to `vs30/configs/viktor_cpt_clustering.yaml`:

```yaml
apply_coastal_distance_mod: true

geology_correlation:
  model: exponential
  phi: 1407

terrain_correlation:
  model: exponential
  phi: 993
```

- [ ] **Step 5: Add FOSTER_2019 to FixedModelVersion and MODEL_VERSION_TO_CONFIG**

In `vs30/constants.py`, add `FOSTER_2019` to the `FixedModelVersion` enum (line 15-20):

```python
class FixedModelVersion(StrEnum):
    """Identifiers for fixed versions of the geology and terrain models."""

    FOSTER_2019 = "foster_2019"
    MODIFIED_FOSTER_2019 = "modified_foster_2019"
    JAEHWI_V1P0 = "jaehwi_v1p0"
    VIKTOR_CPT_CLUSTERING = "viktor_cpt_clustering"
```

And add to `MODEL_VERSION_TO_CONFIG` (line 25-29):

```python
MODEL_VERSION_TO_CONFIG = {
    FixedModelVersion.FOSTER_2019: CONFIGS_DIR / "foster_2019.yaml",
    FixedModelVersion.MODIFIED_FOSTER_2019: CONFIGS_DIR / "modified_foster_2019.yaml",
    FixedModelVersion.JAEHWI_V1P0: CONFIGS_DIR / "jaehwi_v1p0.yaml",
    FixedModelVersion.VIKTOR_CPT_CLUSTERING: CONFIGS_DIR / "viktor_cpt_clustering.yaml",
}
```

- [ ] **Step 6: Update all 6 test fixture YAMLs**

Add the following to each file in `tests/fixtures/`:
- `test_config_small_independent_only.yaml`
- `test_config_small_clustered_only.yaml`
- `test_config_independent_only.yaml`
- `test_config_clustered_only.yaml`
- `test_config_small_both.yaml`
- `test_config_both.yaml`

Append:

```yaml
apply_coastal_distance_mod: true

geology_correlation:
  model: exponential
  phi: 1407

terrain_correlation:
  model: exponential
  phi: 993
```

- [ ] **Step 7: Run test suite to verify new YAML fields don't break anything**

Run: `source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && mamba activate vs30_venv && pytest tests/ -v`
Expected: All PASS (the new YAML fields are ignored by current code)

- [ ] **Step 8: Commit**

```bash
git add vs30/configs/ vs30/constants.py tests/fixtures/
git commit -m "add foster_2019 config and correlation fields to all configs"
```

---

### Task 5: Thread corr_fn Through spatial.py

This is the core plumbing task. Every function that currently does `constants.PHI[model_type]` must instead accept a `corr_fn` callable.

**Files:**
- Modify: `vs30/spatial.py`

**Functions to change (in call-graph order, bottom-up):**

1. `build_covariance_matrix` (line 731): Replace `model_type` parameter with `corr_fn`
2. `compute_spatial_adjustment_for_pixel` (line 855): Replace `model_type` with `corr_fn`; derive `corr_zero` from `corr_fn` when not pre-computed
3. `compute_spatial_adjustments` (line 1099): Replace `model_type` with `corr_fn`; pre-compute `corr_zero` from `corr_fn`
4. `compute_spatial_adjustment_at_points` (line 1280): Same pattern
5. `find_affected_pixels` (line 963): Keep `model_type` — it's only used for progress bar labels, not PHI lookup

- [ ] **Step 1: Update `build_covariance_matrix`**

At `vs30/spatial.py:731`, change the signature and body:

**Before** (line 731 onwards — the parameter list includes `model_type`):
```python
def build_covariance_matrix(
    pixel: PixelData,
    selected_observations: ObservationData,
    model_type: constants.ModelType,
    noisy: bool = False,
    cov_reduc: float = constants.COV_REDUC,
) -> np.ndarray:
```

**After:**
```python
def build_covariance_matrix(
    pixel: PixelData,
    selected_observations: ObservationData,
    corr_fn: Callable[[np.ndarray], np.ndarray],
    noisy: bool = False,
    cov_reduc: float = constants.COV_REDUC,
) -> np.ndarray:
```

And at line 770, change:
```python
    # Before:
    corr = utils.exponential_correlation_function(distance_matrix, constants.PHI[model_type])
    # After:
    corr = corr_fn(distance_matrix)
```

Add `from collections.abc import Callable` to the imports at the top of `spatial.py`.

- [ ] **Step 2: Update `compute_spatial_adjustment_for_pixel`**

At `vs30/spatial.py:855`, change signature:

**Before:**
```python
def compute_spatial_adjustment_for_pixel(
    pixel: PixelData,
    obs_data: ObservationData,
    model_type: constants.ModelType,
    max_dist_m: float = constants.MAX_DIST_M,
    max_points: int = constants.MAX_POINTS,
    noisy: bool = False,
    cov_reduc: float = constants.COV_REDUC,
    corr_zero: float | None = None,
) -> SpatialAdjustmentResult | None:
```

**After:**
```python
def compute_spatial_adjustment_for_pixel(
    pixel: PixelData,
    obs_data: ObservationData,
    corr_fn: Callable[[np.ndarray], np.ndarray],
    max_dist_m: float = constants.MAX_DIST_M,
    max_points: int = constants.MAX_POINTS,
    noisy: bool = False,
    cov_reduc: float = constants.COV_REDUC,
    corr_zero: float | None = None,
) -> SpatialAdjustmentResult | None:
```

Update the corr_zero computation (lines 900-903):
```python
    # Before:
    if corr_zero is None:
        corr_zero = utils.exponential_correlation_function(
            np.array([0.0]), constants.PHI[model_type]
        )[0]
    # After:
    if corr_zero is None:
        corr_zero = corr_fn(np.array([0.0]))[0]
```

Update the call to `build_covariance_matrix` (lines 924-930):
```python
    # Before:
    cov_matrix = build_covariance_matrix(
        pixel, selected_obs, model_type, noisy=noisy, cov_reduc=cov_reduc,
    )
    # After:
    cov_matrix = build_covariance_matrix(
        pixel, selected_obs, corr_fn, noisy=noisy, cov_reduc=cov_reduc,
    )
```

Also update the docstring to document `corr_fn` instead of `model_type`.

- [ ] **Step 3: Update `compute_spatial_adjustments`**

At `vs30/spatial.py:1099`, change signature:

**Before:**
```python
def compute_spatial_adjustments(
    raster_data: RasterData,
    obs_data: ObservationData,
    bbox_result: BoundingBoxResult,
    model_type: constants.ModelType,
    ...
```

**After:**
```python
def compute_spatial_adjustments(
    raster_data: RasterData,
    obs_data: ObservationData,
    bbox_result: BoundingBoxResult,
    corr_fn: Callable[[np.ndarray], np.ndarray],
    ...
```

Update corr_zero pre-computation (lines 1167-1169):
```python
    # Before:
    corr_zero = utils.exponential_correlation_function(
        np.array([0.0]), constants.PHI[model_type]
    )[0]
    # After:
    corr_zero = corr_fn(np.array([0.0]))[0]
```

Update the inner call to `compute_spatial_adjustment_for_pixel` (line 1195-1199):
```python
    # Pass corr_fn instead of model_type
    update_result = compute_spatial_adjustment_for_pixel(
        pixel,
        obs_data,
        corr_fn,
        max_dist_m=max_dist_m,
        ...
```

- [ ] **Step 4: Update `compute_spatial_adjustment_at_points`**

At `vs30/spatial.py:1280`, change signature:

**Before:**
```python
def compute_spatial_adjustment_at_points(
    points: np.ndarray,
    ...
    model_type: constants.ModelType,
    ...
```

**After:**
```python
def compute_spatial_adjustment_at_points(
    points: np.ndarray,
    ...
    corr_fn: Callable[[np.ndarray], np.ndarray],
    ...
```

Update corr_zero pre-computation (lines 1386-1388):
```python
    # Before:
    corr_zero = utils.exponential_correlation_function(
        np.array([0.0]), constants.PHI[model_type]
    )[0]
    # After:
    corr_zero = corr_fn(np.array([0.0]))[0]
```

Update the inner call to `compute_spatial_adjustment_for_pixel` (line 1399-1407):
```python
    # Pass corr_fn instead of model_type
    result = compute_spatial_adjustment_for_pixel(
        pixel,
        obs_data,
        corr_fn,
        max_dist_m=max_dist_m,
        ...
```

- [ ] **Step 5: Verify `find_affected_pixels` needs no changes**

`find_affected_pixels` at line 963 uses `model_type` only for the progress bar label at line 1027. It does NOT do any PHI lookup. No changes needed.

- [ ] **Step 6: Do NOT commit yet**

The callers in `pipeline.py` and `parallel.py` still pass `model_type` where `corr_fn` is now expected. Continue to Tasks 6-8 to complete the full plumbing before committing.

---

### Task 6: Thread corr_fn Through pipeline.py

**Files:**
- Modify: `vs30/pipeline.py`

**Functions to change:**

1. `compute_spatial_adjustment_on_grid` (line 328)
2. `run_in_memory_pipeline_for_model_type` (line 679)
3. `compute_grid` (line 890)
4. `compute_at_locations` (line 1072)

- [ ] **Step 1: Update `compute_spatial_adjustment_on_grid`**

Add `corr_fn` parameter. This function passes it to `spatial.compute_spatial_adjustments`, `parallel.run_parallel_spatial_fit`, and `spatial.find_affected_pixels` (no change for last one).

At `vs30/pipeline.py:328`, change signature:

```python
def compute_spatial_adjustment_on_grid(
    vs30_array: np.ndarray,
    stdv_array: np.ndarray,
    profile: dict,
    observations_df: pd.DataFrame,
    model_values_df: pd.DataFrame,
    model_type: constants.ModelType,
    corr_fn: Callable,
    noisy: bool = True,
    ...
```

Add `from collections.abc import Callable` to pipeline.py imports.

Update call to `parallel.run_parallel_spatial_fit` (lines 450-460):
```python
        updates = parallel.run_parallel_spatial_fit(
            affected_flat_indices=affected_flat_indices,
            raster_data=raster_data,
            obs_data=obs_data,
            corr_fn=corr_fn,
            model_type=model_type,
            ...
```

Update call to `spatial.compute_spatial_adjustments` (lines 462-472):
```python
        updates = spatial.compute_spatial_adjustments(
            raster_data,
            obs_data,
            bbox_result,
            corr_fn,
            ...
```

- [ ] **Step 2: Update `run_in_memory_pipeline_for_model_type`**

Add `corr_fn` parameter at `vs30/pipeline.py:679`:

```python
def run_in_memory_pipeline_for_model_type(
    model_type: constants.ModelType,
    grid_config: config_module.GridConfig,
    corr_fn: Callable,
    ...
```

Pass to `compute_spatial_adjustment_on_grid` (line 858):
```python
        current_vs30, current_stdv = compute_spatial_adjustment_on_grid(
            vs30_array=current_vs30,
            stdv_array=current_stdv,
            profile=profile,
            observations_df=observations_df,
            model_values_df=posterior_df,
            model_type=model_type,
            corr_fn=corr_fn,
            noisy=noisy,
            ...
```

- [ ] **Step 3: Update `compute_grid`**

Add `geology_corr_fn` and `terrain_corr_fn` parameters at `vs30/pipeline.py:890`:

```python
def compute_grid(
    grid_config: config_module.GridConfig,
    output_dir: Path | None = None,
    model_type: constants.ModelType = constants.ModelType.COMBINED,
    geology_categorical_csv: Path | None = None,
    terrain_categorical_csv: Path | None = None,
    clustered_observations_csv: Path | None = None,
    independent_observations_csv: Path | None = None,
    combination_method: constants.CombinationMethod = constants.CombinationMethod.STANDARD_DEVIATION_WEIGHTING,
    combine_ratio: float | None = None,
    noisy: bool = True,
    mvn: bool = True,
    do_bayesian_update: bool = True,
    include_intermediate: bool = False,
    n_proc: int = 1,
    max_spatial_boolean_array_memory_gb: float = 1.0,
    geology_corr_fn: Callable | None = None,
    terrain_corr_fn: Callable | None = None,
) -> dict[str, np.ndarray | dict | None]:
```

Pass the appropriate `corr_fn` when calling `run_in_memory_pipeline_for_model_type` for geology and terrain (search for where `run_in_memory_pipeline_for_model_type` is called in `compute_grid` and pass `corr_fn=geology_corr_fn` or `corr_fn=terrain_corr_fn` accordingly).

- [ ] **Step 4: Update `compute_at_locations`**

Add `geology_corr_fn` and `terrain_corr_fn` parameters at `vs30/pipeline.py:1072`:

```python
def compute_at_locations(
    longitudes: np.ndarray,
    latitudes: np.ndarray,
    ...
    n_proc: int = 1,
    geology_corr_fn: Callable | None = None,
    terrain_corr_fn: Callable | None = None,
) -> pd.DataFrame:
```

Pass these through to `parallel.run_parallel_locations` and `parallel.process_geology_at_points` / `parallel.process_terrain_at_points` (will be done in Task 7 for the parallel side).

- [ ] **Step 5: Do NOT commit yet — continue to Task 7**

---

### Task 7: Thread corr_fn Through parallel.py

**Files:**
- Modify: `vs30/parallel.py`

**Functions to change:**

1. `process_geology_at_points` (line 58): add `corr_fn`
2. `process_terrain_at_points` (line 174): add `corr_fn`
3. `LocationsChunkConfig` (line 255): add `geology_corr_fn` and `terrain_corr_fn`
4. `process_locations_chunk` (line 279): pass corr_fn from config
5. `run_parallel_locations` (line 450): no change needed (passes config)
6. `process_pixels_chunk` (line 387): use `corr_fn` from config_params
7. `run_parallel_spatial_fit` (line 518): accept `corr_fn`, store in config_params

- [ ] **Step 1: Update `process_geology_at_points`**

At `vs30/parallel.py:58`, add `corr_fn` parameter:

```python
def process_geology_at_points(
    points: np.ndarray,
    model_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    corr_fn: Callable,
    noisy: bool = False,
    progress_bar: tqdm | None = None,
) -> tuple[...]:
```

Pass to `spatial.compute_spatial_adjustment_at_points` (line 146):
```python
        geol_mvn_vs30, geol_mvn_stdv = spatial.compute_spatial_adjustment_at_points(
            ...
            corr_fn=corr_fn,
            noisy=noisy,
            ...
        )
```

Add `from collections.abc import Callable` to parallel.py imports.

- [ ] **Step 2: Update `process_terrain_at_points`**

At `vs30/parallel.py:174`, add `corr_fn` parameter:

```python
def process_terrain_at_points(
    points: np.ndarray,
    model_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    corr_fn: Callable,
    noisy: bool = False,
    progress_bar: tqdm | None = None,
) -> tuple[...]:
```

Pass to `spatial.compute_spatial_adjustment_at_points` (line 229):
```python
        terr_mvn_vs30, terr_mvn_stdv = spatial.compute_spatial_adjustment_at_points(
            ...
            corr_fn=corr_fn,
            noisy=noisy,
            ...
        )
```

- [ ] **Step 3: Update `LocationsChunkConfig`**

At `vs30/parallel.py:255`, add corr_fn fields:

```python
@dataclass
class LocationsChunkConfig:
    include_intermediate: bool
    model_type: constants.ModelType
    combination_method: constants.CombinationMethod
    combine_ratio: float | None
    noisy: bool
    geology_corr_fn: Callable | None
    terrain_corr_fn: Callable | None
```

- [ ] **Step 4: Update `process_locations_chunk`**

At `vs30/parallel.py:279`, pass corr_fn from config to `process_geology_at_points` and `process_terrain_at_points`:

```python
    # Process geology model
    if run_geology:
        (...) = process_geology_at_points(
            points,
            geol_model_df,
            observations_df,
            config.geology_corr_fn,
            config.noisy,
        )

    # Process terrain model
    if run_terrain:
        (...) = process_terrain_at_points(
            points, terr_model_df, observations_df, config.terrain_corr_fn, config.noisy
        )
```

- [ ] **Step 5: Update `run_parallel_spatial_fit`**

At `vs30/parallel.py:518`, add `corr_fn` parameter and replace PHI lookup:

```python
def run_parallel_spatial_fit(
    affected_flat_indices: np.ndarray,
    raster_data,
    obs_data: spatial.ObservationData,
    corr_fn: Callable,
    model_type: constants.ModelType,
    max_dist_m: float,
    max_points: int,
    noisy: bool,
    cov_reduc: float,
    n_proc: int,
) -> list[spatial.SpatialAdjustmentResult]:
```

Update corr_zero computation (lines 586-597):
```python
    # Before:
    corr_zero = utils.exponential_correlation_function(
        np.array([0.0]), constants.PHI[model_type]
    )[0]
    config_params = {
        constants.KEY_MODEL_TYPE: model_type,
        ...
    }

    # After:
    corr_zero = corr_fn(np.array([0.0]))[0]
    config_params = {
        constants.KEY_MODEL_TYPE: model_type,  # kept for labels in process_pixels_chunk if needed
        "corr_fn": corr_fn,
        constants.KEY_MAX_DIST_M: max_dist_m,
        constants.KEY_MAX_POINTS: max_points,
        constants.KEY_NOISY: noisy,
        constants.KEY_COV_REDUC: cov_reduc,
        constants.KEY_CORR_ZERO: corr_zero,
    }
```

- [ ] **Step 6: Update `process_pixels_chunk`**

At `vs30/parallel.py:387`, update to use `corr_fn` from config_params:

```python
        update = spatial.compute_spatial_adjustment_for_pixel(
            pixel,
            obs_data,
            config_params["corr_fn"],  # was config_params[constants.KEY_MODEL_TYPE]
            max_dist_m=config_params[constants.KEY_MAX_DIST_M],
            max_points=config_params[constants.KEY_MAX_POINTS],
            noisy=config_params[constants.KEY_NOISY],
            cov_reduc=config_params[constants.KEY_COV_REDUC],
            corr_zero=config_params.get(constants.KEY_CORR_ZERO),
        )
```

- [ ] **Step 7: Do NOT commit yet — continue to Task 8**

---

### Task 8: Wire Up corr_fn in CLI

**Files:**
- Modify: `vs30/cli.py`
- Modify: `vs30/pipeline.py` (compute_at_locations callers)

- [ ] **Step 1: Update `grid` command in cli.py**

At `vs30/cli.py:343-369`, resolve correlation functions from config and pass them:

```python
    # After config_data = yaml.safe_load(f) and CSV path resolution:
    geology_corr_fn = resolve_correlation_function(config_data["geology_correlation"])
    terrain_corr_fn = resolve_correlation_function(config_data["terrain_correlation"])

    pipeline.compute_grid(
        ...
        geology_corr_fn=geology_corr_fn,
        terrain_corr_fn=terrain_corr_fn,
    )
```

- [ ] **Step 2: Update `points` command in cli.py**

At `vs30/cli.py:166-189`, resolve and pass:

```python
    geology_corr_fn = resolve_correlation_function(config_data["geology_correlation"])
    terrain_corr_fn = resolve_correlation_function(config_data["terrain_correlation"])

    run_points_pipeline(
        ...
        geology_corr_fn=geology_corr_fn,
        terrain_corr_fn=terrain_corr_fn,
    )
```

- [ ] **Step 3: Update `run_points_pipeline`**

At `vs30/cli.py:21`, add `geology_corr_fn` and `terrain_corr_fn` parameters:

```python
def run_points_pipeline(
    ...
    n_proc: int = 1,
    lon_column: str = constants.LOCATIONS_LON_COLUMN,
    lat_column: str = constants.LOCATIONS_LAT_COLUMN,
    geology_corr_fn: Callable | None = None,
    terrain_corr_fn: Callable | None = None,
) -> None:
```

Pass them through to `pipeline.compute_at_locations`:

```python
    result_df = pipeline.compute_at_locations(
        ...
        geology_corr_fn=geology_corr_fn,
        terrain_corr_fn=terrain_corr_fn,
    )
```

- [ ] **Step 4: Wire up `compute_at_locations` to pass corr_fn to both paths**

In `vs30/pipeline.py:compute_at_locations`:

**Parallel path** (line 1217-1223): Add corr_fn fields to `LocationsChunkConfig`:

```python
        loc_config = parallel.LocationsChunkConfig(
            include_intermediate=include_intermediate,
            model_type=model_type,
            combination_method=combination_method,
            combine_ratio=combine_ratio,
            noisy=noisy,
            geology_corr_fn=geology_corr_fn,
            terrain_corr_fn=terrain_corr_fn,
        )
```

**Sequential geology path** (line 1261-1267): Add `corr_fn` to the call:

```python
            ) = parallel.process_geology_at_points(
                points,
                geol_model_df,
                observations_df,
                corr_fn=geology_corr_fn,
                noisy=noisy,
                progress_bar=pbar,
            )
```

**Sequential terrain path** (line 1289-1295): Add `corr_fn` to the call:

```python
            ) = parallel.process_terrain_at_points(
                points,
                terr_model_df,
                observations_df,
                corr_fn=terrain_corr_fn,
                noisy=noisy,
                progress_bar=pbar,
            )
```

- [ ] **Step 5: Add config validation**

In the `grid` and `points` commands in `cli.py`, add validation after loading config:

```python
    # Validate required config fields (no defaults — all configs must be explicit)
    for field in ("geology_correlation", "terrain_correlation", "apply_coastal_distance_mod"):
        if field not in config_data:
            raise typer.BadParameter(
                f"Config missing required field '{field}'. "
                "All configs must specify correlation and coastal distance parameters."
            )
```

- [ ] **Step 6: Handle `_custom` commands**

The `grid_custom` and `points_custom` commands don't load from a config YAML, so they have no correlation sections to resolve. For now, provide exponential defaults matching the current behavior. In `pipeline.py`, where `compute_grid` and `compute_at_locations` receive `geology_corr_fn=None`:

```python
    # In compute_grid and compute_at_locations, at the top:
    if geology_corr_fn is None:
        geology_corr_fn = functools.partial(
            utils.exponential_correlation_function, phi=1407
        )
    if terrain_corr_fn is None:
        terrain_corr_fn = functools.partial(
            utils.exponential_correlation_function, phi=993
        )
```

Add `import functools` to `vs30/pipeline.py` imports.

This ensures `_custom` commands continue to work without code changes. A future enhancement could add explicit `--geology-phi` and `--terrain-phi` CLI options to the `_custom` commands.

- [ ] **Step 7: Run the full test suite**

Run: `source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && mamba activate vs30_venv && pytest tests/ -v`
Expected: All PASS (existing tests should work since test fixtures now have the required fields and the exponential correlation is resolved from config identically to the old hard-coded PHI)

- [ ] **Step 8: Commit all corr_fn plumbing (Tasks 5-8 as one atomic commit)**

```bash
git add vs30/spatial.py vs30/pipeline.py vs30/parallel.py vs30/cli.py
git commit -m "replace hard-coded PHI lookup with config-driven correlation callables"
```

---

### Task 9: Add Coastal Distance Flag

**Files:**
- Modify: `vs30/pipeline.py` (`compute_hybrid_geology_arrays`, `run_in_memory_pipeline_for_model_type`, `compute_grid`, `compute_at_locations`)
- Modify: `vs30/parallel.py` (`process_geology_at_points`, `LocationsChunkConfig`, `process_locations_chunk`)
- Modify: `vs30/cli.py` (`grid`, `points`, `run_points_pipeline`)

- [ ] **Step 1: Update `compute_hybrid_geology_arrays` in pipeline.py**

At `vs30/pipeline.py:270`, add `apply_coastal_distance_mod` parameter:

```python
def compute_hybrid_geology_arrays(
    vs30_array: np.ndarray,
    stdv_array: np.ndarray,
    id_array: np.ndarray,
    profile: dict,
    apply_coastal_distance_mod: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
```

Make coast distance computation conditional:

```python
    logger.info("Computing slope array...")
    slope_array = raster.compute_slope_array(profile)

    if apply_coastal_distance_mod:
        logger.info("Computing coast distance array...")
        coast_dist_array = raster.compute_coast_distance_array(profile)
    else:
        logger.info("Skipping coast distance computation (disabled in config)")
        coast_dist_array = np.zeros_like(vs30_array)  # Dummy, unused by mods

    hybrid_vs30, hybrid_stdv = raster.apply_hybrid_geology_modifications(
        vs30_array,
        stdv_array,
        id_array,
        slope_array,
        coast_dist_array,
        mod6=apply_coastal_distance_mod,
        mod13=apply_coastal_distance_mod,
    )

    return hybrid_vs30, hybrid_stdv, slope_array, coast_dist_array
```

- [ ] **Step 2: Thread through `run_in_memory_pipeline_for_model_type`**

Add `apply_coastal_distance_mod` parameter and pass to `compute_hybrid_geology_arrays`:

```python
def run_in_memory_pipeline_for_model_type(
    ...
    apply_coastal_distance_mod: bool = True,
) -> tuple[np.ndarray, np.ndarray, dict]:
```

At the call site (line ~808):
```python
        current_vs30, current_stdv, slope_array, coast_dist_array = (
            compute_hybrid_geology_arrays(
                vs30_array, stdv_array, id_array, profile,
                apply_coastal_distance_mod=apply_coastal_distance_mod,
            )
        )
```

- [ ] **Step 3: Thread through `compute_grid` and `compute_at_locations`**

Add `apply_coastal_distance_mod: bool = True` parameter to both `compute_grid` (line 890) and `compute_at_locations` (line 1072), and pass through to `run_in_memory_pipeline_for_model_type` (grid path) or to the parallel/sequential call sites (points path).

For `compute_at_locations`, update both paths:

**Parallel path** — add to `LocationsChunkConfig` (done in Step 5 below).

**Sequential path** (line 1261-1267 and 1289-1295) — pass to `process_geology_at_points`:

```python
            ) = parallel.process_geology_at_points(
                points,
                geol_model_df,
                observations_df,
                corr_fn=geology_corr_fn,
                noisy=noisy,
                apply_coastal_distance_mod=apply_coastal_distance_mod,
                progress_bar=pbar,
            )
```

(Terrain does not need `apply_coastal_distance_mod` — hybrid modifications only apply to geology.)

- [ ] **Step 4: Update `process_geology_at_points` in parallel.py**

Add `apply_coastal_distance_mod` parameter:

```python
def process_geology_at_points(
    points: np.ndarray,
    model_df: pd.DataFrame,
    observations_df: pd.DataFrame,
    corr_fn: Callable,
    noisy: bool = False,
    apply_coastal_distance_mod: bool = True,
    progress_bar: tqdm | None = None,
) -> tuple[...]:
```

Make coastal distance conditional (lines 114-124):

```python
    slope_at_points = raster.sample_slope_at_points(points)
    if apply_coastal_distance_mod:
        coast_dist_at_points = raster.compute_coastal_distance_at_points(points)
    else:
        coast_dist_at_points = np.zeros(len(points))

    geol_vs30_hybrid, geol_stdv_hybrid = raster.apply_hybrid_geology_modifications(
        geol_vs30, geol_stdv, geol_ids, slope_at_points, coast_dist_at_points,
        mod6=apply_coastal_distance_mod,
        mod13=apply_coastal_distance_mod,
    )
```

Same pattern for observation hybrid modifications (lines 136-144):

```python
        obs_slope = raster.sample_slope_at_points(obs_locs)
        if apply_coastal_distance_mod:
            obs_coast_dist = raster.compute_coastal_distance_at_points(obs_locs)
        else:
            obs_coast_dist = np.zeros(len(obs_locs))
        obs_model_vs30, obs_model_stdv = raster.apply_hybrid_geology_modifications(
            ..., obs_slope, obs_coast_dist,
            mod6=apply_coastal_distance_mod,
            mod13=apply_coastal_distance_mod,
        )
```

- [ ] **Step 5: Update `LocationsChunkConfig` and `process_locations_chunk`**

Add `apply_coastal_distance_mod: bool` to `LocationsChunkConfig`:

```python
@dataclass
class LocationsChunkConfig:
    include_intermediate: bool
    model_type: constants.ModelType
    combination_method: constants.CombinationMethod
    combine_ratio: float | None
    noisy: bool
    geology_corr_fn: Callable | None
    terrain_corr_fn: Callable | None
    apply_coastal_distance_mod: bool
```

In `process_locations_chunk` (line 332-337), pass it to `process_geology_at_points`:

```python
        ) = process_geology_at_points(
            points,
            geol_model_df,
            observations_df,
            corr_fn=config.geology_corr_fn,
            noisy=config.noisy,
            apply_coastal_distance_mod=config.apply_coastal_distance_mod,
        )
```

In `compute_at_locations`, update the `LocationsChunkConfig` construction (line 1217-1223):

```python
        loc_config = parallel.LocationsChunkConfig(
            include_intermediate=include_intermediate,
            model_type=model_type,
            combination_method=combination_method,
            combine_ratio=combine_ratio,
            noisy=noisy,
            geology_corr_fn=geology_corr_fn,
            terrain_corr_fn=terrain_corr_fn,
            apply_coastal_distance_mod=apply_coastal_distance_mod,
        )
```

- [ ] **Step 6: Wire up in CLI**

In `cli.py` `grid` and `points` commands, read `apply_coastal_distance_mod` from config. Use direct key access (no default — spec requires all configs to be explicit):

```python
    apply_coastal_distance_mod = config_data["apply_coastal_distance_mod"]

    pipeline.compute_grid(
        ...
        apply_coastal_distance_mod=apply_coastal_distance_mod,
    )
```

Add `apply_coastal_distance_mod: bool = True` to `run_points_pipeline` signature and pass through to `pipeline.compute_at_locations`. The `_custom` commands do not pass this parameter, so `pipeline.compute_grid` / `pipeline.compute_at_locations` default to `True` (matching current behavior).

- [ ] **Step 7: Run full test suite**

Run: `source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && mamba activate vs30_venv && pytest tests/ -v`
Expected: All PASS (test fixtures have `apply_coastal_distance_mod: true` so behavior is unchanged)

- [ ] **Step 8: Commit**

```bash
git add vs30/pipeline.py vs30/parallel.py vs30/cli.py
git commit -m "add apply_coastal_distance_mod config flag for coastal distance modifications"
```

---

### Task 10: Create Foster 2019 Observation CSV

**Files:**
- Create: `vs30/resources/observations/foster_2019_measured_vs30_independent_observations.csv`
- Create: `dev/observations/foster_2019/create_foster_2019_observations.py`

- [ ] **Step 1: Create the creation script**

Create `dev/observations/foster_2019/create_foster_2019_observations.py`:

```python
"""
Create the observation CSV for the original Foster (2019) model.

The original model excludes ALL Kaiser Q3 stations. This script filters
the modified_foster_2019 observations by removing Q3 broadband stations
(those with 3-character station names that the modified version retained).

Can be run from the refactored environment:
    mamba activate vs30_venv
    python dev/observations/foster_2019/create_foster_2019_observations.py

Alternatively, can be run from the legacy environment to generate from
raw sources (see create_modified_foster_2019_observations.py for details).
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).parent.parent.parent.parent
OBSERVATIONS_DIR = REPO_ROOT / "vs30" / "resources" / "observations"

INPUT_CSV = OBSERVATIONS_DIR / "modified_foster_2019_measured_vs30_independent_observations.csv"
OUTPUT_CSV = OBSERVATIONS_DIR / "foster_2019_measured_vs30_independent_observations.csv"


def main():
    print("Creating Foster (2019) observation CSV from modified version")
    print("=" * 70)

    df = pd.read_csv(INPUT_CSV, comment="#")
    print(f"Loaded {len(df)} observations from modified_foster_2019 CSV")

    # The modified version retains Kaiser Q3 stations with 3-character names.
    # The original excludes ALL Q3 stations. Drop the additionally retained ones.
    q3_broadband_mask = (
        (df["source"] == "kaiseretal")
        & (df["q"] == 3.0)
        & (df["station"].str.len() == 3)
    )
    n_dropped = q3_broadband_mask.sum()
    print(f"Dropping {n_dropped} Q3 broadband stations (3-char station names)")

    filtered = df[~q3_broadband_mask].copy()
    print(f"Remaining: {len(filtered)} observations")

    # Write with comment header for provenance
    with open(OUTPUT_CSV, "w") as f:
        f.write("# Foster (2019) independent observations\n")
        f.write("# Generated by: dev/observations/foster_2019/create_foster_2019_observations.py\n")
        f.write(f"# Source: {INPUT_CSV.name} ({len(df)} rows)\n")
        f.write(f"# Filter: removed {n_dropped} Kaiser Q3 broadband (3-char station) rows\n")
        filtered.to_csv(f, index=False)

    print(f"\nWrote {len(filtered)} observations to {OUTPUT_CSV.name}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the creation script**

```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv && \
python dev/observations/foster_2019/create_foster_2019_observations.py
```

Expected: Creates `vs30/resources/observations/foster_2019_measured_vs30_independent_observations.csv` with ~412 observations (459 - 47 = 412).

- [ ] **Step 3: Verify the output**

```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv && \
python -c "
import pandas as pd
df = pd.read_csv('vs30/resources/observations/foster_2019_measured_vs30_independent_observations.csv', comment='#')
print(f'Total rows: {len(df)}')
print(f'Sources: {df.source.value_counts().to_dict()}')
# Verify no Q3 stations remain
kaiser_q3 = df[(df.source == 'kaiseretal') & (df.q == 3.0)]
assert len(kaiser_q3) == 0, f'Found {len(kaiser_q3)} Q3 Kaiser stations — should be 0'
print('Verification passed: no Q3 Kaiser stations present')
"
```

- [ ] **Step 4: Commit**

```bash
git add vs30/resources/observations/foster_2019_measured_vs30_independent_observations.csv dev/observations/foster_2019/
git commit -m "add foster_2019 observation CSV excluding all Kaiser Q3 stations"
```

---

### Task 11: Remove Dead PHI Constants

**Files:**
- Modify: `vs30/constants.py`

- [ ] **Step 1: Verify PHI is no longer referenced**

Search for any remaining references to `PHI_GEOLOGY`, `PHI_TERRAIN`, `constants.PHI[`, or `constants.PHI `:

```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv && \
grep -rn "PHI_GEOLOGY\|PHI_TERRAIN\|constants\.PHI\[" vs30/ tests/
```

Expected: No results (all references were replaced in Tasks 5-8).

- [ ] **Step 2: Remove the constants**

In `vs30/constants.py`, remove:
- Lines 62-67 (`PHI_GEOLOGY`, `PHI_TERRAIN` and their comments)
- Lines 69-71 (the comment about PHI dict)
- Lines 376-380 (`PHI` dictionary)

- [ ] **Step 3: Run full test suite**

Run: `source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && mamba activate vs30_venv && pytest tests/ -v`
Expected: All PASS

- [ ] **Step 4: Commit**

```bash
git add vs30/constants.py
git commit -m "remove dead PHI_GEOLOGY, PHI_TERRAIN, and PHI dict constants"
```

---

### Task 12: Regression Test with modified_foster_2019

Verify that the modified_foster_2019 model produces identical output before and after all changes.

**Files:**
- No new files

- [ ] **Step 1: Run existing regression tests**

```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv && \
pytest tests/ -v
```

Expected: All PASS

- [ ] **Step 2: Smoke-test the new foster_2019 model version**

Run a small grid computation with the foster_2019 model to verify it executes without errors:

```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv && \
vs30 grid foster_2019 \
    --grid-xmin 1569500 --grid-xmax 1572500 \
    --grid-ymin 5178500 --grid-ymax 5181500 \
    --grid-dx 100 --grid-dy 100 \
    --output-dir /tmp/vs30_foster_2019_test \
    --include-intermediate
```

Expected: Runs to completion, produces raster files in `/tmp/vs30_foster_2019_test/`.

- [ ] **Step 3: Run the same domain with modified_foster_2019 and compare**

```bash
source /home/arr65/miniforge-pypy3/etc/profile.d/conda.sh && \
source /home/arr65/miniforge-pypy3/etc/profile.d/mamba.sh && \
mamba activate vs30_venv && \
vs30 grid modified_foster_2019 \
    --grid-xmin 1569500 --grid-xmax 1572500 \
    --grid-ymin 5178500 --grid-ymax 5181500 \
    --grid-dx 100 --grid-dy 100 \
    --output-dir /tmp/vs30_modified_foster_2019_test \
    --include-intermediate
```

Compare the outputs to confirm the two models produce different results (which they should, due to different correlation functions and coastal distance behavior).

- [ ] **Step 4: Verify Matern kappa=0.5 equivalence with a quick unit test (optional)**

If not already confirmed by the unit tests in Task 2, verify that the Matern function with kappa=0.5 matches the exponential function at the terrain phi value (993 m). This confirms the terrain correlation is equivalent between models.

---

## Implementation Notes

### Critical Verification

The Matern parameterization must be verified against R's gstat output. The spec documents that gstat uses range as a direct scale parameter (NOT multiplied by `sqrt(2*kappa)`). Before declaring the foster_2019 model complete, compute correlation values from the R code at specific distances (e.g., 100m, 1000m, 5000m, 20000m) and compare against the Python implementation. See spec section "Implementation note on Matérn parameterization" for details.

### Picklability

All correlation callables use `functools.partial` (not lambdas) because they must be picklable for `multiprocessing.Pool` workers in `parallel.py`. The tests in Task 3 verify this explicitly.

### Nugget Interaction

The Matern function's nugget causes `corr_zero ≈ 0.75` (vs ≈1.0 for exponential). The existing `compute_spatial_adjustment_for_pixel` already uses `corr_zero` to shrink prior variance: `initial_var = (pixel.stdv**2) * corr_zero`. This means with the Matern+nugget, prior variance is reduced to ~75%, which is the intended behavior. This operates independently from omega noise weighting (`noisy=true`).

### Backward Compatibility

All existing configs are updated to include the new required fields. Test fixtures are updated in Task 4. There are no defaults — configs that don't specify `geology_correlation` and `terrain_correlation` will fail with a clear error from the validation added in Task 8.

# vs30 Package Code Review

This review covers the `vs30/` package and its tests under `tests/`. It does **not** cover `VsViewer/`.

The findings are grouped by category. Within each category, items are ordered roughly by impact — fix the top items first.

Throughout, source locations use `path:line_number` so they can be opened directly. Snippets show enough context that an implementer can locate and apply each change without re-reading the surrounding code.

---

## 1. Import-style violations

The repository convention is:

> Imports should always be at the top of the module, and imported as `from vs30 import constants` so call sites use `constants.THING`. Limited exceptions are allowed — `from tqdm import tqdm`, `import numpy as np`, `import pandas as pd`, `import geopandas as gpd`.

The codebase mostly follows this. The remaining violations:

### 1.1 `from vs30.cli import resolve_correlation_function` — `tests/test_utils.py:14`

This is the clearest violation. A function is imported directly from a `vs30` submodule.

**Fix.** Replace with the parent-import form, then qualify the call site:

```python
# tests/test_utils.py:14 — BEFORE
from vs30.cli import resolve_correlation_function
# ...
fn = resolve_correlation_function(config)
```

```python
# tests/test_utils.py:14 — AFTER
from vs30 import cli
# ...
fn = cli.resolve_correlation_function(config)
```

(Update both usages at lines 217 and 229.)

### 1.2 Non-top-level import in `spatial.py:1024`

```python
# vs30/spatial.py:1020-1026
if nproc > 1 and n_chunks > 1:
    # Parallel processing
    actual_nproc = min(nproc, n_chunks)
    logger.info(f"Using {actual_nproc} parallel workers")
    from vs30 import parallel          # <-- inside function

    with parallel.spawn_context.Pool(processes=actual_nproc) as pool:
```

The reason this is buried inside the function is the import cycle: `parallel.py` imports `spatial`, so `spatial.py` cannot import `parallel` at the top.

**Fix.** Break the cycle by moving `spawn_context` (and `single_threaded_blas`, `resolve_nproc`) into a small standalone module — for example `vs30/multiprocess.py` — that has no dependency on either `spatial` or `parallel`. Then `spatial.py` can `from vs30 import multiprocess` at the top, and `parallel.py` keeps its imports of both.

If a standalone module feels heavyweight, an alternative is to move the spawn-pool plumbing currently in `parallel.find_affected_pixels` (the only consumer of `parallel.spawn_context` from `spatial`) up into the caller in `pipeline.py`, so `spatial.find_affected_pixels` returns the chunk arguments and the pool is constructed by `pipeline`. This removes the need for `spatial` to know about `parallel` at all.

### 1.3 Function/class imports from non-exempt third-party submodules

These appear in production code:

- `vs30/utils.py:5` — `from scipy.special import gamma, kv`
- `vs30/gapfill.py:19` — `from scipy.ndimage import maximum_filter`
- `vs30/gapfill.py:20` — `from scipy.spatial import KDTree`
- `vs30/raster.py:19` — `from osgeo import gdal` (this is the standard convention — *acceptable*)

And in tests:

- `tests/test_benchmarks.py:32` — `from pyproj import Transformer`
- `tests/test_gapfill.py:10` — `from rasterio.transform import Affine`

**Fix.** Switch to parent-form imports:

```python
# vs30/utils.py
import scipy.special
# ...
rho = (2 ** (1 - kappa) / scipy.special.gamma(kappa)) * (scaled ** kappa) * scipy.special.kv(kappa, scaled)
```

```python
# vs30/gapfill.py
import scipy.ndimage
import scipy.spatial
# ...
neighborhood_2d = scipy.ndimage.maximum_filter(fillable_2d, size=struct_size)
tree = scipy.spatial.KDTree(valid_locations)
```

The same treatment applies to `Transformer` and `Affine` in tests. Note: `from osgeo import gdal` follows the `osgeo` package's own convention and should be left alone.

If the user considers these standard-library/scientific-library class imports (e.g. `Path`, `Callable`, `dataclass`, `StrEnum`) as falling under the "limited exceptions" clause, only the `scipy.*` imports are unambiguously in scope here.

### 1.4 Trailing whitespace and blank line in `tests/conftest.py`

Minor: `from vs30 import constants, cli ` (line 15) has a trailing space. Line 76 of the same file is an empty line *inside* the body of `load_fixed_model_config`, immediately after the `"""` docstring close — visual noise.

---

## 2. Circular / fragile module structure

### 2.1 `parallel.py` ↔ `spatial.py` import cycle

Already covered in §1.2. The deeper issue is that `parallel.py` and `spatial.py` are tightly coupled:

- `parallel.py` imports `spatial` to construct `ObservationData` / `PixelData` and call `compute_spatial_adjustment_for_pixel`.
- `spatial.py` needs `parallel.spawn_context` for its own multiprocessing fan-out in `find_affected_pixels`.

Either:
- (A) keep all multiprocessing in `parallel.py` (move `find_affected_pixels`'s parallel branch into `parallel`), or
- (B) extract a minimal `multiprocess.py` shared by both (see §1.2).

(B) is the smaller change and removes the deferred import.

---

## 3. Unnecessary complexity in multiprocessing data shuttling

### 3.1 Dataclass→dict→dataclass round-trip with the `KEY_*` constants

`vs30/parallel.py:454-491` rebuilds `ObservationData` and `PixelData` by reading values out of dicts that were just constructed from those same dataclasses (lines 615-636). The justification is a comment:

```python
# vs30/parallel.py:456
# Reconstruct ObservationData from dict (dataclasses can't always be pickled cleanly)
```

Standard Python dataclasses ARE picklable. If `ObservationData` and `PixelData` contain only NumPy arrays and primitive types (which they do — see `spatial.py:29-105`), they pickle fine through `multiprocessing.Pool.imap`. The dict round-trip exists for no reason and creates a maintenance burden:

- A whole block of `KEY_*` string constants in `constants.py:346-360` (`KEY_LOCATIONS`, `KEY_MODEL_VS30`, `KEY_MODEL_STDV`, `KEY_RESIDUALS`, `KEY_OMEGA`, `KEY_LOCATION`, `KEY_STDV`, `KEY_INDEX`, `KEY_MODEL_TYPE`, `KEY_MAX_DIST_M`, `KEY_MAX_POINTS`, `KEY_NOISY`, `KEY_COV_REDUC`, `KEY_CORR_ZERO`) exists solely to label dict keys that are never persisted, never crossed a serialization boundary outside this round-trip, and are immediately consumed by the same code that wrote them.
- Inconsistency: in the same dicts, `obs_data.vs30` is keyed under `constants.ObservationColumn.VS30` (= `"vs30"`), while `obs_data.model_vs30` is keyed under `constants.KEY_MODEL_VS30`. There's no semantic reason for the difference.

**Fix.**

1. Pass dataclass instances directly through `pool.imap`. Replace the dict-construction in `parallel.run_parallel_spatial_fit` (lines 614-636) and the dict-deconstruction in `parallel.process_pixels_chunk` (lines 454-476) with direct dataclass passing:

   ```python
   # parallel.run_parallel_spatial_fit (replacement for 614-636)
   grid_locs = raster_data.get_coordinates()
   pixels = []
   for i, flat_idx in enumerate(affected_flat_indices):
       valid_idx = np.searchsorted(raster_data.valid_flat_indices, flat_idx)
       if valid_idx < len(grid_locs):
           pixels.append(
               spatial.PixelData(
                   location=grid_locs[valid_idx],
                   vs30=float(raster_data.vs30.flat[flat_idx]),
                   stdv=float(raster_data.stdv.flat[flat_idx]),
                   index=int(flat_idx),
               )
           )

   corr_zero = corr_fn(np.array([0.0]))[0]
   chunks = np.array_split(np.arange(len(pixels)), n_chunks)
   chunk_args = [
       ([pixels[i] for i in chunk], chunk_id, obs_data, corr_fn,
        max_dist_m, max_points, noisy, cov_reduc, corr_zero)
       for chunk_id, chunk in enumerate(chunks) if len(chunk) > 0
   ]
   ```

2. Update `process_pixels_chunk` to receive the dataclasses directly:

   ```python
   def process_pixels_chunk(args):
       pixels, chunk_id, obs_data, corr_fn, max_dist_m, max_points, noisy, cov_reduc, corr_zero = args
       updates = []
       for pixel in pixels:
           result = spatial.compute_spatial_adjustment_for_pixel(
               pixel, obs_data, corr_fn,
               max_dist_m=max_dist_m, max_points=max_points,
               noisy=noisy, cov_reduc=cov_reduc, corr_zero=corr_zero,
           )
           if result is not None:
               vs30, stdv, _ = result
               updates.append((pixel.index, vs30, stdv))
       return chunk_id, updates
   ```

3. Delete every `KEY_*` constant in `constants.py:347-360` — they have no other callers.

The `corr_fn` and other scalar config values can be passed as ordinary positional/keyword arguments rather than packed into a `config_params` dict at all.

### 3.2 `np.searchsorted` is used as if it were a "find" — fragile

`vs30/parallel.py:618`:

```python
valid_idx = np.searchsorted(raster_data.valid_flat_indices, flat_idx)
if valid_idx < len(grid_locs):
    ...
```

`np.searchsorted` returns an *insertion point*, not a "found" indicator. If `flat_idx` is not in `valid_flat_indices` but the insertion point is < `len(grid_locs)`, the code silently uses the wrong coordinate. There is no equality check against `valid_flat_indices[valid_idx]`. The current callers always pass indices that are in `valid_flat_indices`, so this is latent rather than active — but it's a footgun.

**Fix.** Either add an equality assertion:

```python
valid_idx = np.searchsorted(raster_data.valid_flat_indices, flat_idx)
assert valid_idx < len(raster_data.valid_flat_indices) and \
       raster_data.valid_flat_indices[valid_idx] == flat_idx, \
       f"flat_idx {flat_idx} not in valid_flat_indices"
```

…or precompute a `dict` from `flat_idx → valid_idx` once and look up directly.

### 3.3 Dead "chunking" inside `compute_spatial_adjustments` — `spatial.py:1085-1203`

The function defines `chunk_size`, `n_chunks`, slices `affected_flat_indices` etc. into per-chunk arrays (`chunk_flat_indices`, `chunk_affected_locs`, …) on lines 1144-1172, then iterates **one pixel at a time** inside the chunk on lines 1174-1199. The chunking has no effect on memory footprint or progress reporting (the single `tqdm` is updated per pixel, not per chunk).

**Fix.** Collapse to a single loop:

```python
def compute_spatial_adjustments(
    raster_data: RasterData,
    obs_data: ObservationData,
    bbox_result: BoundingBoxResult,
    corr_fn: Callable[[np.ndarray], np.ndarray],
    max_spatial_boolean_array_memory_gb: float,  # now unused — drop it
    max_dist_m: float = constants.MAX_DIST_M,
    max_points: int = constants.MAX_POINTS,
    noisy: bool = False,
    cov_reduc: float = constants.COV_REDUC,
) -> tuple[np.ndarray, np.ndarray]:
    affected_flat_indices = np.where(bbox_result.mask)[0]
    affected_valid_indices = np.where(bbox_result.mask[raster_data.valid_flat_indices])[0]

    grid_locs = raster_data.get_coordinates()
    affected_locs = grid_locs[affected_valid_indices]
    affected_vs30 = raster_data.vs30.flat[affected_flat_indices]
    affected_stdv = raster_data.stdv.flat[affected_flat_indices]

    updated_vs30 = raster_data.vs30.copy()
    updated_stdv = raster_data.stdv.copy()
    corr_zero = corr_fn(np.array([0.0]))[0]
    n_updated = 0

    for i, flat_idx in enumerate(tqdm(affected_flat_indices, desc="Spatial adjustment", unit="pixel")):
        pixel = PixelData(
            location=affected_locs[i],
            vs30=float(affected_vs30[i]),
            stdv=float(affected_stdv[i]),
            index=flat_idx,
        )
        result = compute_spatial_adjustment_for_pixel(
            pixel, obs_data, corr_fn,
            max_dist_m=max_dist_m, max_points=max_points,
            noisy=noisy, cov_reduc=cov_reduc, corr_zero=corr_zero,
        )
        if result is not None:
            vs30, stdv, _ = result
            updated_vs30.flat[flat_idx] = vs30
            updated_stdv.flat[flat_idx] = stdv
            n_updated += 1

    logger.info(f"Completed: {n_updated:,} pixels updated")
    return updated_vs30, updated_stdv
```

`max_spatial_boolean_array_memory_gb` becomes unused here; remove from the signature and from the call site at `pipeline.py:577`.

(The chunking *inside* `find_affected_pixels` is genuine — it controls the size of the broadcasted boolean arrays — and should stay.)

---

## 4. Dead / unused code

These are defined but have no production caller. The reviewer should verify with `grep -rn` before deleting in case I missed an indirect reference, but the spot-checks all came up empty.

### 4.1 `raster.create_category_id_raster` — `vs30/raster.py:214-277`

Only `create_category_id_array` is called. `create_category_id_raster` duplicates `pipeline.write_id_raster`'s job (write 1-band uint8 raster) but is never invoked.

**Fix.** Delete it. The pipeline already calls `create_category_id_array` and then `write_id_raster` (`pipeline.py:895, 905`).

### 4.2 Unused constants in `constants.py`

These have no readers in `vs30/` or `tests/`:

- `RASTER_BAND_VS30` (line 378)
- `RASTER_BAND_STDV` (line 379)
- `GEOTIFF_TILED` (line 384)
- `GEOTIFF_BIGTIFF` (line 385)

(Only `GEOTIFF_DRIVER` and `GEOTIFF_COMPRESSION` are used, in `raster.py:134, 142`.)

**Fix.** Delete them. If they were intended to be passed as part of the rasterio profile, they should either be wired in or removed.

### 4.3 File-based slope/coast-distance fallback in `prepare_observation_data` is unreachable

`spatial.py:489-529` (the `else` branch when `has_in_memory_arrays` is False) reads slope/coast-distance from on-disk rasters and creates them via `raster.create_slope_raster` / `raster.create_coast_distance_raster` if missing. The only caller, `pipeline.compute_spatial_adjustment_on_grid` (`pipeline.py:498-508`), always passes `slope_array` and `coast_dist_array`, so `has_in_memory_arrays` is always True.

**Fix.** Either:

- (A) Remove the dead branch (`spatial.py:489-529`), the `output_dir` parameter (`spatial.py:326`), and `raster.create_slope_raster` and `raster.create_coast_distance_raster` (which only exist for this path). Keep the in-memory `compute_slope_array` and `compute_coast_distance_array`.

- (B) If the file-based path is intentionally retained for a future use case, document that and add a test exercising it. Without a caller and without a test it is liability code.

(A) is recommended unless there's a load-bearing reason for (B) that isn't visible from the source.

### 4.4 Diagnostic block in `compute_spatial_adjustment_for_pixel`

`spatial.py:21-26, 852-925` — env-var-gated timing instrumentation (`VS30_MVN_DIAG`, `VS30_MVN_DIAG_*`). It's wired up in 3 places (module-level state, the function body, and `parallel.run_parallel_spatial_fit:663-666`). It writes to `/tmp` files.

This is "production debugging code". It currently looks well-isolated, but it pollutes the function with `_diag_active`, `t0`-`t4` timestamps, a global counter, and a sampling loop in the middle of the hot path. No test exercises it.

**Suggestion.** Move it behind a context manager or `@contextlib.contextmanager` profiling wrapper, or remove it entirely and reproduce on demand with a profiler when needed (`scalene`, `py-spy`). At minimum, document somewhere user-facing that the env vars exist.

### 4.5 `mvn` is in `points_custom`/`grid_custom` but missing from yaml configs

`cli.points_custom` (line 312) and `cli.grid_custom` (line 544) take `mvn: bool` as a *required* CLI option. The fixed-version `cli.points` and `cli.grid` do not — and the YAML configs don't contain a `mvn` key, so `cfg["mvn"]` would `KeyError` if `points`/`grid` ever read it. Currently `mvn` is just hard-coded `True` for the fixed-version path.

**Fix.** Either:

- Add `mvn` to the config YAMLs and to the required-fields list in `cli.load_model_config:85-91`, then thread it through, or
- Document that fixed-version `points`/`grid` runs always include MVN and that toggling it requires `*_custom`.

The first option is cleaner — it puts the toggle in the config rather than hidden in a CLI dispatch path.

---

## 5. Inconsistencies

### 5.1 `pd.read_csv` parameters differ across observation reads

The same observation CSVs are read with different parameters depending on caller:

- `pipeline.py:78` (in `_collect_observation_csvs`): `pd.read_csv(csv, comment="#")`
- `pipeline.py:177` (in `compute_categorical_vs30_updates`): `pd.read_csv(..., skipinitialspace=True, comment="#")`
- `pipeline.py:231`: same as above
- `pipeline.py:155, 891, 1393, 1412` (categorical model CSVs): `pd.read_csv(..., skipinitialspace=True)` — no `comment="#"`

If observation CSVs include leading whitespace after a separator, the two paths will produce different column-strip behaviour.

**Fix.** Centralise the read in two helpers — one for observations, one for categorical models — and use them everywhere:

```python
# vs30/pipeline.py — at module level
def _read_observations_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", skipinitialspace=True)

def _read_categorical_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#", skipinitialspace=True)
```

…and replace every call site with one of these.

### 5.2 `print(...)` mixed with `logger.info(...)` for the same message

`pipeline.py:546-551`:

```python
print(f"  find_affected_pixels: {t_bbox_elapsed:.1f}s "
      f"({bbox_result.n_affected_pixels:,} affected pixels)")
logger.info(
    f"Found {bbox_result.n_affected_pixels:,} affected pixels "
    f"in {t_bbox_elapsed:.1f}s"
)
```

`pipeline.py:584-585`:

```python
print(f"  compute_spatial_adjustments: {t_spatial_elapsed:.1f}s")
logger.info(f"Spatial adjustments completed in {t_spatial_elapsed:.1f}s")
```

`spatial.py:1048-1054` also uses `print` for a "single chunk" status update, while every other status in that function goes through `logger.info`/`tqdm`.

**Fix.** Drop the `print` calls. Anything important should go through `logger.info`. If the user needs human-facing one-line progress when there are no `tqdm` bars, route it through the logger with a console handler at INFO level.

### 5.3 `compress` value: constant vs. literal

- `raster.py:142` uses `constants.GEOTIFF_COMPRESSION`
- `raster.py:556, 642`, `pipeline.py:707, 750` use `"deflate"` literal

**Fix.** Use `constants.GEOTIFF_COMPRESSION` everywhere or hard-code `"deflate"` everywhere. The constant exists; honour it.

### 5.4 `_collect_observation_csvs` uses `csv.exists()`, callers usually rely on existence

`pipeline.py:71-81`:

```python
csvs = [
    csv
    for csv in [clustered_observations_csv, independent_observations_csv]
    if csv is not None and csv.exists()
]
```

The CLI already guarantees these paths exist (Typer's `exists=True` option). Inside `points_pipeline` the same paths are passed straight through. Silently dropping a non-existent path here is masking a bug rather than handling one. If a non-None path doesn't exist, that's something the caller has gotten wrong and should hear about.

**Fix.** Drop the `.exists()` check; trust the caller. If you want a guard, raise:

```python
csvs = [csv for csv in [clustered_observations_csv, independent_observations_csv] if csv is not None]
for csv in csvs:
    if not csv.exists():
        raise FileNotFoundError(csv)
```

### 5.5 `cli.load_model_config` and `tests/conftest.load_fixed_model_config` are near-duplicates

`cli.load_model_config` (`cli.py:58-113`) and `conftest.load_fixed_model_config` (`tests/conftest.py:58-95`) share most logic. Differences:

- conftest uses `config_data.get(key)` instead of `config_data[key]`.
- conftest skips the required-fields validation (`cli.py:85-96`).
- conftest opens the file without `encoding="utf-8"`.

**Fix.** Make `cli.load_model_config` the single source of truth; have conftest call it. The validation should run in tests too.

```python
# tests/conftest.py
def load_fixed_model_config(version):
    return cli.load_model_config(version)
```

---

## 6. Code quality / simplifiability

### 6.1 `combine_model_arrays` does redundant work — `pipeline.py:595-662`

```python
# pipeline.py:633-639
if (
    combination_method is constants.CombinationMethod.RATIO
    and combine_ratio is None
):
    raise ValueError(
        "combination_method is set to 'ratio' but combine_ratio is not provided"
    )

# pipeline.py:641-645 — make a copy in float32 four times
geol_vs30 = np.array(geol_vs30, dtype=np.float32, copy=True)
geol_stdv = np.array(geol_stdv, dtype=np.float32, copy=True)
terr_vs30 = np.array(terr_vs30, dtype=np.float32, copy=True)
terr_stdv = np.array(terr_stdv, dtype=np.float32, copy=True)
```

- The `RATIO` validation duplicates the one already inside `utils.combine_vs30_models` (`utils.py:155-158`). Drop it.
- `np.array(x, dtype=np.float32, copy=True)` is more idiomatically `x.astype(np.float32, copy=True)` (assuming `x` is already an `ndarray`), and may be a no-op copy when `x.dtype == np.float32`. Replacing four lines with `astype` is fine, but the deeper question is whether copies are needed at all here — the only mutation that follows is `arr[arr == nodata] = np.nan`, and the inputs are typically rasters that the caller doesn't reuse afterwards.

**Fix.**

```python
def combine_model_arrays(
    geol_vs30, geol_stdv, terr_vs30, terr_stdv,
    combination_method, combine_ratio=None,
    nodata=constants.NODATA_VALUE,
):
    geol_vs30 = geol_vs30.astype(np.float32, copy=True)
    geol_stdv = geol_stdv.astype(np.float32, copy=True)
    terr_vs30 = terr_vs30.astype(np.float32, copy=True)
    terr_stdv = terr_stdv.astype(np.float32, copy=True)
    for arr in (geol_vs30, geol_stdv, terr_vs30, terr_stdv):
        arr[arr == nodata] = np.nan
    return utils.combine_vs30_models(
        geol_vs30, geol_stdv, terr_vs30, terr_stdv,
        combination_method, combine_ratio,
    )
```

### 6.2 Three near-identical raster writers in `pipeline.py`

`write_vs30_raster`, `write_single_band_raster`, `write_id_raster` (`pipeline.py:670-780`) all do:

1. `output_path.parent.mkdir(parents=True, exist_ok=True)`
2. Build/copy a profile.
3. `with rasterio.open(...) as dst: dst.write(...); dst.descriptions = (...)`.

The differences are dtype/count/nodata/descriptions.

**Fix.** Collapse to a single helper:

```python
def write_raster(
    output_path: Path,
    profile: dict,
    bands: list[np.ndarray],
    band_descriptions: tuple[str, ...],
    *,
    dtype: str = "float32",
    nodata: float | None = constants.NODATA_VALUE,
    compress: str = constants.GEOTIFF_COMPRESSION,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_profile = profile.copy()
    write_profile.update(
        {"dtype": dtype, "count": len(bands), "nodata": nodata, "compress": compress}
    )
    with rasterio.open(output_path, "w", **write_profile) as dst:
        for i, band in enumerate(bands, start=1):
            dst.write(band.astype(dtype) if band.dtype.name != dtype else band, i)
        dst.descriptions = band_descriptions
    logger.info(f"Wrote raster: {output_path}")
```

Call sites become:

```python
write_raster(path, profile, [vs30, stdv],
             (constants.BAND_DESCRIPTION_VS30, constants.BAND_DESCRIPTION_STDV))

write_raster(path, profile, [slope],
             (constants.BAND_DESCRIPTION_SLOPE,), nodata=constants.NODATA_VALUE)

write_raster(path, profile, [id_array],
             (constants.BAND_DESCRIPTION_ID_INDEX,),
             dtype="uint8", nodata=constants.RASTER_ID_NODATA_VALUE)
```

### 6.3 `assign_to_category_*` dispatch is repeated three times

```python
# pipeline.py:193-196
if model_type == constants.ModelType.GEOLOGY:
    model_ids = category.assign_to_category_geology(obs_locs)
else:  # terrain
    model_ids = category.assign_to_category_terrain(obs_locs)

# pipeline.py:249-252 — same again

# spatial.py:401-406 — same with explicit ValueError
```

**Fix.** Add one helper in `category.py`:

```python
def assign_to_category(points: np.ndarray, model_type: constants.ModelType) -> np.ndarray:
    if model_type == constants.ModelType.GEOLOGY:
        return assign_to_category_geology(points)
    if model_type == constants.ModelType.TERRAIN:
        return assign_to_category_terrain(points)
    raise ValueError(f"Unsupported model_type for category assignment: {model_type}")
```

Then call `category.assign_to_category(...)` in all three places.

### 6.4 Loading observations for the "compute Bayesian update" path is duplicated

`pipeline.compute_categorical_vs30_updates` has two near-identical blocks for clustered (171-223) and independent (225-254) observations. Each:

1. Logs "Loading X observations from: …"
2. `pd.read_csv(...)` with the same parameters
3. Calls `utils.validate_csv_columns(...)` with the same `REQUIRED` list and a different label
4. Logs "Loaded N observations"
5. Builds `obs_locs` from `[easting, northing]`
6. Calls `assign_to_category_geology` or `_terrain` based on `model_type` (this is §6.3)
7. Sets `df[STANDARD_ID_COLUMN] = model_ids`

**Fix.** Lift the common parts into:

```python
def _load_and_assign_observations(
    csv_path: Path, model_type: constants.ModelType, label: str
) -> pd.DataFrame:
    logger.info(f"Loading {label} observations from: {csv_path}")
    df = pd.read_csv(csv_path, skipinitialspace=True, comment="#")  # see §5.1
    utils.validate_csv_columns(df, constants.ObservationColumn.REQUIRED, f"{label} observations CSV")
    logger.info(f"Loaded {len(df)} {label} observations")
    obs_locs = df[[constants.ObservationColumn.EASTING, constants.ObservationColumn.NORTHING]].values
    df[constants.STANDARD_ID_COLUMN] = category.assign_to_category(obs_locs, model_type)
    return df
```

The body of `compute_categorical_vs30_updates` then shrinks to ~30 lines.

### 6.5 Defensive `is None` guards that exist only for type narrowing

Several `raise ValueError(...) if x is None` guards exist solely because the type checker can't prove non-None — not because None is genuinely a runtime possibility:

- `pipeline.py:1191-1192` — `profile is None` after the function has just populated it
- `pipeline.py:1434-1437` — `geol_model_df / terr_model_df is None` inside an already-guarded `run_geology and run_terrain` branch
- `pipeline.py:1464-1467, 1501-1504` — same
- `pipeline.py:1614-1623` — `local_profile is None` and `not isinstance(local_vs30, np.ndarray)` after `grid_pipeline` was just called and is documented to populate them
- `spatial.py:438-441` — `slope_array is None or coast_dist_array is None` inside an `if has_in_memory_arrays` branch where `has_in_memory_arrays` is *defined as* both being non-None (see `spatial.py:384`)

These dilute the signal: a real `None` would be hidden among the noise.

**Fix.** Tighten the types instead. For example, `compute_model_grid` should return a non-Optional `profile`; `grid_pipeline`'s `result` dict should be a `TypedDict` whose `combined_vs30`/`combined_stdv` are `np.ndarray`. Once the types are right, the guards can be deleted. If the type checker still complains, use `assert` (silent in optimized runs) rather than `raise ValueError`.

### 6.6 `grid_points_in_bbox` per-observation loop

`spatial.py:629-634`:

```python
obs_to_grid_indices = []
for obs_idx in range(in_bbox.shape[0]):
    obs_to_grid_indices.append(np.where(in_bbox[obs_idx])[0] + start_grid_idx)
```

**Fix.** Turn into a list comprehension (cosmetic, but more idiomatic):

```python
obs_to_grid_indices = [
    np.where(in_bbox[obs_idx])[0] + start_grid_idx
    for obs_idx in range(in_bbox.shape[0])
]
```

### 6.7 `category.get_vs30_for_ids` is more complicated than it needs to be

`category.py:474-531`. Builds a `dict[id, (mean, stdv)]`, looks up each `cid` with `dict.get(cid, (np.nan, np.nan))`, then unpacks first/second elements into separate arrays via list comprehensions.

**Fix.** Use pandas reindexing:

```python
def get_vs30_for_ids(
    category_ids: np.ndarray,
    categorical_model_df: pd.DataFrame,
) -> pd.DataFrame:
    mean_col, stdv_col = raster.select_vs30_columns_by_priority(
        list(categorical_model_df.columns)
    )
    indexed = categorical_model_df.set_index(constants.STANDARD_ID_COLUMN)
    reindexed = indexed.reindex(category_ids)
    return pd.DataFrame(
        {
            constants.COL_CATEGORY_VS30_MEAN: reindexed[mean_col].to_numpy(dtype=np.float64),
            constants.COL_CATEGORY_VS30_STDV: reindexed[stdv_col].to_numpy(dtype=np.float64),
        }
    )
```

NaN propagation for missing IDs is automatic with `reindex`.

### 6.8 `update_with_independent_data` uses `iterrows` for two nested loops

`category.py:231-270` iterates rows with `df.iterrows()` and writes back with `df.at[idx, col] = value`. Acceptable for small categorical models (a few dozen rows × a few hundred observations) but `iterrows` is slow. More importantly, the inner sequential update loop:

```python
for _, observation_row in observations_for_category_df.iterrows():
    new_variance = compute_bayesian_posterior_variance(...)
    current_mean = compute_bayesian_posterior_mean(...)
    current_std = np.sqrt(new_variance)
    current_n += 1
```

…is purely numeric. Convert the rows to NumPy arrays once and loop over them:

```python
obs_vs30 = observations_for_category_df[constants.ObservationColumn.VS30].to_numpy()
obs_unc  = observations_for_category_df[constants.ObservationColumn.UNCERTAINTY].to_numpy()
for vs30_value, uncertainty in zip(obs_vs30, obs_unc):
    new_variance = compute_bayesian_posterior_variance(current_std, current_n, uncertainty, current_mean, vs30_value)
    current_mean = compute_bayesian_posterior_mean(current_mean, current_n, vs30_value)
    current_std  = np.sqrt(new_variance)
    current_n   += 1
```

Cleaner and ~10x faster on the typical observation count.

### 6.9 The fill-gaps inner loop in `points_pipeline` is too long

`pipeline.py:1556-1654` is ~100 lines of densely nested logic with a `while … else` clause (line 1645) that's easy to misread. The body for one fillable point is the entire loop body.

**Fix.** Extract a helper:

```python
def _fill_one_point_via_local_grid(
    e: float, n: float,
    gapfill_grid_config: config.GridConfig,
    grid_pipeline_kwargs: dict,
) -> tuple[float, float]:
    """Return (fill_vs30, fill_stdv); both are NaN if no donor was found."""
    half_width = constants.GAPFILL_LOCAL_GRID_SIZE_M
    while half_width <= constants.GAPFILL_MAX_LOCAL_GRID_HALF_WIDTH_M:
        local_config = gapfill.create_local_grid_config(e, n, gapfill_grid_config, half_width)
        local_result = grid_pipeline(grid_config=local_config, output_dir=None, **grid_pipeline_kwargs)
        local_vs30, local_stdv = gapfill.fill_nodata_grid(
            local_result["combined_vs30"], local_result["combined_stdv"],
            local_result["geology_ids"], local_result["profile"],
        )
        row, col = rasterio.transform.rowcol(local_result["profile"]["transform"], e, n)
        fill_vs30 = local_vs30[row, col]
        fill_stdv = local_stdv[row, col]
        if not np.isnan(fill_vs30):
            return fill_vs30, fill_stdv
        half_width += constants.GAPFILL_LOCAL_GRID_EXPANSION_M
        logger.info(f"  Gap-fill: expanding local grid to {half_width * 2}m for point ({e:.0f}, {n:.0f})")
    logger.warning(f"  Gap-fill: no valid donor found for point ({e:.0f}, {n:.0f})")
    return np.nan, np.nan
```

The defensive `is None` / `isinstance` checks (lines 1614-1623) go away with §6.5.

### 6.10 `compute_cluster_weighted_mean_and_stddev` recomputes `np.log(vs30)` twice

`category.py:333-382`. Inside the loop, each cluster does `np.log(cluster_sites[VS30].values).sum()`. After the loop, the outer `log_stddev` calculation does `np.log(category_sites[VS30].values)` over all rows again.

**Fix.** Compute `log_vs30` once outside the loop and index into it with `cluster_mask`:

```python
log_vs30_all = np.log(category_sites[constants.ObservationColumn.VS30].values)
weights = np.repeat(1.0 / effective_n, len(category_sites))
weighted_log_vs30_sum = 0.0

for cluster_label in cluster_counts.index:
    cluster_mask = (
        category_sites[constants.ObservationColumn.CLUSTER].values == cluster_label
    )
    cluster_log = log_vs30_all[cluster_mask]
    if cluster_label == constants.CLUSTER_UNCLUSTERED_LABEL:
        weighted_log_vs30_sum += cluster_log.sum()
    else:
        weighted_log_vs30_sum += cluster_log.sum() / cluster_log.size
        weights[cluster_mask] /= cluster_log.size

log_geometric_mean = weighted_log_vs30_sum / effective_n
log_stddev = np.sqrt(np.sum(weights * (log_vs30_all - log_geometric_mean) ** 2))
return float(np.exp(log_geometric_mean)), float(log_stddev)
```

---

## 7. Comments

The user's rule: comments should be genuinely informative, not restate what the code says.

### 7.1 Step-narration comments that restate the next line

Many comments paraphrase the immediate next statement. Removing them makes the code denser to read but no less clear. Examples to delete:

- `category.py:33` `# load QMAP polygons` — followed by `gpd.read_file(... GEOLOGY_SHAPEFILE_PATH)`
- `category.py:38` `# Build point GeoDataFrame` — followed by `gpd.GeoDataFrame(geometry=...)`
- `category.py:42` `# Spatial join` — followed by `gpd.sjoin(...)`
- `category.py:80` `# Handle nodata values` — followed by `if src.nodata is not None: ...`
- `category.py:184` `# Make a working copy to avoid modifying the input DataFrame` — followed by `.copy()` (the `.copy()` already says it)
- `category.py:216, 238, 252, 261, 305, 306, 319, 322, 397, 401, 409, 423, 431, 436, 444` — virtually all of the section-banner comments in this file
- `parallel.py:119, 122, 127, 135, 146, 154-160, 250, 253, 258, 362, 391, 410, 469, 470, 547-548, 559, 613, 660-661` — same pattern: a comment for each step that just renames the next call
- `pipeline.py:471, 481, 484, 496, 534, 553, 641, 647, 871, 893, 916, 961, 1129, 1153, 1176, 1345, 1448, 1462, 1499, 1530` — the numbered-step / underline-banner comments

The Stage banners like `# === STAGE 4: ===` at `pipeline.py:84-86, 271-273, 328-330, 399-401, 590-592, 665-667, 783-785, 1007-1009, 1244-1246` *do* serve as visual separators. Keeping them is reasonable; if you want to be strict, the function names already tell you what they do.

**Fix.** Remove the comments that just paraphrase the next line. Keep section banners only at the top level.

### 7.2 Comments that point at line numbers

`parallel.py:155` `# spatial.py:405-436` and `parallel.py:160` `# spatial.py:480-488` — line-number references will rot when the target file changes (it's already drifted: line 405 in spatial.py is no longer the relevant call as of this review).

**Fix.** Replace with a function-name reference, e.g. "see spatial.prepare_observation_data".

### 7.3 Test comments stating the obvious assertion

`tests/test_gapfill.py:28-29, 36-37, 45, 54-55` (and other instances) have comments like:

```python
# result[0] should evaluate to True
assert result[0], "On-land nodata pixel with valid GID should be fillable"
```

The assertion message and the assertion expression already convey the intent. Drop the preceding comment.

### 7.4 Useful comments to keep

For balance, these comments are *good* and should not be touched:

- `utils.py:78-79` (`del sill, nugget  # retained for config compatibility only`) — explains a non-obvious choice
- `spatial.py:480-488` — explains a legacy compatibility hack with a real reason
- `raster.py:459-463` (the math-pixel-alignment comment in `compute_coast_distance_array`) — non-obvious geometry
- `constants.py:226-231` (`LEGACY_OBS_SLOPE_NODATA_SENTINEL`) — explains *why* a magic value exists
- `spatial.py:789-792` (the einsum explanation) — useful pointer for a reader who doesn't know einsum
- `pipeline.py:108-113` (Bayesian-update ordering rationale) — explains the *why*

The pattern: keep comments that explain *why* a non-obvious choice was made. Remove comments that explain *what* the next line does.

---

## 8. Smaller observations

### 8.1 `validate_observations` and `validate_csv_columns` overlap — `spatial.py:292-316`

`validate_observations` reimplements the missing-columns check that `utils.validate_csv_columns` (`utils.py:182-204`) already does, then adds two positivity checks.

**Fix.** Compose them:

```python
def validate_observations(observations: pd.DataFrame) -> None:
    utils.validate_csv_columns(observations, constants.ObservationColumn.REQUIRED, "Observations")
    if not np.all(observations[constants.ObservationColumn.VS30] > 0):
        raise ValueError("Vs30 must be positive")
    if not np.all(observations[constants.ObservationColumn.UNCERTAINTY] > 0):
        raise ValueError("Uncertainty must be positive")
```

### 8.2 `validate_observations` is called once and `prepare_observation_data` does its own filter

`spatial.py:482` calls `validate_observations` and then `spatial.prepare_observation_data` re-filters by `~np.isnan(model_vs30) & ~np.isnan(model_stdv)` (`spatial.py:425`). The two checks are not mutually exclusive — `validate_observations` rejects bad input *data*, and `prepare_observation_data` filters out points that fall outside the model. That's fine; just noting the responsibilities are split for a reason and shouldn't be merged.

### 8.3 `_compute_valid_mask` and `RasterData.from_arrays` could be simpler

`spatial.py:109-141`. Sufficient as-is, but the function is only used internally by `RasterData.from_arrays`. It could be a `@staticmethod` of `RasterData` rather than a module-level function, since nothing else calls it.

### 8.4 `dump_dir = Path("/tmp/vs30_test_dumps")` in test helper

`tests/conftest.py:115`. Hardcoded `/tmp` path inside a test assertion helper. This silently accumulates files across runs and is OS-specific (won't work on Windows).

**Fix.** Use `tmp_path_factory` (a built-in pytest fixture) for shared scratch directories, or remove the dump entirely (it appears to be a debug aid; if so, gate it on an env var or remove).

### 8.5 Test files use literal column names instead of `constants.*`

`tests/test_category.py:144-146, 154-157, 168-173, 178, 184, 195, 207-209, 217, 231-235`, and similar elsewhere. Hardcoded `"posterior_mean_vs30_km_per_s_independent_observations"`, `"id"`, `"enforced_min_sigma"`, etc. instead of `constants.COL_*`.

**Fix.** Use the constants. If the constant name ever changes, the test breaks visibly at the constants definition rather than at every occurrence:

```python
# tests/test_category.py
from vs30 import constants

# fixture:
return pd.DataFrame({
    constants.STANDARD_ID_COLUMN: [1, 2, 3],
    constants.COL_MEAN: [200.0, 300.0, 400.0],
    constants.COL_STDV: [0.5, 0.4, 0.3],
})

# assertions:
assert constants.COL_POSTERIOR_MEAN_INDEPENDENT in result.columns
```

### 8.6 `pipeline.compute_categorical_vs30_updates` debug logging on line 213 prints an arbitrary slice

```python
logger.info(
    f"Unique category IDs in observations: {sorted(unique_assigned_ids[unique_assigned_ids != constants.RASTER_ID_NODATA_VALUE])[:20]}"
)
```

`[:20]` silently truncates if there are more than 20 unique IDs. Either drop the truncation or note "(showing first 20)". As written, the user has no signal that they're seeing a partial list.

### 8.7 `cli.py:115` magic top-level comment

`# CLI helper shared by `points` and `points_custom` to handle CSV I/O and column merging.` precedes `def run_points_pipeline(...)`. This belongs in the function's docstring, not as a free-standing comment.

### 8.8 `BAND_DESCRIPTION_*` constants aren't always used

`raster.py:561` `dst.descriptions = (constants.BAND_DESCRIPTION_COAST_DISTANCE,)` — uses the constant.
`raster.py:646` `dst.descriptions = (constants.BAND_DESCRIPTION_SLOPE,)` — uses the constant.
But `pipeline.write_vs30_raster` accepts strings as parameters with the constants as defaults — fine. Just confirm callers actually use the constants when overriding (they do: `pipeline.py:957-958, 1210-1211, 1230-1231`).

No fix needed; observation only.

### 8.9 `constants.HYBRID_VS30_PARAMS` and `HYBRID_SIGMA_REDUCTION_FACTORS` are inconsistent

`constants.py:206-220`. `HYBRID_VS30_PARAMS` is a list of dataclass instances keyed by `gid` (2, 3, 4, 6). `HYBRID_SIGMA_REDUCTION_FACTORS` is a `dict[int, float]` keyed by *the same gids* (2, 3, 4, 6). They could be unified into a single dataclass:

```python
@dataclass
class HybridGeologyParams:
    gid: int
    slope_limits: tuple[float, float]
    vs30_values: tuple[float, float]
    sigma_reduction: float

HYBRID_GEOLOGY_PARAMS = [
    HybridGeologyParams(gid=2, slope_limits=(-1.85, -1.22), vs30_values=(242, 418), sigma_reduction=0.4888),
    HybridGeologyParams(gid=3, slope_limits=(-2.70, -1.35), vs30_values=(171, 228), sigma_reduction=0.7103),
    HybridGeologyParams(gid=4, slope_limits=(-3.44, -0.88), vs30_values=(252, 275), sigma_reduction=0.9988),
    HybridGeologyParams(gid=6, slope_limits=(-3.56, -0.93), vs30_values=(183, 239), sigma_reduction=0.9348),
]
```

`raster.apply_hybrid_geology_modifications` then iterates once instead of twice and the relationship between the two tables is explicit.

### 8.10 `apply_hybrid_geology_modifications` parameter explosion

`raster.py:750-766`. The function takes 8 hybrid-mod parameters (`hybrid_mod6_dist_min`, `hybrid_mod6_dist_max`, `hybrid_mod6_vs30_min`, `hybrid_mod6_vs30_max`, ditto for mod13). Every caller passes the constants (`HYBRID_MOD6_DIST_MIN`, …) as defaults; the only callers that override are tests. The names also confuse: "mod6" applies to GID 4, "mod13" applies to GID 10 (per the docstring). The numbering 6/13 is from a legacy R model and conveys nothing to a reader of this code.

**Fix.** Pass the parameters in a dataclass or just hard-code from constants inside the function (tests can monkeypatch the module-level constants if they need to override). At minimum, rename `mod6/mod13` → `gid4/gid10`.

### 8.11 `parallel.process_geology_at_points` mid-function `if` for spatial fit creates duplication with grid path

`parallel.process_geology_at_points` (lines 63-204) and `parallel.process_terrain_at_points` (lines 207-288) duplicate the categorical-lookup → spatial-adjustment scaffold. The terrain version is a strict subset of the geology version (no hybrid mods). It would be possible to merge them with `model_type` switching, but doing so without complicating the signature is hard — accept the duplication.

The redundancy worth fixing: the **spatial-adjustment-on-points** preparation (computing residuals, slope/coast samples for observations) is duplicated between `parallel.py:147-194` and `spatial.py:319-559` (`prepare_observation_data`). Both reproduce the same legacy-NODATA sentinel logic and the same `apply_hybrid_geology_modifications` call. Extract this into a single helper used by both points-mode and grid-mode paths.

### 8.12 `setup.py` and `pyproject.toml` coexist

Worth confirming this is intentional. `pyproject.toml` is normally enough on modern Python. The `setup.py` (`/home/arr65/src/Vs30/setup.py`) appears to handle shapefile extraction at install time; if so, that should be in `pyproject.toml`'s `[tool.setuptools]` or in a custom build hook, not in a separate `setup.py`. Out of scope of code review, but flag for follow-up.

---

## 9. Suggested ordering of fixes

A reasonable sequence that minimises rebase pain:

1. **§1.1 — `from vs30.cli import resolve_correlation_function`** (1-line fix).
2. **§4 — Dead code removal** (constants, unused functions).
3. **§5.1 — Centralise `pd.read_csv` calls.**
4. **§5.2/§5.3 — Print → logger; literal `"deflate"` → constant.**
5. **§3.3 — Remove dead chunking in `compute_spatial_adjustments`.**
6. **§6.5 — Tighten types and remove defensive `None` guards.**
7. **§3.1 — Eliminate the dataclass→dict round-trip and the `KEY_*` constants.**
8. **§1.2 / §2.1 — Resolve the import cycle by extracting `multiprocess.py`.**
9. **§6.2 — Collapse the three raster writers.**
10. **§6.3, §6.4, §6.7, §6.8, §6.10 — Local code-quality cleanups (each independent).**
11. **§7 — Comment cleanup.**
12. **§8 — Smaller observations as time permits.**

Each step should be a separate commit/PR. After step 5, run the full test suite (`pytest --runslow`) to make sure the chunking removal doesn't change numerical output (it shouldn't — but that's the most behaviour-affecting change in the list).

---

## Out of scope

- `VsViewer/` (excluded by the user).
- The numerical correctness of the MVN math, the Bayesian update formulas, or the model-combination weighting. The math is documented and tested against R `gstat` reference values (`tests/test_utils.py:140-208`); I have not audited the formulas line-by-line.
- The yaml configs themselves — only `cli.load_model_config`'s handling of them.
- Performance benchmarking. Several suggestions touch hot-path code (e.g. §6.8, §3.3); each should be benchmarked before/after with a representative dataset.

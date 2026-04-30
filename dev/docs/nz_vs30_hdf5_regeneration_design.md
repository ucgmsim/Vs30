# NZ Vs30 HDF5 Regeneration — Design

**Date:** 2026-05-01
**Branch:** `vs30_refactor`
**Status:** Design (pre-implementation)

## 1. Goal

Regenerate the two HDF5 files at `/home/arr65/src/nzcvm_data/vs30/`
(`NZ_Vs30.h5`, `NZ_Vs30_HD_With_Offshore.h5`) so that they carry Vs30 values
sampled from the refactored `jaehwi_v1p0` model, on the same WGS84 lat/lon
grid the files already define.

> **Renaming the misnamed dataset.** The existing files store their Vs30
> values in a dataset named `elevation` — a template misnomer carried
> over from an earlier file format, not an actual elevation dataset. The
> regenerated files rename this to `vs30` to end the confusion. Any
> consumer code that currently reads `elevation` from these files needs a
> small update; the search-and-replace is in scope for this work (see
> §6).

## 2. Existing grid

Both HDF5 files share an identical 3001 × 3001 grid stored as separate
`latitude` and `longitude` 1-D arrays:

- Latitude: −48.0° → −33.0°, step 0.005°
- Longitude: 165.0° → 180.0°, step 0.005°

The grid is regular in WGS84 lat/lon but **not** axis-aligned in NZTM —
its NZTM footprint is a curved quadrilateral (1403 km wide × 1696 km tall)
and the longitude spacing in metres varies with latitude:

| Adjacent neighbour | Distance |
|---|---|
| North–south | ~555 m everywhere |
| East–west at lat = −33° | ~467 m |
| East–west at lat = −40.5° | ~424 m |
| East–west at lat = −48° | ~373 m (minimum on the grid) |

## 2.1 What the existing files currently contain

Both files have an identical `(latitude, longitude, elevation)` schema with
`elevation` actually carrying Vs30. Their value distributions, however, are
dominated by discrete sentinels — not what the modern Bayesian/MVN
pipeline produces:

| File | min | median | max | Sentinel observation |
|---|---:|---:|---:|---|
| `NZ_Vs30.h5` | 108 | 500 | 1221 | ~50 % of cells exactly 500.00 |
| `NZ_Vs30_HD_With_Offshore.h5` | 50 | 50 | 847 | >75 % of cells exactly 50.00 |

This pattern is consistent with a categorical lookup of the form
"geology/terrain unit → fixed Vs30", not with `jaehwi_v1p0`'s continuous
MVN-adjusted output. Regenerating with `jaehwi_v1p0` therefore changes the
**content** as well as the **method**: the new files will have smoothly
varying values everywhere on land, with no sentinel plateaus. This is by
design and worth flagging to downstream consumers.

What distinguishes the two existing files (besides the value range) is
not yet established. The plain `NZ_Vs30.h5` looks like a land-only
categorical map with 500 m/s as the offshore default; the
`HD_With_Offshore` variant looks like a soft-sediment / offshore-included
version with 50 m/s as the offshore default. Investigation deferred to
§6.

## 3. Approach

Three stages:

1. **Run `vs30 grid jaehwi_v1p0` at NZTM 100 m on `FULL_NZ_GRID_CONFIG`.**
   Produces a multi-band GeoTIFF (`combined_vs30` + `combined_stdv`) at
   10 600 × 15 200 = 161 M cells, EPSG:2193, with `fill_gaps=true` per the
   model's config.

2. **Reproject the GeoTIFF to the WGS84 lat/lon grid via
   `rasterio.warp.reproject`** with `Resampling.bilinear`. Source = NZTM
   raster, destination = a fresh 3001 × 3001 lat/lon raster whose transform
   places pixel centres at exactly the latitudes and longitudes stored in
   the HDF5 file.

3. **Write the resampled arrays back to HDF5** mirroring the existing file
   structure (`latitude`, `longitude`, plus new `vs30` and `stdv` 2-D
   datasets; `ncols`, `nrows` root attrs). Decision pending: whether to
   write new files alongside the originals or overwrite — see §6.

## 4. Key design decisions

### 4.1 NZTM grid spacing — `dx = dy = 100 m`

The smallest WGS84-grid neighbour spacing is ~373 m (E–W at lat = −48°), so
100 m gives:

- Bilinear resampling smear at category/coastal discontinuities ≤ ¼ of the
  smallest target cell — sub-pixel, generally invisible.
- Source spacing 10× finer than the MVN exponential correlation length
  (`phi ≈ 1 km`), so the spatial adjustment is well-resolved.
- The QMAP and IwahashiPike rasters that drive the categorical model don't
  contain genuine detail finer than ~50–100 m, so going below 100 m would
  refine raster representation without adding model content.
- 100 m is the canonical NZ Vs30 product resolution and matches
  `FULL_NZ_GRID_CONFIG`, exercising the pipeline on its tested footprint.

Cost: 161 M cells. Peak memory likely 5–10 GB across the several float32
grids the pipeline holds simultaneously (vs30, stdv, slope, coast distance,
ids). Wall time not measured at 100 m; extrapolating from the 5000 m
benchmark (14 s) is dominated by sublinear factors so projection is loose
— roughly 1–3 hours expected.

### 4.2 NZTM bbox — `FULL_NZ_GRID_CONFIG`

The WGS84 grid's true NZTM footprint extends well beyond
`FULL_NZ_GRID_CONFIG` (×1.5 cell count if we widened the bbox to fully
cover it). We use `FULL_NZ_GRID_CONFIG` anyway:

- The 5000 m `jaehwi_v1p0` benchmark shows the actual NZ-land valid pixels
  sit comfortably inside `FULL_NZ_GRID_CONFIG` with a nodata border of
  20–55 km on every side. At 100 m that's 200+ source pixels of implicit
  buffer — far more than any resampling kernel needs.
- WGS84 cells whose centres transform to NZTM coordinates outside
  `FULL_NZ_GRID_CONFIG` (a strip near lon = 180° plus a southern strip)
  are all over open ocean east/south of NZ and would be nodata regardless.
  `rasterio.warp.reproject` will assign nodata to these target cells,
  which is the correct result.
- Saves ~35 % of cells, memory, and wall time vs the wider buffered bbox.

The tradeoff is an explicit no: lat/lon cells whose centres fall outside
the canonical NZ NZTM footprint will be nodata in the output. Empirically
this is fine because no NZ land falls in those regions (mainland NZ ends
at ~178.5°E; the Chatham Islands at ~183.5°E sit outside the WGS84 grid
entirely).

### 4.3 Resampling kernel — bilinear

Vs30 is a continuous quantity, so bilinear is the standard choice. The
~¼-target-cell smear at coastlines and category boundaries is acceptable.
Nearest-neighbour was considered (preserves discrete jumps) but produces
visibly blocky outputs and isn't worth it at this resolution mismatch.
Cubic was considered (smoother) but can overshoot near sharp edges and
needs a 4×4 source neighbourhood — the buffer is sufficient for it but the
quality gain over bilinear is marginal for Vs30.

For the `stdv` band we apply the same bilinear kernel — uncertainties are
also smooth in regions where the model is defined.

### 4.4 Why grid + reproject, not points pipeline directly

`pipeline.points_pipeline` could in principle take all 9 M lat/lon centres
directly and avoid any resampling. We don't do this because:

- The points-mode performance investigation
  (`points_perf_investigation_design.md`) is in flight and only validates
  up to ~100 k query points. 9 M is 90× larger than tested. Memory and
  wall-time behaviour at that scale is unknown.
- The grid pipeline is the canonical, well-tested code path at the full
  NZ × 100 m footprint.
- The bilinear smear introduced by reprojection is small (≤ ¼ target cell)
  and is the only artefact added relative to the points-direct route.

If the points-direct route becomes attractive later (e.g. after the perf
investigation finishes), it can be revisited as an alternative
implementation that reuses the same plan structure.

## 5. Output structure

The existing files have `latitude`, `longitude`, `elevation` datasets and
`ncols`/`nrows` root attrs, where `elevation` is actually Vs30 (see §1).
The regenerated files fix the misnomer:

- `latitude` (3001,) — copied verbatim from the source HDF5
- `longitude` (3001,) — copied verbatim from the source HDF5
- `vs30` (3001, 3001) — Vs30 from `jaehwi_v1p0`, float64 to match the
  source dtype, NaN over nodata
- Root attrs: `ncols=3001`, `nrows=3001`, plus new provenance attrs
  (model version, source NZTM grid spec, reprojection kernel, vs30
  commit hash, generation date)

The pipeline also produces a `combined_stdv` band. Whether to write it
into the regenerated file as a `stdv` dataset is an open decision (§6).

## 6. Open decisions

| Decision | Options | Default unless changed |
|---|---|---|
| Output filename(s) | (a) overwrite the existing files in place; (b) write new files alongside (e.g. `NZ_Vs30_jaehwi_v1p0.h5`) | (b) — least destructive; the originals are easy A/B references and the discrete-sentinel content is hard to reconstruct if lost. |
| What to do about the `_HD_With_Offshore` variant | The two source files have qualitatively different content (offshore default 500 vs 50; >50 % vs >75 % cells at sentinel). `jaehwi_v1p0` produces one Vs30 grid, not two, so we can't naturally regenerate both from a single model run. Options: (a) regenerate only `NZ_Vs30.h5` and leave the variant untouched; (b) write `jaehwi_v1p0` output to one filename only and let the user decide if a separate "with offshore" derivative is still needed. | (a) — produce one regenerated file matching `NZ_Vs30.h5`'s name pattern. The offshore variant's purpose needs to be understood before we can sensibly regenerate it. |
| Include `stdv` dataset | Pipeline produces both vs30 and stdv. Existing files only carry one band. | Include `stdv` — once we've broken schema by renaming `elevation`, adding `stdv` is a free win and downstream uncertainty quantification benefits. |
| `dbscan_nproc` for the run | The pipeline only parallelises DBSCAN clustering (`dbscan_nproc`); MVN is single-process multi-threaded BLAS per the recent grid-mode perf investigation. | `dbscan_nproc=6` per the workstation's "leave 2 cores free" preference. |
| Updates to consumer code that reads `elevation` from these files | Any caller that currently does `f["elevation"][...]` against `NZ_Vs30.h5` needs to read `f["vs30"][...]` from the regenerated file. The set is small enough to fix exhaustively. | Audit and update all such call sites as part of this work, in the same branch as the regeneration script. |

## 7. Risks

| Risk | Mitigation |
|---|---|
| 100 m run takes longer than projected. | Time a 1000 m run first if a tighter estimate is wanted before launching the full job. |
| Peak memory exceeds expectations and OOMs the workstation. | Pipeline doesn't currently support tile-based grid runs. If memory is tight, fall back to a coarser NZTM grid (e.g. 200 m, ~40 M cells, ¼ memory) and accept ½-target-cell bilinear smear. |
| Bilinear produces visible nodata bleed at the coast. | Output looks visually fine in existing 5000 m benchmarks where the nodata border is wide. If problematic, switch to GDAL warp with explicit nodata-aware masking. |
| The existing HDF5 files' `latitude`/`longitude` arrays don't exactly match the assumed 0.005° regular grid (e.g. floating-point drift). | Use the file's actual `latitude` and `longitude` arrays as-is when constructing the destination transform; don't reconstruct from `arange`. |

## 8. Reproducibility

- Source code commit at run time: record in HDF5 root attrs.
- Source NZTM GeoTIFF: keep alongside the HDF5 output for diagnostics.
- The whole regeneration is one script-driven sequence — to be implemented
  as a follow-up plan.

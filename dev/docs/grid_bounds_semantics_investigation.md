# Grid Bounds Semantics — Investigation

**Date:** 2026-05-02
**Branch:** `vs30_refactor`
**Status:** Findings (no code changes proposed yet)
**Investigator:** Claude (Opus 4.7) under direction of Andrew Ridden-Harper

## 1. Question

What do `grid_xmin`, `grid_xmax`, `grid_ymin`, `grid_ymax` mean to the
refactored `vs30 grid` CLI?

Two competing interpretations exist in widespread GIS code:

- **(A) Pixel-edge convention.** `xmin/xmax/ymin/ymax` are the *outer*
  bounds of the raster extent. The leftmost pixel's left edge sits at
  `xmin`, its centre at `xmin + dx/2`. Number of pixels is
  `nx = (xmax - xmin) / dx`. This is GDAL's `outputBounds` convention,
  rasterio's `transform.from_bounds`, and R's `raster::raster()`.
- **(B) Pixel-centre convention.** `xmin/xmax/ymin/ymax` are the
  *centres* of the edge pixels. Number of pixels is
  `nx = (xmax - xmin) / dx + 1`. The actual extent runs from
  `xmin - dx/2` to `xmax + dx/2`. This is the convention netCDF /
  cf-conventions / many scientific gridded-data packages use.

Comments and docstrings in the refactored repo currently claim **(B)**.
This document tests that claim by tracing the code, inspecting source
and output rasters, and comparing against four legacy codebases. The
finding is that the refactored code unambiguously uses **(A)**, that the
docstrings are wrong, and that the chosen `FULL_NZ_GRID_CONFIG` values
(`1060050, 2120050, 4730050, 6250050`) are systematically *50 m
misaligned* with the bundled IwahashiPike source raster — every pixel
of the full-NZ output sits exactly on an IwahashiPike pixel edge, so
nearest-neighbour resampling sees a tie at every pixel.

## 2. TL;DR

1. **The code unambiguously uses pixel-edge convention (A).** Proof:
   `vs30/raster.py:238-240` calls `rasterio.transform.from_bounds(xmin,
   ymin, xmax, ymax, nx, ny)`, which by definition treats those four
   numbers as the **outer** extent. `nx = round((xmax-xmin)/dx)` (no
   `+1`) confirms this.

2. **Three places in the refactored repo claim pixel-centre semantics
   and are wrong:**
   - `vs30/constants.py:258-261` — comment added 2026-05-01 in commit
     `43a033f`: "*The xmin/xmax/ymin/ymax values are pixel centres (not
     pixel edges) on the 100m NZTM grid.*"
   - `vs30/gapfill.py:216-219` — `create_local_grid_config` docstring:
     "*Snaps the point to the nearest pixel center… The local grid uses
     the same spacing and origin alignment as gapfill_grid_config to
     ensure pixel centers match the full grid.*"
   - The implicit assumption that `FULL_NZ_GRID_CONFIG`'s `..050`
     coordinates align with IwahashiPike pixel centres.

3. **`FULL_NZ_GRID_CONFIG` is 50 m misaligned with the bundled
   IwahashiPike raster in both x and y.** Empirically, generating a
   100×100 pixel terrain raster over Christchurch with the FULL_NZ
   alignment vs. an IwahashiPike-aligned grid changes
   **21.6 %** of pixel category IDs (terrain) and **2.8 %** (geology).

4. **The misalignment is inherited from the legacy pre-refactor
   codebase** (`/home/arr65/src/pre-refactor-Vs30-for-comparison`),
   which uses the same bounds and the same pixel-edge formula. Jaehwi's
   published output (`V1.0_26Mar.tif`) was also produced with these
   bounds. So the refactor preserves this quirk faithfully — it is not
   a regression.

5. **The official Foster (2019) GeoTIFF does NOT have this
   misalignment.** Its origin is `(1000000, 6338400)`, identical to
   IwahashiPike, with pixel centres ending in `…50`. So when comparing
   refactored full-NZ output against the published Foster (2019) map,
   every pixel is sampling a different IwahashiPike pixel than the
   reference did — at boundaries this produces small but real
   discrepancies, and `tests/test_benchmarks.py` already has docstrings
   acknowledging "*categorical/hybrid edge-case differences between the
   legacy R pipeline and the refactored Python one*" (lines 95-99).
   That phrasing now has a concrete mechanism: the 50 m grid offset.

6. **The benchmarks themselves use a *different* alignment from
   `FULL_NZ_GRID_CONFIG`.** `tests/test_benchmarks.py:38-45` defines
   `BENCHMARK_NZ_GRID = (1060100, 4730100, 2120100, 6250100)` — i.e.
   `..100` not `..050`, at 5 km resolution. So *within the refactored
   repo* there are already two different "canonical NZ" grids in
   active use.

The remainder of this document gives the evidence in detail.

## 3. Evidence: what the refactored code actually does

### 3.1 The decisive line of code

`vs30/raster.py:186-240`, `create_category_id_array`:

```python
nx = round((xmax - xmin) / dx)
ny = round((ymax - ymin) / dy)
dst_transform = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, nx, ny)
```

`rasterio.transform.from_bounds(west, south, east, north, width, height)`
is documented as: *"Make an Affine transform that maps pixel coordinates
to spatial coordinates of a raster. west, south, east, north are the
bounds of the raster."* These are the **outer** bounds — the resulting
transform places the upper-left **corner** of the upper-left pixel at
`(west, north)`. The pixel centre of pixel `(0,0)` lands at
`(west + dx/2, north - dy/2)`.

`nx = round((xmax - xmin) / dx)` (no `+1`) is consistent only with the
outer-bound convention. The pixel-centre convention would require
`nx = round((xmax - xmin) / dx) + 1` so that `nx` centres at spacing
`dx` fit between `xmin` and `xmax` inclusive.

### 3.2 Empirical verification

Running this against `FULL_NZ_GRID_CONFIG`:

```python
xmin, xmax = 1060050, 2120050
ymin, ymax = 4730050, 6250050
dx = dy = 100
nx = round((xmax - xmin) / dx)   # 10600
ny = round((ymax - ymin) / dy)   # 15200
t  = rasterio.transform.from_bounds(xmin, ymin, xmax, ymax, nx, ny)
```

Output:

| Quantity                              | Value                  |
|---------------------------------------|------------------------|
| `t.c` (UL corner x — outer edge)      | `1060050.0`            |
| `t.f` (UL corner y — outer edge)      | `6250050.0`            |
| Pixel `(0,0)` CENTRE                  | `(1060100.0, 6250000.0)` |
| Pixel `(15199, 10599)` CENTRE         | `(2120000.0, 4730100.0)` |
| OUTER pixel CORNERS (UL, LR)          | `(1060050, 6250050)`, `(2120050, 4730050)` |

The numbers `1060050` etc. appear as the **outer pixel edges**, not as
pixel centres. The pixel centres are at `..00`, not `..50`.

### 3.3 Cross-check: the gapfill snap arithmetic is consistent with (A)

`vs30/gapfill.py:238-254`, `create_local_grid_config`:

```python
snap_e = grid_xmin + round((easting - grid_xmin) / grid_dx) * grid_dx
snap_n = grid_ymin + round((northing - grid_ymin) / grid_dy) * grid_dy
return GridConfig(
    grid_xmin = snap_e - half_width,
    grid_xmax = snap_e + half_width,
    grid_ymin = snap_n - half_width,
    grid_ymax = snap_n + half_width,
    grid_dx   = grid_dx,
    grid_dy   = grid_dy,
)
```

`snap_e` lands on `grid_xmin + n*dx`. With
`FULL_NZ_GRID_CONFIG.grid_xmin = 1060050`, that's `..050, ..150, ..250,
…` — i.e. on the **outer pixel edges** of the FULL_NZ grid (which are
also at `..050, ..150, …`). It is *not* on FULL_NZ pixel centres
(which are at `..100, ..200, …`).

The local grid built around `snap_e` then uses
`(snap_e - half_width, snap_e + half_width)` as bounds. Under
convention (A) those are pixel edges, and the centre pixel of an
odd-pixel-wide local grid has its centre at `snap_e`. So `snap_e` is
also the centre pixel's centre **of the local grid** — but at
`..050/..150/…`, which is **not** a FULL_NZ pixel centre.

Net effect: the local grids the gapfill / consistency tests build
are *50 m offset* from the FULL_NZ grid in both x and y. The
`create_local_grid_config` docstring says they "ensure pixel centers
match the full grid"; they do not. They happen to align with
**IwahashiPike** pixel centres (which are also at `..50`), which is why
`tests/test_grid_points_consistency.py` agrees to `pytest.approx`
default tolerance — a fortunate, undocumented side-effect.

### 3.4 What `BENCHMARK_NZ_GRID` does instead

`tests/test_benchmarks.py:38-45`:

```python
BENCHMARK_NZ_GRID = config.GridConfig(
    grid_xmin=1060100,
    grid_xmax=2120100,
    grid_ymin=4730100,
    grid_ymax=6250100,
    grid_dx=5000,
    grid_dy=5000,
)
```

`..100` instead of `..050`. With `dx=dy=5000` this gives `nx=212`,
`ny=304`, pixel centres at `(1062600, 6247600), (1067600, 6247600), …`.
Within the refactored repo, this grid is already 50 m offset from the
`FULL_NZ_GRID_CONFIG` grid at the same (sub-)pixel level.

The three benchmark TIFFs in `tests/benchmarks/` (`jaehwi_v1p0.tif`,
`modified_foster_2019.tif`, `viktor_cpt_clustering.tif`) were generated
with this `..100` alignment. Their `gdalinfo` confirms it:

```
Origin = (1060100.000000000000000, 6250100.000000000000000)
Pixel Size = (5000, -5000)
```

## 4. Evidence: what `FULL_NZ_GRID_CONFIG` *should* align with

`vs30/resources/geospatial/IwahashiPike.tif`:

```
Origin     = (1000000.0, 6338400.0)         # outer UL edge
Pixel Size = (100, -100)
Size       = 11264 × 16384
UL pixel CENTRE = (1000050, 6338350)
LR pixel CENTRE = (2126350, 4700050)
```

Pixel centres are at `…50` in both axes (every centre `(c_x, c_y)`
satisfies `c_x mod 100 == 50` and `c_y mod 100 == 50`).

`FULL_NZ_GRID_CONFIG` produces output pixel centres at `…00` in both
axes. They are **systematically 50 m offset** from IwahashiPike pixel
centres at every pixel. Verified empirically:

```
FULL_NZ pixel (0,0) CENTRE = (1060100, 6250000)
Nearest IwahashiPike CENTRE = (1060050, 6250050)
Distance: 50 m in x AND 50 m in y
Fractional IwahashiPike pixel coord of FULL_NZ centre: col=601.0, row=884.0
  (For perfect alignment, the fractional col would be int+0.5, not int.0.)
```

A fractional pixel coordinate of `601.0` means the FULL_NZ centre falls
*exactly on the boundary* between IwahashiPike columns 600 and 601.
GDAL's nearest-neighbour resampling treats this as a tie. Same for
rows.

### 4.1 Empirical impact on terrain/geology classification

I rasterised a 100×100 pixel (10 km × 10 km) test region around
Christchurch with two grids that differ only by a 50 m shift:

| Grid                                      | Convention             | Pixel CENTRES                       |
|-------------------------------------------|------------------------|-------------------------------------|
| `(1565050, 5175050, 1575050, 5185050)`    | FULL_NZ-aligned        | `..00` (offset from IwahashiPike)   |
| `(1565000, 5175000, 1575000, 5185000)`    | IwahashiPike-aligned   | `..50` (matches IwahashiPike)       |

Result:

| Layer                | Differing pixels | % differing |
|----------------------|------------------|-------------|
| Terrain (16 classes) | 2 165 / 10 000   | **21.6 %**  |
| Geology (~50 classes from QMAP) | 284 / 10 000 | **2.8 %**   |

That is, ~22 % of terrain pixels in central Christchurch get classified
into the wrong IwahashiPike class because GDAL has to break a tie at
every pixel. The terrain `id` then drives the categorical Vs30 lookup
and the slope/coast hybrid mods, so this propagates downstream.

Geology has fewer differences because QMAP polygons are typically much
larger than 100 m, so a 50 m shift only affects pixels near polygon
boundaries.

## 5. Evidence: legacy codebases

### 5.1 `pre-refactor-Vs30-for-comparison` (modified_foster_2019, viktor_cpt_clustering source)

`vs30/params.py:38-43`:

```python
@dataclass
class GridParams:
    xmin: int = 1060050
    xmax: int = 2120050
    dx:   int = 100
    ymin: int = 4730050
    ymax: int = 6250050
    dy:   int = 100
```

`vs30/params.py:48-53`:

```python
def update(self):
    self.nx = round((self.xmax - self.xmin) / self.dx)
    self.ny = round((self.ymax - self.ymin) / self.dy)
```

`vs30/model_geology.py:143-154` (rasterise QMAP) and
`vs30/model.py:42-56` (`gdal.Warp`) both pass
`outputBounds=[xmin, ymin, xmax, ymax]` straight through. GDAL's
`outputBounds` is **outer-bound** by definition.

`run_legacy_wrapper_clustered_observation_data.py:14-17` confirms the
default values used in production:

```python
grid_xmin = 1060050
grid_xmax = 2120050
grid_ymin = 4730050
grid_ymax = 6250050
```

**Conclusion for pre-refactor:** pixel-edge convention (A), bounds at
`..050`, pixel centres at `..00`. Same misalignment with IwahashiPike
as the refactored code. The refactor preserves this faithfully.

### 5.2 `jaehwi_fork_vs30/Vs30_2026` (jaehwi_v1p0 source)

`vs30/params.py` defaults differ: `(1060100, 2120100, 4730000, 6250100)`,
with the suggestive comment `#x plus50 y minus 50` on line 37.

`vs30/params.py:48-53` uses the same `nx = round((xmax-xmin)/dx)`
formula — pixel-edge convention (A).

But the published reference output `V1.0_26Mar.tif` (in
`/home/arr65/data/vs30/grid_models/downloaded/jaehwi_v1p0/` and
`/home/arr65/data/vs30/jaehwi_versions_from_slack/`) has:

```
Origin     = (1060050.0, 6250050.0)
Pixel Size = (100, -100)
Size       = 10600 × 15200
```

i.e. it was generated with `--xmin 1060050 --xmax 2120050 --ymin
4730050 --ymax 6250050`, **not** with the `params.py` defaults. This
matches what the refactored `FULL_NZ_GRID_CONFIG` produces exactly.

The earlier sub-agent investigation of this codebase concluded "pixel
centres", citing the `+50/−50` offset of the params.py defaults from
the `_full_land_grid` reference (`1060050`) as a "smoking gun". That
reasoning is incorrect: the `nx = (xmax-xmin)/dx` formula (no `+1`) is
already conclusive proof of pixel-edge convention regardless of what
the default bound values are. The `+50/−50` offset of *defaults* from
the *reference* is just a config glitch — they produce different grids
and jaehwi did not use the defaults.

**Conclusion for jaehwi_fork:** pixel-edge convention (A); the
published model output uses the same `..050` bounds as everything else
in this story.

### 5.3 `Kevin_Foster_R_code_vs30_model/Vs30_NZ` (foster_2019 R source)

`R/MAP_NZGD00_regions.R:17`: `NZ = extent(1020000, 2260000, 4730000,
6220000)`. R's `extent()` is the GDAL convention — outer bounds.

`R/tileSlopes.R:41-95` builds an 11 × 16 tile grid covering
`[1000000, 2126400] × [4700000, 6338400]` at 100 m resolution. The
arithmetic only works under pixel-edge convention: 11 tiles ×
1024 pixels × 100 m = 1 126 400 m exactly = `xmax - xmin`. Under
pixel-centre convention you would be one pixel short.

**Conclusion for Foster R:** pixel-edge convention (A). Different
default extent from anything Python — wider on the east-west axis,
shifted north-south. Round-number bounds → pixel centres at `..50`,
matching IwahashiPike.

### 5.4 Foster (2019) official GeoTIFF

`/home/arr65/data/vs30/grid_models/downloaded/foster_2019/foster_2019.tif`:

```
Origin     = (1000000.0, 6338400.0)         # IDENTICAL to IwahashiPike
Pixel Size = (100, -100)
Size       = 11264 × 16384                  # IDENTICAL to IwahashiPike
UL pixel CENTRE = (1000050, 6338350)
LR pixel CENTRE = (2126350, 4700050)
```

**Conclusion for Foster official:** same alignment as IwahashiPike,
pixel centres at `..50`. So the published Foster (2019) map is
*not* on the same grid as the refactored `FULL_NZ_GRID_CONFIG` output —
they are 50 m offset in x and y.

## 6. Cross-codebase summary table

| Codebase / artefact                        | Convention | NZ-wide xmin (UL edge) | NZ-wide ymax (UL edge) | Pixel centre x parity | Aligned with IwahashiPike? |
|--------------------------------------------|-----------|------------------------|------------------------|------------------------|----------------------------|
| **Refactored `FULL_NZ_GRID_CONFIG`**       | (A) edge  | 1 060 050              | 6 250 050              | …00                    | **No (50 m offset)**       |
| Refactored `BENCHMARK_NZ_GRID` (5 km)      | (A) edge  | 1 060 100              | 6 250 100              | …600 (5km)             | n/a (different resolution) |
| Refactored bundled `IwahashiPike.tif`      | (A) edge  | 1 000 000              | 6 338 400              | …50                    | yes (it's the source)      |
| Refactored bundled `slope.tif`             | (A) edge  | 1 060 040 (270 m)      | 6 250 100              | …175 (270 m)           | n/a                        |
| pre-refactor `params.py` defaults          | (A) edge  | 1 060 050              | 6 250 050              | …00                    | No (50 m offset)           |
| pre-refactor wrapper scripts               | (A) edge  | 1 060 050              | 6 250 050              | …00                    | No (50 m offset)           |
| jaehwi_fork `params.py` defaults           | (A) edge  | 1 060 100              | 6 250 100              | …50                    | yes (matches IwahashiPike) |
| jaehwi `V1.0_26Mar.tif` (actual output)    | (A) edge  | 1 060 050              | 6 250 050              | …00                    | **No (50 m offset)**       |
| Foster R original NZ extent                | (A) edge  | 1 020 000              | 6 220 000              | …50                    | yes                        |
| Foster (2019) official GeoTIFF             | (A) edge  | 1 000 000              | 6 338 400              | …50                    | yes (same as IwahashiPike) |

## 7. Inconsistencies found

1. **Comment vs. code, `vs30/constants.py:258-261`.** Comment says
   pixel-centres; code (everywhere) uses pixel-edges. The comment was
   added 2026-05-01 in commit `43a033f` and is already wrong. **Fix:
   rewrite or delete the comment** — the bound values are pixel
   *edges*, and the `..050` ending was inherited from the legacy
   defaults, not from a deliberate centre-vs-edge design choice.

2. **Docstring vs. code, `vs30/gapfill.py:207-219`.** Docstring claims
   `snap_e` is a "pixel center in the reference grid config" and the
   local grid "ensure[s] pixel centers match the full grid." It does
   not. `snap_e` is a FULL_NZ pixel *edge*, and the local grid centres
   are 50 m offset from the FULL_NZ centres. **Fix: rewrite the
   docstring** to either describe the actual behaviour or — better —
   change the snap to land on FULL_NZ pixel centres so the docstring
   becomes true (this would also fix the alignment-with-IwahashiPike
   issue described next, modulo a one-pixel boundary handling change).

3. **`FULL_NZ_GRID_CONFIG` is 50 m misaligned with the bundled
   `IwahashiPike.tif`.** Every output pixel sits exactly on an
   IwahashiPike pixel boundary, forcing GDAL to break a tie.
   Empirically, this changes ~22 % of terrain category IDs in
   Christchurch. **Possible fixes (not all equivalent):**
   - Change `FULL_NZ_GRID_CONFIG` to `(1060000, 4730000, 2120100,
     6250100)`, dx=dy=100. Now nx=10601, ny=15201; pixel centres at
     `(1060050, 6250050), …, (2120050, 4730050)`, perfectly aligned
     with IwahashiPike. **Caveat:** this changes the output raster
     shape by one pixel in each dimension, which will almost certainly
     break the existing benchmarks and may mismatch downstream
     consumers (e.g. NZ_Vs30 HDF5).
   - Keep the 10600 × 15200 shape but shift to `(1060000, 4730000,
     2120000, 6250000)` (round numbers). Pixel centres at `(1060050,
     6249950), …, (2119950, 4730050)`, also IwahashiPike-aligned.
     Shifts the entire output grid by 50 m east of where it currently
     is, but preserves shape.
   - Document the misalignment as a deliberate inherited quirk and
     leave it alone — accept that the refactored code is bit-faithful
     to the legacy pre-refactor and to jaehwi's published output, and
     that any deviation from Foster (2019) at boundaries comes from
     this 50 m offset plus tie-breaking.

4. **Two "canonical NZ" grids in the refactored repo.**
   `FULL_NZ_GRID_CONFIG` (`..050`) and `BENCHMARK_NZ_GRID` (`..100`)
   are different by 50 m. The CLI help text references
   `FULL_NZ_GRID_CONFIG`; the regression tests use `BENCHMARK_NZ_GRID`.
   At the 5 km resolution the benchmarks run at this is cosmetic, but
   it means the test bench is verifying alignment-X output against an
   alignment-X reference and tells you nothing about whether
   alignment-Y output is correct. **Fix: pick one alignment and use it
   everywhere**, or document why the benchmark has its own.

5. **`BENCHMARK_NZ_GRID` 5 km centres do not land on IwahashiPike pixel
   centres either.** With xmin=1060100 dx=5000, the UL pixel centre is
   at (1062600, 6247600), 50 m offset from the nearest IwahashiPike
   centre. So nearest-neighbour IwahashiPike sampling has the same
   tie-break ambiguity at 5 km that it does at 100 m. The benchmark
   reference TIFF was generated with the same code path and the same
   tie-breaks, so the benchmark passes — but it is testing that the
   code reproduces *itself*, not that it produces correctly-aligned
   output.

## 8. Side note: what the `..050` choice actually expresses

Both legacy codebases and the refactored code use NZ-wide bounds ending
in `..050`. Under pixel-edge convention this puts pixel centres at
`..00`. Under pixel-centre convention this would put pixel centres at
`..050` (matching IwahashiPike).

It is plausible that whoever first chose `1060050` *intended* convention
(B) — the value lines up neatly with IwahashiPike under (B). But every
implementation since has used (A), so the value has acquired a (A)
meaning that doesn't match the original intent.

The recently-added "pixel centres" comment in `constants.py` reads as
an attempt to recover the *intended* meaning. Unfortunately the *code*
follows the (A) interpretation, and code wins over comments.

## 9. Recommended next steps

In rough order of effort vs payoff:

1. **Fix the docstrings/comments now.** They are actively misleading.
   No code change. Cheap.
2. **Add a unit test** that asserts `FULL_NZ_GRID_CONFIG` produces a
   transform whose pixel centres match the bundled `IwahashiPike.tif`
   pixel centres. It will fail. Make the test the canonical
   regression for whatever alignment we land on.
3. **Decide** whether to keep the legacy 50 m misalignment as-is
   (rationale: bit-faithful to jaehwi's V1 output and to pre-refactor
   benchmarks) or fix it (rationale: aligned with Foster (2019)
   official, no tie-break ambiguity, no spurious classification flips).
   This is a science-level decision, not a code one.
4. **If fixing**: shift `FULL_NZ_GRID_CONFIG` to round-number bounds
   `(1060000, 4730000, 2120000, 6250000)` (preserves 10600 × 15200
   shape but shifts the output footprint 50 m east). Regenerate the
   three benchmark TIFFs in `tests/benchmarks/` from a known-good run.
   Update `BENCHMARK_NZ_GRID` to match (or document the deliberate
   difference). Re-derive any cached HDF5/NZ_Vs30 outputs that bake in
   the old grid.
5. **If keeping**: write down the misalignment in a top-level
   `KNOWN_QUIRKS.md` or add it to `CLAUDE.md`, so the next person who
   compares against Foster (2019) does not re-discover this from
   scratch.

## Appendix A: Reproducer scripts

The empirical results in this document can be reproduced verbatim via
the snippets in §3.2, §3.3, and §4.1. They depend only on the bundled
`IwahashiPike.tif` and `qmap` shapefile, both already in
`vs30/resources/geospatial/`.

## Appendix B: References to evidence

| Section | File / artefact                                                                                  |
|---------|--------------------------------------------------------------------------------------------------|
| §3.1    | `vs30/raster.py:186-240`                                                                          |
| §3.2    | empirical, against `vs30/constants.py:262-269` (FULL_NZ_GRID_CONFIG)                              |
| §3.3    | `vs30/gapfill.py:207-254`                                                                         |
| §3.4    | `tests/test_benchmarks.py:38-45`, `tests/benchmarks/{jaehwi_v1p0,modified_foster_2019,viktor_cpt_clustering}.tif` |
| §4      | `vs30/resources/geospatial/IwahashiPike.tif`                                                     |
| §4.1    | empirical                                                                                         |
| §5.1    | `/home/arr65/src/pre-refactor-Vs30-for-comparison/vs30/params.py`, `…/run_legacy_wrapper_*.py`    |
| §5.2    | `/home/arr65/src/jaehwi_fork_vs30/Vs30_2026/vs30/params.py`, `/home/arr65/data/vs30/jaehwi_versions_from_slack/V1.0_26Mar.tif` |
| §5.3    | `/home/arr65/src/Kevin_Foster_R_code_vs30_model/Vs30_NZ/R/MAP_NZGD00_regions.R`, `…/R/tileSlopes.R` |
| §5.4    | `/home/arr65/data/vs30/grid_models/downloaded/foster_2019/foster_2019.tif`                       |

# Issue hunt, October 2026

A review of `vs30_refactor` for wrong results, crashes, input handling, speed and documentation, run on 7–9 October 2026 against commit `ae73207`. It found 32 issues, all closed by `5569310`; the October 2026 wrap-up then merged `vs30_refactor` into `master`. Each fix's commit message explains its problem and change. This note keeps what the commits don't: an index from issue to commit, what is still true, why the design decisions went the way they did, and what was checked and behaved correctly.

## Fixes

| Issue | Fix | Commit |
|---|---|---|
| — | Points gap-fill shows one progress bar over the points | `bd13ff1` |
| — | Points gap-fill takes the valid pixel nearest the query pixel's centre, with its own 50 km cap. This fixed an effective hang at ~11% of coastal gap sites | `ae73207` |
| #1 | Grids with no valid pixels skip the MVN. It used to raise `ZeroDivisionError`, losing the whole `points` run | `b8a5ac3` |
| #2 | `points` samples slope on cell edges the way GDAL does, so `points` and `grid` agree | `262a299` |
| #3 | Observations take slope and coast distance at their exact location in both modes, so a pixel no longer depends on the grid's extent. Grid benchmarks regenerated | `f539ab8` |
| #4 | Relative nugget (1e-8) on the MVN observation block. Failed updates keep the prior, with a warning | `87e1fa9` |
| #5 | `viktor_cpt_clustering` documented as it behaves, with no numerical change | `1568598` |
| #6 | No −32767 placeholders in `points` intermediate columns | `d70a337` |
| #7, #21 | `vs30 grid MODEL OUTPUT_DIR`, with full-NZ default bounds, bounds validation and a warning when off the national lattice | `452af58` |
| #8, #12, #13 | Config values validated at load, with one-line errors | `5db91fa` |
| #8 | Tracebacks without local variables; output location checked before computing | `8333f75`, `a165585` |
| #9–#11, #13 | `points` input: bad rows get blank results and a warning, spaced headers are read, and colliding columns are kept as `<name>_input` | `02259e6` |
| #13 | `points` validates observations like `grid` | `a86e9e2` |
| #14 | Coast distance computed over the grid plus 20 km, not all of NZ | `006f19e` |
| #15 | MVN observation covariance built once per run (up to 5,000 observations) | `3887b84` |
| #16 | Geology rasterizes only the polygons overlapping the grid; `--nproc` caps DBSCAN and BLAS threads | `e6fc856`, `4b9e60a` |
| #17, #19 | Usage.md states units and default outputs; `--help` text fixed | `c859919`, `3ffae3f` |
| #18 | Categorical columns named with their real units | `3766af1` |
| #20 | Consistency fixes: prior stdv kept when there's no update; non-positive model values dropped in grid mode; `coast_distance.tif` written only when used; slope nodata; the GID 4 coastal-over-slope rule; `--mvn-chunk-memory-gb`; coast-distance and gap-fill docs | `fbdd24a`, `1a7015a`, `f5b18ad`, `15dab16`, `b144eb1`, `568e2ed`, `234eb3a` |
| #22 | CLI logging, with `--verbose` | `9ba5f43` |
| #25 | Repo formatted with ruff, and CI checks formatting | `f59e7c8`, `3d9e5e6`, `6eab017` |
| #27 | `MODEL_ARG_HELP` loses its leading underscore | `fddc7f0` |
| #28 | Ruff's lint rules pinned with `select` after ruff 0.16 expanded its defaults | `1050d39` |
| #29–#31 | Upstream releases: depend on `qcore-utils[cli]`; drop unused `ty: ignore` comments; avoid affine 3's deprecated `*` | `7684bad`, `9a525c0`, `b1aefa9` |
| #32 | CLI tests turn off Rich styling under GitHub Actions | `5569310` |
| #23, #24, #26 | No change needed. #23 and #24 weren't real problems (see below); #26 removed stale scratch notes | — |

## Still true

- **`viktor_cpt_clustering` collapses to soft-soil values.**
  - All 15 geology categories end at 137–181 m/s (crystalline 750 → 170, volcanic 635 → 148), and the 16 terrain categories at 148–178 m/s (alpine summits 519 → 167). The Aoraki/Mt Cook summit gets 171 m/s.
  - This is the legacy design: CPT-inferred Vs30 replace the prior of every category they fall in, and a CPT can only sample soft ground.
  - DBSCAN's effective counts (crystalline 29, alluvium 63) swamp `N_PRIOR = 3`, so no parameter change fixes it.
  - Documented rather than changed; see the decisions below.
- **viktor's MVN covariance is still built per pixel.** Its 35,706 observations exceed the 5,000 limit of `3887b84`, so its MVN stays slow.
- **Grid and points define coast distance differently, by design.**
  - Grid measures from the pixel centre to the nearest sea-pixel centre; points measure the exact distance to the coastline.
  - Grid comes out about 45 m larger on average. That gives up to 0.9% (GID 4) and 1.25% (GID 10) Vs30 difference 8–20 km from the coast.
- **Gap-fill has three documented limits:**
  - The 1:1.5M coastline leaves some reclaimed port land unfilled (e.g. 1750250, 5430750).
  - Harbour pixels with geology id 255 take land values, because only id 0 is excluded.
  - Grid-mode donors are limited to the requested extent. A 3 km and a 6 km `jaehwi_v1p0` grid differed by up to 49% at one gap-filled pixel.
- **`apply_alluvium_slope_mod` has no effect when the coastal adjustment is on**, as in `modified_foster_2019` and `viktor_cpt_clustering`. Their YAMLs say so.
- **The bundled observation CSVs still contain duplicates.**
  - They're mostly the same datum reported twice: Wotherspoon measurements plus Kaiser Q1 at the same Christchurch stations, exact duplicate rows in `jaehwi_v1p0`, and 1,348 duplicate CPT rows in `viktor_cpt_clustering`.
  - The #4 nugget makes them harmless in the MVN, but they double-count in the Bayesian categorical update.
  - Cleaning them is a data-curation decision for the observation generators, and it would move the benchmarks.
- **Two warnings are not problems (#23, #24).**
  - A single-site `vs30 points` run prints a NumPy DeprecationWarning from inside pyproj, which already falls back correctly on the future `TypeError`.
  - pyogrio imports `shapely.geos` inside `try/except`.
  - Both warnings show only in pytest output.

## Why the decisions went this way

### #3: observation covariates at the exact location

The residual `ln(obs / model(x_obs))` belongs to the observation and the model, not to the output grid. The MVN treats it as the model's error at the observation point, so evaluating the model at a pixel centre made results depend on the grid's spacing and alignment.

Legacy sampled the national 100 m rasters (nearest neighbour) in both modes and had no sub-grids, so it was self-consistent. The refactor's points mode switched to exact sampling while grid mode kept pixel values, which split the two.

"Pixel centre everywhere" can't be made spacing-independent: a 5 km grid would take covariates up to 3.5 km from the observation. It also needs two code paths. So every observation uses its exact location.

The cost was that the grid benchmarks moved near observations, all within `MAX_DIST_M`:
- `modified_foster_2019`: 152 pixels, by up to 10% in Vs30 and 29% in stdv
- `jaehwi_v1p0`: 90 pixels, by up to 1.4% and 0.5%
- `viktor_cpt_clustering`: 673 pixels, by up to 13% and 51%

The benchmarks were regenerated from the refactored pipeline. On the national 100 m lattice, 112 of 458 observation model values changed, with a median change of 0.3% and a maximum of 9.6%.

### #4: a nugget rather than merging duplicates

The fix had three requirements:
- remove the failure for any user data
- leave the bundled `noisy: true` outputs unchanged
- stay faithful to `noisy: false`, which Foster's paper defines as zero-nugget kriging

Two conflicting values at one point have no exact solution. With a 1e-8 relative nugget they act as one observation with the averaged residual, while single observations are still honoured to eight digits. `noisy: true` outputs moved by at most 5e-7.

Merging duplicates would need a radius, custom data can reintroduce them, and it shifts the `noisy: true` benchmarks. So merging became the data-curation question above.

Neither earlier implementation handled this case: Foster's R code screened duplicates within 2 m from the data, and legacy used an unguarded matrix inverse.

### #5: document `viktor_cpt_clustering`, don't change it

The refactor's remit was reproduction, and the collapse is a property of the model's design rather than a pipeline parameter. A Bayesian blend wouldn't help, since `N_PRIOR = 3` against 29 effective clusters still gives the data mean.

The sensible redesigns are modelling decisions that change the benchmark:
- restrict the CPT update to soft-sediment categories, or
- start from the Foster posteriors plus measured data, and add CPTs as a second clustered update

So the model was documented as it behaves instead.

### Grid CLI shape

`vs30 grid MODEL OUTPUT_DIR`, with bounds and spacing defaulting to the full-NZ grid, follows the rule from `e8cca4f`: few required values are positional, many are options. Three reasons:
- The full-NZ map is the primary product.
- The package already treats the full-NZ lattice as canonical.
- Six hand-typed numbers were the main source of misaligned grids (#7).

The defaults therefore come with bounds validation, and a warning when custom bounds are off the national lattice.

## What was checked and behaved correctly

These checks were made at `ae73207`, before the fixes.

- **Edge-case sites:** all four models at city, coastal, lake, offshore, Sydney, Stewart Island and Chatham Islands sites gave NaN where expected, with no −32767 or negative values in the final Vs30 and stdv.
- **Speed:** 2,000 random on-land sites with `foster_2019_approx` took 3.5 s; 50,000 Christchurch sites took 98 s using 1.1 GB.
- **Points vs grid:** `points` matched `grid` within 0.11% for `foster_2019_approx` away from the slope cell-edge columns (#2), and gap-filled `jaehwi_v1p0` values matched `grid` exactly.
- **Input handling:** custom `--lon-column`/`--lat-column`, a byte-order mark, a missing trailing newline, duplicate rows, a single site, and a header-only file.
- **Clean errors** for:
  - an unknown model name
  - a missing input file
  - an output path that is a directory
  - missing grid options
  - a missing required YAML field
  - a misspelt correlation model
- **Configs:** absolute CSV paths, `standard_deviation_weighting`, `mvn: false`, `combine_ratio: 0`, and `fill_gaps: true` on `foster_2019_approx`.
- **Internals:**
  - the Bayesian update and the cluster-weighted mean and stdv match the legacy formulas
  - category IDs are consistently 1-indexed
  - QMAP has no overlapping polygons
  - terrain IDs agree between grid and points, including on 50 m-offset grids
  - KDTree reshaping, sentinel filtering and chunking are correct
  - combination weights, and the NODATA→NaN conversion before combining, are correct

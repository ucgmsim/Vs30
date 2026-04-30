# Provenance of `HYBRID_GEOLOGY_PARAMS.sigma_reduction`

## Background

`vs30/constants.py` defines a per-geology-category sigma reduction
factor as part of `HYBRID_GEOLOGY_PARAMS`:

```python
HYBRID_GEOLOGY_PARAMS: list[HybridGeologyParams] = [
    HybridGeologyParams(gid=2, slope_limits=[-1.85, -1.22], vs30_values=[242, 418], sigma_reduction=0.4888),
    HybridGeologyParams(gid=3, slope_limits=[-2.70, -1.35], vs30_values=[171, 228], sigma_reduction=0.7103),
    HybridGeologyParams(gid=4, slope_limits=[-3.44, -0.88], vs30_values=[252, 275], sigma_reduction=0.9988),
    HybridGeologyParams(gid=6, slope_limits=[-3.56, -0.93], vs30_values=[183, 239], sigma_reduction=0.9348),
]
```

These four numbers are a **multiplicative reduction applied to the
categorical posterior standard deviation** for the four
slope-modifiable geology categories (peat-derived/fill,
fluvial-estuarine, alluvium, beach-bar-dune respectively). They were
flagged in `CODE_REVIEW.md` §5.1 as having a non-obvious behaviour:
they are applied even when the per-pixel slope-based Vs30 update is
*not* applied (specifically, for GID 4 when
`apply_alluvium_slope_mod=False`). This document records what those
numbers represent, where they came from, and why the asymmetry exists.

## Origin

The values originate in Kevin Foster's R implementation of the model:

```
/home/arr65/src/Kevin_Foster_R_code_vs30_model/Vs30_NZ/R/MODEL_AhdiAK_noQ3_hyb09c.R
```

at line 77:

```r
sigmaReducFac = c(0.4888, 0.7103, 0.9988, 0.9348)
# Added this on 20171223. Based on slope-Vs30 plots.
# (See new SIGMA notes added to plot subtitles in slopePlotDetail.R output.)
```

So:

- They were added on **2017-12-23**.
- They were derived **empirically from slope-Vs30 scatter plots**
  (`slopePlotDetail.R`), not from a formal optimisation. The author
  inspected the plots for each slope-modifiable geology category and
  picked a per-category reduction reflecting how much the
  slope-Vs30 relationship tightens the categorical sigma in
  expectation.
- A reduction of ~1 (e.g. GID 4's 0.9988) means the slope-Vs30 plot
  for that category was nearly as diffuse as the categorical prior —
  i.e. slope barely helps. A reduction of ~0.5 (GID 2's 0.4888) means
  slope substantially constrains Vs30 for that category.

## How the R code uses them

The R code applies the reduction **once at lookup-table construction
time**, not per pixel
(`MODEL_AhdiAK_noQ3_hyb09c.R:80-93`):

```r
# here, modify lookup function to incorporate sigma reduction factors....
lookupDF.orig <- AhdiAK_noQ3_hyb09c_lookup()
lookupDF.mod  <- lookupDF.orig
for (i in seq(nrow(AhdiAK_noQ3_hyb09c_hybDF))) {
  lookupDF.mod$stDv_AhdiAK_noQ3[lookupDF.mod$groupID == ...slopeUnits[i]] <-
    lookupDF.mod$stDv_AhdiAK_noQ3[...] * sigmaReducFac[i]
}
# finally, new version of lookup function.
AhdiAK_noQ3_hyb09c_lookup <- function(){ return(lookupDF.mod) }
```

This is the key semantic point: in the R model the reduction is
**baked into the categorical sigma at lookup time**. Every consumer of
the lookup inherits the tightened sigma. Whether the pixel later goes
through slope-based Vs30 interpolation, the coastal-distance branch,
or no further modification at all, the categorical sigma it gets is
already the reduced one.

## How Jaehwi's Python port preserves this

Jaehwi's port
(`/home/arr65/src/jaehwi_fork_vs30/Vs30_2026/vs30/model_geology.py`)
applies the reduction at use time rather than lookup time, but
preserves the same unconditional semantics
(`model_geology.py:243-246`):

```python
# sigma reduction factors
if geol.hybrid:
    model = np.copy(model)
    # still updating g06 even if using coast function instead of hybrid
    model[HYBRID_SRF[0] - 1, 1] *= HYBRID_SRF[1]
```

The inline comment ("still updating g06 even if using coast function
instead of hybrid" — `g06` = `06_alluvium` = GID 4) explicitly
documents the asymmetry: even when the coastal-distance branch
replaces the slope-based hybrid for alluvium, the sigma reduction
still runs.

## How the refactored Python preserves this

`vs30/raster.py:apply_hybrid_geology_modifications` reproduces the
same per-use unconditional pattern, but folds the sigma reduction
into the same per-spec loop as the slope-based Vs30 interpolation
(`raster.py:689-702`):

```python
for spec in constants.HYBRID_GEOLOGY_PARAMS:
    mask = id_array == spec.gid
    stdv_array[mask] *= spec.sigma_reduction          # always applies

    if spec.gid == 4 and not apply_alluvium_slope_mod:
        continue                                      # skip Vs30 update only

    if np.any(mask):
        interpolated_val = np.interp(
            safe_log_slope[mask], spec.slope_limits, spec.vs30_values_log10
        )
        vs30_array[mask] = 10**interpolated_val
```

The asymmetry — sigma reduction for GID 4 always runs, but the
Vs30 slope interpolation for GID 4 is gated by
`apply_alluvium_slope_mod` — directly mirrors the R behaviour: the
sigma is tied to the geology category, not to whether the slope-based
Vs30 path is exercised for any given pixel.

## Empirical validation

To rule out the alternative interpretation (that the sigma reduction
should only apply when the slope mod is actually used, i.e. that the
asymmetry is a latent bug), the symmetric-guard variant was tested:

```python
for spec in constants.HYBRID_GEOLOGY_PARAMS:
    mask = id_array == spec.gid

    # Symmetric guard: skip BOTH stdv reduction AND vs30 update.
    if spec.gid == 4 and not apply_alluvium_slope_mod:
        continue

    stdv_array[mask] *= spec.sigma_reduction
    if np.any(mask):
        ...
```

The `jaehwi_v1p0` benchmark is the only fixed model that exercises
this path (it sets `apply_alluvium_slope_mod: false`). Of the 10 676
valid pixels in the 5 km benchmark domain, 1329 (12.4 %) are GID 4 —
plenty to detect a behavioural change.

Per-pixel residuals against the legacy `tests/benchmarks/jaehwi_v1p0.tif`
reference:

| Code path | Stdv max residual | Stdv mean residual |
|---|---|---|
| **Asymmetric** (current — always reduce stdv) | **9.51e-5** | **3.17e-5** |
| Symmetric (skip stdv when slope-off) | 7.22e-4 | 1.06e-4 |

The asymmetric form matches the legacy reference ~7.6× more tightly
on max stdv residual and ~3.3× more tightly on mean. The Vs30 column
is identical to within float-noise either way (max ~6e-5 vs ~10e-5),
because the asymmetry only affects sigma — the slope-mod-off branch
already keeps Vs30 unchanged in both variants.

## Conclusion

`HYBRID_GEOLOGY_PARAMS.sigma_reduction` is **not a fudge factor**. It
is an empirically calibrated per-category constant that represents
the average sigma tightening obtained by the slope-Vs30 relationship,
chosen by visual inspection of slope-Vs30 plots in the original R
model (Kevin Foster, 2017-12-23). It is applied unconditionally to
the categorical sigma for the four slope-modifiable geology
categories, regardless of whether the slope-based Vs30 path runs for
any individual pixel — matching the R semantics where the reduction
is baked into the lookup table at startup.

GID 4's reduction of 0.9988 (a ~0.12 % tightening, almost a no-op)
reflects the alluvium slope-Vs30 relationship being only marginally
informative beyond the categorical prior. The other three reductions
(0.4888 for GID 2, 0.7103 for GID 3, 0.9348 for GID 6) reflect more
informative slope relationships in those categories.

## References

- `vs30/raster.py:689-702` — current implementation.
- `vs30/constants.py` — `HYBRID_GEOLOGY_PARAMS` definition.
- `Kevin_Foster_R_code_vs30_model/Vs30_NZ/R/MODEL_AhdiAK_noQ3_hyb09c.R:54, 60-78, 80-93` — canonical R source.
- `Kevin_Foster_R_code_vs30_model/Vs30_NZ/R/slopePlotDetail.R` — script that generated the slope-Vs30 plots used to choose the values.
- `jaehwi_fork_vs30/Vs30_2026/vs30/model_geology.py:32, 243-246` — Jaehwi's Python port preserving the asymmetric semantics.
- `CODE_REVIEW.md` §5.1 — review item that prompted this investigation.

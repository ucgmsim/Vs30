# The modified_foster_2019 Model

`modified_foster_2019` is a variant of the published Foster et al. (2019)
Vs30 model. It uses the same categorical priors and posteriors as the
original, but adds coastal-distance hybrid modifications, swaps the
spatial correlation function, retains some otherwise-excluded
observations, and skips the Bayesian update step in favour of
pre-computed posteriors loaded from CSV.

## Summary

| Aspect | Foster (2019) | modified_foster_2019 |
|--------|---------------|----------------------|
| Coastal-distance modifications | None | Alluvium and Floodplain adjusted by distance to coast |
| Alluvium (G06) hybrid | Slope-based Vs30 interpolation | Coastal-distance interpolation (slope mod skipped) |
| Floodplain (G13) hybrid | None | Coastal-distance interpolation |
| Spatial correlation | Matérn (κ=0.9, range=20 km, nugget=0.05) | Exponential (φ_geology=1407 m, φ_terrain=993 m, no nugget) |
| Kaiser Q3 observations | All excluded | Q3 stations with 3-character names retained |
| Bayesian update | Performed from priors + observations | Skipped; pre-computed posteriors loaded from CSV |

## 1. Coastal-distance modifications (new)

The original model has no coastal-distance modifications — its only
geomorphological adjustment is slope-based.

`modified_foster_2019` adds a linear-interpolation coastal-distance Vs30
adjustment for two geology categories:

**Alluvium (G06)** — slope modification skipped; instead:

- < 8 km from coast → 240 m/s
- \> 20 km from coast → 500 m/s
- 8–20 km → linear interpolation

The original slope adjustment produced 252–275 m/s; the coastal version
produces a much wider 240–500 m/s range. The physical rationale is that
near-coastal alluvial deposits tend to be younger Holocene sediments
with lower Vs30, while inland equivalents include older, more
consolidated Pleistocene deposits.

**Floodplain (G13)** — no original modification; modified version uses:

- < 8 km from coast → 197 m/s (the categorical posterior)
- \> 20 km from coast → 500 m/s
- 8–20 km → linear interpolation

## 2. Spatial correlation function

The original uses a Matérn variogram (κ=0.9, range=20 km, nugget=0.05);
`modified_foster_2019` uses a simple exponential `ρ(d) = exp(-d/φ)` with
φ_geology=1407 m and φ_terrain=993 m, no nugget.

The two correspond to similar **effective ranges** (≈4.2 km geology,
≈3.0 km terrain), so the models agree at long range. They differ at
short distances: the Matérn has a rounder peak near zero and an
explicit nugget, while the exponential drops more steeply.

The Matérn nugget means the original's predictions don't exactly match
observations even at the observation location — the nugget absorbs some
of the observation's influence as measurement noise. The exponential
form has no nugget, but uses Worden et al. (2018) correlation
adjustment factors via a `noisy` flag to achieve a similar effect.

## 3. Kaiser Q3 observation filtering

The original excludes all Q3 stations: *"We do not use Q3 data for V_S30
modeling here"* (paper p. 1869).

`modified_foster_2019` retains Q3 stations with 3-character names —
roughly 60 additional stations. The reasoning is that 3-character
station codes identify GeoNet broadband seismometers (e.g. "WEL",
"BHW"), whose Q3 rating likely reflects the indirect Vs30 estimation
method rather than genuinely poor data quality.

This affects only the MVN spatial-conditioning step — not the
categorical posteriors, which are loaded directly from CSV (see §4).

## 4. Bayesian update strategy

The original performs Bayesian updating from scratch — sequentially
updating each category prior using observations within that category.
The published posteriors (Tables 1 and 2 of the paper) are the output.

`modified_foster_2019` skips the Bayesian step entirely and loads the
original R code's pre-computed posteriors directly from CSV. The
categorical model values (Vs30 mean and σ per category) are therefore
**identical** between the two models — the difference is purely
operational.

## What's the same

Everything else: prior weight κ₀=ν₀=3, minimum σ=0.5 (log-space),
covariance reduction factor a=1.5, slope-modified categories
(G04/G05/G06/G09) and their slope interpolation parameters, sigma
reduction factors (0.4888, 0.7103, 0.9988, 0.9348), equal-weight
geometric-mean model combination, and the measurement uncertainties
for McGann/Wotherspoon and Kaiser Q1/Q2 stations.

## References

- Foster, K. M., Bradley, B. A., McGann, C. R., and Wotherspoon, L. M.
  (2019). A V_{S30} Map for New Zealand Based on Geologic and Terrain
  Proxy Variables and Field Measurements. *Earthquake Spectra*, 35(4),
  1865–1897.
- Original R code: [fostergeotech/Vs30_NZ](https://github.com/fostergeotech/Vs30_NZ).
- Worden, C. B. et al. (2018). Spatial and spectral interpolation of
  ground-motion intensity measure observations. *BSSA*, 108, 866–875.

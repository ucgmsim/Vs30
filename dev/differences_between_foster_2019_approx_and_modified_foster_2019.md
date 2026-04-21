# Differences Between Original Foster (2019) and modified_foster_2019

This document compares the original Foster (2019) Vs30 model — as described in the
published paper (Foster et al., 2019, *Earthquake Spectra* 35(4)) and implemented in
Kevin Foster's R code (`fostergeotech/Vs30_NZ`) — with the `modified_foster_2019`
model version implemented in this Python codebase.

## Summary of Differences

| Aspect | Original Foster (2019) | modified_foster_2019 |
|--------|----------------------|----------------------|
| Coastal distance modifications | None | Alluvium and Floodplain adjusted by distance to coast |
| Alluvium (G06) hybrid treatment | Slope-based Vs30 interpolation | Coastal distance-based Vs30 interpolation (slope modification skipped) |
| Floodplain (G13) hybrid treatment | No special modification | Coastal distance-based Vs30 interpolation |
| Spatial correlation function | Matérn (κ=0.9, range=20 km, nugget=0.05) | Exponential (φ_geology=1407 m, φ_terrain=993 m, no nugget) |
| Kaiser Q3 observation filtering | All Q3 stations excluded | Q3 stations with 3-character names retained |
| Bayesian updating | Performed from scratch using all observations | Skipped; pre-computed posteriors loaded from CSV |


## Difference 1: Coastal Distance Modifications (New Feature)

**This is the largest difference between the two models.**

The original Foster (2019) model has **no modifications based on distance to the
coastline**. The only geomorphological adjustment is slope-based (Section
"Topographic Slope-Based Modification" in the paper; `MODEL_AhdiAK_noQ3_hyb09c.R`
in the R code). A search of the entire R codebase confirms zero references to coastal
distance, coastal proximity, or distance-based Vs30 adjustments.

The `modified_foster_2019` model adds coastal distance modifications for two geology
categories:

### Alluvium (G06, internal GID 4)

- **Original**: Vs30 varies with topographic slope via piecewise-linear interpolation
  in log-log space (slope limits log₁₀ = [-3.44, -0.88], Vs30 = [252, 275] m/s).
- **Modified**: Slope modification is **skipped** (controlled by `apply_alluvium_slope_mod: false`
  in the config; see `raster.py:827`). Instead, Vs30
  is linearly interpolated based on distance from coast:
  - Distance < 8 km from coast → Vs30 = 240 m/s
  - Distance > 20 km from coast → Vs30 = 500 m/s
  - Between 8–20 km → linear interpolation

This is a substantial change. The original slope-based adjustment for alluvium produced
a narrow Vs30 range (252–275 m/s), while the coastal distance modification produces a
much wider range (240–500 m/s). The physical rationale is that alluvial deposits near
the coast tend to be younger, less consolidated, and have lower Vs30, while alluvium
far from the coast is older and stiffer.

### Floodplain (G13, internal GID 10)

- **Original**: No special modification. Uses the Bayesian posterior value directly
  (posterior Vs30 = 197 m/s, σ = 0.20 from Table 1 of the paper).
- **Modified**: Vs30 linearly interpolated based on distance from coast:
  - Distance < 8 km from coast → Vs30 = 197 m/s
  - Distance > 20 km from coast → Vs30 = 500 m/s
  - Between 8–20 km → linear interpolation

**Code locations:**
- Coastal distance parameters: `vs30/constants.py:171–179`
- Coastal distance modification logic: `vs30/raster.py:692–734`
- Slope skip for alluvium: `vs30/raster.py:827` (gated by `apply_alluvium_slope_mod`)
- Coastal distance application: `vs30/raster.py:838–852` (gated by `apply_coastal_distance_mod`)


## Difference 2: Spatial Correlation Function

The models use fundamentally different correlation functions for the MVN (Multivariate
Normal) spatial conditioning step.

### Original Foster (2019) — Matérn Variogram

The R code fits a **Matérn variogram** to empirical variograms of normalized residuals
(`fitVariogram_AhdiAK_noQ3_hyb09c.R`):

```
Model:  Matérn
Sill:   0.15
Range:  20,000 m (20 km scale parameter)
Nugget: 0.05
κ:      0.9
```

The Matérn correlation function with κ=0.9 is smoother than exponential (which
corresponds to κ=0.5) and has a shape that transitions more gradually between short-
and long-range correlations. The R code converts this variogram to a piecewise-linear
correlation function via `approxfun()` over 128 logarithmically spaced distance points
(`mvn_params.R`).

The **effective ranges** (distance at which correlation ≈ 0.05) reported in the paper
are approximately **4.2 km** for the geology model and **3.0 km** for the terrain
model (paper p. 1886).

### modified_foster_2019 — Simple Exponential

The Python code uses a simple **exponential correlation function**:

```
ρ(d) = exp(-d / φ)
```

with `φ_geology = 1407 m` and `φ_terrain = 993 m` (`constants.py:66–67`). There is no
nugget term. A minimum distance of 0.1 m is enforced to prevent singularities
(`MIN_DIST_ENFORCED`, `constants.py:75`).

For an exponential model, the practical range (distance to 5% correlation) is
approximately 3φ:
- Geology: 3 × 1407 = 4221 m ≈ **4.2 km** ✓
- Terrain: 3 × 993 = 2979 m ≈ **3.0 km** ✓

The **effective ranges match** the paper's values, so the models agree at long range.
However, the shapes of the correlation functions differ at short to medium distances:

- The Matérn (κ=0.9) has a rounder peak near zero distance, meaning correlation decays
  more slowly at very short distances compared to exponential.
- The exponential model drops off more steeply at short distances.
- The Matérn has a **nugget of 0.05**, introducing a discontinuity at zero distance
  (correlation starts at σ²/(σ²+τ²) = 0.15/0.20 = 0.75 rather than 1.0). The
  exponential model in the Python code has **no nugget**, so correlation at zero
  distance is 1.0 (modulo the MIN_DIST_ENFORCED correction).

### Impact

The practical effect of this difference is most visible near observation points:

- **With nugget (original)**: Predictions don't exactly match observations even at
  the observation location. The nugget absorbs some of the observation's influence
  as measurement noise.
- **Without nugget (modified)**: The MVN approach uses the `noisy` parameter
  (`constants.py:60`) with Worden et al. (2018) correlation adjustment factors
  (omega weighting) to achieve a similar effect. This is functionally equivalent
  to a location-dependent nugget, which is actually the paper's stated advantage
  of MVN over kriging (paper p. 1889).


## Difference 3: Observation Data Filtering

### Kaiser et al. Q3 Data

- **Original**: All Q3 stations are excluded. The paper states: *"We do not use Q3
  data for V_{S30} modeling here"* (p. 1869). The R code creates a `noQ3` subset
  that drops all Kaiser Q3 entries.

- **Modified**: Q3 stations with exactly 3-character station names are **retained**.
  From `create_modified_foster_2019_observations.py:64–67`:
  ```python
  kaiseretal_filtered = kaiseretal_unfiltered[
      (kaiseretal_unfiltered.q != 3)
      | (kaiseretal_unfiltered.station.str.len() == 3)
  ].copy()
  ```
  The rationale is that 3-character station codes identify broadband seismometers on
  rock, which are considered reliable even with a Q3 quality flag. This adds
  approximately 60 additional stations to the observation dataset.

### Impact

Since `modified_foster_2019` sets `do_bayesian_update: false`, this filtering
difference does **not** affect the categorical model posteriors (those come from the
original R code's Q3-excluded posteriors). The difference only affects the **MVN
spatial conditioning step**, where additional observations near these stations will
pull predictions toward the measured values.


## Difference 4: Bayesian Update Strategy

- **Original**: The R code performs Bayesian updating from scratch. It starts with
  prior distributions from Ahdi et al. (2017b) for geology and Yong et al. (2012)
  for terrain, then sequentially updates each category using observations falling
  within that category. The posterior values (Tables 1 and 2 in the paper) are the
  output.

- **Modified**: The Python code sets `do_bayesian_update: false` in the config and
  loads the R code's pre-computed posteriors directly from CSV files
  (`geology_model_posterior_from_modified_foster_2019_mean_and_standard_deviation.csv`
  and the terrain equivalent). The Bayesian update code path is never executed.

### Impact

This means the categorical model values (Vs30 mean and σ per geology/terrain
category) are **identical** between the two models. The difference is purely
operational — the modified version trusts the R code's posteriors rather than
recomputing them.


## Differences That Are the Same (Confirmed Matches)

The following aspects were verified to be consistent between the two models:

| Aspect | Value | Source |
|--------|-------|--------|
| Prior weight (κ₀ = ν₀) | 3 | Paper p. 1880; `constants.py:100` |
| Minimum σ | 0.5 (log-space) | Paper p. 1880; `constants.py:104` |
| Covariance reduction factor (a) | 1.5 | Paper Eq. 6; `constants.py:60` |
| Slope-modified categories | G04, G05, G06*, G09 | Paper Table 3; `constants.py:212–217` |
| Slope interpolation parameters | Match Table 3 | Paper Table 3; `constants.py:212–217` |
| Sigma reduction factors | 0.4888, 0.7103, 0.9988, 0.9348 | R code; `constants.py:221–226` |
| Model combination | Equal weight geometric mean (w=0.5) | Paper Eq. 8; config `combine_ratio: 1.0` |
| Combination σ formula | Paper Eq. 9 | Paper Eq. 9; `utils.py:113–125` |
| Measurement uncertainties | McGann/Wotherspoon: 0.2, Kaiser Q1: 0.1, Q2: 0.2 | Paper p. 1871 |

*G06 slope modification is present in code but skipped when coastal distance
modification is active (default behavior).


## Origin of the Modifications

The modifications were introduced during or after Viktor Polak's translation of the
R code to Python (first Python commit 2021-02-04). The coastal distance modifications
and observation filtering changes are not documented in the original paper or R code.
The specific motivation for these changes is not recorded in the commit history,
but the physical reasoning is plausible:

- **Coastal distance for alluvium/floodplain**: Near-coastal alluvial and floodplain
  deposits in New Zealand tend to be younger Holocene sediments with lower Vs30,
  while inland equivalents may include older, more consolidated Pleistocene deposits.
  A distance-to-coast proxy captures this age/consolidation gradient that surface
  geology maps do not distinguish.

- **Retaining Q3 broadband stations**: 3-character station codes (e.g., "WEL", "BHW")
  identify GeoNet broadband seismometers, many on well-characterized rock sites.
  Their Q3 rating likely reflects the indirect nature of the Vs30 estimate (from
  site period or geology) rather than genuinely poor data quality.


## References

- Foster, K. M., Bradley, B. A., McGann, C. R., and Wotherspoon, L. M. (2019).
  A V_{S30} Map for New Zealand Based on Geologic and Terrain Proxy Variables and
  Field Measurements. *Earthquake Spectra*, 35(4), 1865–1897.
- Original R code: `fostergeotech/Vs30_NZ`
  (local copy: `/home/arr65/src/Kevin_Foster_R_code_vs30_model/Vs30_NZ`)
- Worden, C. B. et al. (2018). Spatial and spectral interpolation of ground-motion
  intensity measure observations. *BSSA*, 108, 866–875.

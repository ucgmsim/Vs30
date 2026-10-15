# Vs30 uncertainty findings and deferred work

Status: deferred at the user's request, 2026-09-19. The immediate objective is
to populate central Vs30 estimates using `andrew-latest`. Do not mistake this
document for an implemented uncertainty model.

## Initial database population policy

Calculate all numerically supported CPT/SPT-to-Vs and Vs-to-Vs30 combinations.
Store central estimates in m/s and **leave `vs30_stddev` SQL NULL**, including
for profiles reaching 30 m. NULL means not evaluated reliably, not zero
uncertainty. Retain correlation identifiers and a separate run audit so these
rows can be selected for later recalculation without touching measurements.
Downstream consumers must not replace NULL sigma with zero or use these rows as
zero-uncertainty observations in a weighted/spatial model.

The user initially requested resolving uncertainty before population, then
explicitly changed priority to population first and documentation for follow-up.
No full uncertainty implementation or new calibration is part of this run.

## Confirmed implementation findings

### Boore 2011

`vs_calc/vs30_correlations.py::boore_2011` implements the central relation

    log10(Vs30) = C0 + C1 log10(VsZ) + C2 [log10(VsZ)]^2

consistently with equation 2 and Table 2 of
[Boore, Thompson & Cadet (2011)](https://www.daveboore.com/pubs_online/regional_correlations_of_vs30_and_vsz_bssa_101.pdf).
The final table column is residual standard deviation in **log10** units. The
current uncertainty expression mixes logarithm bases, uses an incorrect
derivative, and adds a derivative squared without an input-variance factor.
Its result must not be interpreted as a valid total Vs30 standard deviation.

The table used here covers averaging depths 5–29 m. The publication discusses
regional differences; its KiK-net relation is not an independently calibrated
NZ relation. The code's existing central predictions are retained for now.

### Missing propagation, including Boore 2004 and 30 m profiles

`VsProfile.calc_vsz` integrates travel time but does not propagate `vs_sd`.
`boore_2004` returns only its tabulated regression scatter, not total propagated
uncertainty. `VsProfile.calc_vs30` explicitly returns zero standard deviation
at 30 m even when the predicted Vs profile is uncertain. Reaching 30 m removes
the extrapolation step; it does not remove CPT/SPT-to-Vs uncertainty.

### Input uncertainty definitions need an audit

- Vs-profile plots use `Vs * exp(±vs_sd)`, implying natural-log standard
  deviations. The database estimate columns do not currently specify units.
- Robertson 2009 and Hegazy 2006 return assumed `0.2` uncertainties; comments
  explicitly say published values are unavailable in this implementation.
- Andrus variants transform a reported m/s residual using `log(1 + SD/Vs)`.
  This is not generally an exact conversion to lognormal standard deviation.
- Kwak 2015 explicitly contains a TODO and reuses Brandenberg's within-site
  uncertainty expression. The correct source-specific treatment is unresolved.
- SPT routines collapse between-site and within-site components into a single
  per-depth standard deviation, losing information needed for depth covariance.
- Brandenberg has no dedicated gravel branch and currently uses its clay branch
  for gravel; Kwak has a gravel branch. Retain the soil names in the audit so
  these cases can be selected explicitly for review.

These observations come from the source code; they are not claims that every
original publication lacks an uncertainty model. Re-read the primary papers
before changing coefficients or inventing missing scatter.

## Proposed implementation after review

1. Audit each of the nine input correlations against primary publications:
   formula, calibration domain, units, residual distribution, within/between-site
   components, and whether reported predictions are medians or means.
2. Define a documented output convention, preferably `sigma_ln(Vs30)` to match
   the existing logarithmic profile convention. Update database documentation
   and consumers together; do not silently change the meaning of populated data.
3. Choose a depth-covariance model. Densely sampled CPT points are not independent
   realizations of correlation-model error. Independent errors can spuriously
   make uncertainty decrease merely by increasing sampling density. Use published
   decompositions where possible, and compare explicit assumptions otherwise.
4. Propagate through the travel-time average. For interval thicknesses `h_i`,
   speeds `v_i`, travel time `T = sum(h_i/v_i)`, and `VsZ = Z/T`, the first-order
   log-space sensitivity is `w_i = (h_i/v_i)/T`. Thus
   `Var[ln(VsZ)] ≈ w.T @ Cov[ln(v)] @ w`.
5. Propagate through Boore. Under an explicit independence assumption between
   input-profile errors and extrapolation residuals, the first-order 2011 result is

       Var[ln(Vs30)] ≈ (C1 + 2*C2*log10(VsZ))^2 * Var[ln(VsZ)]
                       + (ln(10)*sigma_RES_log10)^2

   For 2004 the derivative is its slope `b`; verify its published scatter units
   first. At 30 m, return the propagated profile uncertainty without adding an
   extrapolation residual. These propagation formulas are proposed derivations,
   not a claim that Boore supplies a complete CPT/SPT uncertainty model.
6. Validate analytical approximations against reproducibly seeded Monte Carlo
   calculations on representative profiles before deciding whether full-run
   simulation is necessary.

Do not silently include uncertainty in groundwater, soil classification, unit
weights, SPT efficiency, borehole diameter, or measurement extraction. Those
require additional distributions and dependencies. Between-correlation spread
also is not interchangeable with the residual uncertainty of one correlation.

## Required tests and decision checkpoint

- Independently check the published central values and residual scatters.
- Check depth boundaries (5, 10, 29, 30 m), integer truncation, and invalid input.
- Check constant-velocity profiles and common multiplicative errors with known
  answers; zero input uncertainty should recover extrapolation-only scatter.
- Confirm nonzero profile uncertainty survives the 30 m path.
- Check resampling invariance under the selected covariance model.
- Compare analytical and Monte Carlo distributions across shallow/deep and
  highly contrasting layered profiles.
- Re-run representative real CPT/SPT cases and check downstream plots/exports.

Planning allowance from the review: roughly 3–5 working days of engineering
effort, plus production computation, if published information supports a
practical model. New empirical calibration is a separate project. The first
checkpoint should present unsupported uncertainties and depth-covariance
assumptions for review before fixing production defaults.

## Prioritise later recalculation

1. All NULL uncertainty values after a defensible model is implemented.
2. Shallow extrapolations, sparse SPTs, and large gaps/surface extensions.
3. Estimates relying on missing metadata, default clay, or non-core soil mapping.
4. Geological-domain mismatches: the Andrus age variants and McGann loess model
   are labelled alternatives, not confirmation that the site matches that geology.
5. Source anomalies, extreme predictions, questionable groundwater and unit
   weights. Inspect sources rather than clipping outputs without justification.

## Other known issues to revisit

The population QA also found a numerical cancellation issue in corrected cone
resistance. Decimal inputs such as `qc=0.04 MPa`, `u2=-0.2 MPa`, and area ratio
`0.8` cancel to zero but can leave a tiny positive floating-point residual.
The batch now rejects these numerical-zero inputs rather than importing their
near-zero central estimates. The preliminary run is preserved but invalidated.

- CPT hydrostatic pore pressure currently starts updating at sample index 1.
  If the first sample is already below groundwater, its effective stress is
  overestimated. The initial population preserves this existing calculation.
- `VsProfile` can modify input arrays while truncating to integer depth; CPT
  correlations can modify cached arrays. Batch execution must isolate arrays
  so correlation ordering does not change results.
- CPT unit weight was recalculated over the entire profile at every depth;
  hoisting that calculation once is a performance-only change, with regression
  tests required to show unchanged numerical results.
- The supplied NZGD unit-weight lookup includes silt/gravel saturated weights
  below unsaturated weights. Preserve and flag these until scientifically reviewed.
- The existing Boore tables and interpolation approach do not establish validity
  for every geological setting or poorly sampled near-surface interval.

See [the population runbook](vs30_database_population.md) for execution policy,
and [effective-stress validation](effective_stress_validation.md) for the earlier
SPT validation. That validation did not establish the correctness of Vs30 sigma.

# Vs30

**Vs30** is the time-averaged shear-wave velocity in the top 30 metres of
ground. It is a key input for seismic site-response analysis: sites with
low Vs30 (soft sediment) amplify earthquake ground motion, while
high-Vs30 sites (rock) do not. National seismic hazard maps, building
codes, and ground-motion prediction equations all depend on knowing Vs30.

This package produces Vs30 maps and point estimates for New Zealand by
combining three layers of information:

1. **Categorical priors** from QMAP geology and IwahashiPike terrain,
   each category calibrated to a log-normal Vs30 distribution.
2. **Bayesian posterior update** of those priors using measured Vs30
   observations (surface-wave surveys, CPT-derived inferences,
   seismometer station metadata).
3. **MVN spatial conditioning** that pulls pixels within ~10 km of an
   observation toward the observed value, using an exponential (or
   Matérn) spatial correlation.

Output is a two-band GeoTIFF on the NZTM2000 grid (band 1 = Vs30 mean,
band 2 = log-space standard deviation).

- **Codebase:** https://github.com/ucgmsim/Vs30
- **Scientific reference:** Foster et al. (2019), *Earthquake Spectra*.

## Supported Models

Four model versions ship with the package. They differ in the
observations they use, whether the Bayesian update runs live from priors
or reuses pre-computed posteriors, and which spatial correlation kernel
is applied.

![Model comparison](images/model_comparison.png)

| Version | Observations | Update | Notes |
|---------|--------------|--------|-------|
| `foster_2019_approx` | 412 independent (Foster-derived, no Kaiser Q3) | Uses pre-computed Foster posteriors | Faithful reproduction of the published Foster (2019) paper map. Matérn correlation for geology. |
| `modified_foster_2019` | 412 independent (same as above) | Uses pre-computed Foster posteriors | Same base observations as `foster_2019_approx`, but the refactored pipeline's exponential correlation instead of Matérn. |
| `jaehwi_v1p0` | 671 independent (McGann 276 + Wotherspoon 36 + Kaiser 359) | Live Bayesian update from raw priors | Reproduces Jaehwi's v1.0 output. GID 4 alluvium slope modification is off. |
| `viktor_cpt_clustering` | ~35 700 CPT-derived (DBSCAN-clustered) | Live Bayesian update from raw priors | CPT-derived Vs30 across Canterbury and central NZ pulls those regions toward lower values — visible as the more uniform map above. |

Higher-resolution maps:
[foster_2019_approx](images/foster_2019_approx.png),
[modified_foster_2019](images/modified_foster_2019.png),
[jaehwi_v1p0](images/jaehwi_v1p0.png),
[viktor_cpt_clustering](images/viktor_cpt_clustering.png).

## Reference Pages

- [Differences between foster_2019_approx and modified_foster_2019](differences_between_foster_2019_approx_and_modified_foster_2019.md)
  — Observation sets, correlation functions, and hybrid modifications
  distinguishing the two Foster-based variants.
- [Legacy Python code bug analysis](legacy_python_code_bug_analysis.md)
  — Bugs found in the pre-refactor codebase and how the refactor
  addresses them.
- [Why we cannot reproduce v1.0 grid exactly](why_we_cannot_reproduce_v1p0_grid_exactly.md)
  — Float32 precision shortcuts in Jaehwi's grid-mode implementation
  make V1.0_26Mar.tif non-reproducible even by the legacy code itself.

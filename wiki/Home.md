# Vs30

The time-averaged shear-wave velocity in the top 30 metres of
ground (Vs30) is an important quantity for understanding seismic
hazard, as it indicates how the ground will behave in an earthquake.

Vs30 can be measured at specific sites with geotechnical investigations, 
such as the seismic cone penetration test (sCPT). However, These 
measurements are slow and expensive, so it is not feasible to measure 
Vs30 at the large number of sites required to accurately sample large regions. 
As Vs30 across large regions is needed to understand seismic hazard, models 
have been developed to infer Vs30 from geological and topological maps, 
as well as geotechnical investigations that do not probe the shear wave velocity, 
such as cone penetration tests (CPTs) and Standard Penetration Tests (SPTs).

This package queries modelled Vs30 at specified locations using any of
the models described below.

## Quick Start

Query Vs30 at a list of sites from a CSV with `lon`/`lat` columns:

```bash
vs30 points modified_foster_2019 sites.csv results.csv
```

See the [Usage page](Usage.md) for grid maps, model version selection,
and parameter overrides.

## Supported Models

### `foster_2019_approx`

A near-identical reproduction of the Vs30 map from Foster et al. (2019).
Minor implementation differences between this pipeline and the original
prevent an exact match, but typical differences are only a few percent,
so for practical purposes this model can be considered representative
of the published Foster et al. (2019) map (see comparison figure below).

### `modified_foster_2019`

The model developed by Foster et al. (2019), with several modifications:

- alluvium and floodplain Vs30 adjusted by distance from the coast
- exponential spatial correlation in place of Matérn
- some lower-quality Vs30 observations retained rather than excluded

### `jaehwi_v1p0`

The model developed by Foster et al. (2019) but with the inclusion of
additional Vs30 measurements from the New Zealand National Seismic
Hazard Model (NSHM).

### `viktor_cpt_clustering`

Further developments of `modified_foster_2019` by Viktor Polak to include
~35,700 CPT-derived Vs30 estimates, clustered with DBSCAN to avoid
over-representing densely surveyed regions.

![Model comparison](images/model_comparison.png)

The left panel is the **reference**: the Vs30 map from Foster et al. (2019)
as downloaded from the supplementary data. The four right panels show each
model as a log-ratio difference map relative to that reference,
`ln(model / reference)`, following the seismic-hazard convention for
model comparisons. Red pixels are where a model predicts higher Vs30
(stiffer ground) than Foster 2019; blue pixels are where it predicts
lower Vs30 (softer ground). Zero (white) means agreement.

#### Higher-resolution maps. #### 
Each model links to two single-panel diff
variants: the **shared-scale** version uses the same ±1.0 ln-units
colourbar as the composite above (so models remain directly comparable),
and the **autoscaled** version rescales to each panel's 1st–99th
percentile so small within-model features (e.g. MVN residuals near
observation sites) become visible.

- Reference: [foster_2019 (published)](images/reference_foster_2019.png)
- foster_2019_approx: [shared scale](images/foster_2019_approx_diff.png) ·
  [autoscaled](images/foster_2019_approx_diff_autoscale.png)
- modified_foster_2019: [shared scale](images/modified_foster_2019_diff.png) ·
  [autoscaled](images/modified_foster_2019_diff_autoscale.png)
- jaehwi_v1p0: [shared scale](images/jaehwi_v1p0_diff.png) ·
  [autoscaled](images/jaehwi_v1p0_diff_autoscale.png)
- viktor_cpt_clustering: [shared scale](images/viktor_cpt_clustering_diff.png) ·
  [autoscaled](images/viktor_cpt_clustering_diff_autoscale.png)

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

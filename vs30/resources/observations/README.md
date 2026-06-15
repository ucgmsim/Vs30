# Vs30 Observation Datasets

Prepared site-observation datasets used to refine the Vs30 model predictions. Each model version selects its files in `vs30/configs/<version>.yaml` via two keys that differ in how nearby points are grouped:

- **`independent_observations_csv`** — used one point at a time (no clustering).
- **`clustered_observations_csv`** — DBSCAN first merges nearby points into single pseudo-observations, so densely sampled areas don't over-weight the result.

## CSV format

All files share `easting,northing,vs30,uncertainty`; the Foster files add `source,station,q`, the CPT file adds a leading `index` and `source`, and the Jaehwi file has only the four core columns. Coordinates are NZTM2000 (EPSG:2193, meters); `vs30` is in m/s; `uncertainty` is a natural-log-scale standard deviation (e.g. 0.2 ≈ 20%).

## `foster_2019_approx_measured_vs30_independent_observations.csv`

Independent observations for the `foster_2019_approx` model: measured Vs30 from the McGann et al. (2015) and Wotherspoon et al. (2015) Canterbury surface-wave surveys, plus a quality-screened nationwide subset of Kaiser et al. (2017). In the legacy code this package supersedes, these sources were assembled by `sites_load.py` loaders.

## `modified_foster_2019_measured_vs30_independent_observations.csv`

Independent observations for the `modified_foster_2019` model — the fuller measured set that `foster_2019_approx` is derived from. That model starts from precomputed Foster (2019) posteriors (no Bayesian update), so its observations feed only the MVN spatial adjustment.

## `viktor_inferred_vs30_from_cpt.csv`

Clustered observations for the `viktor_cpt_clustering` model: ~35,700 Vs30 values *inferred* from cone penetration test (CPT) soundings (`source = cpt` in legacy code).

## `jaehwi_v1p0_independent_observations.csv`

Independent observations (671 stations) for the `jaehwi_v1p0` model, combining McGann, Wotherspoon, and Kaiser/GeoNet sources. Produced from Jaehwi's `Vs30_2026` fork (legacy code this package supersedes) by running `sites_load_NSHM2022.load_vs()`.

## Generation

These CSVs are static, pre-generated inputs; the scripts that produced them are maintainer-only and not part of the installed package.

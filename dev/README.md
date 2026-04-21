# Developer Resources

This directory contains raw data files, processing scripts, and validation
tools used during development and maintenance of the vs30 package. These
are for maintainers and developers only — typical users should not need to
run anything here.

## Structure

- `scripts/` — Development and maintenance scripts.
  - `generators/` — Produce committed fixtures, benchmarks, or CSVs.
    Re-run when inputs change.
  - `comparison/` — Reusable tools for comparing rasters or points-mode
    output between codebases.
  - `investigations/` — Historical diagnostics kept for future reference.
    May reference legacy codebases or stale paths; treat as read-only.
- `observations/` — Raw observation data and scripts that produce the
  prepared CSVs used by the pipeline.
  - `foster_2019_approx/` — Script for the foster_2019_approx observations.
  - `modified_foster_2019/` — Script for the modified_foster_2019 observations.
- `jaehwi_v1p0_reproduction/` — Self-contained investigation bundle
  (scripts + reference data) from reproducing Jaehwi's v1.0 output.
- `docs/` — Maintainer-only design notes and investigations. User-facing
  scientific context lives in the repo's `wiki/` directory instead.

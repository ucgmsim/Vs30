# Developer Resources

This directory contains raw data files, processing scripts, and validation tools
used during development and maintenance of the vs30 package. These are for
maintainers and developers only — typical users should not need to run anything
here.

## Structure

- `observations/` — Raw observation data and scripts that produce the prepared
  CSVs used by the pipeline.
  - `jaehwi_v1p0/` — Data and script for the Jaehwi v1.0 model observations.
  - `foster_2019/` — Script that generated the Foster 2019 model observations
    (requires the legacy codebase environment).
- `validation/` — Scripts for comparing pipeline outputs against the legacy
  codebase or between different runs.

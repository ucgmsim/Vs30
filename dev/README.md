# Developer Resources

This directory contains raw data files, processing scripts, and validation tools
used during development and maintenance of the vs30 package. These are for
maintainers and developers only — typical users should not need to run anything
here.

## Structure

- `observations/` — Raw observation data and scripts that produce the prepared
  CSVs used by the pipeline.
  - `jaehwi_v1p0/` — Data and script for the Jaehwi v1.0 model observations.
  - `foster_2019_approx/` — Script that generates the foster_2019_approx model
    observations (approximate reproduction of Foster et al. 2019; requires the
    legacy codebase environment for raw-source regeneration).
- `compare_rasters.py` — Compares any two GeoTIFFs:
  - `stats` prints pixel-level comparison statistics over the overlapping region.
  - `diff` saves signed, absolute, and log-space difference GeoTIFFs.

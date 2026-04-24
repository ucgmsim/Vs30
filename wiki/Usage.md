# Usage

The package installs a `vs30` CLI entry point with two main modes:

- **`vs30 points`** — query Vs30 at a list of specific lat/lon sites. Fast; best for a handful to a few thousand locations.
- **`vs30 grid`** — run the full pipeline on a regular raster grid and write GeoTIFFs. Appropriate for producing maps.

Each mode has a `-custom` variant (`points-custom`, `grid-custom`) that exposes every scientific parameter individually. Use the plain commands with a `--version` for routine work; reach for `-custom` only when you need to override a single parameter from a model's config.

## Installation

Clone and install in editable mode:

```bash
pip install -e /path/to/Vs30
```

The install step unpacks bundled shapefiles from `vs30/resources/geospatial/shapefiles.tar.xz`.

## Point Queries

Prepare an input CSV with WGS84 longitude and latitude columns (defaults: `lon`, `lat`). For example, `sites.csv`:

```csv
site_id,lon,lat
wellington,174.7762,-41.2865
christchurch,172.6362,-43.5321
```

Run the pipeline with one of the supported model versions:

```bash
vs30 points modified_foster_2019 sites.csv results.csv
```

The output CSV contains geology/terrain category IDs, per-model Vs30 values, and the combined Vs30 used as the final answer.

Useful options:

- `--lon-column` / `--lat-column` — override the default column names.
- `--include-intermediate` — also write the geology-only and terrain-only Vs30 values.
- `--nproc 6` — limit the number of parallel workers (default: all cores).

## Grid Pipeline

The grid command writes a directory of GeoTIFFs (category IDs, pre-MVN Vs30, post-MVN Vs30, and the combined geology+terrain result).

A full 100 m New Zealand grid:

```bash
vs30 grid \
    --version modified_foster_2019 \
    --grid-xmin 1060050 --grid-xmax 2120050 \
    --grid-ymin 4730050 --grid-ymax 6250050 \
    --grid-dx 100 --grid-dy 100 \
    --output-dir ./vs30_out \
    --nproc 6
```

Coordinates are NZTM2000 (EPSG:2193), in metres. Reduce the domain or coarsen the spacing (`--grid-dx 400 --grid-dy 400`) for faster test runs — a 400 m national grid finishes in ~20 minutes on a typical workstation.

## Model Versions

Any command that takes `--version` (or a `version` argument, for `points`) accepts one of:

- `foster_2019_approx`
- `modified_foster_2019`
- `jaehwi_v1p0`
- `viktor_cpt_clustering`

See the [Home page](Home.md) for a description of each.

## Custom Parameters

`points-custom` and `grid-custom` expect every scientific parameter to be provided explicitly (observation CSVs, correlation kernels, hybrid modifier toggles, etc.). This is mainly useful for ablation experiments — e.g. running `modified_foster_2019`'s config but with the coastal-distance modifier disabled:

```bash
vs30 points-custom \
    --geology-csv vs30/resources/observations/modified_foster_2019/geology.csv \
    --terrain-csv vs30/resources/observations/modified_foster_2019/terrain.csv \
    --combination-method ratio --combine-ratio 1.0 \
    --noisy --mvn --do-bayesian-update \
    --no-apply-coastal-distance-mod \
    --no-apply-alluvium-slope-mod \
    --fill-gaps \
    --locations-csv sites.csv --output-csv results.csv
```

Run `vs30 points-custom --help` or `vs30 grid-custom --help` for the full option list.

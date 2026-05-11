# Usage

The package installs a `vs30` CLI entry point with two main modes:

- **`vs30 points`** — query Vs30 at a list of specific lat/lon sites. Fast; best for a handful to a few thousand locations.
- **`vs30 grid`** — run the full pipeline on a regular raster grid and write GeoTIFFs. Appropriate for producing maps.

Both modes accept either a bundled model version name or a path to a custom YAML config file with the same schema. See [Custom Model Configurations](#custom-model-configurations) for how to derive your own config from a bundled one.

## Installation

Clone and install in editable mode:

```bash
pip install -e /path/to/Vs30
```

The install step unpacks bundled shapefiles from `vs30/resources/geospatial/shapefiles.tar.xz`.

## Point Queries

Prepare an input CSV with WGS84 longitude and latitude columns (defaults: `longitude`, `latitude`; override with `--lon-column` / `--lat-column`). For example, `sites.csv`:

```csv
site_id,longitude,latitude
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

## Grid Pipeline

The grid command writes a directory of GeoTIFFs (category IDs, pre-MVN Vs30, post-MVN Vs30, and the combined geology+terrain result).

A full 100 m New Zealand grid:

```bash
vs30 grid \
    --model modified_foster_2019 \
    --grid-xmin 1060100 --grid-xmax 2120100 \
    --grid-ymin 4730100 --grid-ymax 6250100 \
    --grid-dx 100 --grid-dy 100 \
    --output-dir ./vs30_out
```

Coordinates are NZTM2000 (EPSG:2193), in metres. The `--grid-xmin/xmax/ymin/ymax` values are the **outer edges** of the grid (pixel-edge convention), not pixel centres — so the leftmost pixel's left edge sits at `xmin` and its centre at `xmin + dx/2`. Reduce the domain or coarsen the spacing (`--grid-dx 400 --grid-dy 400`) for faster test runs — a 400 m national grid finishes in ~20 minutes on a typical workstation.

If the spatial-adjustment step runs out of memory on a large grid, lower `--max-spatial-boolean-array-memory-gb` (default: 1.0). It caps the size of the per-chunk boolean arrays used to find observations near each pixel — smaller values trade a small amount of speed for a lower peak memory footprint.

## Model Versions

Any command that takes a model argument (positional `model` for `points`, `--model` for `grid`) accepts one of:

- `foster_2019_approx`
- `modified_foster_2019`
- `jaehwi_v1p0`
- `viktor_cpt_clustering`

See the [Home page](Home.md) for a description of each.

## Custom Model Configurations

Where the commands take a model argument (positional `model` for `points`, `--model` for `grid`), you can pass either a bundled model version name (see [Model Versions](#model-versions)) or a path to a YAML config file with the same schema. The YAML option is the simplest way to override pipeline parameters — toggle a hybrid modifier, swap the correlation kernel, point at your own observation CSVs — without modifying the package.

The recommended workflow is to copy one of the bundled configs from `vs30/configs/` and edit it:

```bash
cp vs30/configs/modified_foster_2019.yaml ./my_custom_config.yaml
# Edit my_custom_config.yaml — e.g. set `apply_coastal_distance_mod: false`

vs30 points ./my_custom_config.yaml sites.csv results.csv

vs30 grid \
    --model ./my_custom_config.yaml \
    --grid-xmin 1060100 --grid-xmax 2120100 \
    --grid-ymin 4730100 --grid-ymax 6250100 \
    --grid-dx 100 --grid-dy 100 \
    --output-dir ./vs30_out
```

The path to the YAML file may be absolute or relative to the current working directory. All fields from the bundled configs must be present — they are validated when the YAML is loaded, and a missing field raises an error.

The CSV path fields inside the YAML (`geology_categorical_csv`, `terrain_categorical_csv`, `clustered_observations_csv`, `independent_observations_csv`) resolve in one of two ways:

- An **absolute path** is used as-is, so you can supply your own CSVs anywhere on disk.
- A **relative path** is resolved against the bundled `vs30/resources/` tree, so you can keep referring to bundled inputs by filename.

This means you can keep all the bundled resources and only override the one or two parameters you actually want to change.

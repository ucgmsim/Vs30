# Usage

The `vs30` command-line tool produces Vs30 estimates from geological and topographic data. It has two modes:

- **`vs30 points`** — Vs30 at sites specified by latitude and longitude in a CSV.
- **`vs30 grid`** — Vs30 across a regular raster grid, written as GeoTIFFs.

If you're new to Vs30 or want a description of each available model, see the [Home page](Home.md) first.

## Available model names

Both modes take a **model** argument that selects which Vs30 model to run. There are four bundled options:

- `foster_2019_approx`
- `modified_foster_2019`
- `jaehwi_v1p0`
- `viktor_cpt_clustering`

See the [Home page](Home.md) for a description of each. The examples below all use `modified_foster_2019`; any of the four can be substituted.

## Vs30 at specific sites: `vs30 points`

### Input file

Prepare a CSV listing the sites you want Vs30 for, with WGS84 longitude and latitude columns. By default these columns must be named `longitude` and `latitude`; if your file uses different names, pass them with `--lon-column` and `--lat-column`.

For example, `sites.csv`:

```csv
site_id,longitude,latitude
wellington,174.7762,-41.2865
christchurch,172.6362,-43.5321
```

Any additional columns (like `site_id` above) are preserved in the output.

### Running

The basic form takes a model name, the input CSV, and the output CSV:

```bash
vs30 points modified_foster_2019 sites.csv results.csv
```

If you've created a custom model (see [Custom configurations](#custom-configurations)), pass the path to its YAML config file in place of the model name:

```bash
vs30 points ./my_custom_config.yaml sites.csv results.csv
```

### Output

The output CSV contains every column from the input plus four new columns:

- `easting`, `northing` — the input coordinates converted to NZTM2000 (EPSG:2193).
- `vs30` — the final Vs30 estimate (the combined geology + terrain value).
- `stdv` — the estimated standard deviation of `vs30`.

To also write the intermediate per-model values (geology and terrain category IDs, the separate per-model Vs30 estimates, etc.), pass the `--include-intermediate` flag.

### All options

For the complete list of options:

```bash
vs30 points --help
```

## Vs30 across a region: `vs30 grid`

### Inputs

The grid command needs only a bounding box, a spacing, and an output directory — it doesn't take an input CSV (the grid itself defines the locations). The bounding-box and spacing values are in NZTM2000 metres (EPSG:2193).

### Running

A full 100 m New Zealand grid:

```bash
vs30 grid \
    --model modified_foster_2019 \
    --grid-xmin 1060100 --grid-xmax 2120100 \
    --grid-ymin 4730100 --grid-ymax 6250100 \
    --grid-dx 100 --grid-dy 100 \
    --output-dir ./vs30_out
```

If you've created a custom model (see [Custom configurations](#custom-configurations)), pass the path to its YAML config file in place of the model name:

```bash
vs30 grid \
    --model ./my_custom_config.yaml \
    --grid-xmin 1060100 --grid-xmax 2120100 \
    --grid-ymin 4730100 --grid-ymax 6250100 \
    --grid-dx 100 --grid-dy 100 \
    --output-dir ./vs30_out
```

The `--grid-xmin/xmax/ymin/ymax` values are the **outer edges** of the grid (pixel-edge convention), not pixel centres. So the leftmost pixel's left edge sits at `xmin` and its centre at `xmin + dx/2`.

### Output

The command writes a directory of GeoTIFF rasters covering the bounding box at the chosen spacing:

- the geology and terrain category IDs
- the geology and terrain Vs30 before MVN spatial adjustment
- the geology and terrain Vs30 after MVN adjustment
- the combined geology + terrain Vs30 (the final result)

### Practical tips

- **Shorter test runs**: while you're confirming your inputs are right, reduce the bounding box or coarsen the spacing (e.g. `--grid-dx 400 --grid-dy 400`).
- **Memory**: if the spatial-adjustment step runs out of memory on a large grid, lower `--max-spatial-boolean-array-memory-gb` (default: 1.0). It caps the size of the per-chunk boolean arrays used to find observations near each pixel — smaller values trade a small amount of speed for a lower peak memory footprint.

### All options

For the complete list of options:

```bash
vs30 grid --help
```

## Custom configurations

Each bundled model version is just a YAML config file living in `vs30/configs/`. To customize the pipeline — for example to toggle the coastal-distance modifier or swap the correlation kernel — copy a bundled config, edit it, and pass the path to `vs30 points` or `vs30 grid` in place of a bundled model name.

A typical workflow:

```bash
cp vs30/configs/modified_foster_2019.yaml ./my_custom_config.yaml
# edit my_custom_config.yaml — e.g. set `apply_coastal_distance_mod: false`

vs30 points ./my_custom_config.yaml sites.csv results.csv

vs30 grid \
    --model ./my_custom_config.yaml \
    --grid-xmin 1060100 --grid-xmax 2120100 \
    --grid-ymin 4730100 --grid-ymax 6250100 \
    --grid-dx 100 --grid-dy 100 \
    --output-dir ./vs30_out
```

The path to the YAML file can be absolute or relative to your current working directory. All fields from the bundled config must be present — they're validated when the YAML is loaded, and a missing field raises an error.

### Path fields inside the YAML

A few YAML fields point at CSV data files: `geology_categorical_csv`, `terrain_categorical_csv`, `clustered_observations_csv`, and `independent_observations_csv`. You can leave these set to the bundled file names (which is what the bundled configs do) or replace them with paths to your own CSVs. The resolution rules:

- An **absolute path** is used as-is, so you can supply your own CSVs anywhere on disk.
- A **bundled data file name** is automatically resolved under one of two subdirectories: `vs30/resources/categorical_vs30_mean_and_stddev/` for the geology/terrain CSVs, or `vs30/resources/observations/` for the observation CSVs. This lets you refer to bundled CSVs by file name.

In practice you'll usually keep most of the bundled config intact and only edit the one or two parameters you actually want to change.

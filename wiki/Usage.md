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
- `vs30` — the final Vs30 estimate in m/s (the combined geology + terrain value).
- `stdv` — the uncertainty of `vs30`, given as the standard deviation of ln(Vs30). It has no units; 0.3 means roughly ±30%.

To also write the intermediate per-model values (geology and terrain category IDs, the separate per-model Vs30 estimates, etc.), pass the `--include-intermediate` flag.

Rows whose coordinates are missing or invalid are kept, with blank results, and a warning lists their line numbers. Sites outside the model's data coverage (offshore, on water, or outside New Zealand) also get blank results, and a warning says how many. If one of your input columns has the same name as an output column (for example your own `vs30`), it's kept as `vs30_input`.

### All options

For the complete list of options:

```bash
vs30 points --help
```

By default the command prints a few progress lines; add `--verbose` (`-v`) to also see each step.

## Vs30 across a region: `vs30 grid`

### Inputs

The grid command needs only a model and an output directory — it doesn't take an input CSV (the grid itself defines the locations). By default the grid covers all of New Zealand at 100 m spacing; to compute a smaller region or a coarser grid, set the bounding box and spacing, in NZTM2000 metres (EPSG:2193).

### Running

A full 100 m New Zealand grid (a long run; the command starts by printing the grid's size, 10600 x 15200 pixels):

```bash
vs30 grid modified_foster_2019 ./vs30_out
```

A smaller region, for example a 20 km box around Wellington:

```bash
vs30 grid modified_foster_2019 ./wellington_out \
    --grid-xmin 1740100 --grid-xmax 1760100 \
    --grid-ymin 5420100 --grid-ymax 5440100
```

If you've created a custom model (see [Custom configurations](#custom-configurations)), pass the path to its YAML config file in place of the model name:

```bash
vs30 grid ./my_custom_config.yaml ./vs30_out
```

The `--grid-xmin/xmax/ymin/ymax` values are the **outer edges** of the grid (pixel-edge convention), not pixel centres. So the leftmost pixel's left edge sits at `xmin` and its centre at `xmin + dx/2`. The extent must be a whole number of pixels. For a region to line up pixel-for-pixel with the full-NZ grid, keep its edges at `1060100` plus a multiple of `dx` (east) and `4730100` plus a multiple of `dy` (north); otherwise the command warns.

### Output

By default the command writes three GeoTIFF rasters covering the bounding box at the chosen spacing:

- `combined_vs30.tif` — the combined geology + terrain Vs30 (the final result)
- `geology_vs30_slope_and_coastal_distance_and_spatially_adjusted_with_uncertainty.tif` — the geology model after MVN spatial adjustment
- `terrain_vs30_spatially_adjusted_with_uncertainty.tif` — the terrain model after MVN spatial adjustment

Each has two bands: Vs30 in m/s, and its uncertainty as the standard deviation of ln(Vs30) (no units; 0.3 means roughly ±30%).

Pass `--include-intermediate` to also write the intermediate products: the geology and terrain category IDs (`gid.tif`, `tid.tif`), each model's Vs30 before MVN adjustment, the slope and (for models that use it) coastal-distance rasters behind the geology modifications (coastal distance is capped at 20 km, beyond which it no longer changes the result), for models that update the categories from observations, the updated category tables as CSV; and, for models that gap-fill, the combined Vs30 before gap-filling.

### Practical tips

- **Shorter test runs**: while you're confirming your inputs are right, reduce the bounding box or coarsen the spacing (e.g. `--grid-dx 400 --grid-dy 400`).
- **CPU use**: by default the command uses every CPU core. `--nproc 4` (for example) caps both the clustering of observations and the linear algebra in the spatial adjustment, keeping the machine responsive, usually with little loss of speed.
- **Memory**: if the spatial-adjustment step runs out of memory on a large grid, lower `--max-spatial-intermediate-array-memory-gb` (default: 4.0). It caps the per-chunk (indices, distances) arrays produced by the KDTree query during MVN spatial adjustment. Smaller values trade a small amount of speed for a lower peak memory footprint.

### All options

For the complete list of options:

```bash
vs30 grid --help
```

By default the command prints a few progress lines; add `--verbose` (`-v`) to also see each step.

## Custom configurations

Each bundled model version is just a YAML config file living in `vs30/configs/`. To customize the pipeline — for example to toggle the coastal-distance modifier or swap the correlation kernel — copy a bundled config, edit it, and pass the path to `vs30 points` or `vs30 grid` in place of a bundled model name.

A typical workflow:

```bash
cp vs30/configs/modified_foster_2019.yaml ./my_custom_config.yaml
# edit my_custom_config.yaml — e.g. set `apply_coastal_distance_mod: false`

vs30 points ./my_custom_config.yaml sites.csv results.csv

vs30 grid ./my_custom_config.yaml ./vs30_out
```

The path to the YAML file can be absolute or relative to your current working directory. All fields from the bundled config must be present, and no others. The config is checked when it's loaded, before any computation: a missing or unknown field, a value of the wrong kind (e.g. `"false"` in quotes instead of `false`), or a CSV path that doesn't exist stops the run with an error naming the problem.

### Path fields inside the YAML

A few YAML fields point at CSV data files: `geology_categorical_csv`, `terrain_categorical_csv`, `clustered_observations_csv`, and `independent_observations_csv`. You can leave these set to the bundled file names (which is what the bundled configs do) or replace them with paths to your own CSVs. The resolution rules:

- An **absolute path** is used as-is, so you can supply your own CSVs anywhere on disk.
- A **bundled data file name** is automatically resolved under one of two subdirectories: `vs30/resources/categorical_vs30_mean_and_stddev/` for the geology/terrain CSVs, or `vs30/resources/observations/` for the observation CSVs. This lets you refer to bundled CSVs by file name.

In practice you'll usually keep most of the bundled config intact and only edit the one or two parameters you actually want to change.

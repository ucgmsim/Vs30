#!/usr/bin/env python
"""
Calculate Vs30 over a region (default) or specify points at which to output for instead.
"""

from functools import partial
import math
from multiprocessing import Pool
import os
from shutil import rmtree
import sys
from time import time

import numpy as np
import pandas as pd
from pyproj import Transformer

from vs30 import (
    model,
    model_geology,
    model_terrain,
    mvn,
    params,
    sites_cluster,
    sites_load,
)

# work on ~50 points per process
SPLIT_SIZE = 50
MODEL_MAPPING = {"geology": model_geology, "terrain": model_terrain}
wgs2nztm = Transformer.from_crs(4326, 2193, always_xy=True)
p_paths, p_sites, p_grid, p_ll, p_geol, p_terr, p_comb, nproc = params.load_args()
pool = Pool(nproc)

# working directory/output setup
if os.path.exists(p_paths.out):
    if p_paths.overwrite:
        rmtree(p_paths.out)
    else:
        sys.exit("output exists")
os.makedirs(p_paths.out)


def array_split(array):
    """
    Split dataframes and numpy arrays for multiprocessing.Pool.map
    """
    return np.array_split(array, math.ceil(len(array) / SPLIT_SIZE))


# input locations
if p_ll is not None:
    print("loading locations...")
    table = pd.read_csv(
        p_ll.ll_path,
        usecols=(p_ll.lon_col_ix, p_ll.lat_col_ix),
        names=["longitude", "latitude"],
        engine="c",
        skiprows=p_ll.skip_rows,
        dtype=np.float64,
        sep=p_ll.col_sep,
    )
    table["easting"], table["northing"] = wgs2nztm.transform(
        table.longitude.values, table.latitude.values
    )
    table_points = table[["easting", "northing"]].values

# measured sites
print("loading measured sites...")
sites = sites_load.load_vs(source=p_sites.source)
sites_points = np.column_stack((sites.easting.values, sites.northing.values))

# model loop
tiffs = []
tiffs_mvn = []
for model_setup in [p_geol, p_terr]:
    if model_setup.update == "off":
        continue
    print(model_setup.name, "model...")
    t = time()
    model_module = MODEL_MAPPING[model_setup.name]

    print("    model measured update...")
    sites[f"{model_setup.letter}id"] = np.concatenate(
        pool.map(model_module.model_id, array_split(sites_points))
    )
    if model_setup.update == "prior":
        model_table = model_module.model_prior()
    elif model_setup.update == "posterior_paper":
        model_table = model_module.model_posterior_paper()
    elif model_setup.update == "posterior":
        model_table = model_module.model_prior()
        if p_sites.source == "cpt":
            sites = sites_cluster.cluster(sites, model_setup.letter, nproc=nproc)
            model_table = model.cluster_update(model_table, sites, model_setup.letter)
        else:
            model_table = model.posterior(model_table, sites, f"{model_setup.letter}id")

    print("    model at measured sites...")
    (
        sites[f"{model_setup.name}_vs30"],
        sites[f"{model_setup.name}_stdv"],
    ) = model_module.model_val(
        sites[f"{model_setup.letter}id"].values,
        model_table,
        model_setup,
        paths=p_paths,
        points=sites_points,
        grid=p_grid,
    ).T

    if p_ll is not None:
        print("    model points...")
        table[f"{model_setup.letter}id"] = np.concatenate(
            pool.map(model_module.model_id, array_split(table_points))
        )
        (
            table[f"{model_setup.name}_vs30"],
            table[f"{model_setup.name}_stdv"],
        ) = model_module.model_val(
            table[f"{model_setup.letter}id"].values,
            model_table,
            model_setup,
            paths=p_paths,
            points=table_points,
            grid=p_grid,
        ).T
        print("    measured mvn...")
        (
            table[f"{model_setup.name}_mvn_vs30"],
            table[f"{model_setup.name}_mvn_stdv"],
        ) = np.concatenate(
            pool.map(
                partial(mvn.mvn_table, sites=sites, model_name=model_setup.name),
                array_split(table),
            )
        ).T
    else:
        print("    model map...")
        tiffs.append(
            model_module.model_val_map(p_paths, p_grid, model_table, model_setup)
        )
        print("    measured mvn...")
        tiffs_mvn.append(mvn.mvn_tiff(p_paths, p_grid, model_setup.name, sites))

    print(f"{time()-t:.2f}s")

if p_geol.update != "off" and p_terr.update != "off":
    # combined model
    print("combining geology and terrain...")
    t = time()
    if p_ll is not None:
        for prefix in ["", "mvn_"]:
            table[f"{prefix}vs30"], table[f"{prefix}stdv"] = model.combine_models(
                p_comb,
                table[f"geology_{prefix}vs30"],
                table[f"geology_{prefix}stdv"],
                table[f"terrain_{prefix}vs30"],
                table[f"terrain_{prefix}stdv"],
            )
    else:
        model.combine_tiff(p_paths.out, "combined.tif", p_grid, p_comb, *tiffs)
        model.combine_tiff(p_paths.out, "combined_mvn.tif", p_grid, p_comb, *tiffs_mvn)
    print(f"{time()-t:.2f}s")

# save point based data
sites.to_csv(os.path.join(p_paths.out, "measured_sites.csv"), na_rep="NA", index=False)
if p_ll is not None:
    table.to_csv(os.path.join(p_paths.out, "vs30points.csv"), na_rep="NA", index=False)

print("complete.")

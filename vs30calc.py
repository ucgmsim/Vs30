#!/usr/bin/env python
"""
Calculate Vs30 over a region (default) or specify points at which to output for instead.
"""

from shutil import rmtree
import os
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

wgs2nztm = Transformer.from_crs(4326, 2193, always_xy=True)
a = params.load_args()

# working directory/output setup
if os.path.exists(a["paths"].out):
    if a["paths"].overwrite:
        rmtree(a["paths"].out)
    else:
        sys.exit("output exists")
os.makedirs(a["paths"].out)

# input locations
if a["ll"] is not None:
    print("loading locations...")
    table = pd.read_csv(
        a["ll"].ll_path,
        usecols=(a["ll"].lon_col_ix, a["ll"].lat_col_ix),
        names=["longitude", "latitude"],
        engine="c",
        skiprows=a["ll"].skip_rows,
        dtype=np.float64,
        sep=a["ll"].col_sep,
    )
    table["easting"], table["northing"] = wgs2nztm.transform(
        table.longitude.values, table.latitude.values
    )
    table_points = table[["easting", "northing"]].values

# measured sites
print("loading measured sites...")
sites = sites_load.load_vs(source=a["sites"].source)
sites_points = np.column_stack((sites.easting.values, sites.northing.values))

# model loop
tiffs = []
tiffs_mvn = []
for s in [a["geol"], a["terr"]]:
    if s.update == "off":
        continue
    print(s.name, "model...")
    t = time()
    c = eval(f"model_{s.name}")

    print("    model measured update...")
    sites[f"{s.letter}id"] = c.model_id(sites_points, a["paths"], grid=a["grid"])
    if s.update == "prior":
        m = c.model_prior()
    elif s.update == "posterior_paper":
        m = c.model_posterior_paper()
    elif s.update == "posterior":
        m = c.model_prior()
        if a["sites"].source == "cpt":
            sites = sites_cluster.cluster(sites, s.letter)
            m = model.cluster_update(m, sites, s.letter)
        else:
            m = model.posterior(m, sites, f"{s.letter}id")

    print("    model at measured sites...")
    sites[f"{s.name}_vs30"], sites[f"{s.name}_stdv"] = c.model_val(
        sites[f"{s.letter}id"].values,
        m,
        s,
        paths=a["paths"],
        points=sites_points,
        grid=a["grid"],
    ).T

    if a["ll"] is not None:
        print("    model points...")
        table[f"{s.letter}id"] = c.model_id(table_points, a["paths"], a["grid"])
        table[f"{s.name}_vs30"], table[f"{s.name}_stdv"] = c.model_val(
            table[f"{s.letter}id"].values,
            m,
            s,
            paths=a["paths"],
            points=table_points,
            grid=a["grid"],
        ).T
        print("    measured mvn...")
        table[f"{s.name}_mvn_vs30"], table[f"{s.name}_mvn_stdv"] = mvn.mvn(
            table_points,
            table[f"{s.name}_vs30"],
            table[f"{s.name}_stdv"],
            sites,
            s.name,
        )
    else:
        print("    model map...")
        tiffs.append(c.model_val_map(a["paths"], a["grid"], m, s))
        print("    measured mvn...")
        tiffs_mvn.append(mvn.mvn_tiff(a["paths"], a["grid"], s.name, sites))

    print(f"{time()-t:.2f}s")

if a["geol"].update != "off" and a["terr"].update != "off":
    # combined model
    print("combining geology and terrain...")
    t = time()
    if a["ll"] is not None:
        for prefix in ["", "mvn_"]:
            table[f"{prefix}vs30"], table[f"{prefix}stdv"] = model.combine_models(
                a["comb"],
                table[f"geology_{prefix}vs30"],
                table[f"geology_{prefix}stdv"],
                table[f"terrain_{prefix}vs30"],
                table[f"terrain_{prefix}stdv"],
            )
    else:
        model.combine_tiff(a["paths"].out, "combined.tif", a["grid"], a["comb"], *tiffs)
        model.combine_tiff(
            a["paths"].out, "combined_mvn.tif", a["grid"], a["comb"], *tiffs_mvn
        )
    print(f"{time()-t:.2f}s")

# save point based data
sites.to_csv(
    os.path.join(a["paths"].out, "measured_sites.csv"), na_rep="NA", index=False
)
if a["ll"] is not None:
    table.to_csv(
        os.path.join(a["paths"].out, "vs30points.csv"), na_rep="NA", index=False
    )

print("complete.")

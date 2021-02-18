#!/usr/bin/env python

from argparse import ArgumentParser
from shutil import rmtree
import os, sys
from time import time

import numpy as np
import pandas as pd
from pyproj import Transformer

from vs30 import model, model_geology, model_terrain, mvn, sites_cluster, sites_load

PREFIX = "/mnt/nvme/work/plotting_data/Vs30/"
wgs2nztm = Transformer.from_crs(4326, 2193, always_xy=True)

parser = ArgumentParser()
arg = parser.add_argument
arg("--out", help="output location", default="./vs30map")
arg("--overwrite", help="overwrite output location", action="store_true")
arg("--mapdata", help="location to map sources", type=str, default=PREFIX)
# point options
arg("--ll", help="locations from file instead of running over a grid, space separated longitude latitude columns")
arg("--lon", help="ll file column containing longitude", type=int, default=0)
arg("--lat", help="ll file column containing latitude", type=int, default=1)
arg("--sep", help="ll file column separator", default=" ")
arg("--head", help="ll file rows to skip", type=int, default=0)
# grid options (used with points as well)
arg("--xmin", help="minimum easting", type=int, default=1000000)
arg("--xmax", help="maximum easting", type=int, default=2126400)
arg("--ymin", help="minimum northing", type=int, default=4700000)
arg("--ymax", help="maximum northing", type=int, default=6338400)
arg("--xd", help="horizontal spacing", type=int, default=100)
arg("--yd", help="vertical spacing", type=int, default=100)
# model update options
arg(
    "--gupdate",
    help="geology model updating",
    choices=["off", "prior", "posterior", "posterior_paper"],
    default="posterior_paper",
)
arg(
    "--tupdate",
    help="terrain model updating",
    choices=["off", "prior", "posterior", "posterior_paper"],
    default="posterior_paper",
)
# geology model has a few parametric processing options
parser.add_argument("--no-g6mod", dest="g6mod", action="store_false")
parser.add_argument("--no-g13mod", dest="g13mod", action="store_false")
parser.add_argument("--no-ghybrid", dest="ghybrid", action="store_false")
# combination arguments
parser.add_argument(
    "--stdv-weight",
    help="use standard deviation for model combination",
    action="store_true",
)
parser.add_argument(
    "--k", help="k factor for stdv based weight combination", type=float, default=1
)
# measured site arguments
parser.add_argument(
    "--cpt", help="use CPT based data for observed", action="store_true"
)
parser.add_argument("--no-downsample", dest="dsmcg", action="store_false")

# process arguments
args = parser.parse_args()
# add a few shared details and derivatives
args.nx = round((args.xmax - args.xmin) / args.xd)
args.ny = round((args.ymax - args.ymin) / args.yd)


# working directory/output setup
if os.path.exists(args.out):
    if args.overwrite:
        rmtree(args.out)
    else:
        sys.exit("output exists")
os.makedirs(args.out)

# input locations
if args.ll is not None:
    print("loading locations...")
    table = pd.read_csv(
        args.ll,
        usecols=(args.lon, args.lat),
        names=["longitude", "latitude"],
        engine="c",
        skiprows=args.head,
        dtype=np.float32,
        sep=args.sep,
    )
    table["easting"], table["northing"] = wgs2nztm.transform(
        table.longitude.values, table.latitude.values
    )
    table_points = table[["easting", "northing"]].values

# measured sites
print("loading sites...")
sites = sites_load.load_vs(cpt=args.cpt, downsample_mcgann=args.dsmcg)
points = np.column_stack((sites.easting.values, sites.northing.values))

# model loop
tiffs = []
tiffs_mvn = []
specs = [
    {"update": args.gupdate, "class": model_geology, "letter": "g", "name": "geology"},
    {"update": args.tupdate, "class": model_terrain, "letter": "t", "name": "terrain"},
]
for s in specs:
    if s["update"] != "off":
        print(s["name"], "model...")
        t = time()

        print("    model measured update...")
        sites[f'{s["letter"]}id'] = s["class"].model_id(points, args)
        if s["update"] == "prior":
            m = s["class"].model_prior()
        elif s["update"] == "posterior_paper":
            m = s["class"].model_posterior_paper()
        elif s["update"] == "posterior":
            m = s["class"].model_prior()
            if args.cpt:
                sites = sites_cluster.cluster(sites, s["letter"])
                m = model.cluster_update(m, sites, s["letter"])
            else:
                m = model.posterior(m, sites, f'{s["letter"]}id')

        print("    model at measured sites...")
        sites[f'{s["name"]}_vs30'], sites[f'{s["name"]}_stdv'] = (
            s["class"]
            .model_val(sites[f'{s["letter"]}id'], m, args=args, points=points)
            .T
        )

        if args.ll is not None:
            print("    model points...")
            table[f'{s["letter"]}id'] = s["class"].model_id(table_points, args)
            table[f'{s["name"]}_vs30'], table[f'{s["name"]}_stdv'] = (
                s["class"]
                .model_val(
                    table[f'{s["letter"]}id'].values, m, args=args, points=table_points
                )
                .T
            )
            print("    measured mvn...")
            table[f'{s["name"]}_mvn_vs30'], table[f'{s["name"]}_mvn_stdv'] = mvn.mvn(
                table_points,
                table[f'{s["name"]}_vs30'],
                table[f'{s["name"]}_stdv'],
                sites,
                s["name"],
            )
        else:
            print("    model map...")
            tiffs.append(s["class"].model_val_map(args, m))
            print("    measured mvn...")
            tiffs_mvn.append(mvn.mvn_tiff(args, s["name"], sites))

        print(f"{time()-t:.2f}s")

if args.gupdate != "off" and args.tupdate != "off":
    # combined model
    print("combining geology and terrain...")
    t = time()
    if args.ll is not None:
        for prefix in ["", "mvn_"]:
            table[f"{prefix}vs30"], table[f"{prefix}stdv"] = model.combine(
                args,
                table[f"geology_{prefix}vs30"],
                table[f"geology_{prefix}stdv"],
                table[f"terrain_{prefix}vs30"],
                table[f"terrain_{prefix}stdv"],
            )
    else:
        model.combine_tiff(args, *tiffs)
        model.combine_tiff(args, *tiffs_mvn)
    print(f"{time()-t:.2f}s")

# save point based data
sites.to_csv(os.path.join(args.out, "measured_sites.csv"), na_rep="NA", index=False)
if args.ll is not None:
    table.to_csv(os.path.join(args.out, "vs30points.csv"), na_rep="NA", index=False)

print("complete.")

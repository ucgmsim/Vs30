#!/usr/bin/env python

from argparse import ArgumentParser
from shutil import rmtree
import os, sys
from time import time

import numpy as np

from vs30 import model, model_geology, model_terrain, sites_cluster, sites_load

PREFIX = "/run/media/vap30/Hathor/work/plotting_data/Vs30/"

parser = ArgumentParser()
arg = parser.add_argument
arg("--wd", help="output location", default="./vs30map")
arg("--overwrite", help="overwrite output location", action="store_true")
arg("--mapdata", help="location to map sources", type=str, default=PREFIX)
# grid options
arg("--xmin", help="minimum easting", type=int, default=1000000)
arg("--xmax", help="maximum easting", type=int, default=2126400)
arg("--ymin", help="minimum northing", type=int, default=4700000)
arg("--ymax", help="maximum northing", type=int, default=6338400)
arg("--xd", help="horizontal spacing", type=int, default=100)
arg("--yd", help="vertical spacing", type=int, default=100)
# model update options
arg("--gupdate", help="geology model updating", choices=["off", "prior", "posterior", "posterior_paper"], default="posterior_paper")
arg("--tupdate", help="terrain model updating", choices=["off", "prior", "posterior", "posterior_paper"], default="posterior_paper")
# geology model has a few parametric processing options
parser.add_argument('--g6mod', dest='g6mod', action='store_true')
parser.add_argument('--no-g6mod', dest='g6mod', action='store_false')
parser.set_defaults(g6mod=True)
parser.add_argument('--g13mod', dest='g13mod', action='store_true')
parser.add_argument('--no-g13mod', dest='g13mod', action='store_false')
parser.set_defaults(g13mod=True)
parser.add_argument('--ghybrid', dest='ghybrid', action='store_true')
parser.add_argument('--no-ghybrid', dest='ghybrid', action='store_false')
parser.set_defaults(ghybrid=True)
# combination arguments
parser.add_argument('--stdv-weight', help="use standard deviation for model combination", action='store_true')
parser.add_argument('--k', help="k factor for stdv based weight combination", type=float, default=1)
# measured site arguments
parser.add_argument('--cpt', help="use CPT based data for observed", action="store_true")
parser.add_argument('--downsample', dest='dsmcg', action='store_true')
parser.add_argument('--no-downsample', dest='dsmcg', action='store_false')
parser.set_defaults(dsmcg=True)

# process arguments
args = parser.parse_args()
# add a few shared details and derivatives
args.nx = round((args.xmax - args.xmin) / args.xd)
args.ny = round((args.ymax - args.ymin) / args.yd)


# working directory/output setup
if os.path.exists(args.wd):
    if args.overwrite:
        rmtree(args.wd)
    else:
        sys.exit("output exists")
os.makedirs(args.wd)

# measured sites
print("loading sites...")
sites = sites_load.load_vs(cpt=args.cpt, downsample_mcgann=args.dsmcg)
points = np.column_stack((sites.easting.values, sites.northing.values))

# model loop
tiffs = []
specs = [
    {"update":args.gupdate, "class":model_geology, "letter":"g", "name":"geology"},
    {"update":args.tupdate, "class":model_terrain, "letter":"t", "name":"terrain"},
]
for s in specs:
    if s["update"] != "off":
        print(s["name"], "map...")
        t = time()

        # load model
        sites[f'{s["letter"]}id'] = s["class"].mid(points, args)
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

        # run model for map
        tiffs.append(s["class"].model_map(args, m))
        print(f"{time()-t:.2f}s")

if args.gupdate != "off" and args.tupdate != "off":
    # combined model
    print("combining geology and terrain...")
    t = time()
    model.combine(args, *tiffs)
    print(f"{time()-t:.2f}s")

# save sites

print("complete.")

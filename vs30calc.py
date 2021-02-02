#!/usr/bin/env python

from argparse import ArgumentParser
from tempfile import mkdtemp

from vs30 import model_geology, model_terrain, model_combine

PREFIX = "/run/media/vap30/Hathor/work/plotting_data/Vs30/"

parser = ArgumentParser()
arg = parser.add_argument
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
# process arguments
args = parser.parse_args()
# add a few shared details and derivatives
args.wd = mkdtemp()
args.nx = round((args.xmax - args.xmin) / args.xd)
args.ny = round((args.ymax - args.ymin) / args.yd)

if args.gupdate != "off":
    # geology model
    model = model_geology.model_posterior_paper()
    g_tiff = model_geology.model_map(args, model)
if args.tupdate != "off":
    # terrain model
    model = model_terrain.model_posterior_paper()
    t_tiff = model_terrain.model_map(args, model)
if args.gupdate != "off" and args.tupdate != "off":
    # combined model
    model_combine.combine(args, g_tiff, t_tiff)


print(args.wd)

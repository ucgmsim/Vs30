#!/usr/bin/env python
"""
Plots Vs30 output maps.
For netCDF outputs, not GeoTiff
For a fast plot, set --oversample 1 --dpi 600
"""

from argparse import ArgumentParser
import os
from shutil import rmtree
from tempfile import mkdtemp

from qcore import gmt

# standard vs30 nz wide region
REGION = (1000000, 2126400, 4700000, 6338400)
# topography illumination file on standard vs30 nz wide region
ILLU = "/nesi/project/nesi00213/PlottingData/Topo/vsmapgrid_ngb_i5.nc"

parser = ArgumentParser()
parser.add_argument("map_file", help="path to map file")
parser.add_argument(
    "--out", help="output png path excluding extension", default="vs30map"
)
parser.add_argument("--title", help="set plot title", default="Vs30 Map")
parser.add_argument(
    "--width", help="map width, determines font/line sizing", type=float, default=6
)
parser.add_argument("--dpi", help="resolution", type=int, default=1200)
parser.add_argument("--noillu", help="no topo illumination", action="store_true")
parser.add_argument(
    "--oversample", help="1-8 lowest to highest quality", type=int, default=8
)
args = parser.parse_args()
assert os.path.isfile(args.map_file)

wd = mkdtemp()
cpt = os.path.join(wd, "map.cpt")
ps = os.path.join(wd, "vs30plot.ps")
height = (REGION[3] - REGION[2]) / (REGION[1] - REGION[0]) * args.width

gmt.makecpt("rainbow", cpt, 180, 750, invert=True, continuing=True, continuous=True)
p = gmt.GMTPlot(ps)
p.spacial(
    "X", REGION, sizing="%s/%si" % (args.width, height), x_shift="1i", y_shift="1.5i"
)
if args.noillu:
    p.overlay(args.map_file, cpt)
else:
    p.topo(args.map_file, topo_file_illu=ILLU, cpt=cpt)
p.ticks(major=500000, minor=100000)
p.text((REGION[0] + REGION[1]) / 2, REGION[3], args.title, size="24p", dy="0.2i")
p.cpt_scale(
    "C", "B", cpt, pos="rel_out", dy="0.5i", major=100, minor=20, label="Vs30 [m/s]"
)
p.finalise()
p.png(
    background="white",
    dpi=args.dpi * args.oversample,
    downscale=args.oversample,
    out_name=os.path.abspath(args.out),
)

rmtree(wd)

from argparse import ArgumentParser
from dataclasses import dataclass

PREFIX = "/mnt/nvme/work/plotting_data/Vs30/"


@dataclass
class Paths:
    """
    Locations to things.
    """

    out: str = "./vs30map"
    overwrite: bool = False
    mapdata: str = PREFIX


@dataclass
class Sites:
    """
    Parameters for measured sites.
    """

    choices = ["original", "cpt"]

    source: str = "original"


@dataclass
class Grid:
    """
    Grid parameters for rasterization.
    """

    xmin: int = 1060050
    xmax: int = 2120050
    dx: int = 100
    ymin: int = 4730050
    ymax: int = 6250050
    dy: int = 100

    def __post_init__(self):
        self.update()

    def update(self):
        self.nx = round((self.xmax - self.xmin) / self.dx)
        self.ny = round((self.ymax - self.ymin) / self.dy)


@dataclass
class LLFile:
    """
    Long/Lat file for locations of interest.
    """

    ll: str
    lon: int = 0
    lat: int = 1
    sep: str = " "
    head: int = 0


@dataclass
class Geology:
    """
    Geology model parameters.
    """

    letter = "g"
    name = "geology"
    choices = ["off", "prior", "posterior", "posterior_paper"]

    update: str = "posterior_paper"
    hybrid: bool = True
    mod6: bool = True
    mod13: bool = True


@dataclass
class Terrain:
    """
    Terrain model parameters.
    """

    letter = "t"
    name = "terrain"
    choices = ["off", "prior", "posterior", "posterior_paper"]

    update: str = "posterior_paper"


@dataclass
class Combination:
    """
    Model combination parameters.
    """

    stdv_weight: bool = False
    k: float = -1.0


def load_args():
    parser = ArgumentParser()
    arg = parser.add_argument
    arg("--out", help="output location", type=type(Paths.out), default=Paths.out)
    arg("--overwrite", help="overwrite output location", action="store_true")
    arg(
        "--mapdata",
        help="location to map sources",
        type=type(Paths.mapdata),
        default=Paths.mapdata,
    )
    # point options
    arg(
        "--ll",
        help="locations from file instead of running over a grid, space separated longitude latitude columns",
    )
    arg(
        "--lon",
        help="ll file column containing longitude",
        type=type(LLFile.lon),
        default=LLFile.lon,
    )
    arg(
        "--lat",
        help="ll file column containing latitude",
        type=type(LLFile.lat),
        default=LLFile.lat,
    )
    arg(
        "--sep",
        help="ll file column separator",
        type=type(LLFile.sep),
        default=LLFile.sep,
    )
    arg(
        "--head",
        help="ll file rows to skip",
        type=type(LLFile.head),
        default=LLFile.head,
    )
    # grid options (used with points as well)
    arg("--xmin", help="minimum easting", type=type(Grid.xmin), default=Grid.xmin)
    arg("--xmax", help="maximum easting", type=type(Grid.xmax), default=Grid.xmax)
    arg("--dx", help="horizontal spacing", type=type(Grid.dx), default=Grid.dx)
    arg("--ymin", help="minimum northing", type=type(Grid.ymin), default=Grid.ymin)
    arg("--ymax", help="maximum northing", type=type(Grid.ymax), default=Grid.ymax)
    arg("--dy", help="vertical spacing", type=type(Grid.dy), default=Grid.dy)
    # model update options
    arg(
        "--gupdate",
        help="geology model updating",
        choices=Geology.choices,
        default=Geology.update,
    )
    arg(
        "--tupdate",
        help="terrain model updating",
        choices=Terrain.choices,
        default=Terrain.update,
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
        "--k",
        help="k factor for stdv based weight combination",
        type=type(Combination.k),
        default=Combination.k,
    )
    # measured site arguments
    arg(
        "--source",
        help="measured site dataset",
        choices=Sites.choices,
        default=Sites.source,
    )

    # process arguments
    args = parser.parse_args()
    # argument sets
    p_paths = Paths(out=args.out, overwrite=args.overwrite, mapdata=args.mapdata)
    p_sites = Sites(source=args.source)
    p_grid = Grid(
        xmin=args.xmin,
        xmax=args.xmax,
        dx=args.dx,
        ymin=args.ymin,
        ymax=args.ymax,
        dy=args.dy,
    )
    if args.ll is not None:
        p_ll = LLFile(args.ll, lon=args.lon, lat=args.lat, sep=args.sep, head=args.head)
    else:
        p_ll = None
    p_geol = Geology(
        hybrid=args.ghybrid, mod6=args.g6mod, mod13=args.g13mod, update=args.gupdate
    )
    p_terr = Terrain(update=args.tupdate)
    p_comb = Combination(stdv_weight=args.stdv_weight, k=args.k)

    return {
        "paths": p_paths,
        "sites": p_sites,
        "grid": p_grid,
        "ll": p_ll,
        "geol": p_geol,
        "terr": p_terr,
        "comb": p_comb,
    }

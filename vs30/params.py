"""
Parameters used throughout the vs30 package.
Also parses command line arguments for vs30calc.py.
"""
from argparse import ArgumentParser
from dataclasses import dataclass
from multiprocessing import cpu_count


@dataclass
class PathsParams:
    """
    Locations to resources, ouputs and anything else.
    """

    out: str = "./vs30map"
    overwrite: bool = False


@dataclass
class SitesParams:
    """
    Parameters for measured sites.
    """

    choices = ["original", "cpt"]

    source: str = "original"


@dataclass
class GridParams:
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
        """
        Updates number of columns and rows given other parameters.
        """
        self.nx = round((self.xmax - self.xmin) / self.dx)
        self.ny = round((self.ymax - self.ymin) / self.dy)


@dataclass
class LLFileParams:
    """
    Parameters for loading long/lat locations of interest file.
    """

    ll_path: str
    lon_col_ix: int = 0
    lat_col_ix: int = 1
    col_sep: str = " "
    skip_rows: int = 0


@dataclass
class GeologyParams:
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
class TerrainParams:
    """
    Terrain model parameters.
    """

    letter = "t"
    name = "terrain"
    choices = ["off", "prior", "posterior", "posterior_paper"]

    update: str = "posterior_paper"


@dataclass
class CombinationParams:
    """
    Model combination parameters.
    """

    stdv_weight: bool = False
    k: float = -1.0


def load_args():
    """
    Load arguments from command line.
    """
    parser = ArgumentParser()
    arg = parser.add_argument
    arg("--nproc", help="number of processes to use", type=int, default=cpu_count())
    arg(
        "--out",
        help="output location",
        type=type(PathsParams.out),
        default=PathsParams.out,
    )
    arg("--overwrite", help="overwrite output location", action="store_true")
    # point options
    arg(
        "--ll-path",
        help="locations from file instead of running over a grid, space separated longitude latitude columns",
    )
    arg(
        "--lon-col-ix",
        help="ll file column containing longitude",
        type=type(LLFileParams.lon_col_ix),
        default=LLFileParams.lon_col_ix,
    )
    arg(
        "--lat-col-ix",
        help="ll file column containing latitude",
        type=type(LLFileParams.lat_col_ix),
        default=LLFileParams.lat_col_ix,
    )
    arg(
        "--col-sep",
        help="ll file column separator",
        type=type(LLFileParams.col_sep),
        default=LLFileParams.col_sep,
    )
    arg(
        "--skip-rows",
        help="ll file rows to skip",
        type=type(LLFileParams.skip_rows),
        default=LLFileParams.skip_rows,
    )
    # grid options (used with points as well)
    arg(
        "--xmin",
        help="minimum easting",
        type=type(GridParams.xmin),
        default=GridParams.xmin,
    )
    arg(
        "--xmax",
        help="maximum easting",
        type=type(GridParams.xmax),
        default=GridParams.xmax,
    )
    arg(
        "--dx",
        help="horizontal spacing",
        type=type(GridParams.dx),
        default=GridParams.dx,
    )
    arg(
        "--ymin",
        help="minimum northing",
        type=type(GridParams.ymin),
        default=GridParams.ymin,
    )
    arg(
        "--ymax",
        help="maximum northing",
        type=type(GridParams.ymax),
        default=GridParams.ymax,
    )
    arg(
        "--dy", help="vertical spacing", type=type(GridParams.dy), default=GridParams.dy
    )
    # model update options
    arg(
        "--gupdate",
        help="geology model updating",
        choices=GeologyParams.choices,
        default=GeologyParams.update,
    )
    arg(
        "--tupdate",
        help="terrain model updating",
        choices=TerrainParams.choices,
        default=TerrainParams.update,
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
        type=type(CombinationParams.k),
        default=CombinationParams.k,
    )
    # measured site arguments
    arg(
        "--source",
        help="measured site dataset",
        choices=SitesParams.choices,
        default=SitesParams.source,
    )

    # process arguments
    args = parser.parse_args()
    # argument sets
    p_paths = PathsParams(out=args.out, overwrite=args.overwrite)
    p_sites = SitesParams(source=args.source)
    p_grid = GridParams(
        xmin=args.xmin,
        xmax=args.xmax,
        dx=args.dx,
        ymin=args.ymin,
        ymax=args.ymax,
        dy=args.dy,
    )
    if args.ll_path is not None:
        p_ll = LLFileParams(
            args.ll_path,
            lon_col_ix=args.lon_col_ix,
            lat_col_ix=args.lat_col_ix,
            col_sep=args.col_sep,
            skip_rows=args.skip_rows,
        )
    else:
        p_ll = None
    p_geol = GeologyParams(
        hybrid=args.ghybrid, mod6=args.g6mod, mod13=args.g13mod, update=args.gupdate
    )
    p_terr = TerrainParams(update=args.tupdate)
    p_comb = CombinationParams(stdv_weight=args.stdv_weight, k=args.k)

    return p_paths, p_sites, p_grid, p_ll, p_geol, p_terr, p_comb, args.nproc

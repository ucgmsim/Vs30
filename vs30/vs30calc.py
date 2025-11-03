import time
from pathlib import Path
from functools import partial
import logging
import math
from multiprocessing import Pool
import os
from shutil import copyfile, rmtree
import sys

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

WGS2NZTM = Transformer.from_crs(4326, 2193, always_xy=True)


def array_split(
    array: np.ndarray, n_proc: int, chunk_size: int = 50
) -> list[np.ndarray]:
    """
    Split dataframes and numpy arrays for multiprocessing.Pool.map
    """
    n_chunks = math.ceil(max(n_proc, len(array) / chunk_size))
    return np.array_split(array, n_chunks)


def run_vs30calc(
    p_paths: params.PathsParams,
    p_sites: params.SitesParams,
    p_grid: params.GridParams | None,
    p_ll: params.LLFileParams | None,
    p_geol: params.GeologyParams,
    p_terr: params.TerrainParams,
    p_comb: params.CombinationParams,
    n_procs: int,
):
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        level=logging.INFO,
    )
    logger = logging.getLogger(__name__)
    logger.info(f"Running Vs30Calc with {n_procs} processes")

    # working directory/output setup
    if os.path.exists(p_paths.out):
        if p_paths.overwrite:
            rmtree(p_paths.out)
        else:
            sys.exit("output exists")
    os.makedirs(p_paths.out)

    # input locations
    if p_ll is not None:
        logger.info("loading locations...")
        table = pd.read_csv(
            p_ll.ll_path,
            usecols=(p_ll.lon_col_ix, p_ll.lat_col_ix),
            names=["longitude", "latitude"],
            engine="c",
            skiprows=p_ll.skip_rows,
            dtype=np.float64,
            sep=p_ll.col_sep,
        )
        table["easting"], table["northing"] = WGS2NZTM.transform(
            table.longitude.values, table.latitude.values
        )
        table_points = table[["easting", "northing"]].values

    # measured sites
    logger.info("loading measured sites...")
    measured_sites = sites_load.load_vs(source=p_sites.source)
    measured_sites_points = np.column_stack(
        (measured_sites.easting.values, measured_sites.northing.values)
    )

    MODEL_MAPPING = {"geology": model_geology, "terrain": model_terrain}

    # model loop
    tiffs = []
    tiffs_mvn = []
    for model_setup in [p_geol, p_terr]:
        if model_setup.update == "off":
            continue
        logger.info(f"{model_setup.name} model...")
        model_module = MODEL_MAPPING[model_setup.name]

        logger.info("    model measured update...")
        start_time = time.time()
        measured_sites[f"{model_setup.letter}id"] = model_module.model_id(measured_sites_points)
        if model_setup.update == "prior":
            model_table = model_module.model_prior()
        elif model_setup.update == "posterior_paper":
            model_table = model_module.model_posterior_paper()
        elif model_setup.update == "posterior":
            model_table = model_module.model_prior()
            if p_sites.source == "cpt":
                measured_sites = sites_cluster.cluster(
                    measured_sites, model_setup.letter, nproc=n_procs
                )
                model_table = model.cluster_update(
                    model_table, measured_sites, model_setup.letter
                )
            else:
                model_table = model.posterior(
                    model_table, measured_sites, f"{model_setup.letter}id"
                )
        logger.info(f"    took: {time.time()-start_time:.2f}s")

        logger.info("    model at measured sites...")
        start_time = time.time()
        (
            measured_sites[f"{model_setup.name}_vs30"],
            measured_sites[f"{model_setup.name}_stdv"],
        ) = model_module.model_val(
            measured_sites[f"{model_setup.letter}id"].values,
            model_table,
            model_setup,
            paths=p_paths,
            points=measured_sites_points,
            grid=p_grid,
        ).T
        logger.info(f"    took: {time.time()-start_time:.2f}s")

        if p_ll is not None:
            logger.info("    model points...")
            # if n_procs == 1:
            start = time.time()
            table[f"{model_setup.letter}id"] = model_module.model_id(
                table_points
            )
            logger.info(f"    took: {time.time() - start} ")
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

            logger.info("    measured mvn...")
            start_time = time.time()
            if n_procs == 1:
                (
                    table[f"{model_setup.name}_mvn_vs30"],
                    table[f"{model_setup.name}_mvn_stdv"],
                ) = mvn.mvn_table(
                    table,
                    sites=measured_sites,
                    model_name=model_setup.name,
                ).T
            else:
                with Pool(n_procs) as pool:
                    result = pool.map(
                            partial(
                                mvn.mvn_table,
                                sites=measured_sites,
                                model_name=model_setup.name,
                            ),
                            array_split(table, n_procs, 10_000),
                        )
                    (
                        table[f"{model_setup.name}_mvn_vs30"],
                        table[f"{model_setup.name}_mvn_stdv"],
                    ) = np.concatenate(
                        result,
                    ).T
            logger.info(f"    took: {time.time()-start_time:.2f}s")
        else:
            logger.info("    model map...")
            tiffs.append(
                model_module.model_val_map(p_paths, p_grid, model_table, model_setup)
            )
            logger.info("    measured mvn...")
            tiffs_mvn.append(
                mvn.mvn_tiff(p_paths.out, model_setup.name, measured_sites, n_procs)
            )

    if p_geol.update != "off" and p_terr.update != "off":
        # combined model
        logger.info("combining geology and terrain...")
        start_time = time.time()
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
            model.combine_tiff(
                p_paths.out, "combined_mvn.tif", p_grid, p_comb, *tiffs_mvn
            )
        logger.info(f"{time.time()-start_time:.2f}s")

    # save point based data and copy qgis project files
    measured_sites.to_csv(
        os.path.join(p_paths.out, "measured_sites.csv"), na_rep="NA", index=False
    )
    if p_ll is not None:
        table.to_csv(
            os.path.join(p_paths.out, "vs30points.csv"), na_rep="NA", index=False
        )
        copyfile(
            os.path.join(sites_load.data, "qgis_points.qgz"),
            os.path.join(p_paths.out, "qgis.qgz"),
        )
    else:
        copyfile(
            os.path.join(sites_load.data, "qgis_rasters.qgz"),
            os.path.join(p_paths.out, "qgis.qgz"),
        )

    logger.info("complete.")

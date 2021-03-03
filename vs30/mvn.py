"""
MVN (multivariate normal distribution)
for modifying vs30 values based on proximity to measured values.
"""
import math
import os
from shutil import copyfile

import numpy as np
from osgeo import gdal


def corr_func(distances, model):
    """
    Correlation function by distance.
    phi is 993 for terrain model, 1407 for geology
    """
    if model == "geology":
        phi = 1407
    elif model == "terrain":
        phi = 993
    else:
        raise ValueError("unknown model")
    # r code linearly interpolated from logarithmically spaced distances:
    # d = np.exp(np.linspace(np.log(0.1), np.log(2000e3), 128))
    # c = 1 / np.e ** (d / phi)
    # return np.interp(distances, d, c)
    # minimum distance of 0.1 metres enforced
    return 1 / np.e ** (np.maximum(0.1, distances) / phi)


def tcrossprod(x):
    """
    Matrix cross product (or outer product) from a 1d numpy array.
    Same functionality as the R function tcrossprod(x) with y = NULL.
    https://stat.ethz.ch/R-manual/R-devel/library/base/html/crossprod.html
    """
    return x[:, np.newaxis] * x


def dists(x):
    """
    Euclidean distance from 2d diff array.
    """
    # return np.linalg.norm(x, axis=1)
    # alternative, may be faster
    return np.sqrt(np.einsum("ij,ij->i", x, x))


def xy2complex(x):
    """
    Convert array of 2D coordinates to array of 1D complex numbers.
    """
    c = x[:, 0].astype(np.complex64)
    c.imag += x[:, 1]
    return c


def dist_mat(x):
    """
    Distance matrix between coordinates (complex numbers) or simple values.
    """
    return np.abs(x[:, np.newaxis] - x)


def mvn(
    model_locs,
    model_vs30,
    model_stdv,
    sites,
    model_name,
    cov_reduc=1.5,
    noisy=True,
    max_dist=10000,
):
    """
    Modify model with observed locations.
    noisy: noisy measurements
    """
    # cut not-applicable sites to prevent nan propagation
    sites = sites[~np.isnan(sites[f"{model_name}_vs30"])]

    obs_locs = np.column_stack((sites.easting.values, sites.northing.values))
    obs_model_stdv = sites[f"{model_name}_stdv"].values
    obs_residuals = np.log(sites.vs30.values / sites[f"{model_name}_vs30"].values)

    # Wea equation 33, 40, 41
    if noisy:
        omega_obs = np.sqrt(
            obs_model_stdv ** 2 / (obs_model_stdv ** 2 + sites.uncertainty.values ** 2)
        )
        obs_residuals *= omega_obs

    # default outputs if no sites closeby
    pred = np.log(model_vs30)
    var = model_stdv ** 2 * corr_func(0, model_name)

    # model point to observations
    for i, model_loc in enumerate(model_locs):
        if np.isnan(model_vs30[i]):
            continue
        distances = dists(obs_locs - model_loc)
        wanted = distances < max_dist
        if max(wanted) is False:
            # not close enough to any observed locations
            continue

        # distances between interesting points
        cov_matrix = dist_mat(xy2complex(np.vstack((model_loc, obs_locs[wanted]))))
        # correlation
        cov_matrix = corr_func(cov_matrix, model_name)
        # uncertainties
        cov_matrix *= tcrossprod(np.insert(obs_model_stdv[wanted], 0, model_stdv[i]))

        if noisy:
            omega = tcrossprod(np.insert(omega_obs[wanted], 0, 1))
            np.fill_diagonal(omega, 1)
            cov_matrix *= omega

        # covariance reduction factors
        if cov_reduc > 0:
            cov_matrix *= np.exp(
                -cov_reduc
                * dist_mat(
                    np.insert(
                        np.log(sites.loc[wanted, f"{model_name}_vs30"].values),
                        0,
                        pred[i],
                    )
                )
            )

        inv_matrix = np.linalg.inv(cov_matrix[1:, 1:])
        pred[i] += np.dot(np.dot(cov_matrix[0, 1:], inv_matrix), obs_residuals[wanted])
        var[i] = cov_matrix[0, 0] - np.dot(
            np.dot(cov_matrix[0, 1:], inv_matrix), cov_matrix[1:, 0]
        )

    return model_vs30 * np.exp(pred - np.log(model_vs30)), np.sqrt(var)


def mvn_table(table, sites, model_name):
    """
    Run MVN over DataFrame. multiprocessing.Pool.map friendly.
    """
    # reset indexes for this instance to prevent index errors with split table
    ix0_table = table.reset_index(drop=True)
    return np.column_stack(
        mvn(
            ix0_table[["easting", "northing"]].values,
            ix0_table[f"{model_name}_vs30"],
            ix0_table[f"{model_name}_stdv"],
            sites,
            model_name,
        )
    )


def mvn_tiff(paths, grid, model, sites):
    """
    Run MVN over GeoTIFF.
    """
    # mvn based on original model, modified if in proximity to measured sites
    out_tiff = os.path.join(paths.out, f"{model}_mvn.tif")
    copyfile(os.path.join(paths.out, f"{model}.tif"), out_tiff)
    ds = gdal.Open(out_tiff, gdal.GA_Update)
    trans = ds.GetGeoTransform()
    vs30b = ds.GetRasterBand(1)
    stdvb = ds.GetRasterBand(2)
    vnd = vs30b.GetNoDataValue()
    snd = stdvb.GetNoDataValue()

    # processing chunk/block sizing
    block = vs30b.GetBlockSize()
    nxb = (int)((grid.nx + block[0] - 1) / block[0])
    nyb = (int)((grid.ny + block[1] - 1) / block[1])

    for x in range(nxb):
        xoff = x * block[0]
        # last block may be smaller
        if x == nxb - 1:
            block[0] = grid.nx - x * block[0]
        # reset y block size
        block_y = block[1]

        for y in range(nyb):
            yoff = y * block[1]
            # last block may be smaller
            if y == nyb - 1:
                block_y = grid.ny - y * block[1]

            # determine results on block
            vs30v = vs30b.ReadAsArray(
                xoff=xoff, yoff=yoff, win_xsize=block[0], win_ysize=block_y
            )
            vs30v[vs30v == vnd] = np.nan
            stdvv = stdvb.ReadAsArray(
                xoff=xoff, yoff=yoff, win_xsize=block[0], win_ysize=block_y
            )
            stdvv[stdvv == snd] = np.nan

            locs = np.vstack(
                np.mgrid[
                    trans[0]
                    + (xoff + 0.5) * trans[1] : trans[0]
                    + (xoff + 0.5 + block[0]) * trans[1] : trans[1],
                    trans[3]
                    + (yoff + 0.5) * trans[5] : trans[3]
                    + (yoff + 0.5 + block_y) * trans[5] : trans[5],
                ].T
            ).astype(np.float32)
            vs30v, stdvv = mvn(locs, vs30v.flatten(), stdvv.flatten(), sites, model)

            # write results
            vs30b.WriteArray(vs30v.reshape(block_y, block[0]), xoff=xoff, yoff=yoff)
            stdvb.WriteArray(stdvv.reshape(block_y, block[0]), xoff=xoff, yoff=yoff)
    # close
    vs30b = None
    stdvb = None
    ds = None
    return out_tiff

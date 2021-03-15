"""
MVN (multivariate normal distribution)
for modifying vs30 values based on proximity to measured values.
"""
from functools import partial
from multiprocessing import Pool
import os
from shutil import copyfile

import numpy as np
from osgeo import gdal


def _corr_func(distances, model):
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


def _tcrossprod(x):
    """
    Matrix cross product (or outer product) from a 1d numpy array.
    Same functionality as the R function tcrossprod(x) with y = NULL.
    https://stat.ethz.ch/R-manual/R-devel/library/base/html/crossprod.html
    """
    return x[:, np.newaxis] * x


def _dists(x):
    """
    Euclidean distance from 2d diff array.
    """
    # return np.linalg.norm(x, axis=1)
    # alternative, may be faster
    return np.sqrt(np.einsum("ij,ij->i", x, x))


def _xy2complex(x):
    """
    Convert array of 2D coordinates to array of 1D complex numbers.
    """
    c = x[:, 0].astype(np.complex64)
    c.imag += x[:, 1]
    return c


def _dist_mat(x):
    """
    Distance matrix between coordinates (complex numbers) or simple values.
    """
    return np.abs(x[:, np.newaxis] - x)


def _mvn(
    model_locs,
    model_vs30,
    model_stdv,
    sites,
    model_name,
    cov_reduc=1.5,
    noisy=True,
    max_dist=10000,
    max_points=500,
):
    """
    Modify model with observed locations.
    noisy: whether measurements are noisy True/False
    max_dist: only consider observed locations within this many metres
    max_points: limit observed locations to closest N points within max_dist
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
    var = model_stdv ** 2 * _corr_func(0, model_name)

    # model point to observations
    for i, model_loc in enumerate(model_locs):
        if np.isnan(model_vs30[i]):
            # input of nan is always output of nan
            continue
        # don't recalculate distances if delta distance is too small anyway
        # useful when calculating accross grids or close points
        try:
            movement = _dists(np.atleast_2d(model_loc - prev_model_loc))[0]
            if min_dist - movement > max_dist:
                continue
        except NameError:
            pass
        distances = _dists(obs_locs - model_loc)
        max_points_i = min(max_points, len(distances)) - 1
        min_dist, cutoff_dist = np.partition(distances, [0, max_points_i])[
            [0, max_points_i]
        ]
        prev_model_loc = model_loc
        if min_dist > max_dist:
            # not close enough to any observed locations
            continue
        loc_mask = distances <= min(max_dist, cutoff_dist)

        # distances between interesting points
        cov_matrix = _dist_mat(_xy2complex(np.vstack((model_loc, obs_locs[loc_mask]))))
        # correlation
        cov_matrix = _corr_func(cov_matrix, model_name)
        # uncertainties
        cov_matrix *= _tcrossprod(np.insert(obs_model_stdv[loc_mask], 0, model_stdv[i]))

        if noisy:
            omega = _tcrossprod(np.insert(omega_obs[loc_mask], 0, 1))
            np.fill_diagonal(omega, 1)
            cov_matrix *= omega

        # covariance reduction factors
        if cov_reduc > 0:
            cov_matrix *= np.exp(
                -cov_reduc
                * _dist_mat(
                    np.insert(
                        np.log(sites.loc[loc_mask, f"{model_name}_vs30"].values),
                        0,
                        pred[i],
                    )
                )
            )

        inv_matrix = np.linalg.inv(cov_matrix[1:, 1:])
        pred[i] += np.dot(
            np.dot(cov_matrix[0, 1:], inv_matrix), obs_residuals[loc_mask]
        )
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
        _mvn(
            ix0_table[["easting", "northing"]].values,
            ix0_table[f"{model_name}_vs30"],
            ix0_table[f"{model_name}_stdv"],
            sites,
            model_name,
        )
    )


def _mvn_tiff_worker(tif_path, x_offset, y_offset, x_size, y_size, sites, model_name):
    """
    Works on single tif block as split by mvn_tiff.
    """
    # load tif
    tif_ds = gdal.Open(tif_path, gdal.GA_ReadOnly)
    tif_trans = tif_ds.GetGeoTransform()
    vs30_band = tif_ds.GetRasterBand(1)
    stdv_band = tif_ds.GetRasterBand(2)
    vs30_nd = vs30_band.GetNoDataValue()
    stdv_nd = stdv_band.GetNoDataValue()

    # read pre-mvn data from tif
    vs30_val = vs30_band.ReadAsArray(
        xoff=x_offset, yoff=y_offset, win_xsize=x_size, win_ysize=y_size
    ).flatten()
    vs30_val[vs30_val == vs30_nd] = np.nan
    stdv_val = stdv_band.ReadAsArray(
        xoff=x_offset, yoff=y_offset, win_xsize=x_size, win_ysize=y_size
    ).flatten()
    stdv_val[stdv_val == stdv_nd] = np.nan
    # coordinates for tif data
    locs = np.vstack(
        np.mgrid[
            tif_trans[0]
            + (x_offset + 0.5) * tif_trans[1] : tif_trans[0]
            + (x_offset + 0.5 + x_size) * tif_trans[1] : tif_trans[1],
            tif_trans[3]
            + (y_offset + 0.5) * tif_trans[5] : tif_trans[3]
            + (y_offset + 0.5 + y_size) * tif_trans[5] : tif_trans[5],
        ].T
    ).astype(np.float32)

    # close tif
    vs30_band = None
    stdv_band = None
    tif_ds = None

    # calculate mvn
    vs30_mvn, stdv_mvn = _mvn(locs, vs30_val, stdv_val, sites, model_name)
    return (
        x_offset,
        y_offset,
        vs30_mvn.reshape(y_size, x_size),
        stdv_mvn.reshape(y_size, x_size),
    )


def mvn_tiff(out_dir, model_name, sites, nproc=1):
    """
    Run MVN over GeoTIFF.
    """
    # mvn based on original model, modified if in proximity to measured sites
    in_tiff = os.path.join(out_dir, f"{model_name}.tif")
    out_tiff = os.path.join(out_dir, f"{model_name}_mvn.tif")
    copyfile(in_tiff, out_tiff)
    tif_ds = gdal.Open(out_tiff, gdal.GA_Update)
    nx = tif_ds.RasterXSize
    ny = tif_ds.RasterYSize
    vs30_band = tif_ds.GetRasterBand(1)
    stdv_band = tif_ds.GetRasterBand(2)

    # processing chunk/block sizing
    # usually just lines of nx=nx, ny=1 which is a good size for multiprocessing
    block = vs30_band.GetBlockSize()
    nxb = (int)((nx + block[0] - 1) / block[0])
    nyb = (int)((ny + block[1] - 1) / block[1])

    # asynchronously append jobs to queue
    pool = Pool(nproc)
    jobs = []
    for x in range(nxb):
        xoff = x * block[0]
        # last block may be smaller
        if x == nxb - 1:
            block[0] = nx - x * block[0]
        # reset y block size
        block_y = block[1]

        for y in range(nyb):
            yoff = y * block[1]
            # last block may be smaller
            if y == nyb - 1:
                block_y = ny - y * block[1]

            jobs.append(
                pool.apply_async(
                    partial(
                        _mvn_tiff_worker,
                        in_tiff,
                        xoff,
                        yoff,
                        block[0],
                        block_y,
                        sites,
                        model_name,
                    ),
                    (),
                )
            )

    # collect jobs
    for job in jobs:
        xoff, yoff, vs30_mvn, stdv_mvn = job.get()
        # write results
        vs30_band.WriteArray(vs30_mvn, xoff=xoff, yoff=yoff)
        stdv_band.WriteArray(stdv_mvn, xoff=xoff, yoff=yoff)
    # close
    pool.close()
    pool.join()
    vs30_band = None
    stdv_band = None
    tif_ds = None
    return out_tiff

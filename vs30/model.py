from math import exp, log, sqrt
import os

import numpy as np
from osgeo import gdal, gdalconst

gdal.UseExceptions()

ID_NODATA = 255


def model_val(mid, model, nodata=ID_NODATA):
    """
    Convert IDs returned by the <model>.mid function into model values.
    """
    vals = np.full((len(mid), 2), np.nan, dtype=np.float32)
    valid_idx = mid != nodata
    vals[valid_idx] = model[mid[valid_idx]]

    return vals


def interpolate(points, raster, band=1, nodata=ID_NODATA):
    """
    Returns values of raster at points (nearneighbour).
    points: 2D numpy array of coords in raster srs.
    """
    r = gdal.Open(raster, gdal.GA_ReadOnly)
    t = r.GetGeoTransform()
    b = r.GetRasterBand(band)
    n = b.GetNoDataValue()

    x = np.floor((points[:, 0] - t[0]) / t[1]).astype(np.int32)
    y = np.floor((points[:, 1] - t[3]) / t[5]).astype(np.int32)
    valid = np.where((x >= 0) & (x < r.RasterXSize) & (y >= 0) & (y < r.RasterYSize))
    v = np.full(len(points), nodata, dtype=b.ReadAsArray(win_xsize=1, win_ysize=1).dtype)
    v[valid] = b.ReadAsArray()[y[valid], x[valid]]
    # defined nodata in case nodata in tif is different, so model_val() understands
    # minus 1 because ids in rasters start at 1
    v = np.where(v == n, nodata, v - 1)

    b = None
    r = None
    return v


def resample(
    src, dst, xmin, xmax, ymin, ymax, xd, yd, alg=gdalconst.GRIORA_NearestNeighbour
):
    gdal.Warp(
        dst,
        src,
        creationOptions=["COMPRESS=DEFLATE", "BIGTIFF=YES"],
        outputBounds=[xmin, ymin, xmax, ymax],
        xRes=xd,
        yRes=yd,
        resampleAlg=alg,
    )


def combine(args, a, b):
    """
    Combine geology and terrain models (path to geotiff files).
    """
    # model a
    ads = gdal.Open(a, gdal.GA_ReadOnly)
    a_vs30 = ads.GetRasterBand(1)
    a_stdv = ads.GetRasterBand(2)
    # model b
    bds = gdal.Open(b, gdal.GA_ReadOnly)
    b_vs30 = bds.GetRasterBand(1)
    b_stdv = bds.GetRasterBand(2)
    # output
    driver = gdal.GetDriverByName("GTiff")
    ods = driver.Create(
        os.path.join(args.out, "combined.tif"),
        xsize=args.nx,
        ysize=args.ny,
        bands=2,
        eType=gdal.GDT_Float32,
        options=["COMPRESS=DEFLATE", "BIGTIFF=YES"],
    )
    ods.SetGeoTransform(ads.GetGeoTransform())
    ods.SetProjection(ads.GetProjection())
    o_vs30 = ods.GetRasterBand(1)
    o_stdv = ods.GetRasterBand(2)
    o_vs30.SetDescription("Vs30")
    o_stdv.SetDescription("Standard Deviation")
    vnd = a_vs30.GetNoDataValue()
    snd = a_stdv.GetNoDataValue()
    o_vs30.SetNoDataValue(vnd)
    o_stdv.SetNoDataValue(snd)

    # processing chunk/block sizing
    block = o_vs30.GetBlockSize()
    nxb = (int)((args.nx + block[0] - 1) / block[0])
    nyb = (int)((args.ny + block[1] - 1) / block[1])

    for x in range(nxb):
        xoff = x * block[0]
        # last block may be smaller
        if x == nxb - 1:
            block[0] = args.nx - x * block[0]
        # reset y block size
        block_y = block[1]

        for y in range(nyb):
            yoff = y * block[1]
            # last block may be smaller
            if y == nyb - 1:
                block_y = args.ny - y * block[1]

            # determine results on block
            avv = a_vs30.ReadAsArray(
                xoff=xoff, yoff=yoff, win_xsize=block[0], win_ysize=block_y
            )
            avv[avv == vnd] = np.nan
            asv = a_stdv.ReadAsArray(
                xoff=xoff, yoff=yoff, win_xsize=block[0], win_ysize=block_y
            )
            asv[asv == snd] = np.nan
            bvv = b_vs30.ReadAsArray(
                xoff=xoff, yoff=yoff, win_xsize=block[0], win_ysize=block_y
            )
            bvv[bvv == vnd] = np.nan
            bsv = b_stdv.ReadAsArray(
                xoff=xoff, yoff=yoff, win_xsize=block[0], win_ysize=block_y
            )
            bsv[bsv == snd] = np.nan

            if args.stdv_weight:
                m_a = (asv ** 2) ** -args.k
                m_b = (bsv ** 2) ** -args.k
                w_a = m_g / (m_g + m_t)
                w_b = m_t / (m_g + m_t)
            else:
                w_a = 0.5
                w_b = 0.5

            log_ab = np.log(avv) * w_a + np.log(bvv) * w_b

            # write results
            o_vs30.WriteArray(np.exp(log_ab), xoff=xoff, yoff=yoff)
            o_stdv.WriteArray(
                (
                    w_a * ((np.log(asv) - log_ab) ** 2 + asv ** 2)
                    + w_b * ((np.log(bsv) - log_ab) ** 2 + bsv ** 2)
                )
                ** 0.5
            )
    # close
    o_vs30 = None
    o_stdv = None
    ods = None


def cluster_update(prior, sites, letter):
    # creates a model from the distribution of measured sites as clustered
    # prior: prior model, values only taken if no measurements available for ID
    posterior = prior
    # looping through model IDs
    for m in range(len(posterior)):
        vs_sum = 0
        idtable = sites[sites[f"{letter}id"] == m]
        clusters = idtable[f"{letter}cluster"].value_counts()
        # overall N is one per cluster, clusters labeled -1 are individual clusters
        n = len(clusters)
        if -1 in clusters.index:
            n += clusters[-1] - 1
        if n == 0:
            continue
        w = np.repeat(1 / n, len(idtable))
        for c in clusters.index:
            cidx = idtable[f"{letter}cluster"] == c
            ctable = idtable[cidx]
            if c == -1:
                # values not part of cluster, weight = 1 per value
                vs_sum += sum(np.log(ctable.vs30.values))
            else:
                # values in cluster, weight = 1 / cluster_size per value
                vs_sum += sum(np.log(ctable.vs30)) / len(ctable)
                w[cidx] /= len(ctable)
        posterior[m, 0] = exp(vs_sum / n)
        posterior[m, 1] = np.sqrt(sum(w * (np.log(idtable.vs30.values) - vs_sum / n) ** 2))

    return posterior


def _new_mean(mu_0, n0, var, y):
    return exp((n0 / var * log(mu_0) + log(y) / var) / (n0 / var + 1 / var))


def _new_var(sigma_0, n0, uncertainty):
    return (n0 * sigma_0 * sigma_0 + uncertainty * uncertainty) / (n0 + 1)


def posterior(model, sites, idcol, n_prior=3, min_sigma=0.5):
    """
    model: prior model
    sites: sites containing vs30 and uncertainty
    idcol: model ID column in sites
    n_prior: assume prior model made up of n_prior measurements
    min_sigma: minimum model_stdv allowed
    """

    # new model
    vs30 = model[:, 0]
    stdv = np.maximum(model[:, 1], min_sigma)

    # loop through observed
    n0 = np.repeat(n_prior, len(model))
    for i, r in sites.iterrows():
        m = r[idcol]
        if m == ID_NODATA:
            continue
        var = _new_var(stdv[m], n0[m], r.uncertainty)
        vs30[m] = _new_mean(vs30[m], n0[m], var, r.vs30)
        stdv[m] = sqrt(var)
        n0[m] += 1

    return np.column_stack((vs30, stdv))
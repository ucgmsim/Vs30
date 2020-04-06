#!/usr/bin/env python

import numpy as np
from osgeo import gdal

vs_ahdi = "MVN_Vs30_NZGD00_allNZ_AhdiAK_noQ3_hyb09c_noisyT_minDist0.0km_v6_crp1.5.tif"
vs_yong = "MVN_Vs30_NZGD00_allNZ_YongCA_noQ3_noisyT_minDist0.0km_v7_crp1.5.tif"
sigma_ahdi = "MVN_stDv_NZGD00_allNZ_AhdiAK_noQ3_hyb09c_noisyT_minDist0.0km_v6_crp1.5.tif"
sigma_yong = "MVN_stDv_NZGD00_allNZ_YongCA_noQ3_noisyT_minDist0.0km_v7_crp1.5.tif"


def raster(path, band=1):
    raster = gdal.Open(path, gdal.GA_ReadOnly)
    band = raster.GetRasterBand(band)
    #nodata = band.GetNoDataValue()
    array = band.ReadAsArray()
    #array = np.ma.masked_equal(band.ReadAsArray(), nodata)
    return array

def raster_properties(path, band=1):
    raster = gdal.Open(path, gdal.GA_ReadOnly)
    band = raster.GetRasterBand(band)
    ds = band.GetDataset()
    geo_trans = ds.GetGeoTransform()
    wkt_proj = ds.GetProjection()
    nx = ds.RasterXSize
    ny = ds.RasterYSize
    return {"trans":geo_trans, "proj":wkt_proj, "nx":nx, "ny":ny}

def save_raster(filename, array, prop):
    driver = gdal.GetDriverByName('GTiff')

    dataset = driver.Create(
        filename,
        prop["nx"],
        prop["ny"],
        1,
        gdal.GDT_Float32
    )
    dataset.SetGeoTransform(prop["trans"])
    dataset.SetProjection(prop["proj"])
    dataset.GetRasterBand(1).WriteArray(array)
    dataset.FlushCache()

properties = raster_properties(vs_ahdi)

vsa = np.log(raster(vs_ahdi))
vsy = np.log(raster(vs_yong))
sa = raster(sigma_ahdi) ** 2
sy = raster(sigma_yong) ** 2

vsay = np.exp((vsa + vsy) * 0.5)
sigma = np.sqrt(0.5 * (((vsa - vsay) ** 2) + sa + ((vsy - vsay) ** 2) + sy))

save_raster("AhdiYongWeightedMVN_nTcrp1.5_Vs30.tif", vsay, properties)
save_raster("AhdiYongWeightedMVN_nTcrp1.5_sigma.tif", sigma, properties)

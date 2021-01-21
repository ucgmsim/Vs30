


def save(vals, x0, y0, nx, ny, xd, yd, filename):
    driver = gdal.GetDriverByName("GTiff")
    # https://gdal.org/drivers/raster/gtiff.html
    # TILED=YES much smaller (entire nan blocks), slower with eg: QGIS
    # COMPRESS=DEFLATE smaller, =LZW larger
    gfile = driver.Create(filename, xsize=nx, ysize=ny, bands=1,
                          eType=gdal.GDT_Float32, options=["COMPRESS=DEFLATE"])
    gfile.SetGeoTransform([x0, xd, 0, y0, 0, yd])
    # projection
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(2193)
    gfile.SetProjection(srs.ExportToWkt())
    # data
    gfile.GetRasterBand(1).WriteArray(vals.reshape(ny, nx))
    # close file
    gfile = None

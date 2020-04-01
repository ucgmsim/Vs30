#!/usr/bin/env python

from h5py import File as h5open
import numpy as np
from pyproj import Proj, transform

# new format requires pyproj6
wgs84 = Proj("epsg:4326")
nztm = Proj("epsg:2193")
nzmg = Proj("epsg:27200")

# lon lat pairs
locations = np.array([[174.780278, -41.300278], [177, -37.983333]])

# slope
slope_ni = "/home/vap30/VsMap/Rdata/nzni_9c_slp.nc"
slope_si = "/home/vap30/VsMap/Rdata/nzsi_9c_slp.nc"

# variogram psill, range, kappa, ang1, ang2, ang3, anis1, anis2
vg_nug = 0, 0, 0.0, 0, 0, 0, 1, 1
vg_mat = 1, 1407, 0.5, 0, 0, 0, 1, 1

locations_nztm = np.array(transform(wgs84, nztm, locations[:, 1], locations[:, 0]))[::-1].T
locations_nzmg = np.array(transform(wgs84, nzmg, locations[:, 1], locations[:, 0]))[::-1].T


def nn_ncdf4(locations, path):
    """
    Find nearest value for locations inside NetCDF4 file at path.
    locations: x y pairs in same coordinate system as file at path
    path: netcdf4 file (exported with R function from raster)
    """
    # open file, read x y ranges
    h = h5open(path, "r")
    e = h["easting"][...]
    n = h["northing"][...]
        
    # find closest x and y for each location
    np.digitize()
    
    # return results, value for each location


slope_ni = nn_ncdf4(locations_nzgd, slope_ni)

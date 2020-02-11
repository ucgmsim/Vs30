#!/usr/bin/env python

from glob import glob
from subprocess import call


home = "/home/vap30"
# input and outputs
in_out = [
    ["~/VsMap/tmp/MVkrg_NZGD00_allNZ_*_AhdiAK_noQ3_hyb09c_noisyT_minDist0.0km_v6_crp1.5.tif",
     "~/big_noDB/models/MVN_Vs30_NZGD00_allNZ_AhdiAK_noQ3_hyb09c_noisyT_minDist0.0km_v6_crp1.5.tif"],
    ["~/VsMap/tmp/MVkrg_NZGD00_allNZ_*_YongCA_noQ3_noisyT_minDist0.0km_v7_crp1.5.tif",
     "~/big_noDB/models/MVN_Vs30_NZGD00_allNZ_YongCA_noQ3_noisyT_minDist0.0km_v7_crp1.5.tif"],
    ["~/VsMap/tmp/MVsdv_NZGD00_allNZ_*_AhdiAK_noQ3_hyb09c_noisyT_minDist0.0km_v6_crp1.5.tif",
     "~/big_noDB/models/MVN_stDv_NZGD00_allNZ_AhdiAK_noQ3_hyb09c_noisyT_minDist0.0km_v6_crp1.5.tif"],
    ["~/VsMap/tmp/MVsdv_NZGD00_allNZ_*_YongCA_noQ3_noisyT_minDist0.0km_v7_crp1.5.tif",
     "~/big_noDB/models/MVN_stDv_NZGD00_allNZ_YongCA_noQ3_noisyT_minDist0.0km_v7_crp1.5.tif"],
    ["~/VsMap/tmp/MVres_NZGD00_allNZ_*_AhdiAK_noQ3_hyb09c_noisyT_minDist0.0km_v6_crp1.5.tif",
     "~/big_noDB/models/MVN_resid_NZGD00_allNZ_AhdiAK_noQ3_hyb09c_noisyT_minDist0.0km_v6_crp1.5.tif"],
    ["~/VsMap/tmp/MVres_NZGD00_allNZ_*_YongCA_noQ3_noisyT_minDist0.0km_v7_crp1.5.tif",
     "~/big_noDB/models/MVN_resid_NZGD00_allNZ_YongCA_noQ3_noisyT_minDist0.0km_v7_crp1.5.tif"]
]


for io in in_out:
    tiffs = glob(io[0].replace("~", home))
    if len(tiffs) == 0:
        print("Invalid expression.")
        continue
    # gdal_merge.py is part of gdal
    # not sure about this custom nodata 1.699..., maybe just use nan?
    cmd = ["gdal_merge.py", "-o", io[1].replace("~", home), "-n", "1.69999999999999994e+308", "-a_nodata", "-1.69999999999999994e+308"]
    cmd.extend(tiffs)
    call(cmd)

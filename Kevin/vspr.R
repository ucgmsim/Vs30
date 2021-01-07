# Gathers metadata for points with measured Vs30.
# only columns needed are coordinates, Vs30/lnMeasUncer, Vs30/stDV for model(s),
#                         QualityFlag/StationID for subsetting

library(sp)

source("Kevin/const.R")
source("config.R")
source("Kevin/load_vs.R")
source("Kevin/model_ahdiak.R")
source("Kevin/model_yongca.R")


vspr_run = function() {
    # source coordinates with metadata
    vspr = sp::spTransform(load_vs(downsample_McGann=TRUE), NZTM)

    # remove points in the same location with the same Vs30
    mask = rep(TRUE, length(vspr))
    dup_pairs = sp::zerodist(vspr)
    for (i in seq(dim(dup_pairs)[1])) {
        if(vspr[dup_pairs[i,1],]$Vs30 == vspr[dup_pairs[i,2],]$Vs30) {
            mask[dup_pairs[i,2]] = FALSE
        }
    }
    vspr = vspr[mask, !names(vspr) == "DataSourceW"]

    # add model categories
    vspr$gid_aak = model_ahdiak_get_gid(vspr)
    vspr$gid_yca = model_yongca_get_gid(vspr)
    # create posterior models
    if (POSTERIOR_UPDATE) {
        source("Kevin/bayes.R")
        source("Kevin/model_ahdiak_prior.R")
        source("Kevin/model_yongca_prior.R")
        model_ahdiak <<- bayes_posterior(vspr, vspr$gid_aak, model_ahdiak)
        model_yongca <<- bayes_posterior(vspr, vspr$gid_yca, model_yongca)
    } else {
        source("Kevin/model_ahdiak_posterior.R")
        source("Kevin/model_yongca_posterior.R")
    }

    return(vspr)
}

# Gathers metadata for points with measured Vs30.
# only columns needed are coordinates, Vs30/lnMeasUncer, Vs30/stDV for model(s),
#                         QualityFlag/StationID for subsetting

library(sp)

source("R/const.R")
source("config.R")
source("R/load_vs.R")
source("R/model_ahdiak.R")
source("R/model_yongca.R")


vspr_run = function(outfile="data/vspr.csv", clusters=T) {
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
    # save for Python clustering code to read and update
    vspr = as.data.frame(vspr, row.names=NULL)
    names(vspr)[names(vspr) == "Easting"] = "x"
    names(vspr)[names(vspr) == "Northing"] = "y"
    write.csv(vspr_df, file=outfile, row.names=F)
    if (clusters) {
        system("./python/cluster.py")
        vspr = read.csv("data/vspr.csv")
        # average clusters for posterior update
        # TODO:
    }
    # create posterior models
    if (POSTERIOR_UPDATE) {
        source("R/bayes.R")
        source("R/model_ahdiak_prior.R")
        source("Kevin/model_yongca_prior.R")
        model_ahdiak <<- bayes_posterior(vspr, vspr$gid_aak, model_ahdiak)
        model_yongca <<- bayes_posterior(vspr, vspr$gid_yca, model_yongca)
    } else {
        source("R/model_ahdiak_posterior.R")
        source("R/model_yongca_posterior.R")
    }

    # remove Q3 quality unless station name is 3 chars long.
    vspr = vspr[(vspr$QualityFlag != "Q3" |
                 nchar(as(vspr$StationID, "character")) == 3 |
                 is.na(vspr$QualityFlag)),]

    return(vspr)
}

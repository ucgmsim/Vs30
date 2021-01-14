# Gathers metadata for points with measured Vs30.
# only columns needed are coordinates, Vs30/lnMeasUncer, Vs30/stDV for model(s),
#                         QualityFlag/StationID for subsetting

library(sp)

source("R/const.R")
source("config.R")
source("R/load_vs.R")
source("R/model_ahdiak.R")
source("R/model_yongca.R")


vspr_clustering = function(vspr, model="aak") {
    # originally assuming that clusters are being reduced to single value
    # stdev is actually sqrt(<sum>Wi*(V30i-V30mean)^2)
    # mean is like normal mean but clusters are meaned first and treated as n=1
    # posterior needs only $lnMeasUncer and $Vs30
    out = data.frame(lnMeasUncer=logical(), Vs30=logical(), gid=integer())
    ids = unique(vspr[,paste0("gid_", model)])
    ids = ids[!is.na(ids)]
    for (id in ids) {
        idtable = vspr[which(vspr[,paste0("gid_", model)] == id),]
        clusters = unique(idtable[,paste0("cluster_", model)])
        for (c in clusters) {
            ctable = idtable[which(idtable[,paste0("cluster_", model)] == c),]
            if (c == -1) {
                # not part of a cluster, treat as individual values
                out = rbind(out, data.frame(ctable[c("lnMeasUncer", "Vs30")], gid=id))
            } else {
                # combine values
                # Vs30 is being averaged on a per-cluster basis
                # uncertainty needs overall average per this geo/terrain ID in above formula
                out = rbind(out, data.frame(lnMeasUncer=combined$stdv,
                                            Vs30=sum(ctable$Vs30) / nrow(ctable),
                                            gid=id))
            }
        }
    }
    return(out)
}


vspr_run = function(outfile="data/vspr.csv") {
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
    # work on plain dataframe columns
    vspr = as.data.frame(vspr, row.names=NULL)
    names(vspr)[names(vspr) == "Easting"] = "x"
    names(vspr)[names(vspr) == "Northing"] = "y"
    # remove Q3 quality unless station name is 3 chars long.
    vspr = vspr[(vspr$QualityFlag != "Q3" |
                 nchar(as(vspr$StationID, "character")) == 3 |
                 is.na(vspr$QualityFlag)),]
    write.csv(vspr, file=outfile, row.names=F)
    # save for Python clustering code to read and update
    if (POSTERIOR_CLUSTERS) {
        system("./python/cluster.py")
        vspr = read.csv("data/vspr.csv")
    }
    # create posterior models
    if (POSTERIOR_UPDATE) {
        source("R/bayes.R")
        source("R/model_ahdiak_prior.R")
        source("R/model_yongca_prior.R")
        # average clusters for posterior update
        if (POSTERIOR_CLUSTERS) {
            vsprc = vspr_clustering(vspr, model="aak")
            model_ahdiak <<- bayes_posterior(vsprc, vsprc$gid, model_ahdiak)
            vsprc = vspr_clustering(vspr, model="yca")
            model_yongca <<- bayes_posterior(vsprc, vsprc$gid, model_yongca)
        } else {
            model_ahdiak <<- bayes_posterior(vspr, vspr$gid_aak, model_ahdiak)
            model_yongca <<- bayes_posterior(vspr, vspr$gid_yca, model_yongca)
        }
    } else {
        source("R/model_ahdiak_posterior.R")
        source("R/model_yongca_posterior.R")
    }

    return(vspr)
}

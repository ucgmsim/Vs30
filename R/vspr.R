# Gathers metadata for points with measured Vs30.
# Creates posterior model from measured values.
# only columns needed are coordinates, Vs30/lnMeasUncer, Vs30/stDV for model(s),
#                         QualityFlag/StationID for subsetting

source("config.R")
source("R/load_vs.R")
source("R/model_ahdiak.R")
source("R/model_yongca.R")


cluster_model = function(vspr, model="aak", prior) {
    # creates a model from the distribution of measured sites as clustered
    # prior: prior model, values only taken if no measurements available for ID
    out = prior
    # overwrite prior model looping through IDs
    for (id in seq(length(prior$vs30))) {
        vs_n = 0
        vs_sum = 0
        idtable = vspr[which(vspr[,paste0("gid_", model)] == id),]
        clusters = table(idtable[,paste0("cluster_", model)])
        # overall N is one per cluster, clusters labeled -1 are individual clusters
        n = length(clusters)
        if ("-1" %in% names(clusters)) n = n + unname(clusters["-1"]) - 1
        if (n == 0) next
        w = rep(1/n, nrow(idtable))
        for (c in as.integer(names(clusters))) {
            cidx = which(idtable[,paste0("cluster_", model)] == c)
            ctable = idtable[cidx,]
            if (c == -1) {
                # values not part of cluster, weight = 1 per value
                vs_sum = vs_sum + sum(ctable$Vs30)
            } else {
                # values in cluster, weight = 1 / cluster_size per value
                vs_sum = vs_sum + sum(ctable$Vs30) / nrow(ctable)
                w[cidx] = w[cidx] / nrow(ctable)
            }
        }
        out$vs30[id] = vs_sum / vs_n
        out$stdv[id] = sqrt(sum(w * (idtable$Vs30 - out$vs30[id]) ^ 2))
    }
    return(out)
}


vspr_run = function(outfile="data/vspr.csv", posterior_update=F, clusters=F, cpt=F) {
    # source coordinates with metadata
    vspr = load_vs(cpt=cpt, downsample_McGann=TRUE)

    # add model categories
    vspr$gid_aak = model_ahdiak_get_gid(vspr)
    vspr$gid_yca = model_yongca_get_gid(vspr)
    # work on plain dataframe columns
    vspr = as.data.frame(vspr, row.names=NULL)
    names(vspr)[names(vspr) == "Easting"] = "x"
    names(vspr)[names(vspr) == "Northing"] = "y"
    # save for Python clustering code to read and update
    write.csv(vspr, file=outfile, row.names=F)
    if (clusters) {
        system("./python/cluster.py")
        vspr = read.csv("data/vspr.csv")
    }
    # create posterior models
    if (posterior_update) {
        source("R/bayes.R")
        source("R/model_ahdiak_prior.R")
        source("R/model_yongca_prior.R")
        # average clusters for posterior update
        if (clusters) {
            model_ahdiak = cluster_model(vspr, model="aak", model_ahdiak)
            model_yongca = cluster_model(vspr, model="yca", model_yongca)
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

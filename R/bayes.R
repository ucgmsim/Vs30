
new_mean = function(mu_0, n0, var, y) {
    return(exp((n0 / var * log(mu_0) + log(y) / var) / (n0 / var + 1 / var)))
}
new_var = function(sigma_0, n0, lnMeasUncer) {
    return((n0 * sigma_0 * sigma_0 + lnMeasUncer * lnMeasUncer) / (n0 + 1))
}

bayes_posterior = function(vspr, gids, model, n_prior=3, min_sigma=0.5) {
    # vspr: contains $lnMeasUncer and $Vs30
    # gids: numeric IDs of observed data for model
    # n_prior: assume prior model made up of n_prior measurements
    # min_sigma: minimum model_stdv allowed

    # new models
    update_vs30 = model$vs30
    update_stdv = pmax(model$stdv, min_sigma)

    # loop through observed
    nn_prior = rep(n_prior, length(model$vs30))
    for(o in 1:length(gids)) {
        g = gids[o]
        if (is.na(g)) next
        n0 = nn_prior[g]
        var = new_var(update_stdv[g], n0, vspr$lnMeasUncer[o])
        update_vs30[g] = new_mean(update_vs30[g], n0, var, vspr$Vs30[o])
        update_stdv[g] = sqrt(var)
        nn_prior[g] = n0 + 1
    }

    return(list(vs30=update_vs30, stdv=update_stdv))
}

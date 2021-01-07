# MODEL_AhdiAK_noQ3_hyb09c
# 
# This model is based on the bayes updated model, AhdiAK_noQ3
# The "reference" version of that model was generated assuming 3 prior observations in each group,
# and enforcing a minimum sigma of 0.5 in the prior.

# the model that will be modified needs to be loaded
stopifnot(exists(model_ahdiak))
model_ahdiak_hybrid = model_ahdiak

# define which units have a slope-Vs30 relation to apply
# log10slope and vs30 values define two points of reference for interpolation.
# They were chosen by eye from the model this hybrid is based upon.
hybconf = data.frame(
  gid        = c(     2,               3,               4,               6),
  log10slope = I(list(c(-1.85, -1.22), c(-2.70, -1.35), c(-3.44, -0.88), c(-3.56, -0.93))),
  vs30       = I(list(c(242, 418),     c(171, 228),     c(252, 275),     c(183, 239))),
  sigmafac   = c(     0.4888,          0.7103,          0.9988,          0.9348)
)
# modify standard deviation to incorporate sigma reduction factors
model_ahdiak_hybrid$stdv[hybconf$gid] = model_ahdiak$stdv[hybconf$gid] *
                                        hybconf$sigmafac

model_ahdiak_get_vs30 = function(data, g06mod=T, g13mod=T){
  # first apply geo (parent model)
  vs30out = model_ahdiak$vs30[data$gid]
  
  # then apply hybrid model changes
  for (i in seq(nrow(hybconf))) {
    gid = hybconf$gid[i]
    # g06 only depends on coast distance (below) if mod used
    if (g06mod & gid == 4) next
    w = which(data$gid == gid)
    if (length(w) == 0) next
    vs30out[w] = (10^approx(x=hybconf[[i, "log10slope"]],
                            y=log10(hybconf[[i, "vs30"]]),
                            xout=log10(data[w, "slp09c"]),
                            rule=2)$y)
  }
  if (g06mod) {
      # override G06 (index 4)
      # 240 up to 8km distance, then up to 500 at 20km distance.
      g06 = which(data$gid == 4)
      vs30out[g06] = pmax(240, pmin(500, 240 + (500-240) * (data$coastkm[g06]-8)/(20-8)))
  }
  if (g13mod) {
      # override G13 (index 10)
      # 197 up to 8km distance, then up to 500 at 20km distance.
      g13 = which(data$gid == 10)
      vs30out[g13] = pmax(197, pmin(500, 197 + (500-197) * (data$coastkm[g13]-8)/(20-8)))
  }
  return(vs30out)
}

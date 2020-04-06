# MODEL_AhdiAK_noQ3_hyb09c
# 
# This model is based on the bayes updated model, AhdiAK_noQ3
# The "reference" version of that model was generated assuming 3 prior observations in each group,
# and enforcing a minimum sigma of 0.5 in the prior.
# The summary table is also included here for reference:

source("Kevin/MODEL_AhdiAK_noQ3.R")

###### SUMMARY TABLE FROM out/bayesUpdateSummary_NinvChiSq_DATA_noQ3__nObsPrior003_measPrec1_minSigma0.500.txt

#   | nObs|groupID             | priorVs30| geomMeanVs30_obs| posteriorVs30| mu_sigma| logPriorStDv| posteriorStDv| sigma_sigma|
#   |----:|:-------------------|---------:|----------------:|-------------:|--------:|------------:|-------------:|-----------:|
#   |    9|01_peat             |       161|          163.528|       162.892|    0.008|        0.522|         0.301|       0.054|
#   |   11|04_fill             |       198|          297.317|       272.513|    0.006|        0.500|         0.280|       0.041|
#   |   11|05_fluvialEstuarine |       239|          189.912|       199.502|    0.014|        0.867|         0.439|       0.100|
#   |   25|06_alluvium         |       323|          265.407|       271.051|    0.002|        0.500|         0.243|       0.018|
#   |    0|08_lacustrine       |       326|              NaN|       326.000|       NA|        0.500|         0.500|          NA|
#   |   70|09_beachBarDune     |       339|          199.989|       204.374|    0.001|        0.647|         0.232|       0.009|
#   |    5|10_fan              |       360|          196.538|       246.614|    0.015|        0.500|         0.345|       0.112|
#   |    4|11_loess            |       376|          561.324|       472.749|    0.018|        0.500|         0.355|       0.144|
#   |    0|12_outwash          |       399|              NaN|       399.000|       NA|        0.500|         0.500|          NA|
#   |  252|13_floodGravel      |       448|          195.556|       197.473|    0.000|        0.500|         0.203|       0.004|
#   |    0|14_moraineTill      |       453|              NaN|       453.000|       NA|        0.512|         0.512|          NA|
#   |    0|15_undifSed         |       455|              NaN|       455.000|       NA|        0.545|         0.545|          NA|
#   |    2|16_terrace          |       458|          210.000|       335.280|    0.073|        0.761|         0.603|       0.857|
#   |    0|17_volcanic         |       635|              NaN|       635.000|       NA|        0.995|         0.995|          NA|
#   |    4|18_crystalline      |       750|          649.774|       690.974|    0.028|        0.641|         0.446|       0.227|


########## FOR COMPARISON (NOT USED!!) ---
###### SUMMARY TABLE FROM out/bayesUpdateSummary_NinvChiSq_DATA_Kaiser_all__nObsPrior003_measPrec1_minSigma0.500.txt

#  | nObs|groupID             | priorVs30| geomMeanVs30_obs| posteriorVs30| mu_sigma| logPriorStDv| posteriorStDv| sigma_sigma|
 #  |----:|:-------------------|---------:|----------------:|-------------:|--------:|------------:|-------------:|-----------:|
 #  |    5|01_peat             |       161|          131.203|       141.669|    0.021|        0.522|         0.410|       0.159|
 #  |   20|04_fill             |       198|          346.888|       322.422|    0.006|        0.500|         0.382|       0.052|
 #  |    5|05_fluvialEstuarine |       239|          161.348|       186.961|    0.051|        0.867|         0.642|       0.388|
 #  |  176|06_alluvium         |       323|          301.948|       302.289|    0.001|        0.500|         0.478|       0.025|
 #  |    0|08_lacustrine       |       326|              NaN|       326.000|       NA|        0.500|         0.500|          NA|
 #  |   35|09_beachBarDune     |       339|          276.886|       281.346|    0.005|        0.647|         0.434|       0.048|
 #  |   21|10_fan              |       360|          402.862|       397.237|    0.010|        0.500|         0.491|       0.083|
 #  |   10|11_loess            |       376|          369.658|       371.112|    0.017|        0.500|         0.464|       0.120|
 #  |    3|12_outwash          |       399|          265.629|       325.555|    0.042|        0.500|         0.500|       0.375|
 #  |   33|13_floodGravel      |       448|          212.690|       226.312|    0.005|        0.500|         0.412|       0.045|
 #  |    3|14_moraineTill      |       453|          328.594|       385.815|    0.043|        0.512|         0.506|       0.384|
 #  |   62|15_undifSed         |       455|          766.136|       747.931|    0.004|        0.545|         0.502|       0.047|
 #  |   24|16_terrace          |       458|          318.436|       331.558|    0.010|        0.761|         0.521|       0.086|
 #  |    0|17_volcanic         |       635|              NaN|       635.000|       NA|        0.995|         0.995|          NA|
 #  |   43|18_crystalline      |       750|          843.573|       837.130|    0.006|        0.641|         0.506|       0.058|


#######################################################################
#######################################################################
########        hybrid model params here   ############################
#######################################################################
#######################################################################
# Now define which units have a slope-Vs30 relation to apply:
# 
# One entry for each hybrid unit
AhdiAK_noQ3_hyb09c_hybDF = data.frame(
  slopeUnits      = c(2, 3, 4, 6),
  log10slope_0    = c(-1.85   , -2.70   , -3.44   , -3.56), # these four values define two points of reference
  log10slope_1    = c(-1.22   , -1.35   , -0.88   , -0.93), # for interpolation. They should be chosen by eye
  Vs30_0          = c( 242    ,  171    ,  252    ,  183 ), # from the model this hybrid is based upon. They
  Vs30_1          = c( 418    ,  228    ,  275    ,  239 ), # become inputs to approx / approxfun.
                                                            # (actually my slopePlotDetail.R has been updated
                                                            #  to include endpoints of fit line. 20171205)
  # Added this on 20171223. Based on slope-Vs30 plots. (See new SIGMA notes added to plot subtitles in slopePlotDetail.R output.)
  sigmaReducFac   = c( 0.4888 , 0.7103  , 0.9988  , 0.9348)
)

AhdiAK_noQ3_hyb09c_set_Vs30_hyb = function(data, i) {
  # Use slopes and intercepts entered above to compute Vs30 as f(slope) (using 09c resolution slopes)

  row = AhdiAK_noQ3_hyb09c_hybDF[i,]
  # interpolate
  x = c(row$log10slope_0, row$log10slope_1)
  y = c(log10(row$Vs30_0), log10(row$Vs30_1))
  return(10^approx(x=x, y=y, xout=log10(data$slp09c), rule=2)$y)
}

AhdiAK_noQ3_hyb09c_set_Vs30 <- function(data, g06mod=T, g13mod=T){
  # first apply geo (parent model)
  vs30out = AhdiAK_noQ3_set_Vs30(data)
  
  # then apply hybrid model changes
  for (i in seq(nrow(AhdiAK_noQ3_hyb09c_hybDF))) {
    gid = AhdiAK_noQ3_hyb09c_hybDF$slopeUnits[i]
    if (g06mod & gid == 4) next # gid 10 (g13) not in table anyway
    w = which(data$groupID_AhdiAK %in% gid)
    if (length(w) == 0) next
    vs30out[w] = AhdiAK_noQ3_hyb09c_set_Vs30_hyb(data[w,], i)
  }
  if (g06mod) {
      # override G06 (index 4)
      g06 = which(data$groupID_AhdiAK %in% 4)
      vs30out[g06] = pmax(240, pmin(500, 240 + (500-240) * (data$coastkm[g06]-8)/(20-8)))
  }
  if (g13mod) {
      # (index 10)
      g13 = which(data$groupID_AhdiAK %in% 10)
      vs30out[g13] = pmax(197, pmin(500, 197 + (500-197) * (data$coastkm[g13]-8)/(20-8)))
  }
  return(vs30out)
}


# modify standard deviation to incorporate sigma reduction factors....
stDv_AhdiAk_noQ3_hyb09c = stDv_AhdiAK_noQ3
for (i in seq(nrow(AhdiAK_noQ3_hyb09c_hybDF))) {
  stDv_AhdiAk_noQ3_hyb09c[AhdiAK_noQ3_hyb09c_hybDF$slopeUnits[i]] =
    stDv_AhdiAK_noQ3[AhdiAK_noQ3_hyb09c_hybDF$slopeUnits[i]] * AhdiAK_noQ3_hyb09c_hybDF$sigmaReducFac[i]
}

AhdiAK_noQ3_hyb09c_set_stDv = function(data) {
  return(stDv_AhdiAk_noQ3_hyb09c[data$groupID_AhdiAK])
}

# MODEL_YongCA_noQ3.R

load("data/BayesUpdateTables.Rdata")
Vs30_YongCA_noQ3 = upd_YongCA_noQ3$summary$posteriorVs30
stDv_YongCA_noQ3 = upd_YongCA_noQ3$summary$posteriorStDv
rm(upd_AhdiAK_KaiAll, upd_AhdiAK_KaiNoQ3, upd_AhdiAK_noQ3, upd_AhdiAK_noQ3noMcGann, upd_YongCA_noQ3)

YongCA_noQ3_set_Vs30 = function(data) {
  return(Vs30_YongCA_noQ3[data$groupID_YongCA_noQ3])
}

YongCA_noQ3_set_stDv = function(data) {
  return(stDv_YongCA_noQ3[data$groupID_YongCA_noQ3])
}

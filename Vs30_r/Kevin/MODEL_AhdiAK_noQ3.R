# MODEL_AhdiAK_noQ3.R

load("../Vs30_data/BayesUpdateTables.Rdata")
Vs30_AhdiAK_noQ3 = upd_AhdiAK_noQ3$summary$posteriorVs30
stDv_AhdiAK_noQ3 = upd_AhdiAK_noQ3$summary$posteriorStDv
rm(upd_AhdiAK_KaiAll, upd_AhdiAK_KaiNoQ3, upd_AhdiAK_noQ3, upd_AhdiAK_noQ3noMcGann, upd_YongCA_noQ3)

AhdiAK_noQ3_set_Vs30 = function(data) {
  return(Vs30_AhdiAK_noQ3[data$groupID_AhdiAK])
}

AhdiAK_noQ3_set_stDv = function(data) {
  return(stDv_AhdiAK_noQ3[data$groupID_AhdiAK])
}

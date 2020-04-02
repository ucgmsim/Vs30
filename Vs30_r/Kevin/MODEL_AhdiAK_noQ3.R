# MODEL_AhdiAK_noQ3.R

source("Kevin/MODEL_AhdiAK.R")
AhdiAK_noQ3_setGroupID     <- AhdiAK_setGroupID
groupID_noUpdate           <- AhdiAK_lookup()$groupID

# updTableFile <- "Rdata/BayesUpdateTables_201708.Rdata"
# updTableFile <- "Rdata/BayesUpdateTables_20171129.Rdata"
# updTableFile <- "Rdata/BayesUpdateTables_201711.Rdata"
updTableFile <- "../Vs30_data/BayesUpdateTables.Rdata"



if(file.exists(updTableFile)) {
  load(updTableFile)

  AhdiAK_noQ3_lookup <- function() {
    return(data.frame(
      groupID                = upd_AhdiAK_noQ3$summary$groupID,
      Vs30_AhdiAK_noQ3        = upd_AhdiAK_noQ3$summary$posteriorVs30,
      stDv_AhdiAK_noQ3        = upd_AhdiAK_noQ3$summary$posteriorStDv))
  }

} else {
  AhdiAK_noQ3_lookup <- function() {
    return(data.frame(
      groupID                 = groupID_noUpdate,
      Vs30_AhdiAK_noQ3     = rep(NA,length(groupID_noUpdate)),
      stDv_AhdiAK_noQ3     = rep(NA,length(groupID_noUpdate))))
  }
  warning("No Bayesian update tables to use. Run modelsUpdate.R!")
}

AhdiAK_noQ3_set_Vs30 <- function(data) {
  lookup <- AhdiAK_noQ3_lookup()
  Vs30   <- lookup$Vs30_AhdiAK_noQ3
  names(Vs30)  <- as.character(lookup$groupID)
  Vs30out <- Vs30[as.character(data$groupID_AhdiAK_noQ3)]
  names(Vs30out) <- NULL
  return(Vs30out)
}

AhdiAK_noQ3_set_stDv <- function(data) {
  lookup <- AhdiAK_noQ3_lookup()
  stDv   <- lookup$stDv_AhdiAK_noQ3
  names(stDv)  <- as.character(lookup$groupID)
  stDvOut <- stDv[as.character(data$groupID_AhdiAK_noQ3)]
  names(stDvOut) <- NULL
  return(stDvOut)
}



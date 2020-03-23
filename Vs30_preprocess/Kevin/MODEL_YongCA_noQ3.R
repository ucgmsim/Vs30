# MODEL_YongCA_noQ3.R

source("Kevin/MODEL_YongCA.R")
YongCA_noQ3_setGroupID     <- YongCA_setGroupID
groupID_noUpdate           <- YongCA_lookup()$groupID

# updTableFile <- "Rdata/BayesUpdateTables_201708.Rdata"
# updTableFile <- "Rdata/BayesUpdateTables_20171129.Rdata"
# updTableFile <- "Rdata/BayesUpdateTables_201711.Rdata"
updTableFile <- "~/VsMap/Rdata0/BayesUpdateTables.Rdata"


if(file.exists(updTableFile)) {
  load(updTableFile)

  YongCA_noQ3_lookup <- function() {
    return(data.frame(
      groupID                = upd_YongCA_noQ3$summary$groupID,
      Vs30_YongCA_noQ3        = upd_YongCA_noQ3$summary$posteriorVs30,
      stDv_YongCA_noQ3        = upd_YongCA_noQ3$summary$posteriorStDv))
  }

} else {
  YongCA_noQ3_lookup <- function() {
    return(data.frame(
      groupID                 = groupID_noUpdate,
      Vs30_YongCA_noQ3     = rep(NA,length(groupID_noUpdate)),
      stDv_YongCA_noQ3     = rep(NA,length(groupID_noUpdate))))
  }
  warning("No Bayesian update tables to use. Run modelsUpdate.R!")
}

YongCA_noQ3_set_Vs30 <- function(data) {
  lookup <- YongCA_noQ3_lookup()
  Vs30   <- lookup$Vs30_YongCA_noQ3
  names(Vs30)  <- as.character(lookup$groupID)
  Vs30out <- Vs30[as.character(data$groupID_YongCA_noQ3)]
  names(Vs30out) <- NULL
  return(Vs30out)
}

YongCA_noQ3_set_stDv <- function(data) {
  lookup <- YongCA_noQ3_lookup()
  stDv   <- lookup$stDv_YongCA_noQ3
  names(stDv)  <- as.character(lookup$groupID)
  stDvOut <- stDv[as.character(data$groupID_YongCA_noQ3)]
  names(stDvOut) <- NULL
  return(stDvOut)
}



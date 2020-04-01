# MODEL_YongCA

source("Kevin/IP_levels.R")

YongCA_lookup <- function() {
  groupID     <- IPlevels[[1]]$category # just using numeric identifiers here, not group names. Use R/IP_levels.R if needed.
  Vs30_YongCA <- c(519,
                   393,
                   547,
                   459,
                   402,
                   345,
                   388,
                   374,
                   497,
                   349,
                   328,
                   297,
                   500,  # No data in Yong et al. This is my guess for incised terraces.
                   209,
                   363,
                   246)
  
  stDv_YongCA <- c(0.3521,  # All sigmas in this vector were chosen
                   0.4161,  # based on Yong et al. (2012) figure 9, the
                   0.4695,  # "bandplot" showing scatter for each category.
                   0.3540,  # The work done to estimate these is contained
                   0.3136,  # in the folder "Yong---digitizing".
                   0.2800,
                   0.4161,
                   0.3249,
                   0.3516,
                   0.2800,
                   0.2736,
                   0.2931,
                   0.5,    # guess
                   0.1749,
                   0.2800,
                   0.2206)

  return(data.frame(groupID, Vs30_YongCA, stDv_YongCA))
}


YongCA_setGroupID <- function(){} # this is all done in vspr.R.


YongCA_set_Vs30 <- function(data) {
  lookup <- YongCA_lookup()
  Vs30   <- lookup$Vs30_YongCA
  names(Vs30)  <- as.character(lookup$groupID)
  Vs30out <- Vs30[as.character(data$groupID_YongCA)]
  names(Vs30out) <- NULL
  return(Vs30out)
}

YongCA_set_stDv <- function(data) {
  lookup <- YongCA_lookup()
  stDv   <- lookup$stDv_YongCA
  names(stDv)  <- as.character(lookup$groupID)
  stDvOut <- stDv[as.character(data$groupID_YongCA)]
  names(stDvOut) <- NULL
  return(stDvOut)
}

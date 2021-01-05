# bayes.R
#
# replaces former bayes.R, now bayesMC.R
#
# 
library("geoR")


bayesPosterior <- function(data, groupIDmod, groupIDobs, Vs30mod, stDvMod) {

  
  # get the actual Vs30 observations
  Vs30vec <- data$Vs30
  # measurement uncertainty
  # measPrecision is set to zero for perfectly precise measurements
  # or one to choose values from lnMeasUncer without modification.
  measUncerVec <- data$lnMeasUncer * measPrecision

  # Compute the model predictions first.
  # This is also done in vspr.R - It's done again here.
  # The reason for this is that relying on previously-computed model estimates creates a circular
  # dependency, preventing quick model updates by re-running everything.
  # 
  # metadata inputs for classification...
  age1 <- data$age1
  age2 <- data$age2
  dep  <- data$dep

  SIMPLE_NAME = data$SIMPLE_NAME
  MAIN_ROCK   = data$MAIN_ROCK
  UNIT_CODE   = data$UNIT_CODE
  DESCRIPTION = data$DESCRIPTION
  MAP_UNIT    = data$MAP_UNIT
  
  # summarize columns of new model summary table (including bayes parameters for troubleshooting/review)
  posteriorVs30 <- posteriorStDv <- 
        logMeanVs30_obs <- logStDvVs30_obs <-
    geomMeanVs30_obs <- nObs <-
        logPriorVs30 <- logPriorStDv <- priorVs30 <-
    sigma_sigma <- mu_sigma <- 
                rep(NA, length(groupIDmod))
  
  # every step of bayesian updating stored here.
  updatesByGroup <- list()
  
  for (i in 1:length(groupIDmod)) {
    # i <- 1   # test
    groupIDi <- groupIDmod[i]
    idx_i <- which(as.character(groupIDobs) == as.character(groupIDi))
    n <- length(idx_i)
    nObs[i] <- n
    
    
    
    # contextual variable name   ......... # Gelman (2014) Section 3.3 variable name
    priorVs30[i]    <- Vs30mod[i]      # exp(mu)
    logPriorStDv[i] <- stDvMod[i]      # sigma
    logPriorVs30[i] <- log(priorVs30[i])   # mu
    
    logPriorStDv[i] <- max(logPriorStDv[i], minSigma) # enforcing a minimum permissible uncertainty in priors from imported models.
    
    

    Vs30byGroup      <- Vs30vec[idx_i]
    measUncerByGroup <- measUncerVec[idx_i]
    
    
    # statistics for the data - included in summary output table.
    logMeanVs30_obs[i] <- sum(log(Vs30byGroup))   /   n
    geomMeanVs30_obs[i] <- exp(logMeanVs30_obs[i])        #  I chose this variable name to distinguish it from the logMeanVs30 which is just the log-space value of the geometric mean.
    logStDvVs30_obs[i] <- sqrt(sum((log(Vs30byGroup) -   logMeanVs30_obs[i])^2)/n)
    
    
    
    # initializing output dataframe:
    updatesByGroup[[i]] <- data.frame()

    # initialize "byGroup" output vectors
    
    bGpriorVs30 <- bGlogPriorVs30 <- bGpriorStDv <- 
         bGobsVs30 <- bGlogObsVs30 <- bGlnMeasUncer <- 
         bGpostVs30 <- bGlogPostVs30 <- bGpostStDv <- 
         # new...
         bGsigma_sigma <- bGmu_sigma <- 
         bGn <- 
         c()


    
    
    # kappa_0 represents the number of degree of freedom i.e. number of prior measurements
    # used to constrain mean. higher kappa_0 -> greater influence of prior.
    # Similarly, nu_0 represents the number of degree of freedom i.e. number of prior measurements
    # used to constrain mean. higher nu_0 -> greater influence of prior.
    kappa_0 <- nu_0  <- nObsPrior # VARIANCE OF INVERSE-CHI-SQUARED DISTRIBUTION IS UNSTABLE IS nu_0 IS TOO LOW.
    
    # set first prior. This will be updated in loops below.
    sigma_0 <- logPriorStDv[i]
    mu_0    <- logPriorVs30[i]
    
    
    
    if(n>0) {
      for(obsIdx in 1:n) { # one per datapoint
          #obsIdx <- 1
          measUncer_obs              <- measUncerByGroup[obsIdx]
          y                          <- log(Vs30byGroup[obsIdx])
          
          bGpriorVs30[obsIdx]       <- exp(mu_0)
          bGlogPriorVs30[obsIdx]    <- mu_0
          bGpriorStDv[obsIdx]       <- sigma_0
          bGobsVs30[obsIdx]         <- exp(y)
          bGlogObsVs30[obsIdx]      <-  y
          bGlnMeasUncer[obsIdx]     <- measUncer_obs
  
          
          # UPDATING:
          
          sigma_n_sq <- priorToPostVar(mu_0, sigma_0, kappa_0, nu_0, 
                                       y, measUncer_obs)
          nu_n    <- nu_0    + 1
          
          #################################################################
          # "The mean of the uncertainty" - posterior sigma!!
          #################################################################
          # The following features of the posterior distribution for Sigma
          # are from the Gelman appendix:
          invChiSqMean <- nu_n/(nu_n-2)*sigma_n_sq
          invChiSqMode <- nu_n/(nu_n+2)*sigma_n_sq
          
          # ONE of the above central measures---mean or mode---
          # needs to be chosen as "best" indicator of posterior sigma!
          # postVariance <- invChiSqMean
          # postVariance <- invChiSqMode
          postVariance <- sigma_n_sq
          thisPosteriorStDv <- sqrt(postVariance)

          #################################################################
          # "the uncertainty of the uncertainty" - sigma_sigma!          
          #################################################################
          # also from the Gelman appendix:
          # N.B. nu must be >= 4 for variance to be guaranteed non-negative
          invChiSqVar  <-  2*nu_n*nu_n*sigma_n_sq^2 / (((nu_n - 2)^2) * (nu_n-4))  
          sigma_sigma_i <- sqrt(invChiSqVar) 

                  
          mean_hyperparameters <- priorToPostMean(mu_0, kappa_0, sig = thisPosteriorStDv, y)
            mu_n  <- thisPosteriorLogVs30        <-    mean_hyperparameters$mu_n
            sigma_sq_over_kappa_n <- mu_sigma_i  <-    mean_hyperparameters$sigma_sq_over_kappa_n  # THIS is the value that was erroneously reported as posterior sigma before. It actually should approach 1/n and is unimportant.
                  
          
          # posterior becomes the new prior for next loop:
          kappa_n <- kappa_0 + 1
                  
          sigma_0 <- thisPosteriorStDv
          mu_0    <- thisPosteriorLogVs30
          kappa_0 <- kappa_n
          nu_0    <- nu_n
          
          # populating output dataframe
          bGpostVs30[obsIdx]        <- exp(thisPosteriorLogVs30)
          bGlogPostVs30[obsIdx]     <- thisPosteriorLogVs30
          bGpostStDv[obsIdx]        <- thisPosteriorStDv
          bGn[obsIdx]               <- nu_0 
          
          # new:
          bGsigma_sigma[obsIdx]  <- sigma_sigma_i   # "the uncertainty of the uncertainty"
          bGmu_sigma[obsIdx]   <- mu_sigma_i   # the uncertainty of the sigma OF the distribution describing the posterior mean. Should quickly approach zero and not be very interesting/useful.
      }
    updatesByGroup[[i]] <- data.frame(
      n            = bGn           ,
      priorVs30    = bGpriorVs30   ,
      logPriorVs30 = bGlogPriorVs30,
      priorStDv    = bGpriorStDv   ,
      obsVs30      = bGobsVs30     ,
      logObsVs30   = bGlogObsVs30  ,
      lnMeasUncer  = bGlnMeasUncer ,
      postVs30     = bGpostVs30    ,
      logPostVs30  = bGlogPostVs30 ,
      postStDv     = bGpostStDv    ,
      sigma_sigma  = bGsigma_sigma  ,
      mu_sigma     = bGmu_sigma)
    
    posteriorVs30[i] <- exp(mu_0)
    posteriorStDv[i] <- sigma_0
    sigma_sigma[i]   <- sigma_sigma_i
    mu_sigma[i]      <- mu_sigma_i
    }
    else{
      updatesByGroup[[i]] <- c()
      posteriorVs30[i] <- Vs30mod[i]
      posteriorStDv[i] <- max(stDvMod[i], minSigma) # enforcing a minimum permissible uncertainty in priors from imported models.
    }
      
  }
  logPostVs30 <- log(posteriorVs30)
  summaryTable <- data.frame(  nObs,
                               groupID = groupIDmod,
                               priorVs30,
                               geomMeanVs30_obs,
                               posteriorVs30,
                               mu_sigma,
                               
                               # logPriorVs30,
                               # logStDvVs30_obs,
                               # logPostVs30,
                               
                               logPriorStDv,
                               posteriorStDv,
                               sigma_sigma
                               
                               )


  return(list(summary = summaryTable, byGroup = updatesByGroup))
}


#  # The below were verified after determining that Michael Jordan's course notes
#  # (Lemma 6: Michael I. Jordan, Stat260 Lecture 5 (Feb 8 2010):
#  # The Conjugate Prior for the Normal Distribution)   IS WRONG
#  # The source is Kevin P. Murphy, murphyk@cs.ubc.ca - "Conjugate Analysis
#  # of the Gaussian Distribution" - Notes last updated 03 October 2007.  EQUATIONS 20 and 24
#
# priorToPostMean <- function(mu_0, sigma_0, n, x, sigma) {
#   return(
#     (sigma_0*sigma_0*x / 
#        (sigma*sigma/n + sigma_0*sigma_0))
#     +
#       (sigma*sigma*mu_0 /
#          (sigma*sigma + sigma_0*sigma_0*n)))
# }
# 
# priorToPostStDv <- function(mu_0, sigma_0, n, x, sigma) {
#   return(sqrt(sigma_0*sigma_0 * sigma*sigma /  # sqrt ---> don't mix up sigma and variance!
#                            (n * sigma_0*sigma_0 + sigma*sigma))) }

# The equations BELOW are updating functions derived from ANOTHER source
# because Brendon was concerned I did not verify. As expected these are equivalent to
# commented-out functions above.
#
# THE PROBLEM HERE is that mu_0 and sigma_0 are not THE PRIOR DISTRIBUTION PARAMETERS
# but instead THE HYPERPARAMETERS DESCRIBING PRIOR KNOWLEDGE OF THE MEAN, with the
# standard deviation assumed FIXED!! But this approach is not what we actually want.
# 
# The conventional (fixed-sigma) approach, implemented in the equations below and above,
# is discussed in Gelman (2014), Bayesian Data Analysis 3rd Ed. section 2.5
# The CORRECT approach when we need to get hyperparameters for SIGMA as well as MEAN
# is outlined in Gelman, Section 3.3.
# 
# 
# priorToPostMean <- function(mu_0, sigma_0, x, sigma) {
#   n <- length(x)
#   return(
#     (mu_0/sigma_0/sigma_0 + sum(x)/sigma/sigma)
#           /
#        (1/sigma_0/sigma_0 + n/sigma/sigma))
# }
# 
# priorToPostStDv <- function(mu_0, sigma_0, x, sigma) {
#   n <- length(x)
#   return(sqrt(
#     1 /
#         (1/sigma_0/sigma_0  +  n/sigma/sigma)
#       )) }



#########################################################################################################################
#### BAYESIAN UPDATING HELPER FUNCTIONS #################################################################################
#########################################################################################################################
# 
# Approach: Gelman (2014), Bayesian Data Analysis 3rd Edition - section 3.3
#
# These functions take as input mu_0, sigma_0, kappa_0 and nu_0. Output are mu_1 ~ N(mu_0,)and sigma_
#
priorToPostMean <- function(mu_0, kappa_0, sig, y) {
# This corresponds to Gelman Eq. 3.8, the posterior distribution of mu that is conditional on sigma.
# It returns the pair of posterior hyperparamaters that describe the posterior distribution of mu_1 conditioned on sigma.
# The form of the posterior distribution is Gaussian.
  n <- length(y)
  mu_n <- (
    (kappa_0/sig/sig*mu_0 + sum(c(y))/sig/sig)
          /
       (kappa_0/sig/sig + n/sig/sig))
  sigma_sq_over_kappa_n <- 1 / (kappa_0/sig/sig + n/sig/sig)
  return(list(mu_n = mu_n, sigma_sq_over_kappa_n = sigma_sq_over_kappa_n))
}


priorToPostVar <- function(mu_0, sigma_0, kappa_0, nu_0, 
                           y, measUncer) {
  # This corresponds to Gelman Eq. 3.9, the marginal posterior distribution of sigma that is conditioned just on the data.
  # It returns the pair of posterior hyperparameters that describe the posterior distribution of sigma^2 conditioned on the data.
  # The form of the posterior distribution is scaled inverse-chi^2.
  if(length(y) != length(measUncer)) {
    stop("expects y and measUncer to be same length")
  }
  
  if(length(y) > 1) {
    # for a vector of inputs, sampling statistic is computed directly
    # AND measUncer IS IGNORED!! This corresponds to Gelman equations directly.
    # This is for updating a geology model in one step, with all data.
    n <- length(y)
    y_bar <- sum(y)/n
    nu_n <- nu_0 + n
    s_sq <- 1/(n-1)*sum((y-y_bar)^2)
    nu_n_sigma_n_sq <- nu_0*sigma_0*sigma_0 + (n - 1)*s_sq 
                           + kappa_0*n/(kappa_0 + n)*(y_bar - mu_0)^2
  } else {
    # for a SINGLE value of y and a SINGLE measUncer value,
    # (n-1)*s_sq is just measUncer^2 in the limit, so this term is replaced. 
    # In other words the sampling variance is represented by measUncer.
    # (However, the weighting is handled using n=1 so that the updated distribution is
    # a proper compromise based on nu_0.)
    n <- 1
    y_bar <- y
    nu_n <- nu_0 + 1
    nu_n_sigma_n_sq <- nu_0*sigma_0*sigma_0 + measUncer^2 
                           + kappa_0*n/(kappa_0 + n)*(y_bar - mu_0)^2
  }
  
  sigma_n_sq <- nu_n_sigma_n_sq / nu_n
  return(sigma_n_sq)
}
  
  

scaled_Inv_chi2_centralTendencies <- function(nu, s_sq) {  # from Gelman appendix
  mean <- nu/(nu-2)*s_sq
  variance <-  2*nu*nu*s_sq^2 / (((nu - 2)^2) * (nu-4))  # nu must be >= 4 for variance to be guaranteed non-negative
  mode <- nu/(nu+2)*s_sq
  return(list(
    mean = mean,
    variance = variance,
    mode = mode
  ))
}

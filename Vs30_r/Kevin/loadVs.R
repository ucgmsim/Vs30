library(rgdal)
library(sp)
library(maptools)

source("Kevin/downSampleMcG.R")
source("Kevin/geodetic.R")

# Load each datasource.
# As of 2017-11 this comprises 3 sources
#     McGann Vs30 map (McGann, submitted, 2016 SDEE "Development of regional Vs30 Model and typical profiles... Christchurch CPT correlation")
#     Kaiser et al.
#     Internal communication with Wotherspoon: Characterised Vs30 Canterbury_June2017_KFed.csv
#     SUPERSEDED:  Canterbury region SMS sites (Hoby)
#     SUPERSEDED:  Internal communication with Wotherspoon: Vs30_May2016_Canterbury.csv (email thread around 23 June 2016)
#
# The McGann data is downsampled here by calling downSampleMcGann.

loadVs <- function(downSampleMcGann=TRUE){
  
  # load each Vs data source.
  McGannVsPts_NZGD49       <- loadMcGannVs()
  # WotherspoonVsPts_NZGD49  <- loadWotherspoonVs()  # SUPERSEDED
  WotherspoonVsPts201711_NZGD49  <- loadWotherspoonVs201711()
  # CantSMSvsPts_NZGD49      <- loadCantSMSvs()      # SUPERSEDED BY KAISER ET AL.
  
  
  # Kaiser et al.
  #
  # 20170329 - contains Kaiser flatfile with duplicates!
  # KaiserEtAlPts_NZGD49     <- loadKaiserEtAlVs("in/Vs/201703_allNZ/20170329_vs_allNZ.ll")
  # This file has had duplicates removed (description in header)
  KaiserEtAlPts_NZGD49     <- loadKaiserEtAlVs("../Vs30_data/20170817_vs_allNZ_duplicatesCulled.ll")
  
  # If flag is set, apply downsampling criterion to McGann CPT-based data
  if(downSampleMcGann) {
    McGannVsPts_NZGD49_DS <- downSampleMcG(inputSPDF = McGannVsPts_NZGD49)
    McGannVsPts_NZGD49    <- McGannVsPts_NZGD49_DS
  }

  # Bind data together (can be separated using DataSource field)
  VsPts_NZGD49 <- rbind(McGannVsPts_NZGD49,
                        # WotherspoonVsPts_NZGD49,   # SUPERSEDED
                        WotherspoonVsPts201711_NZGD49,
                        # CantSMSvsPts_NZGD49,       # SUPERSEDED BY KAISER ET AL.
                        KaiserEtAlPts_NZGD49)  
  VsPts_NZGD49$lnMeasUncer <- assignMeasurementUncertainty(VsPts_NZGD49)
  
  
  colnames(VsPts_NZGD49@coords) <- c("Easting","Northing")
  
  return(VsPts_NZGD49)
}



 
assignMeasurementUncertainty <- function(inputPoints) {
  lnMeasUncer <- rep(NA,length(inputPoints))
  
  lnMeasUncer[inputPoints$DataSource=="McGannCPT"]   <- 0.2
  lnMeasUncer[inputPoints$DataSource=="CantSMS"]     <- 0.3
  # lnMeasUncer[inputPoints$DataSource=="Wotherspoon"] <- 0.2 # superseded
  lnMeasUncer[inputPoints$DataSource=="Wotherspoon201711"] <- 0.2
  
  
  # Kaiser et al. states:
  #   "[Q1, Q2 and Q3] correspond to approximate uncertainties
  #    of <10%, 10-20% and >20% respectively."
  #
  #
  # I choose the following values:
  #
  #     10%    :   ln(1.1) ~= 0.1
  #     20%    :   ln(1.2) ~= 0.2
  #     50%    :   ln(1.5) ~= 0.5
  #
  lnMeasUncer[(inputPoints$DataSource=="KaiserEtAl") &
                (inputPoints$QualityFlag=="Q1")]       <- 0.1
  lnMeasUncer[(inputPoints$DataSource=="KaiserEtAl") &
                (inputPoints$QualityFlag=="Q2")]       <- 0.2
  lnMeasUncer[(inputPoints$DataSource=="KaiserEtAl") &
                (inputPoints$QualityFlag=="Q3")]       <- 0.5
  return(lnMeasUncer)
}






loadMcGannVs <- function() {
  # ====================================================================================
  # Loading McGann CPT-based Vs30 points -----------------------------------------
  McGannCPT <- read.table(
    "../Vs30_data/McGann_cptVs30data.csv",
    header = TRUE, sep = ",")  #read in Chris McGann's Vs data
  
  # The following lines create a dataframe with just McGann's NZMG data,
  # and put it in a SpatialPoints object for later use with over()
  en <- data.frame(McGannCPT$NZMG.Easting,
                   McGannCPT$NZMG.Northing)
  McGannPts_NZGD49 <- SpatialPoints(coords = en, proj4string = crsNZGD49(), bbox = NULL)
  # the CRS projection arguments above are copied verbatim from the QMAP
  # shapefile as imported. This assumes that the NZMG coordinate system
  # of McGann data is exactly equivalent.

  McGannVsPts_NZGD49 <- SpatialPointsDataFrame(
    coords = McGannPts_NZGD49,
    data = data.frame(Vs30       = McGannCPT$Vs30,
                      DataSource = rep("McGannCPT",dim(McGannCPT)[[1]]),
                      DataSourceW= rep(NA, nrow(McGannCPT)),
                      StationID  = rep(NA, dim(McGannCPT)[[1]]),
                      QualityFlag= rep(NA, dim(McGannCPT)[[1]])))
  return(McGannVsPts_NZGD49)
}












loadCantSMSvs <- function() {
  # ====================================================================================
  # Loading Canterbury Strong-motion station (SMS) Vs30 values from Hoby -------------
  CantSMS_ll <- read.table(
    "in/Vs/20160623 Hoby/cantstations.ll.csv",
    header = TRUE, sep = ",",
    strip.white = TRUE)         #read in Canterbury strong motion station
  #locations provided by Hoby
  
  CantSMS_Vs <- read.table(
    "in/Vs/20160623 Hoby/cantstations.vs30.csv",
    header = TRUE, sep = ",",
    strip.white = TRUE)  #read in Canterbury strong motion station
  
  cc <- merge(CantSMS_ll, CantSMS_Vs, by = "StationID")
  rm(CantSMS_ll, CantSMS_Vs)
  
  en <- data.frame(cc$Longitude,cc$Latitude)
  CantSMSpts_WGS84 <- SpatialPoints(coords = en, proj4string = crsWGS84(), bbox = NULL)
  
  CantSMSpts_NZGD49  <- convert2NZGD49(CantSMSpts_WGS84)
  
  CantSMSvsPts_NZGD49 <- SpatialPointsDataFrame(
    CantSMSpts_NZGD49,
    data = data.frame(
      Vs30    = cc$Vs30,
      DataSource = rep("CantSMS",dim(cc)[[1]]),
      DataSourceW= rep(NA, nrow(cc)),
      StationID  = cc$StationID,
      QualityFlag = rep(NA,dim(cc)[[1]])))
  
  return(CantSMSvsPts_NZGD49)
}






loadWotherspoonVs201711 <- function(){
  # ====================================================================================
  # Loading values from Liam Wotherspoon's spreadsheet -----------------------
  
  WotherspoonCSV  <- read.table(
    "../Vs30_data/Characterised Vs30 Canterbury_June2017_KFed.csv",
    header = TRUE, sep = "\t",
    strip.white = TRUE)         #read in Liam Wotherspoon summary spreadsheet
                                # for Canterbury region
  
  
  en <- data.frame(WotherspoonCSV$Long, WotherspoonCSV$Lat)
  WotherspoonPts_WGS84 <- SpatialPoints(coords = en, proj4string = crsWGS84(), bbox = NULL)
  WotherspoonPts_NZGD49  <- convert2NZGD49(WotherspoonPts_WGS84)
  
  WotherspoonPts <- WotherspoonPts_NZGD49
  
  WotherspoonVsPts_NZGD49 <- SpatialPointsDataFrame(
    WotherspoonPts_NZGD49,
    data = data.frame(
      Vs30 = WotherspoonCSV$Median.VS30..m.s.,
      StationID    = WotherspoonCSV$Site,
      DataSource   = rep("Wotherspoon201711",dim(WotherspoonCSV)[[1]]),
      DataSourceW  = WotherspoonCSV$Source,
      QualityFlag  = rep(NA,dim(WotherspoonCSV)[[1]])
    ))
  
  return(WotherspoonVsPts_NZGD49)
}


# SUPERSEDED
# loadWotherspoonVs <- function(){
#   # ====================================================================================
#   # Loading values from Liam Wotherspoon's spreadsheet -----------------------
#   
#   WotherspoonCSV  <- read.table(
#     "in/Vs/20160623 Liam/Vs30_May2016_Canterbury.csv",
#     header = TRUE, sep = ",",
#     strip.white = TRUE)         #read in Liam Wotherspoon summary spreadsheet
#   # for Canterbury region
#   
#   
#   en <- data.frame(WotherspoonCSV$Long, WotherspoonCSV$Lat)
#   WotherspoonPts_WGS84 <- SpatialPoints(coords = en, proj4string = crsWGS84(), bbox = NULL)
#   WotherspoonPts_NZGD49  <- convert2NZGD49(WotherspoonPts_WGS84)
#   
#   WotherspoonPts <- WotherspoonPts_NZGD49
#   
#   WotherspoonVsPts_NZGD49 <- SpatialPointsDataFrame(
#     WotherspoonPts_NZGD49,
#     data = data.frame(
#       Vs30 = WotherspoonCSV$Vs30..m.s.,
#       StationID    = WotherspoonCSV$Site,
#       DataSource   = rep("Wotherspoon",dim(WotherspoonCSV)[[1]]),
#       QualityFlag  = rep(NA,dim(WotherspoonCSV)[[1]])
#     ))
#   
#   return(WotherspoonVsPts_NZGD49)
# }





loadKaiserEtAlVs <- function(whichKaiserFile){
  # ==================================================================
  # Loading values from Kaiser et al. (2017) -----------------------
  # Some one-time reformatting was already done on the XLSX file
  
  # THERE ARE DUPLICATE POINTS IN KAISER ET AL.
  # THIS ISSUE IS TOO TEDIOUS TO HANDLE WITH AUTOMATED CODE.
  # UPDATED VS FILE IS GENERATED MANUALLY.
  
  kaiserEtAl  <- read.table(
    whichKaiserFile,
    header = TRUE, sep = ",",
    strip.white = TRUE)         #read in all NZ data from Kaiser et al .ll file
  
  en <- data.frame(kaiserEtAl$Lon_geonet, kaiserEtAl$Lat_geonet)
  kaiserEtAl_WGS84   <- SpatialPoints(coords = en, proj4string = crsWGS84(), bbox = NULL)
  kaiserEtAl_NZGD49  <- convert2NZGD49(kaiserEtAl_WGS84)
  
  kaiserEtAl_NZGD49.spdf <- SpatialPointsDataFrame(
    kaiserEtAl_NZGD49,
    data = data.frame(
      Vs30            = kaiserEtAl$Vs30,
      StationID       = kaiserEtAl$Station,
      DataSource      = rep("KaiserEtAl",dim(kaiserEtAl)[[1]]),
      DataSourceW     = rep(NA,nrow(kaiserEtAl)),
      QualityFlag     = kaiserEtAl$Q_Vs30
  ))
  
  return(kaiserEtAl_NZGD49.spdf)
}





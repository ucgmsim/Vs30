library(rgdal)
library(sp)

source("R/const.R")
source("R/downsample_McGann.R")


load_vs = function(cpt=F, downsample_McGann=TRUE){
    # load measured sites
    # - McGann Vs30 map (McGann, submitted, 2016 SDEE "Development of regional Vs30 Model and typical profiles... Christchurch CPT correlation")
    # - Kaiser et al.
    # - Internal communication with Wotherspoon: Characterised Vs30 Canterbury_June2017_KFed.csv
    # McGann data is downsampled (downSampleMcGann)

    if (cpt) {
        cpt = read.table("data/cptvs30.ssv", header=T, fill=T, row.names=NULL)
        # remove blank columns
        cpt = cpt[which(!is.na(cpt$Vs30)),]
        names(cpt) = c("Easting", "Northing", "Vs30")
        coordinates(cpt) = ~ Easting + Northing
        crs(cpt) = NZTM
        # default uncertainty
        cpt$lnMeasUncer = 0.5
        return(cpt)
    } else {
        # load each Vs data source.
        mcgann = loadvs_McGann("data/McGann_cptVs30data.csv", downsample=downsample_McGann)
        wotherspoon = loadvs_Wotherspoon("data/Characterised Vs30 Canterbury_June2017_KFed.csv")
        kaiseretal = loadvs_KaiserEtAl("data/20170817_vs_allNZ_duplicatesCulled.ll")

        # Bind data together (can be separated using DataSource field)
        vspoints = rbind(mcgann, wotherspoon, kaiseretal)  
        colnames(vspoints@coords) = c("Easting", "Northing")

        # remove points in the same location with the same Vs30
        mask = rep(TRUE, length(vspr))
        dup_pairs = sp::zerodist(vspr)
        for (i in seq(dim(dup_pairs)[1])) {
            if(vspr[dup_pairs[i,1],]$Vs30 == vspr[dup_pairs[i,2],]$Vs30) {
                mask[dup_pairs[i,2]] = FALSE
            }
        }
        vspr = vspr[mask, !names(vspr) == "DataSourceW"]

        # remove Q3 quality unless station name is 3 chars long.
        vspr = vspr[(vspr$QualityFlag != "Q3" |
                         nchar(as(vspr$StationID, "character")) == 3 |
                         is.na(vspr$QualityFlag)),]

        # in NZTM from NZMG
        return(sp::spTransform(vspoints, NZTM))
    }
}


loadvs_McGann = function(path, downsample=TRUE) {
  # McGann CPT-based Vs30 points
  table = read.table(path, header=TRUE, sep=",")

  # use NZMG
  coords = data.frame(Easting=table$NZMG.Easting, Northing=table$NZMG.Northing)
  coordinates(coords) = ~ Easting + Northing
  crs(coords) = NZMG

  spdf = SpatialPointsDataFrame(
    coords=coords,
    data=data.frame(Vs30=table$Vs30,
                    DataSource="McGannCPT",
                    DataSourceW=NA,
                    StationID=NA,
                    QualityFlag=NA,
                    lnMeasUncer=0.2))

  if(downsample) return(downsample_spdf(inputSPDF=spdf))
  return(spdf)
}


loadvs_Wotherspoon = function(path) {
  # Liam Wotherspoon's spreadsheet
  table = read.table(path, header=TRUE, sep="\t", strip.white=TRUE)

  # as NZMG
  coords = data.frame(Longitude=table$Long, Latitude=table$Lat)
  coordinates(coords) = ~ Longitude + Latitude
  crs(coords) = WGS84
  coords = spTransform(coords, NZMG)

  spdf = SpatialPointsDataFrame(
    coords=coords,
    data=data.frame(Vs30=table$Median.VS30..m.s.,
                    StationID=table$Site,
                    DataSource="Wotherspoon201711",
                    DataSourceW=table$Source,
                    QualityFlag=NA,
                    lnMeasUncer=0.2))

  return(spdf)
}


loadvs_KaiserEtAl = function(path){
  # Kaiser et al. (2017)
  table = read.table(path, header=TRUE, sep=",", strip.white=TRUE)

  coords = data.frame(Longitude=table$Lon_geonet, Latitude=table$Lat_geonet)
  coordinates(coords) = ~ Longitude + Latitude
  crs(coords) = WGS84
  coords = spTransform(coords, NZMG)
  
  spdf = SpatialPointsDataFrame(
    coords=coords,
    data=data.frame(Vs30=table$Vs30,
                    StationID=table$Station,
                    DataSource="KaiserEtAl",
                    DataSourceW=NA,
                    QualityFlag=table$Q_Vs30,
                    lnMeasUncer=NA))

  # Kaiser et al. states:
  #   "[Q1, Q2 and Q3] correspond to approximate uncertainties
  #    of <10%, 10-20% and >20% respectively."
  #
  # Kevin choose the following values:
  #     10%    :   ln(1.1) ~= 0.1
  #     20%    :   ln(1.2) ~= 0.2
  #     50%    :   ln(1.5) ~= 0.5
  spdf[spdf$QualityFlag=="Q1", "lnMeasUncer"] = 0.1
  spdf[spdf$QualityFlag=="Q2", "lnMeasUncer"] = 0.2
  spdf[spdf$QualityFlag=="Q3", "lnMeasUncer"] = 0.5

  return(spdf)
}

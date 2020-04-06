library(rgdal)
library(sp)

source("Kevin/downsample_McGann.R")

WGS84 = "+proj=longlat +ellps=WGS84 +datum=WGS84 +no_defs"
NZTM = "+proj=tmerc +lat_0=0 +lon_0=173 +k=0.9996 +x_0=1600000 +y_0=10000000 +ellps=GRS80 +towgs84=0,0,0,0,0,0,0 +units=m +no_defs"
NZMG = "+proj=nzmg +lat_0=-41 +lon_0=173 +x_0=2510000 +y_0=6023150 +units=m +no_defs +ellps=intl +towgs84=59.47,-5.04,187.44,0.47,-0.1,1.024,-4.5993"


load_vs = function(downsample_McGann=TRUE){
  # load measured sites
  # - McGann Vs30 map (McGann, submitted, 2016 SDEE "Development of regional Vs30 Model and typical profiles... Christchurch CPT correlation")
  # - Kaiser et al.
  # - Internal communication with Wotherspoon: Characterised Vs30 Canterbury_June2017_KFed.csv
  # McGann data is downsampled (downSampleMcGann)
  
  # load each Vs data source.
  mcgann = loadvs_McGann("../Vs30_data/McGann_cptVs30data.csv", downsample=downsample_McGann)
  wotherspoon = loadvs_Wotherspoon("../Vs30_data/Characterised Vs30 Canterbury_June2017_KFed.csv")
  kaiseretal = loadvs_KaiserEtAl("../Vs30_data/20170817_vs_allNZ_duplicatesCulled.ll")

  # Bind data together (can be separated using DataSource field)
  vspoints = rbind(mcgann, wotherspoon, kaiseretal)  
  colnames(vspoints@coords) = c("Easting", "Northing")

  # in NZMG
  return(vspoints)
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

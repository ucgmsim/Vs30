source("config.R")

# NZTM
load(paste0(PREFIX, "aak_map.Rdata"))

model_ahdiak_get_gid = function(coords) {
    return(over(coords, aak_map)$groupID_AhdiAK)
}

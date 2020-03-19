# MODEL_AhdiAK

AhdiAK_lookup <- function() {
  groupID     <- c(
    "01_peat",
    "04_fill",
    "05_fluvialEstuarine",
    "06_alluvium",
    "08_lacustrine",
    "09_beachBarDune",
    "10_fan",
    "11_loess",
    "12_outwash",
    "13_floodplain",
    "14_moraineTill",
    "15_undifSed",
    "16_terrace",
    "17_volcanic",
    "18_crystalline")
  Vs30_AhdiAK <- c(
    161,
    198,
    239,
    323,
    326,
    339,
    360,
    376,
    399,
    448,
    453,
    455,
    458,
    635,
    750)
  
  stDv_AhdiAK <- c(
    0.522,
    0.314,
    0.867,
    0.365,
    0.135,
    0.647,
    0.338,
    0.380,
    0.305,
    0.432,
    0.512,
    0.545,
    0.761,
    0.995,
    0.641)
  return(data.frame(groupID, Vs30_AhdiAK, stDv_AhdiAK))
}


AhdiAK_setGroupID <- function(data){
  
  # for testing:
  # load("~/big_noDB/geo/QMAP_Seamless_July13K_NZGD00.Rdata")
  # uMRSN <- data.frame(map_NZGD00)
  # for (str in c('SIMPLE_NAME','MAIN_ROCK','UNIT_CODE', 'DESCRIPTION', 'MAP_UNIT')) {
  #   assign(str, uMRSN[[str]])
  # }
  
  SIMPLE_NAME <- data$SIMPLE_NAME
  MAIN_ROCK   <- data$MAIN_ROCK
  UNIT_CODE   <- data$UNIT_CODE
  DESCRIPTION <- data$DESCRIPTION
  MAP_UNIT    <- data$MAP_UNIT
  
  
  dep <- unlist(mapply(
    FUN = dep_grep_ID_Ahdi, SIMPLE_NAME,
                            MAIN_ROCK  ,
                            UNIT_CODE  ,
                            DESCRIPTION,
                            MAP_UNIT))
  
  # testing output
  # p <- data.frame(SIMPLE_NAME, 
  #                 MAIN_ROCK, 
  #                 UNIT_CODE,
  #                 DESCRIPTION,
  #                 MAP_UNIT, cats)
  return(dep)
}



AhdiAK_set_Vs30 <- function(data) {
  lookup <- AhdiAK_lookup()
  Vs30   <- lookup$Vs30_AhdiAK
  names(Vs30)  <- as.character(lookup$groupID)
  Vs30out <- Vs30[as.character(data$groupID_AhdiAK)]
  names(Vs30out) <- NULL
  return(Vs30out)
}

AhdiAK_set_stDv <- function(data) {
  lookup <- AhdiAK_lookup()
  stDv   <- lookup$stDv_AhdiAK
  names(stDv)  <- as.character(lookup$groupID)
  stDvOut <- stDv[as.character(data$groupID_AhdiAK)]
  names(stDvOut) <- NULL
  return(stDvOut)
}









######################################################################################
######################################################################################
############################   helper functions below     ###########################
######################################################################################
######################################################################################






































dep_grep_ID_Ahdi <- function(sn, mr, uc, de, mu) {
  
  category <- ""
  
  # this categorisation is more detailed than first go-round.
  # GREPing over multiple fields here is intentional and is based on
  # careful perusal of depTable.csv.
  #
  # In general these keywords are selected based directly on Table 1 in Ahdi et al.
  # rather than assigning to masked index subsets directly, list items are appended for each
  # subsequent grep so that multiple-category units can be scrutinised.
  if(grepl("peat",         mr,      ignore.case = T)) {category <- paste0(category, "peat_")}
  if(grepl("human-made",   sn,      ignore.case = T))  {category <- paste0(category, "fill_")}
  
  
  # There are many "river and estuary deposits" in simple_name; to distinguish between them
  # I assign "estuarine" ONLY to groups with MAIN_ROCK = "sand" or "mud."
  if(grepl("fluvial",      de,      ignore.case = T)) {category <- paste0(category, "fluvial_")}
  if(grepl("estu",         sn,      ignore.case = T)) {
    if(grepl("sand", mr, ignore.case = T) |
       grepl("mud",  mr, ignore.case = T)) {category <- paste0(category, "estuarine_")}  
  }
  
  
  if(grepl("river",        sn,      ignore.case = T)) {category <- paste0(category, 'alluvial_')}
  if(grepl("flood",        de,      ignore.case = T)  |
     grepl("flood",        mu,      ignore.case = T)) {category <- paste0(category, 'floodplain_')}
  if(grepl("lake",         sn,      ignore.case = T)) {category <- paste0(category, 'lacustrine_')}
  if(grepl("beach",        mu,      ignore.case = T)) {category <- paste0(category, 'beach_')}
  if(grepl("dune",         mu,      ignore.case = T)) {category <- paste0(category, 'dune_')}
  if(grepl("fan",          mu,      ignore.case = T)) {category <- paste0(category, 'fan_')}
  if(grepl("windblown",    sn,      ignore.case = T)) {category <- paste0(category, 'loess_')}
  if(grepl("outwash",      mu,      ignore.case = T)) {category <- paste0(category, 'outwash_')} # NB: these generally also classify as alluvial deposits (river)
  # note: "glacigenic," "glaciogenic," "drift" yield no search results for Ahdi category 12
  # note: category 13 and category 7 (flood / course/fine) are too difficult to discriminate. Coarse flood deposits may not be as prevalent in NZ as Alaska.(?)
  if(grepl("moraine",      mu,      ignore.case = T)  |
     grepl("moraine",      de,      ignore.case = T)) {category <- paste0(category, 'moraine_')}
  if(grepl("till",         mr,      ignore.case = T)) {category <- paste0(category, 'till_')}
  if(grepl("terrace",      mu,      ignore.case = T)) {category <- paste0(category, 'terrace_')}
  if(grepl("volcanic",     mr,      ignore.case = T)) {category <- paste0(category, 'volcanic_')}
  if(grepl("igneous",      sn,      ignore.case = T)) {category <- paste0(category, 'igneous_')}
  if(grepl("metamorphic",  sn,      ignore.case = T)) {category <- paste0(category, 'metamorphic_')}
  if(grepl("sediment",     sn,      ignore.case = T)  &
     identical(category,""))  # undifferentiated sediments - i.e. not categorised as one
  { # of the above.... (group 15)
    category <- paste0(category, 'undifSed_')}
  # water
  if(grepl("water", uc, ignore.case = T)) {category <- "water_"}
  if(identical(category,""))   {
    # pick up the stragglers.
    category <- switch(as.character(mr),
                       basalt = "igneous_",
                       peridotite = "igneous_",
                       serpentinite= "igneous_",
                       amphibolite= "igneous_",
                       spilite= "igneous_",
                       mylonite="igneous_",
                       keratophyre="igneous_",
                       gabbro = "igneous_",
                       mudstone = "undifSed_",
                       sandstone = "undifSed_",
                       limestone = "undifSed_",
                       micrite = "undifSed_",
                       melange = "undifSed_",
                       debris = "undifSed_",
                       conglomerate = "undifSed_",
                       breccia = "undifSed_",  # mostly hillslope deposits
                       # gravel = "undifSed_",
                       mud = "estuarine_",
                       unknown = "undifSed_",  # mostly hillslope deposits
                       greywacke = "undifSed_", 
                       coal = "undifSed_", 
                       siltstone = "undifSed_", 
                       boulders = "undifSed_", 
                       `broken formation` = "undifSed",  # melange
                       none = "undifSed",
                       
                       
                       # ANOTHER level of stragglers follows:
                       switch(as.character(sn),
                              ice = "ICE_",
                              "")
    )
    # there are a few gravels identified as till in their description:
    if(identical(category,"") & 
       as.character(mr) == "gravel" & 
       grepl("till", de, ignore.case = T))
    {category <- "till_"}
  }
  
  # After examining multi-category groups individually, lump them as appropriate.
  # All of the lines below are based in individual polygon subgroup examinations.
  
  finalCategory <- switch(as.character(category),
                          fluvial_volcanic_igneous_      =  "17_volcanic",
                          alluvial_fan_                  =  "10_fan",
                          # These units are predominantly non-welded ignimbrite and 
                          # therefore will have Vs like similar *non-cemented* deposits:
                          alluvial_igneous_ =  "06_alluvium",
                          alluvial_terrace_ =  "16_terrace",
                          
                          alluvial_beach_ = "18_crystalline",
                          fan_igneous_ = "09_beachBarDune", # map unit  = Fanthams Peak lava!
                          fluvial_alluvial_lacustrine_ = "06_alluvium",   # these are predominantly gravel, early Pleistocene river/lake/shoreline.
                          dune_terrace_ = "09_beachBarDune",
                          dune_loess_terrace_ = "11_loess",
                          alluvial_floodplain_ =  "06_alluvium"  ,
                          
                          # this is a group of polygons with no map_unit metadata that therefore missed
                          # the "fan" classification  -  it's almost entirely fan gravels and similar.
                          alluvial_lacustrine_ =  "10_fan"  ,
                          
                          # this is all "young terrace/plain alluvium" and main_rock = gravel.
                          # Ahdi's terrace category is "old." 
                          alluvial_floodplain_terrace_ = "13_floodplain"  ,
                          alluvial_outwash_ = "12_outwash" ,
                          moraine_till_ =  "14_moraineTill",
                          alluvial_moraine_ = "14_moraineTill",
                          
                          # "well sorted, fresh, rounded, fine to medium 
                          # gravels in benches and storm beach ridges around major lakes"
                          lacustrine_beach_ =  "09_beachBarDune" ,
                          beach_dune_loess_ =  "09_beachBarDune" ,
                          dune_loess_       =  "11_loess" ,
                          peat_floodplain_  =  "01_peat" ,
                          alluvial_floodplain_outwash_ = "12_outwash",
                          fluvial_alluvial_ =  "06_alluvium",
                          alluvial_outwash_moraine_ = "12_outwash",
                          fluvial_loess_    = "11_loess",
                          alluvial_floodplain_fan_ = "06_alluvium",
                          volcanic_igneous_ = switch(as.character(mr),
                                                     metavolcanics = "18_crystalline", 
                                                     # else - all these are either
                                                     # "metavolcanics" or "volcanic breccia
                                                     "17_volcanic"),
                          estuarine_alluvial_ = "15_undifSed",
                          igneous_ = "18_crystalline",
                          metamorphic_ = "18_crystalline",
                          volcanic_metamorphic_ = "18_crystalline",
                          undifSed_  =  "15_undifSed",
                          terrace_   =  "16_terrace",
                          dune_ = "09_beachBarDune",
                          beach_ = "09_beachBarDune",
                          alluvial_ = "06_alluvium",
                          estuarine_ = "05_fluvialEstuarine",
                          peat_ = "01_peat",
                          loess_ = "11_loess",
                          fill_ = "04_fill",
                          volcanic_ = "17_volcanic",
                          fluvial_ = "05_fluvialEstuarine",
                          moraine_ = "14_moraineTill",
                          till_ = "14_moraineTill",
                          lacustrine_ =  "08_lacustrine" ,
                          fan_ = "10_fan",
                          floodplain_ = "15_undifSed",  # not an error - "flood" search yields bad results. These are conglomerates.
                          water_ = "00_WATER",
                          ICE_ = "00_ICE",
                          "15_undifSed"
  )
  
  # return(category)
  return(finalCategory)
}


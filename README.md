# Vs30

## data
20170817_vs_allNZ_duplicatesCulled.ll : Wotherspoon measured Vs30
Characterised Vs30 Canterbury_June2017_KFed.csv : Kaiser et al measured Vs30
McGann_cptVs30data.csv : McGann measured Vs30


## preprocess
rdata2nc.R : convert Rdata files to interoperable formats
vs_polygons.R : match measured Vs30 to polygons


## create
load_vs.py : loads measured Vs30 points
vspr.py : prepare data
...
join_tiles.py : converts tiles to complete images
weighted.py : mean model

# MODEL_AhdiAK

# names of levels (indexed by order, water index to NA)
# NA 00_water
#  1 01_peat
#  2 04_fill
#  3 05_fluvialEstuarine
#  4 06_alluvium
#  5 08_lacustrine
#  6 09_beachBarDune
#  7 10_fan
#  8 11_loess
#  9 12_outwash
# 10 13_floodplain
# 11 14_moraineTill
# 12 15_undifSed
# 13 16_terrace
# 14 17_volcanic
# 15 18_crystalline

model_ahdiak = list(vs30=c(161, 198, 239, 323, 326, 339, 360,
                           376, 399, 448, 453, 455, 458, 635, 750),
                    stdv=c(0.522, 0.314, 0.867, 0.365, 0.135, 0.647, 0.338,
                           0.380, 0.305, 0.432, 0.512, 0.545, 0.761, 0.995, 0.641))

model_ahdiak_get_vs30 = function(data) {
  return(Vs30_AhdiAK[data$gid])
}

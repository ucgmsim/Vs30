# MODEL_AhdiAK

# names of levels (indexed by order)
#01_peat
#04_fill
#05_fluvialEstuarine
#06_alluvium
#08_lacustrine
#09_beachBarDune
#10_fan
#11_loess
#12_outwash
#13_floodplain
#14_moraineTill
#15_undifSed
#16_terrace
#17_volcanic
#18_crystalline

Vs30_AhdiAK = c(161, 198, 239, 323, 326, 339, 360,
                 376, 399, 448, 453, 455, 458, 635, 750)
stDv_AhdiAK = c(0.522, 0.314, 0.867, 0.365, 0.135, 0.647, 0.338,
                 0.380, 0.305, 0.432, 0.512, 0.545, 0.761, 0.995, 0.641)

AhdiAK_set_Vs30 = function(data) {
  return(Vs30_AhdiAK[data$groupID_AhdiAK])
}

AhdiAK_set_stDv = function(data) {
  return(stDv_AhdiAK[data$groupID_AhdiAK])
}

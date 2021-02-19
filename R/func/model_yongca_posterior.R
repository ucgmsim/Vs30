# use version from Kevin's paper

load("../data/BayesUpdateTables.Rdata")
model_yongca = list(vs30=upd_YongCA_noQ3$summary$posteriorVs30,
                    stdv=upd_YongCA_noQ3$summary$posteriorStDv)
# unload Rdata file contents
rm(upd_AhdiAK_KaiAll,
   upd_AhdiAK_KaiNoQ3,
   upd_AhdiAK_noQ3,
   upd_AhdiAK_noQ3noMcGann,
   upd_YongCA_noQ3)

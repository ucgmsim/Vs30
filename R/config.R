#PLOTRES = "/nesi/project/nesi00213/PlottingData/"
#PREFIX = "/nesi/project/nesi00213/PlottingData/Vs30/"
PLOTRES = "/mnt/nvme/work/plotting_data/"
PREFIX = "/mnt/nvme/work/plotting_data/Vs30/old/"
CPT = F
MVN_OVERWRITE = F # overwrite pre-mvn with mvn
POSTERIOR_UPDATE = F # updates prior models using observed sites (T) or use posterior from paper (F)
POSTERIOR_CLUSTERS = F
# EX mvn_params.R
# note that some if statements have been removed to match default values
# should note down where this was and/or change back if ever modifying below values
# at least applicable to useNoisyMeasurements, note loadVs.R
numVGpoints = 128
useNoisyMeasurements = T
covReducPar = 1.5
useDiscreteVariogram = F

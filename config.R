#PLOTRES = "/nesi/project/nesi00213/PlottingData/"
#PREFIX = "/nesi/project/nesi00213/PlottingData/Vs30/"
PLOTRES = "/run/media/vap30/Hathor/work/plotting_data/"
PREFIX = "/run/media/vap30/Hathor/work/plotting_data/Vs30/"
POSTERIOR_UPDATE = T
# EX mvn_params.R
# note that some if statements have been removed to match default values
# should note down where this was and/or change back if ever modifying below values
# at least applicable to useNoisyMeasurements, note loadVs.R
numVGpoints = 128
useNoisyMeasurements = T
covReducPar = 1.5
useDiscreteVariogram = F

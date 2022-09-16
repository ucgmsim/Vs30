// Defines endpoints and other constants

// Base URL
export const VS_API_URL = process.env.REACT_APP_VS_API_URL;

// CPT Endpoints
export const CREATE_CPTS_ENDPOINT = "/cpt/create";
export const GET_CPT_CORRELATIONS_ENDPOINT = "/cpt/correlations";
export const CPT_MIDPOINT_ENDPOINT = "/plot/cpt/midpoint";

// VsProfile Endpoints
export const VS_PROFILE_CREATE_ENDPOINT = "/vsprofile/create";
export const VS_PROFILE_FROM_CPT_ENDPOINT = "/vsprofile/cpt/create";
export const VS_PROFILE_FROM_SPT_ENDPOINT = "/vsprofile/spt/create";
export const VS_PROFILE_CORRELATIONS_ENDPOINT = "/vsprofile/correlations";
export const VS_PROFILE_MIDPOINT_ENDPOINT = "/plot/vsprofile/midpoint";
export const VS_PROFILE_AVERAGE_ENDPOINT = "/plot/vsprofile/average";
export const VS_PROFILE_VS30_ENDPOINT = "/vsprofile/vs30";

// SPT Endpoints
export const SPT_CREATE_ENDPOINT = "/spt/create"
export const GET_SPT_CORRELATIONS_ENDPOINT = "/spt/correlations"
export const SPT_MIDPOINT_ENDPOINT = "/plot/spt/midpoint"
export const GET_HAMMER_TYPES_ENDPOINT = "/spt/hammertypes"
export const GET_SOIL_TYPES_ENDPOINT = "/spt/soiltypes"

// Plots
export const DEFAULTCOLOURS = [
  "#1f77b4", // Muted Blue
  "#ff7f0e", // Safety Orange
  "#2ca02c", // Cooked Asparagus Green
  "#d62728", // Brick Red
  "#9467bd", // Muted Purple
  "#8c564b", // Chestnut Brown
  "#e377c2", // Raspberyy Yogurt Pink
  "#7f7f7f", // Middle Gray
  "#bcbd22", // Curry Yellow-Green
  "#17becf", // Blue-Teal
];

// Tooltips
export const CPT_FILE = "CSV file format with 4 columns labelled in order [Depth, qc, fs, u]";
export const SPT_FILE = "CSV file format with 2-3 columns labelled in order [Depth, NValue, Soil] in which Soil is optional";
export const VS_FILE = "CSV file format with 3 columns labelled in order [Depth, Vs, Vs_SD]";
export const CPT_REMOVED_ROWS = "Number of rows removed during processing where Depth is > 30m or Qc or Fs is <= 0";
export const VS_REMOVED_ROWS = "Number of rows removed during processing where Depth is > 30m or the ending depth value is not a whole number";

// Errors
export const WEIGHT_ERROR = "Weights sum is not close enough to 1 or are not numbers";
export const FILE_ERROR = "This file has produced an error while processing";
export const NAME_ERROR = "This CPT Name is already taken";
export const REQUEST_ERROR = "Could not reach server. Please try again later";
export const COMPUTE_ERROR = "Error computing Vs30";
export const GWL_ERROR = "Ground Water Level input needs to be a valid number";
export const NAR_ERROR = "Net Area Ratio input needs to be a valid number";
export const BORE_ERROR = "Borehole Diameter input needs to be a valid number";
export const ENERGY_ERROR = "Energy Ratio input needs to be a valid number";
// Defines endpoints and other constants

// Base URL
export const VS_API_URL = process.env.REACT_APP_VS_API_URL;

// CPT Endpoints
export const CREATE_CPTS_ENDPOINT = "/api/cpt/create";
export const GET_CPT_CORRELATIONS_ENDPOINT = "/api/cpt/correlations";
export const CPT_MIDPOINT_ENDPOINT = "/api/plot/cpt/midpoint";

// VsProfile Endpoints
export const VS_PROFILE_CREATE_ENDPOINT = "/api/vsprofile/create";
export const VS_PROFILE_FROM_CPT_ENDPOINT = "/api/vsprofile/cpt/create";
export const VS_PROFILE_FROM_SPT_ENDPOINT = "/api/vsprofile/spt/create";
export const VS_PROFILE_CORRELATIONS_ENDPOINT = "/api/vsprofile/correlations";
export const VS_PROFILE_MIDPOINT_ENDPOINT = "/api/plot/vsprofile/midpoint";
export const VS_PROFILE_AVERAGE_ENDPOINT = "/api/plot/vsprofile/average";
export const VS_PROFILE_VS30_ENDPOINT = "/api/vsprofile/vs30";

// SPT Endpoints
export const SPT_CREATE_ENDPOINT = "/api/spt/create"
export const GET_SPT_CORRELATIONS_ENDPOINT = "/api/spt/correlations"
export const SPT_MIDPOINT_ENDPOINT = "/api/plot/spt/midpoint"
export const GET_HAMMER_TYPES_ENDPOINT = "/api/spt/hammertypes"
export const GET_SOIL_TYPES_ENDPOINT = "/api/spt/soiltypes"

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

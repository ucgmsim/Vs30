// Defines endpoints and other constants

// Base URL
export const VS_API_URL = process.env.REACT_APP_VS_API_URL;

// CPT Endpoints
export const CREATE_CPTS_ENDPOINT = "/api/cpt/create";
export const GET_CORRELATIONS_ENDPOINT = "/api/cpt/correlations";
export const CPT_MIDPOINT_ENDPOINT = "/api/plot/cpt/midpoint";

// VsProfile Endpoints
export const VSPROFILE_FROM_CPT_ENDPOINT = "/api/vsprofile/cpt/create";
export const VS_PROFILE_MIDPOINT_ENDPOINT = "/api/plot/vsprofile/midpoint";

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

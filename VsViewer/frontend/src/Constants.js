// Defines endpoints and other constants

// Base URL
export const VS_API_URL = process.env.REACT_APP_VS_API_URL;

// CPT Endpoints
export const CREATE_CPTS_ENDPOINT = "/api/cpt/create";
export const GET_CORRELATIONS_ENDPOINT = "/api/cpt/correlations";
export const CPT_MIDPOINT_ENDPOINT = "/api/plot/cpt/midpoint";

// VsProfile Endpoints
export const VSPROFILE_FROM_CPT_ENDPOINT = "/api/vsprofile/cpt/create"
export const VS_PROFILE_MIDPOINT_ENDPOINT = "/api/plot/vsprofile/midpoint";

// Plots
export const DEFAULTCOLOURS = ['#1f77b4',
'#ff7f0e',
'#2ca02c',  
'#d62728',    
'#9467bd',       
'#8c564b',         
'#e377c2',
'#7f7f7f',      
'#bcbd22',
'#17becf'
];
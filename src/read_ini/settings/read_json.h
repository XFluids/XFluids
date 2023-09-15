#pragma once

#include "json.hpp"
#include "global_setup.h"

// compute Dmin_json
extern real_t ComputeDminJson();
// read json from file
extern bool ReadJson(std::string filename, nlohmann::json &j);
template <class T>
std::vector<T> IntervalDivision(const T &start, const T &end, const T &period, size_t num);

// ==============================read data from json================================
extern nlohmann::json j_conf;
// run
extern const size_t OutBoundary;
extern const real_t CFLnumber_json;
extern const std::string OutputDir;
extern const real_t StartTime, EndTime;		// NOTE: ignore
extern const size_t nStepmax_json, nOutput; // nOutMax
extern const size_t OutDAT, OutVTI, OutSTL, OutOverTime;
extern const size_t OutInterval, POutInterval, RcalInterval; // PushInterval
extern const std::vector<std::string> OutTimeArrays_json, OutTimeStamps_json;
//--for-Thread-Allocation----------------
extern const size_t BlockSize_json, dim_block_x_json, dim_block_y_json, dim_block_z_json;

// MPI
extern const size_t mx_json;
extern const size_t my_json;
extern const size_t mz_json;
extern const std::vector<int> DeviceSelect_json;

// equations
extern const size_t NumFluid;									 // NOTE: ignore
extern const std::vector<std::string> Fluids_name;				 //, species_name; // Fluid_Names, Spiece_Names
extern const size_t Equ_rho, Equ_energy;
extern const std::vector<size_t> Equ_momentum;
extern const std::string SlipOrder, ODESolver;
extern const bool if_overdetermined_eigen, ReactSources, PositivityPreserving;
// extern const size_t Emax;
// extern const size_t NUM_COP;
// extern const size_t NUM_SPECIES;

// source terms
// extern const size_t NUM_REA;
// extern const string

// mesh
extern const size_t NUM_BISD;
extern const real_t width_xt, width_hlf, mx_vlm, ext_vlm, BandforLevelset; // for multiphase
extern const std::vector<real_t> Refs, DOMAIN_Size, Domain_medg;
extern const std::vector<bool> Dimensions;										  // XYZ Dir
extern const std::vector<size_t> Inner, Bwidth;									  // XYZ Resolution
extern const std::vector<size_t> NBoundarys, Boundarys_json;					  // BoundaryBundles
extern const std::vector<std::vector<size_t>> Boundary_x, Boundary_y, Boundary_z; // BoundaryBundle_xyz
//--Limiters----------------
extern const real_t Yil_limiter_json, Dim_limiter_json;

// init
extern const real_t Ma_json;
extern const bool Mach_Modified; // NOTE: ignore
// Mach_Modified: if post-shock theroy in Ref0 used(rewrite the value from Devesh Ranjan's theroy in Ref1 used by default):
// // Ref0:https://doi.org/10.1016/j.combustflame.2015.10.016 Ref1:https://www.annualreviews.org/doi/10.1146/annurev-fluid-122109-160744
extern const size_t cop_type, blast_type;
extern const real_t bubble_boundary, C_json;										   // bubble_boundary_cells;
extern const real_t xa_json, yb_first_json, zc_first_json, yb_json, zc_json;		   // bubble_shape
extern const std::vector<real_t> blast_pos, cop_pos, cop_instates, blast_upstates, blast_downstates; // states

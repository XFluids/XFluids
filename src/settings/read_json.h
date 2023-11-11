#pragma once

#include <vector>
#include <cstring>
#include <iostream>
#include <cstdlib>
#include <cstdio>
#include <fstream>
#include <iomanip>
#include <cmath>
#include <string>
#include <sstream>
#include <ctime>
#include <cassert>
#include <array>
#include <memory>
#include <fstream>

#include "json.hpp"
#include "global_setup.h"

extern nlohmann::json j_conf;
// ==============================read data from json================================
// pwd
extern const std::string WorkDir_json;
// run
extern const size_t OutTimeMethod_json;
extern const size_t nOutTimeStamps_json;
extern const real_t OutTimeStart_json; // OutTimeBeginning
extern const real_t OutTimeStamp_json; // OutTimeInterval
extern const real_t StartTime_json;
extern const real_t EndTime_json;
extern const size_t NumThread_json;
extern const std::string OutputDir_json;
extern const size_t OutBoundary_json;
extern const size_t OutDIRX_json;
extern const size_t OutDIRY_json;
extern const size_t OutDIRZ_json;
extern const size_t OutDAT_json;
extern const size_t OutVTI_json;
extern const size_t OutSTL_json;
extern const size_t outpos_x_json;
extern const size_t outpos_y_json;
extern const size_t outpos_z_json;
extern const size_t nStepmax_json;
extern const size_t nOutput_json; // nOutMax
extern const size_t OutInterval_json;
extern const size_t POutInterval_json; // PushInterval
extern const size_t BlockSize_json;	   // DtBlockSize
extern const size_t dim_block_x_json;  // blockSize_xyz
extern const size_t dim_block_y_json;
extern const size_t dim_block_z_json;

// MPI
extern const size_t mx_json;
extern const size_t my_json;
extern const size_t mz_json;
extern std::vector<size_t> DeviceSelect_json;

// mesh
extern const real_t Domain_length_json; // DOMAIN_length
extern const real_t Domain_width_json;	// DOMAIN_width
extern const real_t Domain_height_json; // DOMAIN_height
extern const real_t Domain_xmin_json;	// xmin
extern const real_t Domain_ymin_json;	// ymin
extern const real_t Domain_zmin_json;	// zmin
extern const real_t LRef_json;
extern const size_t X_inner_json;
extern const size_t Y_inner_json;
extern const size_t Z_inner_json;
extern const size_t Bwidth_X_json;
extern const size_t Bwidth_Y_json;
extern const size_t Bwidth_Z_json;
extern const real_t CFLnumber_json;
extern const size_t NUM_BISD_json;
extern const real_t width_xt_json;
extern const real_t width_hlf_json;
extern const real_t mx_vlm_json;
extern const real_t ext_vlm_json;
extern const real_t BandforLevelset_json;
extern const std::vector<size_t> NBoundarys_json; // BoundaryBundles
extern const std::vector<size_t> Boundary_x_json; // BoundaryBundle_xyz
extern const std::vector<size_t> Boundary_y_json;
extern const std::vector<size_t> Boundary_z_json;
extern const std::vector<size_t> Boundarys_json;

// fluid
extern const std::string fname_json; // Fluid_Names
// extern const string

// init
extern const size_t Mach_Modified_json; // Mach_modified
extern const real_t Ma_json;			// blast_mach
extern const size_t blast_type_json;
extern const real_t blast_center_x_json;
extern const real_t blast_center_y_json;
extern const real_t blast_center_z_json;
extern const real_t blast_radius_json;
extern const real_t blast_density_out_json;
extern const real_t blast_pressure_out_json;
extern const real_t blast_T_out_json;
extern const real_t blast_u_out_json;
extern const real_t blast_v_out_json;
extern const real_t blast_w_out_json;
extern const real_t blast_density_in_json;
extern const real_t blast_pressure_in_json;
extern const real_t blast_T_in_json;
extern const real_t blast_u_in_json;
extern const real_t blast_v_in_json;
extern const real_t blast_w_in_json;
extern const size_t cop_type_json;
extern const real_t cop_center_x_json;
extern const real_t cop_center_y_json;
extern const real_t cop_center_z_json;
extern const real_t cop_density_in_json;
extern const real_t cop_pressure_in_json;
extern const real_t cop_T_in_json;
extern const real_t cop_y1_in_json;
extern const real_t cop_y1_out_json;
extern const real_t xa_json; // bubble_shape_x
extern const real_t yb_first_json;
extern const real_t zc_first_json;
extern const real_t yb_json;			  // bubble_shape_ratioy
extern const real_t zc_json;			  // bubble_shape_ratioz
extern const real_t bubble_boundary_json; // bubble_boundary_cells;
extern const real_t C_json;

/**
 * @brief divise an interval according to the start, end and period
 *
 * @tparam T
 * @param start
 * @param end
 * @param period
 * @param num
 * @return vector<T>
 */
template <class T>
std::vector<T> IntervalDivision(const T &start, const T &end, const T &period, size_t num)
{
	std::vector<T> tmp_vector;
	tmp_vector.reserve(num);
	for (size_t i = 0; i < num - 1; ++i)
	{
		T tmp = start + i * period;
		tmp_vector.push_back(tmp);
	}
	tmp_vector.push_back(end);

	return tmp_vector;
}
// read json from file
bool ReadJson(std::string filename, nlohmann::json &j);

// compute Dmin_json
real_t ComputeDminJson();

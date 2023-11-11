#include "read_json.h"

using namespace std;
nlohmann::json j_conf;
using interfunc = std::function<int(int *, int)>;
// =======================================================
// // // for json file read
// =======================================================

std::string json_path = "sa-1d-diffusion-mpi.json";
bool is_read_json_success = ReadJson(json_path, j_conf);

// ===================read data from json=======================
// ============three methods for get data from JSON==============
// const size_t dim_block_x_json = j_conf["run"]["blockSize_x"];
// const size_t dim_block_x_json = j_conf.at("run").value("blockSize_x",0);
// const size_t dim_block_x_json = j_conf.at("run").at("blockSize_x");
// ==============================================================
// Run setup
const size_t OutTimeMethod_json = j_conf.at("run").value("OutTimeMethod", 1);
const size_t nOutTimeStamps_json = 1 + j_conf.at("run").value("nOutTimeStamps", 0);
const real_t OutTimeStart_json = j_conf.at("run").value("OutTimeBeginning", 0.0);
const real_t OutTimeStamp_json = j_conf.at("run").value("OutTimeInterval", 0.0);
const real_t StartTime_json = j_conf.at("run").value("StartTime", 0.0);
const real_t EndTime_json = j_conf.at("run").value("EndTime", 0.0);
const size_t NumThread_json = j_conf.at("run").value("NumThread", 8);
const std::string OutputDir_json = j_conf.at("run").at("OutputDir");
const size_t OutBoundary_json = j_conf.at("run").value("OutBoundary", 0);
const size_t OutDIRX_json = j_conf.at("run").value("OutDIRX", DIM_X);
const size_t OutDIRY_json = j_conf.at("run").value("OutDIRY", DIM_Y);
const size_t OutDIRZ_json = j_conf.at("run").value("OutDIRZ", DIM_Z);
const size_t OutDAT_json = j_conf.at("run").value("OutDAT", 1);
const size_t OutVTI_json = j_conf.at("run").value("OutVTI", 0);
const size_t OutSTL_json = j_conf.at("run").value("OutSTL", 0);
const size_t outpos_x_json = j_conf.at("run").value("outpos_x", 0);
const size_t outpos_y_json = j_conf.at("run").value("outpos_y", 0);
const size_t outpos_z_json = j_conf.at("run").value("outpos_z", 0);
const size_t nStepmax_json = j_conf.at("run").value("nStepMax", 10);
const size_t nOutput_json = j_conf.at("run").value("nOutMax", 0);
const size_t OutInterval_json = j_conf.at("run").value("OutInterval", nStepmax_json);
const size_t POutInterval_json = j_conf.at("run").value("PushInterval", 5);
const size_t BlockSize_json = j_conf.at("run").value("DtBlockSize", 4);
const size_t dim_block_x_json = DIM_X ? j_conf.at("run").value("blockSize_x", BlockSize_json) : 1;
const size_t dim_block_y_json = DIM_Y ? j_conf.at("run").value("blockSize_y", BlockSize_json) : 1;
const size_t dim_block_z_json = DIM_Z ? j_conf.at("run").value("blockSize_z", BlockSize_json) : 1;

// MPI setup
const size_t mx_json = DIM_X ? j_conf.at("mpi").value("mx", 1) : 1;
const size_t my_json = DIM_Y ? j_conf.at("mpi").value("my", 1) : 1;
const size_t mz_json = DIM_Z ? j_conf.at("mpi").value("mz", 1) : 1;
vector<size_t> DeviceSelect_json_fault = {1, 1, 0};
vector<size_t> DeviceSelect_json = j_conf.at("mpi").value("DeviceSelect", DeviceSelect_json_fault);

// Mesh setup
const real_t Domain_length_json = j_conf.at("mesh").value("DOMAIN_length", 1.0);
const real_t Domain_width_json = j_conf.at("mesh").value("DOMAIN_width", 1.0);
const real_t Domain_height_json = j_conf.at("mesh").value("DOMAIN_height", 1.0);
const real_t Domain_xmin_json = j_conf.at("mesh").value("xmin", 0.0);
const real_t Domain_ymin_json = j_conf.at("mesh").value("ymin", 0.0);
const real_t Domain_zmin_json = j_conf.at("mesh").value("zmin", 0.0);
const real_t LRef_json = j_conf.at("mesh").value("LRef", 1.0);
const size_t X_inner_json = DIM_X ? j_conf.at("mesh").value("X_inner", 1) : 1;
const size_t Y_inner_json = DIM_Y ? j_conf.at("mesh").value("Y_inner", 1) : 1;
const size_t Z_inner_json = DIM_Z ? j_conf.at("mesh").value("Z_inner", 1) : 1;
const size_t Bwidth_X_json = DIM_X ? j_conf.at("mesh").value("Bwidth_X", 4) : 0;
const size_t Bwidth_Y_json = DIM_Y ? j_conf.at("mesh").value("Bwidth_Y", 4) : 0;
const size_t Bwidth_Z_json = DIM_Z ? j_conf.at("mesh").value("Bwidth_Z", 4) : 0;
const real_t CFLnumber_json = j_conf.at("mesh").value("CFLnumber", 0.6);
const size_t NUM_BISD_json = j_conf.at("mesh").value("NUM_BISD", 1);
const real_t width_xt_json = j_conf.at("mesh").value("width_xt", 4.0);
const real_t width_hlf_json = j_conf.at("mesh").value("width_hlf", 2.0);
const real_t mx_vlm_json = j_conf.at("mesh").value("mx_vlm", 0.5);
const real_t ext_vlm_json = j_conf.at("mesh").value("ext_vlm", 0.5);
const real_t BandforLevelset_json = j_conf.at("mesh").value("BandforLevelset", 6.0);
const vector<size_t> NBoundarys_json_fault = {2, 2, 2};
const vector<size_t> Boundary_x_json_fault = {2, 0, 0, 0, 0, 0, 0, 1};
const vector<size_t> Boundary_y_json_fault = {2, 0, 0, 0, 0, 0, 0, 1};
const vector<size_t> Boundary_z_json_fault = {2, 0, 0, 0, 0, 0, 0, 1};
const vector<size_t> Boundarys_json_fault = {3, 3, 3, 3, 3, 3};
const vector<size_t> NBoundarys_json = j_conf.at("mesh").value("BoundaryBundles", NBoundarys_json_fault);
const vector<size_t> Boundary_x_json = j_conf.at("mesh").value("BoundaryBundle_x", Boundary_x_json_fault);
const vector<size_t> Boundary_y_json = j_conf.at("mesh").value("BoundaryBundle_y", Boundary_y_json_fault);
const vector<size_t> Boundary_z_json = j_conf.at("mesh").value("BoundaryBundle_z", Boundary_z_json_fault);
const vector<size_t> Boundarys_json = j_conf.at("mesh").value("Boundarys", Boundarys_json_fault);

// fluid setup
const string fname_json = j_conf.at("fluid").value("Fluid_Names", "");

// Init setup
const size_t Mach_Modified_json = j_conf.at("init").value("Mach_modified", 1);
const real_t Ma_json = j_conf.at("init").value("blast_mach", 0.0);
const size_t blast_type_json = j_conf.at("init").value("blast_type", 0);
const real_t blast_center_x_json = j_conf.at("init").value("blast_center_x", 0.0);
const real_t blast_center_y_json = j_conf.at("init").value("blast_center_y", 0.0);
const real_t blast_center_z_json = j_conf.at("init").value("blast_center_z", 0.0);
const real_t blast_radius_json = j_conf.at("init").value("blast_radius", 0.0);
const real_t blast_density_out_json = j_conf.at("init").value("blast_density_out", 0.0);
const real_t blast_pressure_out_json = j_conf.at("init").value("blast_pressure_out", 0.0);
const real_t blast_T_out_json = j_conf.at("init").value("blast_tempreture_out", 298.15);
const real_t blast_u_out_json = j_conf.at("init").value("blast_u_out", 0.0);
const real_t blast_v_out_json = j_conf.at("init").value("blast_v_out", 0.0);
const real_t blast_w_out_json = j_conf.at("init").value("blast_w_out", 0.0);
const real_t blast_density_in_json = j_conf.at("init").value("blast_density_in", 0.0);
const real_t blast_pressure_in_json = j_conf.at("init").value("blast_pressure_in", 0.0);
const real_t blast_T_in_json = j_conf.at("init").value("blast_tempreture_in", 298.15);
const real_t blast_u_in_json = j_conf.at("init").value("blast_u_in", 0.0);
const real_t blast_v_in_json = j_conf.at("init").value("blast_v_in", 0.0);
const real_t blast_w_in_json = j_conf.at("init").value("blast_w_in", 0.0);
const size_t cop_type_json = j_conf.at("init").value("cop_type", 0);
const real_t cop_center_x_json = j_conf.at("init").value("cop_center_x", 0.0);
const real_t cop_center_y_json = j_conf.at("init").value("cop_center_y", 0.0);
const real_t cop_center_z_json = j_conf.at("init").value("cop_center_z", 0.0);
const real_t cop_density_in_json = j_conf.at("init").value("cop_density_in", blast_density_out_json);
const real_t cop_pressure_in_json = j_conf.at("init").value("cop_pressure_in", blast_pressure_out_json);
const real_t cop_T_in_json = j_conf.at("init").value("cop_tempreture_in", blast_T_out_json);
const real_t cop_y1_in_json = j_conf.at("init").value("cop_y1_in", 0.0);
const real_t cop_y1_out_json = j_conf.at("init").value("cop_y1_out", 0.0);
real_t Dmin_json = ComputeDminJson();
const real_t xa_json = j_conf.at("init").value("bubble_shape_x", 0.4 * Dmin_json);
const real_t yb_first_json = xa_json / j_conf.at("init").value("bubble_shape_ratioy", 1.0);
const real_t zc_first_json = xa_json / j_conf.at("init").value("bubble_shape_ratioz", 1.0);
const real_t yb_json = j_conf.at("init").value("bubble_shape_y", yb_first_json);
const real_t zc_json = j_conf.at("init").value("bubble_shape_z", zc_first_json);
const real_t bubble_boundary_json = j_conf.at("init").value("bubble_boundary_cells", 2);
real_t bubble_boundary_width_default = mx_json * X_inner_json * bubble_boundary_json;
const real_t C_json = xa_json * j_conf.at("init").value("bubble_boundary_width", bubble_boundary_width_default);

bool ReadJson(string filename, nlohmann::json &j)
{
	ifstream in(filename.c_str());
	if (!in)
	{
		cerr << ": error while reading config.json" << endl;
		exit(EXIT_FAILURE);
	}
	vector<string> vec_in;
	string str;
	while (getline(in, str))
	{
		vec_in.push_back(str);
	}
	in.close();

	stringstream out;

	for (auto iter : vec_in)
	{
		size_t pos_comment = iter.find_first_of("//");
		string tmp = iter.substr(0, pos_comment);
		out << tmp << endl;
	}
	out >> j;

	return true;
}

real_t ComputeDminJson()
{
	real_t Dmin_json = Domain_length_json + Domain_width_json + Domain_height_json;
#if DIM_X
	Dmin_json = std::min(Domain_length_json, Dmin_json);
#endif
#if DIM_Y
	Dmin_json = std::min(Domain_width_json, Dmin_json);
#endif
#if DIM_Z
	Dmin_json = std::min(Domain_height_json, Dmin_json);
#endif

	return Dmin_json;
}

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

#include "read_json.h"

nlohmann::json j_conf;
using interfunc = std::function<int(int *, int)>;
// ===================read data from json=======================
// ============three methods for get data from JSON==============
// const size_t dim_block_x = j_conf["run"]["blockSize_x"];
// const size_t dim_block_x = j_conf.at("run").value("blockSize_x",0);
// const size_t dim_block_x = j_conf.at("run").at("blockSize_x");
// ==============================================================

// // =======================================================
// // // // for json file read
bool is_read_json_success = ReadJson(std::string(IniFile), j_conf);
// Run setup
const std::string OutputDir = j_conf.at("run").value("OutputDir", "output");
const real_t CFLnumber_json = j_conf.at("run").value("CFLnumber", 0.4);
const real_t StartTime = j_conf.at("run").value("StartTime", 0.0); // physical time when the simulation start
const real_t EndTime = j_conf.at("run").value("EndTime", 0.0);	   // physical time when the simulation terminated
const size_t OutDAT = j_conf.at("run").value("OutDAT", 1);
const size_t OutVTI = j_conf.at("run").value("OutVTI", 0);
const size_t OutSTL = j_conf.at("run").value("OutSTL", 0);
const size_t OutOverTime = j_conf.at("run").value("OutOverTime", SBICounts);
const size_t OutBoundary = j_conf.at("run").value("OutBoundary", 0);
const size_t nStepmax_json = j_conf.at("run").value("nStepMax", 10);
const size_t nOutput = j_conf.at("run").value("nOutMax", 0);
const size_t OutInterval = j_conf.at("run").value("OutInterval", nStepmax_json);
const size_t POutInterval = j_conf.at("run").value("PushInterval", 5);
const size_t RcalInterval = j_conf.at("run").value("RcalInterval", 100);
const std::vector<std::string> OutTimeArrays_json = j_conf.at("run").value("OutTimeArrays", std::vector<std::string>{});
const std::vector<std::string> OutTimeStamps_json = j_conf.at("run").value("OutTimeStamps", std::vector<std::string>{});
// // ARA method
const bool UseAdvRange_json = j_conf.at("run").value("UseAdvRange", true);
const size_t OutAdvRange_json = j_conf.at("run").value("OutAdvRange", 1);

// MPI setup
const size_t mx_json = j_conf.at("mpi").value("mx", 1);
const size_t my_json = j_conf.at("mpi").value("my", 1);
const size_t mz_json = j_conf.at("mpi").value("mz", 1);
const std::vector<int> DeviceSelect_json = j_conf.at("mpi").value("DeviceSelect", std::vector<int>{1, 0, 0});

// Equations setup
const std::vector<std::string> Fluids_name = j_conf.at("equations").value("Fluids_Name", std::vector<std::string>{"DeF"}); // Fluid_Names
const size_t NumFluid = Fluids_name.size();
// const std::vector<std::string> species_name = j_conf.at("equations").value("Species_Name", std::vector<std::string>{});
const size_t Equ_rho = j_conf.at("equations").value("Equ_rho", 1);
const size_t Equ_energy = j_conf.at("equations").value("Equ_energy", 1);
const std::vector<size_t> Equ_momentum = j_conf.at("equations").value("Equ_momentum", std::vector<size_t>{1, 1, 1});
// // FDM Flux reconstruction solving eigenmatrix system
const bool if_overdetermined_eigen = j_conf.at("equations").value("if_overdetermined_eigen", false);
// // Positivity Preserving
const bool PositivityPreserving = j_conf.at("equations").value("PositivityPreserving", POSP);
// // Viscosity
const std::vector<real_t> Tnode = j_conf.at("equations").value("ViscosityFittingTnode", std::vector<real_t>{273.15, 500.0, 750.0, 1000.0, 1250.0, 1500.0, 1750.0, 2000.0, 2250.0, 2500.0, 2750.0, 3000.0, 5000.0});
// // ODE Solver
const bool ReactSources = j_conf.at("equations").value("Sources_React", COP_CHEME);
const std::string SlipOrder = j_conf.at("equations").value("SlipOrder", CHEME_SPLITTING);
const std::string ODESolver = j_conf.at("equations").value("ODESolver", "Q2");

// // Test samples based functionity
/*** @brief params of ODETest sample 0D flame autoignition test
 * @param ODETest_json: if enable this sample
 * @param ODETestRange[0]: beginning pressure: Pa
 * @param ODETestRange[1]: beginning Temperature: T
 * @param ODETestRange[2]: equilibrium /delta Temperature to end test
 * @param ODETestRange[3]: time to end test
 * @param ODETestRange[4]: time step /delta t to evolution test
 * @param ODETestRange[5]: max evolution steps to end test
 * */
const bool ODETest_json = j_conf.at("equations").value("if_ODReactionTest", ReactSources);
const std::vector<real_t> ODETestRange = j_conf.at("equations").value("ODETestRange", std::vector<real_t>{101325.0, 1150, 1.0E-5, 0.2, 1.0E-5, 10000});

/*** @brief params of Viscosity_coefficients-Temperature sample  test
 * @param ViscosityTest_json: if enable this test
 * @param ViscosityTestRange[0]: beginning Temperature: T
 * @param ViscosityTestRange[1]: Ending Temperature: T
 * */
const bool ViscosityTest_json = j_conf.at("equations").value("if_ViscosityTest", Visc);
const std::vector<real_t> ViscosityTestRange = j_conf.at("equations").value("ViscosityTestRange", std::vector<real_t>{0, 5000});

// Mesh setup
const size_t NUM_BISD = j_conf.at("mesh").value("NUM_BISD", 1);
const real_t width_xt = j_conf.at("mesh").value("width_xt", 4.0);				// Extending width (cells)
const real_t width_hlf = j_conf.at("mesh").value("width_hlf", 2.0);				// Ghost-fluid update width
const real_t mx_vlm = j_conf.at("mesh").value("mx_vlm", 0.5);					// Cells with volume fraction less than this value will be mixed
const real_t ext_vlm = j_conf.at("mesh").value("ext_vlm", 0.5);					// For cells with volume fraction less than this value, their states are updated based on mixing
const real_t BandforLevelset = j_conf.at("mesh").value("BandforLevelset", 6.0); // half-width of level set narrow band
const real_t Yil_limiter_json = j_conf.at("mesh").value("Yil_limiter", 1.0E10);
const real_t Dim_limiter_json = j_conf.at("mesh").value("Dim_limiter", 2.0E-3);
const std::vector<real_t> Refs = j_conf.at("mesh").value("Refs", std::vector<real_t>{1.0});
const std::vector<real_t> DOMAIN_Size = j_conf.at("mesh").value("DOMAIN_Size", std::vector<real_t>{1.0, 1.0, 1.0}); // XYZ
const std::vector<real_t> Domain_medg = j_conf.at("mesh").value("DOMAIN_Medg", std::vector<real_t>{0.0, 0.0, 0.0}); // XYZ
const std::vector<size_t> Inner = j_conf.at("mesh").value("Resolution", std::vector<size_t>{1, 0, 0});				// XYZ
const std::vector<bool> Dimensions = std::vector<bool>{bool(Inner[0]), bool(Inner[1]), bool(Inner[2])};
const size_t BlockSize_json = j_conf.at("run").value("DtBlockSize", 4);
const size_t dim_block_x_json = Dimensions[0] ? j_conf.at("run").value("blockSize_x", BlockSize_json) : 1;
const size_t dim_block_y_json = Dimensions[1] ? j_conf.at("run").value("blockSize_y", BlockSize_json) : 1;
const size_t dim_block_z_json = Dimensions[2] ? j_conf.at("run").value("blockSize_z", BlockSize_json) : 1;
const std::vector<size_t> Bwidth = j_conf.at("mesh").value("Ghost_width", std::vector<size_t>{4, 4, 4}); // XYZ
/* Simple Boundary settings */
const std::vector<size_t> NBoundarys = j_conf.at("mesh").value("BoundaryBundles", std::vector<size_t>{2, 2, 2});		// BoundaryBundles
const std::vector<size_t> Boundarys_json = j_conf.at("mesh").value("Boundarys", std::vector<size_t>{2, 2, 2, 2, 2, 2}); // simple Boundarys settings
// TODO  // BoundaryBundle_xyz
// const std::vector<std::vector<size_t>> Boundary_x = j_conf.at("mesh").value("BoundaryBundle_x", std::vector<std::vector<size_t>>{NBoundarys[0]});
// const std::vector<std::vector<size_t>> Boundary_y = j_conf.at("mesh").value("BoundaryBundle_y", std::vector<std::vector<size_t>>{NBoundarys[1]});
// const std::vector<std::vector<size_t>> Boundary_z = j_conf.at("mesh").value("BoundaryBundle_z", std::vector<std::vector<size_t>>{NBoundarys[2]});
// std::vector<int> NBoundarys = Stringsplit<int>(configMap.getString("mesh", "BoundaryBundles", "2,2,2"));
// for (size_t ii = 0; ii < NBoundarys[0]; ii++) // X Boundary Bundles
//     Boundary_x.push_back(BoundaryRange(Stringsplit<int>(configMap.getString("mesh", "BoundaryBundle_x" + std::to_string(ii), "2,0,0,0,0,0,0,1"))));
// for (size_t jj = 0; jj < NBoundarys[1]; jj++) // Y Boundary Bundles
//     Boundary_y.push_back(BoundaryRange(Stringsplit<int>(configMap.getString("mesh", "BoundaryBundle_y" + std::to_string(jj), "2,0,0,0,0,0,0,1"))));
// for (size_t kk = 0; kk < NBoundarys[2]; kk++) // Z Boundary Bundles
//     Boundary_z.push_back(BoundaryRange(Stringsplit<int>(configMap.getString("mesh", "BoundaryBundle_z" + std::to_string(kk), "2,0,0,0,0,0,0,1"))));

// Init setup
const real_t Ma_json = j_conf.at("init").value("blast_mach", 0.0);
const bool Mach_Modified = Ma_json > 1;
const size_t cop_type = j_conf.at("init").value("cop_type", 0);
const size_t blast_type = j_conf.at("init").value("blast_type", 1);
const std::vector<real_t> cop_pos = j_conf.at("init").value("cop_center", std::vector<real_t>{0.0, 0.0, 0.0});
const std::vector<real_t> blast_pos = j_conf.at("init").value("blast_center", std::vector<real_t>{0.5, 0.5, 0.5});
// states: density, pressure, tempreture, velocity_u, velocity_v, velocity_w
const std::vector<real_t> blast_upstates = j_conf.at("init").value("blast_upstream", std::vector<real_t>{0.0, 0.0, 298.15, 0.0, 0.0, 0.0});
const std::vector<real_t> blast_downstates = j_conf.at("init").value("blast_downstream", std::vector<real_t>{0.0, 0.0, 298.15, 0.0, 0.0, 0.0});
const std::vector<real_t> cop_instates = j_conf.at("init").value("cop_inside", std::vector<real_t>{blast_downstates[0], blast_downstates[1], blast_downstates[2], 0.0, 0.0, 0.0});
const real_t xa_json = j_conf.at("init").value("bubble_shape_x", 0.4 * ComputeDminJson());
const real_t yb_first_json = xa_json / j_conf.at("init").value("bubble_shape_ratioy", 1.0);
const real_t zc_first_json = xa_json / j_conf.at("init").value("bubble_shape_ratioz", 1.0);
const real_t yb_json = j_conf.at("init").value("bubble_shape_y", yb_first_json);
const real_t zc_json = j_conf.at("init").value("bubble_shape_z", zc_first_json);
const real_t bubble_boundary = j_conf.at("init").value("bubble_boundary_cells", 2); // number of cells for cop bubble boundary
const real_t C_json = xa_json * j_conf.at("init").value("bubble_boundary_width", bubble_boundary);

bool ReadJson(std::string filename, nlohmann::json &j)
{
	// std::cout << "Read json configure: " << filename << std::endl;
	std::ifstream in(filename.c_str());
	if (!in)
	{
		std::cerr << ": error while reading json configure file." << std::endl;
		exit(EXIT_FAILURE);
	}
	std::vector<std::string> vec_in;
	std::string str;
	while (getline(in, str))
	{
		vec_in.push_back(str);
	}
	in.close();

	std::stringstream out;

	for (auto iter : vec_in)
	{
		size_t pos_comment = iter.find_first_of("//");
		std::string tmp = iter.substr(0, pos_comment);
		out << tmp << std::endl;
	}
	out >> j;

	return true;
}

real_t ComputeDminJson()
{
	real_t Dmin = DOMAIN_Size[0] + DOMAIN_Size[1] + DOMAIN_Size[2];

	if (Dimensions[0])
		Dmin = std::min(DOMAIN_Size[0], Dmin);
	if (Dimensions[1])
		Dmin = std::min(DOMAIN_Size[1], Dmin);
	if (Dimensions[2])
		Dmin = std::min(DOMAIN_Size[2], Dmin);

	return Dmin;
}

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

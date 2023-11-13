#pragma once

#include "config/ConfigMap.h"
#include "marcos/marco_thermal.h"
#include "../settings/read_json.h"
#include "../read_grid/readgrid.h"
#ifdef USE_MPI
#include "../mpiPacks/mpiPacks.h"
#endif

struct AppendParas
{
	int argc;
	char **argv;
	AppendParas(){};
	~AppendParas(){};
	AppendParas(int argc, char **argv) : argc(argc), argv(argv){};
	template <typename T>
	std::vector<T> match(std::string option);
	std::vector<std::string> match(std::string option);
};

struct Setup
{
public:
	middle::device_t q;
	Block BlSz;
	IniShape ini;
	Gridread grid;
	AppendParas apa;
	Reaction d_react, h_react;
	Thermal d_thermal, h_thermal;

	//--for-MPI&Device-----------------------
	int myRank, nRanks;
	std::vector<int> DeviceSelect; // for Device counting and selecting
	int OutInterval, POutInterval, NumThread, nStepmax, nOutput; // nOutpu: Number of output files
	int Mem_s, Block_Inner_Cell_Size, Block_Inner_Data_Size;
	int Block_Cell_Size, Block_Data_Size, bytes, cellbytes;

	//--for-Mesh-----------------------------
	real_t Domain_length, Domain_width, Domain_height;
	BConditions Boundarys[6];
	std::vector<BoundaryRange> Boundary_x, Boundary_y, Boundary_z;

	//--for-Running--------------------------
	real_t dt;
	std::string OutputDir, WorkDir;
	int OutTimeMethod, nOutTimeStamps;
	real_t StartTime, EndTime, OutTimeStart, OutTimeStamp, *OutTimeStamps;
	/**set output time with two methods below:
	 * switch method with value of OutTimeMethod:0 or 1
	 * 0. read ./runtime.dat/output_time.dat
	 * 1. set in .ini file will rewrite
	 * 		OutTime=OutTimeStart+x*outTimeStamp (0<=x<=nOutTimeStamps)
	 * 	    EndTime=OutTimeStart+OutTimeStamp*nOutTimeStamps
	 * */
	int outpos_x, outpos_y, outpos_z;
	bool OutBoundary, OutDIRX, OutDIRY, OutDIRZ, OutDAT, OutVTI, OutSTL; // debug to if it transfer;

	//--for-Fluids-----------------------------
	int NUM_BISD;
	bool mach_shock, Mach_Modified;
	// Mach_Modified: if post-shock theroy in Ref0 used(rewrite the value from Devesh Ranjan's theroy in Ref1 used by default):
	// // Ref0:https://doi.org/10.1016/j.combustflame.2015.10.016 Ref1:https://www.annualreviews.org/doi/10.1146/annurev-fluid-122109-160744
	std::vector<std::string> fname; //, species_name // give a name to the fluid
	// material properties: 0:material_kind, 1:phase_indicator, 2:gamma, 3:A, 4:B, 5:rho0, 6:R_0, 7:lambda_0, 8:a(rtificial)s(peed of)s(ound)
	// // material_kind: type of material, 0: gamma gas, 1: water, 2: stiff gas ;// fluid indicator and EOS Parameters
	std::vector<std::vector<real_t>> material_props, species_ratio;
	real_t bubble_boundary; // number of cells for cop bubble boundary
	real_t width_xt;		// Extending width (cells)
	real_t width_hlf;		// Ghost-fluid update width
	real_t mx_vlm;			// Cells with volume fraction less than this value will be mixed
	real_t ext_vlm;			// For cells with volume fraction less than this value, their states are updated based on mixing
	real_t BandforLevelset; // half-width of level set narrow band

	//-----------------*backwardArrhenius------------------//
	bool BackArre = false;
	std::vector<std::vector<int>> reaction_list{NUM_SPECIES}, reactant_list{NUM_REA}, product_list{NUM_REA}, species_list{NUM_REA};

	Setup(int argc, char **argv, int rank = 0, int nranks = 1);
	void ReadIni(ConfigMap configMap);
	void ReWrite();
	void init();
	void print(); // only print once when is_print = false
	void CpyToGPU();
	void ReadSpecies(); // pure or specie names && ratio in mixture for reaction
	void ReadThermal();
	void get_Yi(real_t *yi);
	bool Mach_Shock();
	real_t Enthalpy(const real_t T0, const int n);
	real_t get_Coph(const real_t *yi, const real_t T);
	real_t get_CopGamma(const real_t *yi, const real_t T);
	real_t HeatCapacity(real_t *Hia, const real_t T0, const real_t Ri, const int n);

#ifdef COP_CHEME
	void ReadReactions();
	void IniSpeciesReactions();
	void ReactionType(int flag, int i, int *Nuf, int *Nub);
#endif // end COP_CHEME

#ifdef Visc
	real_t Omega_table[2][37][8];	  // collision integral for viscosity and thermal conductivity(first index=0) & binary diffusion coefficient(first index=1)
	real_t delta_star[8], T_star[37]; // reduced temperature and reduced dipole moment,respectively;
	void ReadOmega_table();
	void GetFitCoefficient();
	void Fitting(real_t *specie_k, real_t *specie_j, real_t *aa, int indicator);
	real_t Omega_interpolated(real_t Tstar, real_t deltastar, int index);
	real_t viscosity(real_t *specie, const real_t T);
	real_t thermal_conductivities(real_t *specie, const real_t T, const real_t PP);
	real_t Dkj(real_t *specie_k, real_t *specie_j, const real_t T, const real_t PP); // PP:pressure,unit:Pa
#endif

#ifdef USE_MPI
	MpiTrans *mpiTrans;
#endif // end USE_MPI
};

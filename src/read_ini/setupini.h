#pragma once

// // global header
#include "../include/global_setup.h"
// // internal header
#include "options.hpp"
#include "inishape/inishape.h"
#include "settings/read_json.h"
#include "src/outformat/outformat.h"
// // external header
#ifdef USE_MPI
#include "../mpiPacks/mpiPacks.h"
#endif
#include "../read_grid/readgrid.h"

struct Setup
{
public:
	//--for-MPI&Device-----------------------
	int myRank, nRanks;
	middle::device_t q;
#ifdef USE_MPI
	MpiTrans *mpiTrans;
#endif
	std::vector<int> DeviceSelect; // for Device counting and selecting

	//--for-Running--------------------------
	AppendParas apa;
	std::string WorkDir;
	int nStepmax;		  // running steps
	int bytes, cellbytes; // memory allocate
	std::vector<OutFmt> OutTimeStamps;

	//--for-Mesh-----------------------------
	Block BlSz;
	Gridread grid;
	BConditions Boundarys[6];
	std::vector<BoundaryRange> Boundary_x, Boundary_y, Boundary_z;

	//--for-Fluids-----------------------------
	IniShape ini;
	bool mach_shock;
	Thermal d_thermal, h_thermal;
	std::vector<std::string> species_name{NumFluid};
	std::vector<std::vector<real_t>> material_props{NumFluid};
	// material properties: 0:material_kind, 1:phase_indicator, 2:gamma, 3:A, 4:B, 5:rho0, 6:R_0, 7:lambda_0, 8:a(rtificial)s(peed of)s(ound)
	// // material_kind: type of material, 0: gamma gas, 1: water, 2: stiff gas ;// fluid indicator and EOS Parameters

	Setup(int argc, char **argv, int rank = 0, int nranks = 1);
	void ReadIni();
	void ReWrite();
	void init();
	void print(); // only print once when is_print = false
	void CpyToGPU();
	void ReadSpecies(); // pure or specie names && ratio in mixture for reaction
	void ReadThermal();
	bool Mach_Shock();

	//--for-viscosity--------------------------
	real_t Omega_table[2][37][8];	  // collision integral for viscosity and thermal conductivity(first index=0) & binary diffusion coefficient(first index=1)
	real_t delta_star[8], T_star[37]; // reduced temperature and reduced dipole moment,respectively;
	void ReadOmega_table();
	void GetFitCoefficient();
	void VisCoeffsAccuracyTest(real_t Tmin, real_t Tmax);
	real_t Omega_interpolated(real_t Tstar, real_t deltastar, int index);
	void Fitting(std::vector<real_t> TT, real_t *specie_k, real_t *specie_j, real_t *aa, int indicator);
	real_t viscosity(real_t *specie, const real_t T);								 // accurate kinematic vicosity coefficient
	real_t thermal_conductivities(real_t *specie, const real_t T, const real_t PP);	 // accurate furier heat transfer coefficient
	real_t Dkj(real_t *specie_k, real_t *specie_j, const real_t T, const real_t PP); // accurate binary diffusion coefficient // PP:pressure,unit:Pa

	//--for-reacting---------------------------
	Reaction d_react, h_react;
	std::vector<std::vector<int>> reaction_list{NUM_SPECIES}, reactant_list{NUM_REA}, product_list{NUM_REA}, species_list{NUM_REA};
	void ReadReactions();
	void IniSpeciesReactions();
	bool ReactionType(int flag, int i, int *Nuf, int *Nub);
};

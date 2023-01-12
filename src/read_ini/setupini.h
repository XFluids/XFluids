#pragma once

#include <vector>
#include <cstring>
#include "global_setup.h"
#include "config/ConfigMap.h" //use readini
#ifdef COP
#include "case_setup.h"
#endif
#ifdef USE_MPI
#include "mpiPacks.h"
#endif

typedef struct
{
	int blast_type, cop_type; // blast_type: 0 for 1d shock , 1 for circular shock //cop_type: 0 for 1d set   , 1 for bubble of cop
	real_t blast_center_x, blast_center_y, blast_center_z, blast_radius,
		blast_density_in, blast_density_out, blast_pressure_in, blast_pressure_out,
		blast_T_in, blast_T_out,
		blast_u_in, blast_v_in, blast_w_in, blast_u_out, blast_v_out, blast_w_out;
	real_t cop_center_x, cop_center_y, cop_center_z, cop_radius, cop_density_in,
		cop_pressure_in, cop_T_in, cop_y1_in, cop_y1_out;
	real_t Ma, xa, yb, zc, C, _xa2, _yb2, _zc2, _xa2_in, _yb2_in, _zc2_in, _xa2_out, _yb2_out, _zc2_out; //  shock much number
	real_t bubble_center_x, bubble_center_y, bubble_center_z, bubbleSz;									 //  Note: Domain_length may be the max value of the Domain size
} IniShape;

typedef struct
{
	real_t *species_chara, *Ri, *Wi, *_Wi, *Hia, *Hib, *species_ratio_in, *species_ratio_out, *xi_in, *xi_out; // Ri=Ru/Wi;
	real_t *fitted_coefficients_visc[NUM_SPECIES], *fitted_coefficients_therm[NUM_SPECIES];				 // length: order_polynominal_fitted
	real_t *Dkj_matrix[NUM_SPECIES * NUM_SPECIES];
} Thermal;

#ifdef COP_CHEME
typedef struct
{
	int *Nu_b_, *Nu_f_, *Nu_d_, *react_type, *third_ind; // for forward && back reaction
	real_t *React_ThirdCoef, *Rargus;					 // 从文件读的Reaction参数数据放在Rargus这个指针的空间里方便GPU调用，排布顺序是A,B,E,AA,BB,EE
	int *reaction_list[NUM_SPECIES], *reactant_list[NUM_REA], *product_list[NUM_REA], *species_list[NUM_REA], *rns, *rts, *pls, *sls;
} Reaction;
#endif // end COP_CHEME

struct Setup
{
public:
	middle::device_t q;
	Block BlSz;
	IniShape ini;
	Thermal d_thermal, h_thermal;
	int Mem_s;
	bool mach_shock;
	int myRank, nRanks;
#ifdef COP_CHEME
	Reaction d_react, h_react;
#endif // end COP

	//--for-Mesh-----------------------------
	int NUM_BISD;
	int BLOCK_ratio[3];
	BConditions Boundarys[6];
	real_t DOMAIN_length; // 设定最长边大小
	real_t Domain_length, Domain_width, Domain_height;
	real_t dt;
	/**set output time with two methods below:
	 * switch method with value of OutTimeMethod:0 or 1
	 * 0. read ./runtime.dat/output_time.dat
	 * 1. set in .ini file will rewrite
	 * 		OutTime=OutTimeStart+x*outTimeStamp (0<=x<=nOutTimeStamps)
	 * 	    EndTime=OutTimeStart+OutTimeStamp*nOutTimeStamps
	 * */
	int OutTimeMethod, nOutTimeStamps;
	real_t StartTime, EndTime, OutTimeStart, OutTimeStamp;
	real_t *OutTimeStamps;
	std::string OutputDir;
	bool OutBoundary, OutDIRX, OutDIRY, OutDIRZ; // debug to if it transfer;
	bool Mach_Modified;			// if post-shock theroy in Ref0 used(rewrite the value from Devesh Ranjan's theroy in Ref1 used by default): Ref0:https://doi.org/10.1016/j.combustflame.2015.10.016 Ref1:https://www.annualreviews.org/doi/10.1146/annurev-fluid-122109-160744
	int outpos_x, outpos_y, outpos_z;
	real_t bubble_boundary;		// number of cells for cop bubble boundary
	real_t width_xt;			// Extending width (cells)
	real_t width_hlf;			// Ghost-fluid update width
	real_t mx_vlm;				// Cells with volume fraction less than this value will be mixed
	real_t ext_vlm;				// For cells with volume fraction less than this value, their states are updated based on mixing
	real_t BandforLevelset;		// half-width of level set narrow band

	//--for-Fluids-----------------------------
	std::string fname[NumFluid];		// give a name to the fluid
	int material_kind[NumFluid];		// type of material, 0: gamma gas, 1: water, 2: stiff gas ;// fluid indicator and EOS Parameters
	real_t material_props[NumFluid][8]; // material properties:1: phase_indicator, 2:gamma, 3:A, 4:B, 5:rho0, 6:R_0, 7:lambda_0, 8:a(rtificial)s(peed of)s(ound)
	std::string species_name[NUM_SPECIES];
#ifdef COP_CHEME
	//-----------------*backwardArrhenius------------------//
	bool BackArre = false;
	//-----------------*backwardArrhenius------------------//
	std::vector<int> reaction_list[NUM_SPECIES]; // 数组的每一个元素都是一个vector
	std::vector<int> reactant_list[NUM_REA];
	std::vector<int> product_list[NUM_REA];
	std::vector<int> species_list[NUM_REA];
#endif // COP_CHEME
	//--for-Host-&&-Device--------------------------
	int nStepmax;
	int nOutput; // Number of output files
	int OutInterval, POutInterval;
	int NumThread;
	int Block_Inner_Cell_Size;
	int Block_Inner_Data_Size;
	int Block_Cell_Size;
	int Block_Data_Size;
	int bytes, cellbytes;

	Setup(ConfigMap &configMap, middle::device_t &Q);
	void ReadIni(ConfigMap &configMap);
	void init();
	void print(); // only print once when is_print = false
	void CpyToGPU();
	void ReadSpecies(); // pure or specie names && ratio in mixture for reaction
	void ReadThermal();
	void get_Yi(real_t *yi);
	bool Mach_Shock();
	real_t HeatCapacity(real_t *Hia, const real_t T0, const real_t Ri, const int n);
	real_t Enthalpy(const real_t T0, const int n);
	real_t get_Coph(const real_t yi[NUM_SPECIES], const real_t T);
	real_t get_CopGamma(const real_t yi[NUM_SPECIES], const real_t T);
#ifdef COP_CHEME
	void ReadReactions();
	void IniSpeciesReactions();
	void ReactionType(int flag, int i, int *Nuf, int *Nub);
#endif // end COP_CHEME

#ifdef Visc
	double Omega_table[2][37][8];	  // collision integral for viscosity and thermal conductivity(first index=0) & binary diffusion coefficient(first index=1)
	double delta_star[8], T_star[37]; // reduced temperature and reduced dipole moment,respectively;
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

#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <sstream>
#include <iostream>
// //  using sample settings
#ifdef COP
#include "case_setup.h"
#include "Eigen_global_definition.h"
#endif

#include "compile_sycl.h"
#include "global_marco.h"
#include "global_undef.h"

#define MAX_SPECIES NUM_SPECIES

// =======================================================
// //    Global Precision settings
#ifdef USE_DOUBLE
#define real_t double // #define real_t double;
#define _DF(a) a
#else
#define real_t float // #define real_t float;
#define _DF(a) a##f
#endif //  USE_DOUBLE

#define Interface_line _DF(0.01)
// =======================================================
// //    Global __device__ constant
const int SPCH_Sz = 9; // number of characteristic of compoent,species_cahra[NUM_SPECIES*SPCH_Sz]
const int order_polynominal_fitted = 4;

const real_t _OT = (_DF(1.0) / _DF(3.0));
const real_t _sxtn = _DF(1.0) / _DF(16.0);
const real_t _twfr = _DF(1.0) / _DF(24.0);
const real_t _twle = _DF(1.0) / _DF(12.0);

//! Sqrt(2)
const real_t Sqrt2 = 1.41421356237309504880;
const real_t pi = _DF(3.1415926535897932384626433832795); // M_PI;
// =======================================================
//    viscosity constant
const real_t kB = _DF(1.3806549 * 1.0e-16); // Boltzmann constant,unit:erg/K=10e-7 J/K
const real_t NA = _DF(6.02214129 * 1.0e23); // Avogadro constant

//! Avogadro's Number @f$ N_{\mathrm{A}} @f$ [number/kmol]
const real_t Avogadro = _DF(6.02214076e26);
//! Boltzmann constant @f$ k @f$ [J/K]
const real_t Boltzmann = _DF(1.380649e-23);

const real_t universal_gas_const = Avogadro * Boltzmann; // J/(K mol)/K, "R0", p V = n R0 T
const real_t Ru = universal_gas_const * _DF(1.0E-3);	 // J/mol/K

const real_t p_atm = _DF(1.01325e5); // unit: Pa = kg m^-1 s^-2
const real_t Tref = _DF(1.0);		 // reference T

// constexpr real_t Gamma = 1.4; // 1.666667;
// // for flux Reconstruction order
#define FLUX_method 2 //  0: local LF; 1: global LF, 2: Roe
#if SCHEME_ORDER > 6
const int stencil_P = 3;	// "2" for <=6 order, "3"" for >6 order
const int stencil_size = 8; // "6" for <=6 order, "8"" for >6 order
#elif SCHEME_ORDER <= 6
const int stencil_P = 2;	// "2" for <=6 order, "3"" for >6 order
const int stencil_size = 6; // "6" for <=6 order, "8"" for >6 order
#endif
// for nodemisonlizing
const real_t Reference_params[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
// 0: l_ref(unit :m), 1: rho_ref(unit :kg/m3)(air), 2: p_ref(unit :Pa)(air),
// 3: T_ref(unit :K), 4:W0_ref(air mole mass,unit :g/mol) 5:μ_ref(unit:Pa.s=kg/(m.s))(air)
// 6: t_ref(unit :s), 7:ReynoldsNumber=rho_ref*u_ref*l_ref/vis_ref

// =======================================================
//    Artificial_type in Flux Reconstruction
#if 1 == Artificial_type // ROE
#define Roe_type _DF(1.0)
#define LLF_type _DF(0.0)
#define GLF_type _DF(0.0)
#elif 2 == Artificial_type // LLF
#define Roe_type _DF(0.0)
#define LLF_type _DF(1.0)
#define GLF_type _DF(0.0)
#elif 3 == Artificial_type // GLF
#define Roe_type _DF(0.0)
#define LLF_type _DF(0.0)
#define GLF_type _DF(1.0)
#endif

// =======================================================
//    global static variables
static std::string outputPrefix = "sample-undefined";

// All kinds of Boundary Conditions
enum BConditions
{
	Inflow = 0,
	Outflow = 1,
	Symmetry = 2,
	Periodic = 3,
	nslipWall = 4,
	viscWall = 5,
	slipWall = 6,
	innerBlock = 7,
#ifdef USE_MPI
	BC_COPY = 99,
	BC_UNDEFINED = 100
#endif // USE_MPI
};

enum BoundaryLocation
{
	XMIN = 0,
	XMAX = 1,
	YMIN = 2,
	YMAX = 3,
	ZMIN = 4,
	ZMAX = 5
};

enum VdeType
{
	ducx = 0,
	dvcx = 1,
	dwcx = 2,
	ducy = 3,
	dvcy = 4,
	dwcy = 5,
	ducz = 6,
	dvcz = 7,
	dwcz = 8
};

enum Specie_chara
{
	geo = 0,		// 0:monoatom,1:nonpolar(linear) molecule,2:polar molecule//极性
	epsilon_kB = 1, // epsilon: Lennard-Jones potential well depth;unit:K//势井深度
	d = 2,			// Lennard-Jones collision diameter ,unit: angstroms,10e-10m//碰撞直径in 4-3;
	mue = 3,		// dipole moment,unit:Debye(m);//偶极距
	alpha = 4,		// polarizability;unit:cubic angstrom//极化率
	Zrot_298 = 5,	// rotational relaxation collision Zrot at 298K;
	Wi = 6,			// molar mass, unit: g/mol;
	Vis = 7,		// Miu; 动力粘度 Pa/s;
	SID = 8
};

//! direction used in directional splitting scheme
enum Direction
{
	XDIR = 1,
	YDIR = 2,
	ZDIR = 3
};

enum MpiCpyType
{
	BorToBuf = 0, // Border指的是与边界相邻的四个网格点
	BufToBC = 1
};

typedef struct
{
	int Xmax, Ymax, Zmax;
	int X_inner, Y_inner, Z_inner;
	int Bwidth_X, Bwidth_Y, Bwidth_Z;
} MeshSize;

typedef struct
{
	//--for-Computational-dimensions---------
	size_t DimS;
	bool DimX, DimY, DimZ;
	real_t DimX_t, DimY_t, DimZ_t, DimS_t;
	//--for-Thread-manage--------------------
	size_t BlockSize, dim_block_x, dim_block_y, dim_block_z;
	//--for-Solving-system-------------------
	size_t num_fluids, num_species, num_cop, num_rea, num_eqn;
	//--for-Chemical-sources-----------------
	bool RSources;
	//--for-Mesh-----------------------------
	MeshSize Ms;
	int Bwidth_X, Bwidth_Y, Bwidth_Z; // Bounadry Width
	int Xmax, Ymax, Zmax, X_inner, Y_inner, Z_inner;
	//--for-Domain-size----------------------
	real_t Domain_xmin, Domain_ymin, Domain_zmin;
	real_t Domain_xmax, Domain_ymax, Domain_zmax;
	real_t Domain_length, Domain_width, Domain_height;
    //--for-hybrid-computation---------------
    #ifdef HYBRID_CALC
        int GlobalOffset_Y;
    #endif
	//--for-Discretization-------------------
	real_t dx, dy, dz, dl, _dx, _dy, _dz, _dl, CFLnumber;
	//--for-Mpi: mx means number of ranks in x direction, myMpiPos_x means location of this rank in x-dir ranks(from 0 to mx-1)
	real_t offx, offy, offz;
	int mx, my, mz, myMpiPos_x, myMpiPos_y, myMpiPos_z;
	// reference parameters, for calulate coordinate while readgrid
	real_t LRef;
	//--for-SBI-over-time-count--------------
	size_t Xe_id, N2_id;
	//--for-viscosity-Kernel-----------------
	real_t Dim_limiter, Yil_limiter;
} Block;

typedef struct
{
	real_t *species_chara, *Ri, *Wi, *_Wi, *Hia, *Hib;
	real_t *Hia_NASA, *Hib_NASA, *Hia_JANAF, *Hib_JANAF;
	real_t *species_ratio_in, *species_ratio_out, *xi_in, *xi_out;				  // Ri=Ru/Wi;
	real_t **Dkj_matrix, **fitted_coefficients_visc, **fitted_coefficients_therm; // length: order_polynominal_fitted
} Thermal;

typedef struct
{
	real_t *React_ThirdCoef, *Rargus;					 // 从文件读的Reaction参数数据放在Rargus这个指针的空间里方便GPU调用，排布顺序是A,B,E,AA,BB,EE
	int *Nu_b_, *Nu_f_, *Nu_d_, *react_type, *third_ind; // for forward && back reaction
	int **reaction_list, **reactant_list, **product_list, **species_list, *rns, *rts, *pls, *sls;
} Reaction;

struct BoundaryRange
{
	BConditions type;
	int xmin, ymin, zmin, xmax, ymax, zmax;
	int tag;
	BoundaryRange(){};
	BoundaryRange(std::vector<int> vec)
	{
		type = BConditions(vec[0]);
		xmin = vec[1], xmax = vec[2], ymin = vec[3];
		ymax = vec[4], zmin = vec[5], zmax = vec[6], tag = vec[7];
	};
};

typedef struct
{
	// primitive variables
	real_t *rho, *p, *u, *v, *w, *e, *y, *hi, *gamma, *T, *H, *Cp, *Ri;
	// sound speed at cell surfaces for Eigen System and Flux Reconstruction
	real_t *c, *_cx, *_cy, *_cz;
	// for GetTheta
	real_t *thetaXe, *thetaN2, *thetaXN;
	//  vis variables
	real_t *Vde[9], *vxs[3], *vx, *viscosity_aver, *thermal_conduct_aver, *Dkm_aver;
	// Error out: varibles of eigen system
	real_t *b1x, *b3x, *c2x, *zix, *b1y, *b3y, *c2y, *ziy, *b1z, *b3z, *c2z, *ziz;
	// Error out: prev for Flux_wall before vis addation; pstv for Flux_wall after vis addation and positive preserving
	real_t *preFwx, *preFwy, *preFwz, *pstFwx, *pstFwy, *pstFwz;
	// Error out: Ertemp1, Ertemp2: temp1,2 for Dim caculate; others for vis Flux and calculating variables of visFlux;
	real_t *Ertemp1, *Ertemp2, *visFwx, *visFwy, *visFwz;
	real_t *Dim_wallx, *hi_wallx, *Yi_wallx, *Yil_wallx;
	real_t *Dim_wally, *hi_wally, *Yi_wally, *Yil_wally;
	real_t *Dim_wallz, *hi_wallz, *Yi_wallz, *Yil_wallz;
} FlowData;

typedef struct
{
	int Mtrl_ind;
	real_t Rgn_ind;			  // indicator for region: inside interface, -1.0 or outside 1.0
	real_t Gamma, A, B, rho0; // Eos Parameters and maxium sound speed
	real_t R_0, lambda_0;	  // gas constant and heat conductivity
} MaterialProperty;

typedef struct
{
	// // box range
	real_t xmin_coor, xmax_coor, ymin_coor, ymax_coor, zmin_coor, zmax_coor;
	// // states inside box
	real_t rho, P, T, u, v, w, gamma, c, *yi, *hi, *ei, e, h, H;
} IniBox;

typedef struct
{
	// // bubble center
	real_t center_x, center_y, center_z;
	// // bubble shape
	real_t C, _xa2, _yb2, _zc2;
	// // states inside bubble
	real_t rho, P, T, u, v, w, gamma, c, *yi, *hi, *ei, e, h, H;
} IniBubble;

struct IniShape
{
	// // cop_type: 0 for 1d set, 1 for bubble of cop
	// // blast_type: 0 for 1d shock, 1 for circular shock
	int cop_type, blast_type, bubble_type;
	// // blast position
	real_t blast_center_x, blast_center_y, blast_center_z;
	// // shock much number
	real_t Ma, tau_H;
	// // blast  states
	real_t blast_density_in, blast_pressure_in, blast_T_in, blast_u_in, blast_v_in, blast_w_in, blast_c_in, blast_gamma_in;
	real_t blast_density_out, blast_pressure_out, blast_T_out, blast_u_out, blast_v_out, blast_w_out, blast_c_out, blast_gamma_out;
	// // bubble position
	real_t cop_center_x, cop_center_y, cop_center_z;
	// // bubble states
	real_t cop_density_in, cop_pressure_in, cop_T_in;
	// // bubble position; NOTE: Domain_length may be the max value of the Domain size
	real_t bubble_center_x, bubble_center_y, bubble_center_z;
	// // bubble shape
	real_t xa, yb, zc, C, _xa2, _yb2, _zc2;

	// // Utils initializing model
	size_t num_box, num_bubble;
	IniBox *iboxs;
	IniBubble *ibubbles;

	// IniShape(){};
	// ~IniShape(){};
	// IniShape(sycl::queue &q, size_t num_box, size_t num_bubble);
};

struct ZdCrtl
{
	double dt;
	int nsteps;
	double P, T;

	ZdCrtl()
	{
		dt = 1.0E-5;
		nsteps = 100;
		P = 101325.0, T = 1001.0;
	};
	ZdCrtl(double Dt, int NSteps, double Pre, double Tmp) : dt(Dt), nsteps(NSteps), T(Tmp), P(Pre){};
};

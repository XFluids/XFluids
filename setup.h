#ifndef SETUP_H
#define SETUP_H

#ifdef USE_DP
typedef double Real;
#else
typedef float Real;
#endif

#define NumFluid 1

#define DIM_X 1
#define DIM_Y 0
#define DIM_Z 0
//Block configure
//maximum number of inside cells in each block in different directions
#define X_inner 128			//224 	//x direction
#define Y_inner 1			//224 	//y direction
#define Z_inner 1					//224 	//z direction
//width of boundaries of one block
#define Bwidth_X 4 	//x direction
#define Bwidth_Y 0 	//y direction
#define Bwidth_Z 0 	//z direction

const Real DOMAIN_length =	1.0;//maxium
const int BLOCK_ratio[3] =	{1, 1, 1};
const int NUM_BISD =	1;	//number of blocks on level 0 in the shortest direction

//All kinds of Boundary Conditions
enum BConditions {
	Inflow,
	Outflow,
	Symmetry,
	Periodic,
	Wall
};
//Boundary conditions: x-, x+, y-, y+, z-, z+
const BConditions Boundarys[6] =	{Inflow, Outflow, Symmetry, Symmetry, Symmetry, Symmetry};

#define Emax			5		//maximum number of equations: 0 density, 1 x_momentum, 2 y_momentum, 3 z_momentum, 4 energy
//x direction
#if DIM_X==1
#define Xmax	(X_inner + 2 * Bwidth_X)	// maximum number of total cells in x direction
#else
#define Xmax			1
#endif
//y direction
#if DIM_Y==1
#define Ymax	(Y_inner + 2 * Bwidth_Y)	// maximum number of total cells in y direction
#else
#define Ymax			1
#endif
//z direction
#if DIM_Z==1
#define Zmax	(Z_inner + 2 * Bwidth_Z)	// maximum number of total cells in z direction
#else
#define Zmax			1
#endif

const int NumThread = 16;

const int bytes = Xmax*Ymax*Zmax*sizeof(Real);
const int cellbytes = Emax*bytes;

const int BlockSize = 4;

const int dim_block_x = DIM_X ? BlockSize/2 : 1;
const int dim_block_y = DIM_Y ? BlockSize/2 : 1;
const int dim_block_z = DIM_Z ? BlockSize : 1;

constexpr Real Gamma = 1.4;//1.666667;

//fluid 1
const char name_1[128] =	"air";
//type of material, 0: gamma gas, 1: water, 2: stiff gas, 3: weakly compressible
const int material_1_kind =	0;
//material properties:1: phase_indicator, 2:gamma, 3:A, 4:B, 5:rho0, 6:R_0, 7:lambda_0, 8:a(rtificial)s(peed of)s(ound)
const Real material_props_1[8] =	{1, 1.4, 0, 0, 0, 0, 0, 0};

//fluid 2
const char name_2[128] =	"helium";
//type of material, 0: gamma gas, 1: water, 2: stiff gas, 3: weakly compressible
const int material_2_kind =	0;
//material properties:1: phase_indicator, 2:gamma, 3:A, 4:B, 5:rho0, 6:R_0, 7:lambda_0, 8:a(rtificial)s(peed of)s(ound)
const Real material_props_2[8] =	{-1, 1.667, 0, 0, 0, 0, 0, 0};

const Real pi = 3.1415926535897932384626433832795; //phi = 4*atan((double)1);
const Real width_xt	= 4.0L;		//Extending width (cells)
const Real width_hlf	= 2.0L;		//Ghost-fluid update width
const Real mx_vlm	= 0.5L;		//Cells with volume fraction less than this value will be mixed
const Real ext_vlm	= 0.5L;		//For cells with volume fraction less than this value, their states are updated based on mixing
//-------------------------------------------------------------------------------------------------
// narrow band level set
//-------------------------------------------------------------------------------------------------
const Real BandforLevelset	= 6.0L;		//half-width of level set narrow band

//--- Courant Friedrichs Lewy Number ------------------------
const Real CFLnumber =	0.6;

const Real EndTime = 0.024;//0.2;

constexpr Real dt = 0.002;
constexpr Real dx = 0.01;

#endif // SETUP_H
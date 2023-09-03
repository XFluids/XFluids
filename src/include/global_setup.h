#pragma once
#include <unistd.h>
#include <iostream>
#include <fstream>
#include <cstring>
// #include <dpct/dpct.hpp>
//  use middleware targeting to multi-backends
#include "../../middleware/middle.hpp"
#include "global_marco.h"

#ifdef USE_DOUBLE
using real_t = double; // #define real_t double;
#define _DF(a) a
#else
using real_t = float; // #define real_t float;
#define _DF(a) a##f
#endif //  USE_DOUBLE

#define CJ 0					  // 1: CJ detonation case
#define NUM_COP (NUM_SPECIES - 1) // number of components
#define Emax (NUM_SPECIES + 4)	  // maximum number of equations(4+NUM_SPECIES): 0 density, 1 x_momentum, 2 y_momentum, 3 z_momentum, 4 energy

const real_t _OT = (1.0 / 3.0);
const int order_polynominal_fitted = 4;
const int SPCH_Sz = 9;									  // number of characteristic of compoent,species_cahra[NUM_SPECIES*SPCH_Sz]
const real_t pi = _DF(3.1415926535897932384626433832795); // M_PI;
const real_t Ru = _DF(8.314510);						  // Ru
const real_t universal_gas_const = _DF(8.314510);		  // J/(K mol), "R0", p V = n R0 T
const real_t p_atm = _DF(1.01325e5);					  // unit: Pa = kg m^-1 s^-2
const real_t Tref = _DF(1.0);							  // reference T
#ifdef Visc
const real_t kB = _DF(1.3806549 * 1.0e-16); // Boltzmann constant,unit:erg/K=10e-7 J/K
const real_t NA = _DF(6.02214129 * 1.0e23); // Avogadro constant
#endif

// All kinds of Boundary Conditions
enum BConditions
{
	Inflow = 0,
	Outflow = 1,
	Symmetry = 2,
	Periodic = 3,
	Wall = 4,
#ifdef USE_MPI
	BC_COPY = 5,
	BC_UNDEFINED = 6
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
	//--for-Thread---------------------------
	int BlockSize;
	int dim_block_x, dim_block_y, dim_block_z;
	//--for-Mesh-----------------------------
	bool OutBC;
	int Xmax, Ymax, Zmax;
	int X_inner, Y_inner, Z_inner;
	int Bwidth_X, Bwidth_Y, Bwidth_Z; // Bounadry Width
	real_t Domain_xmin, Domain_ymin, Domain_zmin, Domain_xmax, Domain_ymax, Domain_zmax;
	real_t dx, dy, dz, dl, _dx, _dy, _dz, _dl, CFLnumber;
	//--for-Mpi: mx means number of ranks in x direction, myMpiPos_x means location of this rank in x-dir ranks(from 0 to mx-1)
	int mx, my, mz, myMpiPos_x, myMpiPos_y, myMpiPos_z;
	real_t offx, offy, offz;
} Block;

#pragma once

#include "read_ini/setupini.h"
//  C++ headers
#include <ctime>
#include <cstdio>
#include <iomanip>
#include <cstdlib>
#include <float.h>
#include <fstream>
#include <string.h>
#include <iostream>
// SYCL headers
#include <sycl/sycl.hpp>

// constexpr real_t Gamma = 1.4; // 1.666667;
// for flux Reconstruction order
#define FLUX_method 2 //  0: local LF; 1: global LF, 2: Roe
#if SCHEME_ORDER > 6
const int stencil_P = 3;    // "2" for <=6 order, "3"" for >6 order
const int stencil_size = 8; // "6" for <=6 order, "8"" for >6 order
#elif SCHEME_ORDER <= 6
const int stencil_P = 2;    // "2" for <=6 order, "3"" for >6 order
const int stencil_size = 6; // "6" for <=6 order, "8"" for >6 order
#endif
// for nodemisonlizing
const real_t Reference_params[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
// 0: l_ref(unit :m), 1: rho_ref(unit :kg/m3)(air), 2: p_ref(unit :Pa)(air),
// 3: T_ref(unit :K), 4:W0_ref(air mole mass,unit :g/mol) 5:μ_ref(unit:Pa.s=kg/(m.s))(air)
// 6: t_ref(unit :s), 7:ReynoldsNumber=rho_ref*u_ref*l_ref/vis_ref
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

typedef struct
{
    real_t *rho, *p, *c, *H, *u, *v, *w, *T, *gamma;
    real_t *hi, *Vde[9], *y;
    real_t *viscosity_aver, *thermal_conduct_aver, *Dkm_aver;
    real_t *b1x, *b3x, *c2x, *zix, *b1y, *b3y, *c2y, *ziy, *b1z, *b3z, *c2z, *ziz;
} FlowData;

typedef struct
{
    int Mtrl_ind;
    real_t Rgn_ind;           // indicator for region: inside interface, -1.0 or outside 1.0
    real_t Gamma, A, B, rho0; // Eos Parameters and maxium sound speed
    real_t R_0, lambda_0;     // gas constant and heat conductivity
} MaterialProperty;

typedef struct
{
    int nbX, nbY, nbZ;    // number of points output along each DIR
    int minX, minY, minZ; // beginning point of output along each DIR
    int maxX, maxY, maxZ; // ending point of output along each DIR

} OutSize;

//-------------------------------------------------------------------------------------------------
//								Pre-claimer
//-------------------------------------------------------------------------------------------------
class FluidSYCL; class SYCLSolver;

class FluidSYCL{
    Setup Fs;

public:
    int error_patched_times;
    real_t *uvw_c_max;
    real_t *d_U, *d_U1, *d_LU, *h_U, *h_U1, *h_LU, *Ubak; // *h_ptr for Err out and h_Ubak for Continued Caculate
    real_t *d_eigen_local, *d_eigen_l, *d_eigen_r;
    real_t *d_FluxF, *d_FluxG, *d_FluxH, *d_wallFluxF, *d_wallFluxG, *d_wallFluxH;
    FlowData d_fstate, h_fstate;

    std::string Fluid_name; // name of the fluid
    MaterialProperty material_property;

    FluidSYCL(Setup &setup) : Fs(setup) { error_patched_times = 0; };
    void initialize(int n);
    void InitialU(sycl::queue &q);
    void AllocateFluidMemory(sycl::queue &q);
    void BoundaryCondition(sycl::queue &q, BConditions  BCs[6], int flag);
    void UpdateFluidStates(sycl::queue &q, int flag);
    real_t GetFluidDt(sycl::queue &q);
    void UpdateFluidURK3(sycl::queue &q, int flag, real_t const dt);
    void ComputeFluidLU(sycl::queue &q, int flag);
    bool EstimateFluidNAN(sycl::queue &q, int flag);
#ifdef COP_CHEME
    void ODESolver(sycl::queue &q, real_t Time); // ChemQ2 or CVODE-of-Sundials in this function
#endif                                           // end COP_CHEME
};

class SYCLSolver{
    Setup Ss;
    OutSize VTI, PLT, CPT;
    real_t physicalTime;
    int Iteration, rank, nranks;

public:
    real_t dt;
    BConditions *d_BCs; // boundary condition indicators
    FluidSYCL *fluids[NumFluid];

    SYCLSolver(Setup &setup);
    void Evolution(sycl::queue &q);
    void AllocateMemory(sycl::queue &q);
    void InitialCondition(sycl::queue &q);
    void CopyToUbak(sycl::queue &q);
    void CopyToU(sycl::queue &q);
    bool Read_Ubak(sycl::queue &q, const int rank, int *Step, real_t *Time);
    void Output_Ubak(const int rank, const int Step, const real_t Time);
    void CopyDataFromDevice(sycl::queue &q, bool error);
    void GetCPT_OutRanks(int *OutRanks, int rank, int nranks);
    void Output(sycl::queue &q, int rank, std::string interation, real_t Time, bool error = false);
    void Output_vti(int rank, std::ostringstream &timeFormat, std::ostringstream &stepFormat, std::ostringstream &rankFormat, bool error);
    void Output_plt(int rank, std::ostringstream &timeFormat, std::ostringstream &stepFormat, std::ostringstream &rankFormat, bool error);
    void Output_cvti(int rank, std::ostringstream &timeFormat, std::ostringstream &stepFormat, std::ostringstream &rankFormat);
    void Output_cplt(int rank, std::ostringstream &timeFormat, std::ostringstream &stepFormat, std::ostringstream &rankFormat);
    void BoundaryCondition(sycl::queue &q, int flag);
    void UpdateStates(sycl::queue &q, int flag);
    real_t ComputeTimeStep(sycl::queue &q);
    bool SinglePhaseSolverRK3rd(sycl::queue &q, int rank, int Step, real_t physicalTime);
    bool RungeKuttaSP3rd(sycl::queue &q, int rank, int Step, real_t Time, int flag);
    void UpdateU(sycl::queue &q, int flag);
    void ComputeLU(sycl::queue &q, int flag);
    static bool isBigEndian()
    {
        const int i = 1;
        return ((*(char *)&i) == 0);
    }
#ifdef COP_CHEME
    void Reaction(sycl::queue &q, real_t Time);
#endif // end COP_CHEME
};

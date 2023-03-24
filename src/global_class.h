#pragma once

#include "read_ini/setupini.h"
//  C++ headers
#include <ctime>
#include <math.h>
#include <cstdio>
#include <iomanip>
#include <cstdlib>
#include <float.h>
#include <fstream>
#include <string.h>
#include <iostream>
// SYCL headers
#include <sycl/sycl.hpp>

using namespace std;
using namespace sycl;

constexpr real_t Gamma = 1.4; // 1.666667;
const double Reference_params[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
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
    real_t *rho, *p, *c, *H, *u, *v, *w, *T;
    real_t *hi, *Vde[9], *y[NUM_SPECIES];
    real_t *viscosity_aver, *thermal_conduct_aver, *Dkm_aver;
} FlowData;

typedef struct
{
    int Mtrl_ind;
    real_t Rgn_ind;           // indicator for region: inside interface, -1.0 or outside 1.0
    real_t Gamma, A, B, rho0; // Eos Parameters and maxium sound speed
    real_t R_0, lambda_0;     // gas constant and heat conductivity
} MaterialProperty;

//-------------------------------------------------------------------------------------------------
//								Pre-claimer
//-------------------------------------------------------------------------------------------------
class FluidSYCL; class SYCLSolver;

class FluidSYCL{
    Setup Fs;

public:
    real_t *uvw_c_max;
    real_t *d_U, *d_U1, *d_LU;
    real_t *d_eigen_local;
    real_t *d_FluxF, *d_FluxG, *d_FluxH, *d_wallFluxF, *d_wallFluxG, *d_wallFluxH;
    FlowData d_fstate, h_fstate;

    std::string Fluid_name; // name of the fluid
    MaterialProperty material_property;

    FluidSYCL(Setup &setup) : Fs(setup){};
    void initialize(int n);
    void InitialU(sycl::queue &q);
    void AllocateFluidMemory(sycl::queue &q);
    void BoundaryCondition(sycl::queue &q, BConditions  BCs[6], int flag);
    void UpdateFluidStates(sycl::queue &q, int flag);
    real_t GetFluidDt(sycl::queue &q);
    void UpdateFluidURK3(sycl::queue &q, int flag, real_t const dt);
    void ComputeFluidLU(sycl::queue &q, int flag);
#ifdef COP
#ifdef React
    void ODESolver(sycl::queue &q, real_t Time); // ChemQ2 or CVODE-of-Sundials in this function
#endif                                           // React
#endif                                           // COP
};

class SYCLSolver{
    Setup Ss;

public:
    real_t dt;
    BConditions *d_BCs; // boundary condition indicators
    FluidSYCL *fluids[NumFluid];

    SYCLSolver(Setup &setup) : Ss(setup), dt(setup.dt)
    {
        for (int n = 0; n < NumFluid; n++)
        {
            fluids[n] = new FluidSYCL(setup);
#if 1 != NumFluid
            fluids[n]->initialize(n);
#endif
        }
    }
    void Evolution(sycl::queue &q);
    void AllocateMemory(sycl::queue &q);
    void InitialCondition(sycl::queue &q);
    void CopyDataFromDevice(sycl::queue &q);
    void Output(sycl::queue &q, real_t Time);
    void Output_vti(sycl::queue &q, int rank, int interation, real_t Time);
    void BoundaryCondition(sycl::queue &q, int flag);
    void UpdateStates(sycl::queue &q, int flag);
    real_t ComputeTimeStep(sycl::queue &q);
    void SinglePhaseSolverRK3rd(sycl::queue &q);
    void RungeKuttaSP3rd(sycl::queue &q, int flag);
    void UpdateU(sycl::queue &q,int flag);
    void ComputeLU(sycl::queue &q, int flag);
    static bool isBigEndian()
    {
        const int i = 1;
        return ((*(char *)&i) == 0);
    }
#ifdef COP
#ifdef React
    void Reaction(sycl::queue &q, real_t Time);
#endif // React
#endif // COP
};


#pragma once

#include "../read_ini/setupini.h" //#include "setup.h"//
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

typedef struct
{
    real_t *rho, *p, *c, *H, *u, *v, *w, *y, *T;
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
    FlowData d_fstate;

    real_t *h_U, *h_U1, *h_LU;
    real_t *h_eigen_local;
    real_t *h_FluxF, *h_FluxG, *h_FluxH, *h_wallFluxF, *h_wallFluxG, *h_wallFluxH;
    FlowData h_fstate;

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
    real_t dt;

public:
    BConditions *d_BCs; // boundary condition indicators
    FluidSYCL *fluids[NumFluid];

    SYCLSolver(sycl::queue &q, Setup &setup);
    ~SYCLSolver(){};
    void Evolution(sycl::queue &q);
    void AllocateMemory(sycl::queue &q);
    void InitialCondition(sycl::queue &q);
    void CopyDataFromDevice(sycl::queue &q);
    void Output(real_t Time);
    void Output_vti(int rank, int interation, real_t Time);
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


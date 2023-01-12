#pragma once

#include "read_ini/setupini.h"
//  C++ headers
#include <ctime>
#include <cstdio>
#include <vector>
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
// #define PositivityPreserving // #ifdef used, use Lax-Friedrichs(one-order) instead high-order schemes avoiding NAN.
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
    // primitive variables
    real_t *rho, *p, *c, *H, *u, *v, *w, *T, *gamma, *e;
    // cop(y) and vis variables
    real_t *y, *thetaXe, *thetaN2, *thetaXN, *Vde[9], *vxs[3], *vx, *hi, *viscosity_aver, *thermal_conduct_aver, *Dkm_aver;
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
class Fluid; class LAMNSS;

#define Interface_line 0.01

class Fluid{
    Setup Fs;
    sycl::queue q;
    std::string outputPrefix, file_name;

public:
    int error_patched_times, rank, nranks, SBIOutIter;
    float MPI_trans_time, MPI_BCs_time;
    long double MemMbSize, MPIMbSize;
    real_t *uvw_c_max, *theta, *sigma, *pVar_max, *interface_point;
    // std::vector<real_t> pTime, Theta, Sigma, thetas[3], Var_max[NUM_SPECIES - 3], Interface_points[6]; // Var_max[3]= {Tmax, YiHO2max, YiH2O2max}
    real_t *d_U, *d_U1, *d_LU, *h_U, *h_U1, *h_LU, *Ubak; // *h_ptr for Err out and h_Ubak for Continued Caculate
    real_t *d_eigen_local_x, *d_eigen_local_y, *d_eigen_local_z, *d_eigen_l, *d_eigen_r;
    real_t *d_FluxF, *d_FluxG, *d_FluxH, *d_wallFluxF, *d_wallFluxG, *d_wallFluxH;
    FlowData d_fstate, h_fstate;

    std::string Fluid_name; // name of the fluid
    MaterialProperty material_property;

    Fluid(Setup &setup);
    ~Fluid();
    void initialize(int n);
    void InitialU(sycl::queue &q);
    void AllocateFluidMemory(sycl::queue &q);
    void BoundaryCondition(sycl::queue &q, BConditions  BCs[6], int flag);
    bool UpdateFluidStates(sycl::queue &q, int flag);
    real_t GetFluidDt(sycl::queue &q, const int Iter, const real_t physicalTime);
    void GetTheta(sycl::queue &q);
    void UpdateFluidURK3(sycl::queue &q, int flag, real_t const dt);
    void ComputeFluidLU(sycl::queue &q, int flag);
    bool EstimateFluidNAN(sycl::queue &q, int flag);
    void ZeroDimensionalFreelyFlame();
    void ODESolver(sycl::queue &q, real_t Time); // ChemQ2 or CVODE-of-Sundials in this function
};

class LAMNSS{
    Setup Ss;
    OutSize VTI, PLT, CPT;

public:
    real_t dt, physicalTime;
    int Iteration, rank, nranks;
    float duration, MPI_trans_time, MPI_BCs_time;
    BConditions *d_BCs; // boundary condition indicators
    Fluid *fluids[NumFluid];

    LAMNSS(Setup &setup);
    virtual ~LAMNSS();
    void Evolution(sycl::queue &q);
    void EndProcess();
    void AllocateMemory(sycl::queue &q);
    void InitialCondition(sycl::queue &q);
    void CopyToUbak(sycl::queue &q);
    void CopyToU(sycl::queue &q);
    bool Read_Ubak(sycl::queue &q, const int rank, int *Step, real_t *Time);
    void Output_Ubak(const int rank, const int Step, const real_t Time);
    void CopyDataFromDevice(sycl::queue &q, bool error);
    void GetCPT_OutRanks(int *OutRanks, int rank, int nranks);
    // void Output_Counts();
    void Output(sycl::queue &q, int rank, std::string interation, real_t Time, bool error = false);
    void Output_vti(int rank, std::ostringstream &timeFormat, std::ostringstream &stepFormat, std::ostringstream &rankFormat, bool error);
    void Output_plt(int rank, std::ostringstream &timeFormat, std::ostringstream &stepFormat, std::ostringstream &rankFormat, bool error);
    void Output_cvti(int rank, std::ostringstream &timeFormat, std::ostringstream &stepFormat, std::ostringstream &rankFormat);
    void Output_cplt(int rank, std::ostringstream &timeFormat, std::ostringstream &stepFormat, std::ostringstream &rankFormat);
    void BoundaryCondition(sycl::queue &q, int flag);
    bool UpdateStates(sycl::queue &q, int flag, const real_t Time, const int Step, std::string RkStep);
    real_t ComputeTimeStep(sycl::queue &q);
    float OutThisTime(std::chrono::high_resolution_clock::time_point start_time);
    bool SinglePhaseSolverRK3rd(sycl::queue &q, int rank, int Step, real_t physicalTime);
    bool RungeKuttaSP3rd(sycl::queue &q, int rank, int Step, real_t Time, int flag);
    void UpdateU(sycl::queue &q, int flag);
    void ComputeLU(sycl::queue &q, int flag);
    bool Reaction(sycl::queue &q, real_t dt, real_t Time, const int Step);
    bool EstimateNAN(sycl::queue &q, const real_t Time, const int Step, const int rank, const int flag);
    static bool isBigEndian()
    {
        const int i = 1;
        return ((*(char *)&i) == 0);
    }
};

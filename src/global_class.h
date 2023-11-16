#pragma once

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
// program headers
#include "read_ini/setupini.h"
#include "marcos/marco_global.h"

//-------------------------------------------------------------------------------------------------
//								Pre-claimer
//-------------------------------------------------------------------------------------------------
class Fluid;
class LAMNSS;

class Fluid
{
    Setup Fs;
    sycl::queue q;
    std::string outputPrefix, file_name;

public:
    int error_patched_times, rank, nranks, SBIOutIter;
    float MPI_trans_time, MPI_BCs_time;
    long double MemMbSize, MPIMbSize;
    real_t *uvw_c_max, *eigen_block_x, *eigen_block_y, *eigen_block_z, *theta, *sigma, *pVar_max, *interface_point;
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
    void BoundaryCondition(sycl::queue &q, BConditions BCs[6], int flag);
    bool UpdateFluidStates(sycl::queue &q, int flag);
    real_t GetFluidDt(sycl::queue &q, const int Iter, const real_t physicalTime);
    void GetTheta(sycl::queue &q);
    void UpdateFluidURK3(sycl::queue &q, int flag, real_t const dt);
    void ComputeFluidLU(sycl::queue &q, int flag);
    bool EstimateFluidNAN(sycl::queue &q, int flag);
    void ZeroDimensionalFreelyFlame();
    void ODESolver(sycl::queue &q, real_t Time); // ChemQ2 or CVODE-of-Sundials in this function
};

class LAMNSS
{
    Setup Ss;
    OutSize VTI, PLT, CPT;

public:
    real_t dt, physicalTime;
    int Iteration, rank, nranks;
    float duration, duration_backup, MPI_trans_time, MPI_BCs_time;
    BConditions *d_BCs; // boundary condition indicators
    std::vector<Fluid *> fluids{NumFluid};

    LAMNSS(Setup &setup);
    virtual ~LAMNSS();
    // Memory manage
    void AllocateMemory(sycl::queue &q);
    void CopyDataFromDevice(sycl::queue &q, bool error);
    // Functionity
    void CopyToU(sycl::queue &q);
    void CopyToUbak(sycl::queue &q);
    bool Read_Ubak(sycl::queue &q, const int rank, int *Step, real_t *Time, float *Time_consumption);
    bool EstimateNAN(sycl::queue &q, const real_t Time, const int Step, const int rank, const int flag);
    void Output_Ubak(const int rank, const int Step, const real_t Time, const float Time_consumption, bool solution = false);
    // Solvers
    void EndProcess();
    void Evolution(sycl::queue &q);
    void InitialCondition(sycl::queue &q);
    real_t ComputeTimeStep(sycl::queue &q);
    void BoundaryCondition(sycl::queue &q, int flag = 0);
    float OutThisTime(std::chrono::high_resolution_clock::time_point start_time);
    bool UpdateStates(sycl::queue &q, int flag = 0, const real_t Time = 0, const int Step = 0, std::string RkStep = "_Ini");
    bool SinglePhaseSolverRK3rd(sycl::queue &q, int rank, int Step, real_t physicalTime);
    bool RungeKuttaSP3rd(sycl::queue &q, int rank, int Step, real_t Time, int flag);
    void UpdateU(sycl::queue &q, int flag);
    void ComputeLU(sycl::queue &q, int flag);
    bool Reaction(sycl::queue &q, real_t dt, real_t Time, const int Step);
    // Output
    static bool isBigEndian()
    {
        const int i = 1;
        return ((*(char *)&i) == 0);
    }
    void GetCPT_OutRanks(int *OutRanks, int rank, int nranks);
    void Output(sycl::queue &q, int rank, std::string interation, real_t Time, bool error = false);
    void Output_vti(int rank, std::ostringstream &timeFormat, std::ostringstream &stepFormat, std::ostringstream &rankFormat, bool error);
    void Output_plt(int rank, std::ostringstream &timeFormat, std::ostringstream &stepFormat, std::ostringstream &rankFormat, bool error);
    void Output_cvti(int rank, std::ostringstream &timeFormat, std::ostringstream &stepFormat, std::ostringstream &rankFormat);
    void Output_cplt(int rank, std::ostringstream &timeFormat, std::ostringstream &stepFormat, std::ostringstream &rankFormat);
};

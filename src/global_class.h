#pragma once

// program headers
#include "diskinfo.hpp"
#include "read_ini/setupini.h"
#include "include/global_setup.h"

//-------------------------------------------------------------------------------------------------
//								Pre-claimer
//-------------------------------------------------------------------------------------------------
class Fluid;
class XFLUIDS;

class Fluid
{
    Setup Fs;
    sycl::queue q;
    std::string file_name;

public:
    long double MemMbSize, MPIMbSize;
    float MPI_trans_time, MPI_BCs_time;
    int error_patched_times, rank, nranks, SBIOutIter;

    real_t *d_U, *d_U1, *d_LU, *h_U, *h_U1, *h_LU, *Ubak;
    real_t *d_FluxF, *d_FluxG, *d_FluxH, *d_wallFluxF, *d_wallFluxG, *d_wallFluxH;

    real_t *yi_min, *yi_max, *Dim_min, *Dim_max;
    real_t *eigen_block_x, *eigen_block_y, *eigen_block_z;
    real_t *uvw_c_max, *theta, *sigma, *pVar_max, *interface_point;
    real_t *d_eigen_local_x, *d_eigen_local_y, *d_eigen_local_z, *d_eigen_l, *d_eigen_r;

    FlowData d_fstate, h_fstate;

    std::string Fluid_name; // name of the fluid
    MaterialProperty material_property;

    Fluid(Setup &setup);
    ~Fluid();
    void initialize(int n);
    void InitialU(sycl::queue &q);
    void AllocateFluidMemory(sycl::queue &q);
    bool EstimateFluidNAN(sycl::queue &q, int flag);
    std::vector<float> ComputeFluidLU(sycl::queue &q, int flag);
    // void UpdateFluidURK3(sycl::queue &q, int flag, real_t const dt);
    std::vector<float> UpdateFluidURK3(sycl::queue &q, int flag, real_t const dt);
    void BoundaryCondition(sycl::queue &q, BConditions BCs[6], int flag);
    real_t GetFluidDt(sycl::queue &q, const int Iter, const real_t physicalTime);
    std::pair<bool, std::vector<float>> UpdateFluidStates(sycl::queue &q, int flag);
    // Counts
    void AllCountsHeader();
    void GetTheta(sycl::queue &q);
    void AllCountsPush(sycl::queue &q, const size_t Iter, const real_t time);
    // ChemQ2 or CVODE-of-Sundials in this function
    void ODESolver(sycl::queue &q, real_t Time);
};

class XFLUIDS
{
    Setup Ss;
    OutSize VTI, PLT, CPT;

public:
    bool ReadCheckingPoint;
    real_t dt, physicalTime;
    int Iteration, rank, nranks;
    float duration, duration_backup;
    float MPI_trans_time, MPI_BCs_time;
    float runtime_boundary, runtime_updatestates, runtime_getdt;
    float runtime_computelu, runtime_updateu, runtime_estimatenan, runtime_rea;

    OutFmt OutAtThis;
    BConditions *d_BCs; // boundary condition indicators
    std::vector<Fluid *> fluids{NumFluid};
    /**
     * @brief LU_rt
     * runtime_eigenxyz, runtime_eigenmpi
     * runtime_fluxxyz,runtime_preservexyz
     * runtime_GetCellCenterDerivative
     * runtime_transport, runtime_viscxyz
     * runtime_updatelu
     */
    std::vector<float> LU_rt, UD_rt, UU_rt;

    XFLUIDS(Setup &setup);
    virtual ~XFLUIDS();
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
    void Hybrid_endProcess();
    void EndProcess();
    void Evolution(sycl::queue &q);
    void InitialCondition(sycl::queue &q);
    real_t ComputeTimeStep(sycl::queue &q);
    void BoundaryCondition(sycl::queue &q, int flag = 0);
    bool UpdateStates(sycl::queue &q, int flag = 0, const real_t Time = 0, const int Step = 0, std::string RkStep = "_Ini");
    bool SinglePhaseSolverRK3rd(sycl::queue &q, int rank, int Step, real_t physicalTime);
    bool RungeKuttaSP3rd(sycl::queue &q, int rank, int Step, real_t Time, int flag);
    void UpdateU(sycl::queue &q, int flag);
    void ComputeLU(sycl::queue &q, int flag);
    bool Reaction(sycl::queue &q, real_t dt, real_t Time, const int Step);
    // Output
    void GetCPT_OutRanks(int *OutRanks, OutSize &CVTI, OutSlice pos);
    void GetSPT_OutRanks(int *OutRanks, std::vector<Criterion> &var);
    std::vector<OutVar> Output_variables(FlowData &data, std::vector<std::string> &sp, size_t error = 2);
    void Output(sycl::queue &q, OutFmt ctrl, size_t error = 0);
    template <typename T = float>
    void Output_vti(std::vector<OutVar> error_vars, OutString &osr, size_t error = 0);
    void Output_plt(int rank, OutString &osr, bool error = false);
    template <typename T = float>
    void Output_svti(std::vector<OutVar> &varout, std::vector<Criterion> &cri, OutString &osr);
    template <typename T = float>
    void Output_cplt(std::vector<OutVar> &varout, OutSlice &pos, OutString &osr);
    template <typename T = float>
    void Output_cvti(std::vector<OutVar> &varout, OutSlice &pos, OutString &osr);
};

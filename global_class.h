#pragma once

#include "setup.h"
#include "fun.h"

// SYCL head files
#include <CL/sycl.hpp>
#include "dpc_common.hpp"

using namespace std;
// using namespace tbb;
using namespace sycl;

//-------------------------------------------------------------------------------------------------
//								Pre-claimer
//-------------------------------------------------------------------------------------------------
class FluidSYCL; class SYCLSolver;

class FluidSYCL{

public:

    std::array<int, 3> WGSize, WISize;

    float *d_uvw_c_max;

    Real dx, dy, dz, dl, dt;

    Real *d_U, *d_U1, *d_LU;
    // Real *d_CnsrvU, *d_CnsrvU1;
    Real *d_eigen_local;

    Real *d_FluxF, *d_FluxG, *d_FluxH;
    Real *d_wallFluxF, *d_wallFluxG, *d_wallFluxH;

    FlowData d_fstate;

	Real *h_U, *h_U1, *h_LU;
	Real *h_eigen_local;

	Real *h_FluxF, *h_FluxG, *h_FluxH;
	Real *h_wallFluxF, *h_wallFluxG, *h_wallFluxH;

	FlowData h_fstate;

	char Fluid_name[128]; 		//name of the fluid
    MaterialProperty material_property;

    MaterialProperty *d_material_property;

    FluidSYCL(){};
    ~FluidSYCL(){};
    FluidSYCL(Real Dx, Real Dy, Real Dz, Real Dl, Real Dt, std::array<int, 3> workitem_size, std::array<int, 3> workgroup_size);

    void initialize(int n);
    void InitialU(sycl::queue &q, Real dx, Real dy, Real dz);
    void test(sycl::queue &q);
    void AllocateFluidMemory(sycl::queue &q);

};

class SYCLSolver{

public:

    // range<3> local_range = range<3>(dim_block_x, dim_block_y, dim_block_z);
    // range<3> global_range = range<3>(X_inner+local_range[0], Y_inner, Z_inner);

    std::array<int, 3> workgroup_size, workitem_size;

    Real domain_length, domain_width, domain_height;
    Real dx, dy, dz, dl, dt;
    BConditions  BCs[6]; //boundary condition indicators

    FluidSYCL *fluids[NumFluid];

    SYCLSolver(sycl::queue &q);
    ~SYCLSolver(){};
    void AllocateMemory(sycl::queue &q);
    void InitialCondition(sycl::queue &q);
    void CopyDataFromDevice(sycl::queue &q);
    void Output(Real Time);
};


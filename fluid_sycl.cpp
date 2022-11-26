#include <math.h>
#include "global_class.h"
#include "global_function.h"
#include "block_sycl.h"

using namespace std;
using namespace sycl;
// using namespace tbb;

/**
 * @brief Construct a new Fluid:: Fluid object //采用初始化列表
 * 
 */
FluidSYCL::FluidSYCL(Real Dx, Real Dy, Real Dz, Real Dl, Real Dt,  std::array<int, 3> workitem_size, std::array<int, 3> workgroup_size)
						: dx(Dx), dy(Dy), dz(Dz), dl(Dl), dt(Dt), WISize(workitem_size), WGSize(workgroup_size)
{

}

void FluidSYCL::initialize(int n)
{
	if(n==0){
		strcpy(Fluid_name, name_1);//give a name to the fluid
		//type of material, 0: gamma gas, 1: water, 2: stiff gas
		material_property.Mtrl_ind=	material_1_kind;
		//fluid indicator and EOS Parameters
		material_property.Rgn_ind	=	material_props_1[0];
		//gamma, A, B, rho0, mu_0, R_0, lambda_0
		material_property.Gamma	=	material_props_1[1];
		material_property.A	=	material_props_1[2];
		material_property.B	=	material_props_1[3];
		material_property.rho0	=	material_props_1[4];		
		material_property.R_0	=	material_props_1[5];
		material_property.lambda_0=	material_props_1[6];
	}

    #if NumFluid==2
	if(n==1){
		strcpy(Fluid_name, name_2);//give a name to the fluid
		material_property.Mtrl_ind = material_2_kind;
		//fluid indicator and EOS Parameters
		material_property.Rgn_ind	=	material_props_2[0];
		//gamma, A, B, rho0, mu_0, R_0, lambda_0
		material_property.Gamma	=	material_props_2[1];
		material_property.A	=	material_props_2[2];
		material_property.B	=	material_props_2[3];
		material_property.rho0	=	material_props_2[4];	
		material_property.R_0	=	material_props_2[5];
		material_property.lambda_0=	material_props_2[6];
	}
    #endif
}

void FluidSYCL::AllocateFluidMemory(sycl::queue &q)
{
	d_material_property = static_cast<MaterialProperty *>(malloc_device(sizeof(MaterialProperty), q));

	q.memcpy(d_material_property, &material_property, sizeof(MaterialProperty)).wait();

  	d_U  = static_cast<Real *>(malloc_device(cellbytes, q));
  	d_U1 = static_cast<Real *>(malloc_device(cellbytes, q));
  	d_LU = static_cast<Real *>(malloc_device(cellbytes, q));
	d_eigen_local = static_cast<Real *>(malloc_device(cellbytes, q));
	d_fstate.rho = static_cast<Real *>(malloc_device(bytes, q));
	d_fstate.p = static_cast<Real *>(malloc_device(bytes, q));
	d_fstate.c = static_cast<Real *>(malloc_device(bytes, q));
	d_fstate.H = static_cast<Real *>(malloc_device(bytes, q));
	d_fstate.u = static_cast<Real *>(malloc_device(bytes, q));
	d_fstate.v = static_cast<Real *>(malloc_device(bytes, q));
	d_fstate.w = static_cast<Real *>(malloc_device(bytes, q));
	d_FluxF  = static_cast<Real *>(malloc_device(cellbytes, q));
	d_FluxG  = static_cast<Real *>(malloc_device(cellbytes, q));
	d_FluxH  = static_cast<Real *>(malloc_device(cellbytes, q));
	d_wallFluxF  = static_cast<Real *>(malloc_device(cellbytes, q));
	d_wallFluxG  = static_cast<Real *>(malloc_device(cellbytes, q));
	d_wallFluxH  = static_cast<Real *>(malloc_device(cellbytes, q));
	// shared
	uvw_c_max  = static_cast<Real *>(malloc_shared(3*sizeof(Real), q));

	cout << "Memory Usage: " << (Real)((long)10*cellbytes + (long)7*bytes)/(Real)(1024*1024*1024)<< " GB\n";

  	// 主机内存
  	h_U  = static_cast<Real *>(malloc(cellbytes));
  	h_U1 = static_cast<Real *>(malloc(cellbytes));
  	h_LU = static_cast<Real *>(malloc(cellbytes));
	h_eigen_local = static_cast<Real *>(malloc(cellbytes));
	h_fstate.rho = static_cast<Real *>(malloc(bytes));
	h_fstate.p = static_cast<Real *>(malloc(bytes));
	h_fstate.c = static_cast<Real *>(malloc(bytes));
	h_fstate.H = static_cast<Real *>(malloc(bytes));
	h_fstate.u = static_cast<Real *>(malloc(bytes));
	h_fstate.v = static_cast<Real *>(malloc(bytes));
	h_fstate.w = static_cast<Real *>(malloc(bytes));
	h_FluxF  = static_cast<Real *>(malloc(cellbytes));
	h_FluxG  = static_cast<Real *>(malloc(cellbytes));
	h_FluxH  = static_cast<Real *>(malloc(cellbytes));
	h_wallFluxF  = static_cast<Real *>(malloc(cellbytes));
	h_wallFluxG  = static_cast<Real *>(malloc(cellbytes));
	h_wallFluxH  = static_cast<Real *>(malloc(cellbytes));
}

void FluidSYCL::InitialU(sycl::queue &q, Real dx, Real dy, Real dz)
{
	InitializeFluidStates(q, WGSize, WISize, d_material_property, d_fstate, d_U, d_U1, d_LU, d_FluxF, d_FluxG, d_FluxH, d_wallFluxF, d_wallFluxG, d_wallFluxH, dx, dy, dz);
}

Real FluidSYCL::GetFluidDt(sycl::queue &q)
{
	return GetDt(q, d_fstate, uvw_c_max, dx, dy, dz);
}

void FluidSYCL::BoundaryCondition(sycl::queue &q, BConditions BCs[6], int flag)
{
    if (flag == 0)
        FluidBoundaryCondition(q, BCs, d_U);
    else
        FluidBoundaryCondition(q, BCs, d_U1);
}

void FluidSYCL::UpdateFluidStates(sycl::queue &q, int flag)
{
    if (flag == 0)
        UpdateFluidStateFlux(q, d_U, d_fstate, d_FluxF, d_FluxG, d_FluxH, material_property.Gamma);
    else
        UpdateFluidStateFlux(q, d_U1, d_fstate, d_FluxF, d_FluxG, d_FluxH, material_property.Gamma);
}

void FluidSYCL::UpdateFluidURK3(sycl::queue &q, int flag, Real const dt)
{
	UpdateURK3rd(q, d_U, d_U1, d_LU, dt, flag);
}

void FluidSYCL::ComputeFluidLU(sycl::queue &q, int flag)
{
    if (flag == 0)
        GetLU(q, d_U, d_LU, d_FluxF, d_FluxG, d_FluxH, d_wallFluxF, d_wallFluxG, d_wallFluxH, 
			material_property.Gamma, material_property.Mtrl_ind, d_fstate, d_eigen_local, dx, dy, dz);
    else
        GetLU(q, d_U1, d_LU, d_FluxF, d_FluxG, d_FluxH, d_wallFluxF, d_wallFluxG, d_wallFluxH, 
			material_property.Gamma, material_property.Mtrl_ind, d_fstate, d_eigen_local, dx, dy, dz);
}
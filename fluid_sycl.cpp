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
	// auto *U = d_U;
	FlowData *fdata = &d_fstate;
	auto *U = d_U;
	auto *U1 = d_U1;
	auto *LU = d_LU;
	auto *FF = d_FluxF;
	auto *GG = d_FluxG;
	auto *HH = d_FluxH;
	auto *Fw = d_wallFluxF;
	auto *Gw = d_wallFluxG;
	auto *Hw = d_wallFluxH;
	auto *eigen = d_eigen_local;

	Real *rho = d_fstate.rho;
	Real *p = d_fstate.p;
	Real *H = d_fstate.H;
	Real *c = d_fstate.c;
	Real *u = d_fstate.u;
	Real *v = d_fstate.v;
	Real *w = d_fstate.w;

	MaterialProperty *mp = d_material_property;

	InitializeFluidStates(q, WGSize, WISize, d_material_property, d_fstate, d_U, d_U1, d_LU, d_FluxF, d_FluxG, d_FluxH, d_wallFluxF, d_wallFluxG, d_wallFluxH, dx, dy, dz);

	// auto local_ndrange = range<3>(WGSize.at(0), WGSize.at(1), WGSize.at(2));	// size of workgroup
	// auto global_ndrange = range<3>(WISize.at(0), WISize.at(1), WISize.at(2));

	// q.submit([&](sycl::handler &h){
	// 	h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=, this](sycl::nd_item<3> index){
    // 		// 利用get_global_id获得全局指标
    // 		int i = index.get_global_id(0) + Bwidth_X - 1;
    // 		int j = index.get_global_id(1) + Bwidth_Y;
	// 		int k = index.get_global_id(2) + Bwidth_Z;

	// 		// SYCL不允许在lambda函数里出现结构体成员
	// 		// InitialStatesKernel(i, j, k, mp, U, U1, LU, FF, GG, HH, Fw, Gw, Hw, u, v, w, rho, p, H, c, dx, dy, dz);
	// 		// testkernel(i,j,k, d_U, F, Fw, eigen, u,v,w, rho, p, H,c, 0.1, 0.1, 0.1);
	// 	});
	// });
}

void FluidSYCL::test(sycl::queue &q)
{
	auto *U = d_U;
	auto *F = d_FluxF;
	auto *Fw = d_wallFluxF;
	auto *eigen = d_eigen_local;
	FlowData *fs = &d_fstate;

  	auto local_ndrange = range<3>(dim_block_x, dim_block_y, dim_block_z);
  	// global_ndrange：工作项的大小 = 1024*1024
  	auto global_ndrange = range<3>(X_inner+local_ndrange[0], Y_inner, Z_inner);
		q.submit([&](sycl::handler &h){
			//Deviceptr
			h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index){
    			// 利用get_global_id获得全局指标
    			int i = index.get_global_id(0) + Bwidth_X - 1;
    			int j = index.get_global_id(1) + Bwidth_Y;
				int k = index.get_global_id(2) + Bwidth_Z;

				ReconstructFluxX(i, j, k, U, F, Fw, eigen, fs->rho, fs->u, fs->v, fs->w, fs->H, 0.01, 0.01);
			});
		});

}
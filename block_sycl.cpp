#include "block_sycl.h"
#include "fun.h"

void InitializeFluidStates(sycl::queue &q, array<int, 3> WG, array<int, 3> WI, MaterialProperty *material, FlowData &fdata, Real* U, Real* U1, Real* LU,
                           Real* FluxF, Real* FluxG, Real* FluxH, Real* FluxFw, Real* FluxGw, Real* FluxHw, 
                           Real const dx, Real const dy, Real const dz)
{
	auto local_ndrange = range<3>(WG.at(0), WG.at(1), WG.at(2));	// size of workgroup
	auto global_ndrange = range<3>(WI.at(0), WI.at(1), WI.at(2));

	// auto local_ndrange = range<3>(WGSize.at(0), WGSize.at(1), WGSize.at(2));	// size of workgroup
	// auto global_ndrange = range<3>(WISize.at(0), WISize.at(1), WISize.at(2));

    // dim3 dim_grid_max(dim_grid.x + DIM_X, dim_grid.y + DIM_Y, dim_grid.z + DIM_Z);
    // InitialStatesKernel<<<dim_grid_max, dim_blk>>>(material, U, U1, LU, CnsrvU, CnsrvU1, FluxF, FluxG, FluxH, FluxFw, FluxGw, FluxHw, 
    //                                                                                 fdata.u, fdata.v, fdata.w, fdata.rho, fdata.p, fdata.H, fdata.c, vof, dx, dy, dz);

	Real *rho = fdata.rho;
	Real *p = fdata.p;
	Real *H = fdata.H;
	Real *c = fdata.c;
	Real *u = fdata.u;
	Real *v = fdata.v;
	Real *w = fdata.w;

	q.submit([&](sycl::handler &h){
		h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index){
    		// 利用get_global_id获得全局指标
    		int i = index.get_global_id(0) + Bwidth_X - 1;
    		int j = index.get_global_id(1) + Bwidth_Y;
			int k = index.get_global_id(2) + Bwidth_Z;

			// SYCL不允许在lambda函数里出现结构体成员
			InitialStatesKernel(i, j, k, material, U, U1, LU, FluxF, FluxG, FluxH, FluxFw, FluxGw, FluxHw, u, v, w, rho, p, H, c, dx, dy, dz);

			// testkernel(i,j,k, d_U, F, Fw, eigen, u,v,w, rho, p, H,c, 0.1, 0.1, 0.1);
		});
	});
}
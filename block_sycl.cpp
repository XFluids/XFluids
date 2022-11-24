#include "block_sycl.h"
#include "fun.h"

void InitializeFluidStates(sycl::queue &q, array<int, 3> WG, array<int, 3> WI, MaterialProperty *material, FlowData &fdata, Real* U, Real* U1, Real* LU,
                           Real* FluxF, Real* FluxG, Real* FluxH, Real* FluxFw, Real* FluxGw, Real* FluxHw, 
                           Real const dx, Real const dy, Real const dz)
{
	auto local_ndrange = range<3>(WG.at(0), WG.at(1), WG.at(2));	// size of workgroup
	auto global_ndrange = range<3>(Xmax, Ymax, Zmax);

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
		// auto out = sycl::stream(1024, 256, h);
		h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index){
    		// 利用get_global_id获得全局指标
    		int i = index.get_global_id(0) + Bwidth_X - 1;
    		int j = index.get_global_id(1) + Bwidth_Y;
			int k = index.get_global_id(2) + Bwidth_Z;

			// int ii = index.get_group(0)*index.get_local_range(0) + index.get_local_id(0) + Bwidth_X - 1;
			// int jj = index.get_group(1)*index.get_local_range(1) + index.get_local_id(1) + Bwidth_Y;

			// SYCL不允许在lambda函数里出现结构体成员
			InitialStatesKernel(i, j, k, material, U, U1, LU, FluxF, FluxG, FluxH, FluxFw, FluxGw, FluxHw, u, v, w, rho, p, H, c, dx, dy, dz);

			// testkernel(i,j,k, d_U, F, Fw, eigen, u,v,w, rho, p, H,c, 0.1, 0.1, 0.1);
		});
	});
}

void FluidBoundaryCondition(sycl::queue &q, BConditions  *BCs, Real*  d_UI)
{
	// //x direction
    // constexpr int dim_block_x = DIM_X ? WarpSize : 1;
    // constexpr int dim_block_y = DIM_Y ? WarpSize : 1;
	// constexpr int dim_block_z = 1;//DIM_Z ? WarpSize : 1;

    #if DIM_X
	auto local_ndrange_x = range<3>(Bwidth_X, dim_block_y, dim_block_z);	// size of workgroup
	auto global_ndrange_x = range<3>(Bwidth_X, Ymax, Zmax);

	q.submit([&](sycl::handler &h){
		h.parallel_for(sycl::nd_range<3>(global_ndrange_x, local_ndrange_x), [=](sycl::nd_item<3> index){
    		int i = index.get_global_id(0) + Bwidth_X - 1;
    		int j = index.get_global_id(1) + Bwidth_Y;
			int k = index.get_global_id(2) + Bwidth_Z;

			FluidBCKernelX(i, j, k, &(BCs[0]), d_UI, 0, 0, Bwidth_X, 1);
			FluidBCKernelX(i, j, k, &(BCs[1]), d_UI, Xmax-Bwidth_X, X_inner, Xmax-Bwidth_X-1, -1);
		});
	});
    #endif

    // #if DIM_Y
	// dim3 dim_blk_y(dim_block_x, Bwidth_Y, dim_block_z);
    // dim3 dim_grid_y(X_inner/dim_block_x + DIM_X, 1, Z_inner/dim_block_z + DIM_Z);
    // FluidBCKernelY<<<dim_grid_y, dim_blk_y>>>(BCs[2], d_UI, 0, 0, Bwidth_Y, 1);
    // FluidBCKernelY<<<dim_grid_y, dim_blk_y>>>(BCs[3], d_UI, Ymax-Bwidth_Y, Y_inner, Ymax-Bwidth_Y-1, -1);
    // #endif

    // #if DIM_Z
	// dim3 dim_blk_z(1, dim_block_y, Bwidth_Z);
    // dim3 dim_grid_z(X_inner/1 + DIM_X, Y_inner/dim_block_y + DIM_Y, 1);
    // FluidBCKernelZ<<<dim_grid_z, dim_blk_z>>>(BCs[4], d_UI, 0, 0, Bwidth_Z, 1);
    // FluidBCKernelZ<<<dim_grid_z, dim_blk_z>>>(BCs[5], d_UI, Zmax-Bwidth_Z, Z_inner, Zmax-Bwidth_Z-1, -1);
    // #endif
}
#include "include/global_class.h"
#include "sycl_kernels.hpp"

using namespace std;
using namespace sycl;

void InitializeFluidStates(sycl::queue &q, Block bl, IniShape ini, MaterialProperty material, Thermal *thermal, FlowData &fdata, real_t *U, real_t *U1, real_t *LU,
						   real_t *FluxF, real_t *FluxG, real_t *FluxH, real_t *FluxFw, real_t *FluxGw, real_t *FluxHw)
// real_t const dx, real_t const dy, real_t const dz
{
	auto local_ndrange = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z); // size of workgroup
	auto global_ndrange = range<3>(bl.Xmax, bl.Ymax, bl.Zmax);

	real_t *rho = fdata.rho;
	real_t *p = fdata.p;
	real_t *H = fdata.H;
	real_t *c = fdata.c;
	real_t *u = fdata.u;
	real_t *v = fdata.v;
	real_t *w = fdata.w;
	real_t *y = fdata.y;
	real_t *T = fdata.T;

	q.submit([&](sycl::handler &h)
			 {
		// auto out = sycl::stream(1024, 256, h);
		h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index){
    		// 利用get_global_id获得全局指标
    		int i = index.get_global_id(0);
    		int j = index.get_global_id(1);
			int k = index.get_global_id(2);

			InitialStatesKernel(i, j, k, bl, ini, material, thermal, U, U1, LU, FluxF, FluxG, FluxH, FluxFw, FluxGw, FluxHw, u, v, w, rho, p, y, T, H, c);
		}); });
}

real_t GetDt(sycl::queue &q, Block bl, FlowData &fdata, real_t *uvw_c_max)
{
	real_t *rho = fdata.rho;
	real_t *c = fdata.c;
	real_t *u = fdata.u;
	real_t *v = fdata.v;
	real_t *w = fdata.w;

	auto local_ndrange = range<1>(bl.BlockSize); // size of workgroup
	auto global_ndrange = range<1>(bl.Xmax * bl.Ymax * bl.Zmax);

	for (int n = 0; n < 3; n++)
		uvw_c_max[n] = 0;

	q.submit([&](sycl::handler &h)
			 {
      	// define reduction objects for sum, min, max reduction
		// auto reduction_sum = reduction(sum, sycl::plus<>());
    	auto reduction_max_x = reduction(&(uvw_c_max[0]), sycl::maximum<>());
		auto reduction_max_y = reduction(&(uvw_c_max[1]), sycl::maximum<>());
		auto reduction_max_z = reduction(&(uvw_c_max[2]), sycl::maximum<>());
      
		h.parallel_for(sycl::nd_range<1>(global_ndrange, local_ndrange), reduction_max_x, reduction_max_y, reduction_max_z, 
	  	[=](nd_item<1> index, auto& temp_max_x, auto& temp_max_y, auto& temp_max_z)
		{
        	auto id = index.get_global_id();
			// if(id < Xmax*Ymax*Zmax)
        	temp_max_x.combine(u[id]+c[id]);
        	temp_max_y.combine(v[id]+c[id]);
			temp_max_z.combine(w[id]+c[id]);
      	}); })
		.wait();

	real_t dtref = 0.0;
#if DIM_X
	dtref += uvw_c_max[0] / bl.dx;
#endif
#if DIM_Y
	dtref += uvw_c_max[1] / bl.dy;
#endif
#if DIM_Z
	dtref += uvw_c_max[2] / bl.dz;
#endif
	return bl.CFLnumber / dtref;
}

void UpdateFluidStateFlux(sycl::queue &q, Block bl, Thermal *thermal, real_t *UI, FlowData &fdata, real_t *FluxF, real_t *FluxG, real_t *FluxH, real_t const Gamma)
{
	auto local_ndrange = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z); // size of workgroup
	auto global_ndrange = range<3>(bl.Xmax, bl.Ymax, bl.Zmax);

	real_t *rho = fdata.rho;
	real_t *p = fdata.p;
	real_t *H = fdata.H;
	real_t *c = fdata.c;
	real_t *u = fdata.u;
	real_t *v = fdata.v;
	real_t *w = fdata.w;
	real_t *y = fdata.y;
	real_t *T = fdata.T;

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0);
    		int j = index.get_global_id(1);
			int k = index.get_global_id(2);

			UpdateFuidStatesKernel(i, j, k, bl,thermal, UI, FluxF, FluxG, FluxH, rho, p, c, H, u, v, w, y, T, Gamma); }); });

	q.wait();
}

void UpdateURK3rd(sycl::queue &q, Block bl, real_t *U, real_t *U1, real_t *LU, real_t const dt, int flag)
{
	auto local_ndrange = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z); // size of workgroup
	auto global_ndrange = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner);

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0) + bl.Bwidth_X;
			int j = index.get_global_id(1) + bl.Bwidth_Y;
			int k = index.get_global_id(2) + bl.Bwidth_Z;

			UpdateURK3rdKernel(i, j, k, bl, U, U1, LU, dt, flag); }); });

	q.wait();
}

void GetLU(sycl::queue &q, Block bl, Thermal *thermal, real_t *UI, real_t *LU, real_t *FluxF, real_t *FluxG, real_t *FluxH,
		   real_t *FluxFw, real_t *FluxGw, real_t *FluxHw,
		   real_t const Gamma, int const Mtrl_ind, FlowData &fdata, real_t *eigen_local)
{
	real_t *rho = fdata.rho;
	real_t *p = fdata.p;
	real_t *H = fdata.H;
	real_t *c = fdata.c;
	real_t *u = fdata.u;
	real_t *v = fdata.v;
	real_t *w = fdata.w;
	real_t *y = fdata.y;
	real_t *T = fdata.T;

	bool is_3d = DIM_X * DIM_Y * DIM_Z ? true : false;

	auto local_ndrange = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z);
	auto global_ndrange_max = range<3>(bl.Xmax, bl.Ymax, bl.Zmax);
	auto global_ndrange_inner = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner);

	event e;

#if DIM_X
	// proceed at x directiom and get F-flux terms at node wall
	auto global_ndrange_x = range<3>(bl.X_inner + local_ndrange[0], bl.Y_inner, bl.Z_inner);

	e = q.submit([&](sycl::handler &h)
				 { h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), [=](sycl::nd_item<3> index)
								  {
    		int i = index.get_global_id(0);
    		int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			GetLocalEigen(i, j, k, bl, 1.0, 0.0, 0.0, eigen_local, u, v, w, c); }); });

	q.submit([&](sycl::handler &h)
			 {
		h.depends_on(e);
		h.parallel_for(sycl::nd_range<3>(global_ndrange_x, local_ndrange), [=](sycl::nd_item<3> index)
					   {
    		int i = index.get_global_id(0) + bl.Bwidth_X - 1;
			int j = index.get_global_id(1) + bl.Bwidth_Y;
			int k = index.get_global_id(2) + bl.Bwidth_Z;
			ReconstructFluxX(i, j, k, bl,thermal,Gamma, UI, FluxF, FluxFw, eigen_local, rho, u, v, w,y,T, H); }); });

	q.wait();
#endif

#if DIM_Y
	// proceed at y directiom and get G-flux terms at node wall
	auto global_ndrange_y = range<3>(bl.X_inner, bl.Y_inner + local_ndrange[1], bl.Z_inner);

	e = q.submit([&](sycl::handler &h)
				 { h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), [=](sycl::nd_item<3> index)
								  {
    		int i = index.get_global_id(0);
    		int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			GetLocalEigen(i, j, k, bl, 0.0, 1.0, 0.0, eigen_local, u, v, w, c); }); });

	q.submit([&](sycl::handler &h)
			 {
		h.depends_on(e);
		h.parallel_for(sycl::nd_range<3>(global_ndrange_y, local_ndrange), [=](sycl::nd_item<3> index)
					   {
    		int i = index.get_global_id(0) + bl.Bwidth_X;
			int j = index.get_global_id(1) + bl.Bwidth_Y - 1;
			int k = index.get_global_id(2) + bl.Bwidth_Z;
			ReconstructFluxY(i, j, k, bl,thermal,Gamma,UI, FluxG, FluxGw, eigen_local, rho, u, v, w,y,T, H); }); });

	q.wait();
#endif

#if DIM_Z
	// proceed at y directiom and get G-flux terms at node wall
	auto global_ndrange_z = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner + local_ndrange[2]);

	e = q.submit([&](sycl::handler &h)
				 { h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), [=](sycl::nd_item<3> index)
								  {
    		int i = index.get_global_id(0);
    		int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			GetLocalEigen(i, j, k, bl, 0.0, 0.0, 1.0, eigen_local, u, v, w, c); }); });

	q.submit([&](sycl::handler &h)
			 {
		h.depends_on(e);
		h.parallel_for(sycl::nd_range<3>(global_ndrange_y, local_ndrange), [=](sycl::nd_item<3> index)
					   {
    		int i = index.get_global_id(0) + bl.Bwidth_X;
			int j = index.get_global_id(1) + bl.Bwidth_Y;
			int k = index.get_global_id(2) + bl.Bwidth_Z - 1;
			ReconstructFluxZ(i, j, k, bl, thermal, Gamma, UI, FluxH, FluxHw, eigen_local, rho, u, v, w, y, T, H); }); });

	q.wait();
#endif

// update LU from cell-face fluxes
#if NumFluid == 2
// UpdateFluidMPLU<<<dim_grid, dim_blk>>>(LU, FluxFw, FluxGw, FluxHw, phi, tag_bnd_hlf, Rgn_ind, x_swth, y_swth, z_swth, exchange, dx, dy, dz);
#else
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_inner, local_ndrange), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0) + bl.Bwidth_X;
			int j = index.get_global_id(1) + bl.Bwidth_Y;
			int k = index.get_global_id(2) + bl.Bwidth_Z;

			UpdateFluidLU(i, j, k, bl, LU, FluxFw, FluxGw, FluxHw); }); })
		.wait();
#endif
}

void FluidBoundaryCondition(sycl::queue &q, Block bl, BConditions BCs[6], real_t *d_UI)
{
#if DIM_X
	auto local_ndrange_x = range<3>(bl.Bwidth_X, bl.dim_block_y, bl.dim_block_z); // size of workgroup
	auto global_ndrange_x = range<3>(bl.Bwidth_X, bl.Ymax, bl.Zmax);

	BConditions BC0 = BCs[0], BC1 = BCs[1];

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_x, local_ndrange_x), [=](sycl::nd_item<3> index)
							  {
    		int i0 = index.get_global_id(0) + 0;
			int i1 = index.get_global_id(0) + bl.Xmax - bl.Bwidth_X;
			int j = index.get_global_id(1);
			int k = index.get_global_id(2);

			FluidBCKernelX(i0, j, k, bl, BC0, d_UI, 0, bl.Bwidth_X, 1);
			FluidBCKernelX(i1, j, k, bl, BC1, d_UI, bl.X_inner, bl.Xmax - bl.Bwidth_X - 1, -1); }); });
#endif

#if DIM_Y
	auto local_ndrange_y = range<3>(bl.dim_block_x, bl.Bwidth_Y, bl.dim_block_z); // size of workgroup
	auto global_ndrange_y = range<3>(bl.Xmax, bl.Bwidth_Y, bl.Zmax);

	BConditions BC2 = BCs[2], BC3 = BCs[3];

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_y, local_ndrange_y), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0);
			int j0 = index.get_global_id(1) + 0;
			int j1 = index.get_global_id(1) + bl.Ymax - bl.Bwidth_Y;
			int k = index.get_global_id(2);

			FluidBCKernelY(i, j0, k, bl, BC2, d_UI, 0, bl.Bwidth_Y, 1);
			FluidBCKernelY(i, j1, k, bl, BC3, d_UI, bl.Y_inner, bl.Ymax - bl.Bwidth_Y - 1, -1); }); });
#endif

#if DIM_Z
	auto local_ndrange_z = range<3>(bl.dim_block_x, bl.dim_block_y, bl.Bwidth_Z); // size of workgroup
	auto global_ndrange_z = range<3>(bl.Xmax, bl.Ymax, bl.Bwidth_Z);

	BConditions BC4 = BCs[4], BC5 = BCs[5];

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_z, local_ndrange_z), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0);
			int j = index.get_global_id(1);
    		int k0 = index.get_global_id(2) + 0;
			int k1 = index.get_global_id(2) + bl.Zmax - bl.Bwidth_Z;

			FluidBCKernelZ(i, j, k0, bl, BC4, d_UI, 0, bl.Bwidth_Z, 1);
			FluidBCKernelZ(i, j, k1, bl, BC5, d_UI, bl.Z_inner, bl.Zmax - bl.Bwidth_Z - 1, -1); }); });
#endif
}
#ifdef Visc
void GetCellCenterDerivative(sycl::queue &q, Block bl, FlowData &fdata, BConditions BC[6])
{
	auto local_ndrange = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z); // size of workgroup
	auto global_ndrange = range<3>(bl.X_inner + 4, bl.Y_inner + 4, bl.Z_inner + 4); // NOTE：这里的4是由求微分的算法确定的,内点网格向两边各延伸两个点

	real_t *V[3] = {fdata.u, fdata.v, fdata.w};
	real_t **Vde = fdata.Vde;

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index)
							  { GetInnerCellCenterDerivativeKernel(index, bl, V, Vde); }); });
	q.wait();

#if DIM_X
	auto local_ndrange_x = range<3>(bl.Bwidth_X, bl.dim_block_y, bl.dim_block_z); // size of workgroup
	auto global_ndrange_x = range<3>(bl.Bwidth_X, bl.Ymax, bl.Zmax);

	real_t *Vde_x[4] = {fdata.Vde[ducy], fdata.Vde[ducz], fdata.Vde[dvcy], fdata.Vde[dwcz]};
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_x, local_ndrange_x), [=](sycl::nd_item<3> index)
							  {
    		int i0 = index.get_global_id(0) + 0;
			int i1 = index.get_global_id(0) + bl.Xmax - bl.Bwidth_X;
			int j = index.get_global_id(1);
			int k = index.get_global_id(2);

			CenterDerivativeBCKernelX(i0, j, k, bl, BC[0], Vde_x, 0, bl.Bwidth_X, 1);
			CenterDerivativeBCKernelX(i1, j, k, bl, BC[1], Vde_x, bl.X_inner, bl.Xmax - bl.Bwidth_X - 1, -1); }); });
#endif // DIM_X

#if DIM_Y
	auto local_ndrange_y = range<3>(bl.dim_block_x, bl.Bwidth_Y, bl.dim_block_z); // size of workgroup
	auto global_ndrange_y = range<3>(bl.Xmax, bl.Bwidth_Y, bl.Zmax);

	real_t *Vde_y[4] = {fdata.Vde[dvcx], fdata.Vde[dvcz], fdata.Vde[ducx], fdata.Vde[dwcz]};
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_y, local_ndrange_y), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0);
			int j0 = index.get_global_id(1) + 0;
			int j1 = index.get_global_id(1) + bl.Ymax - bl.Bwidth_Y;
			int k = index.get_global_id(2);

			CenterDerivativeBCKernelY(i, j0, k, bl, BC[2], Vde_y, 0, bl.Bwidth_Y, 1);
			CenterDerivativeBCKernelY(i, j1, k, bl, BC[3], Vde_y, bl.Y_inner, bl.Ymax - bl.Bwidth_Y - 1, -1); }); });
#endif // DIM_Y

#if DIM_Z
	auto local_ndrange_z = range<3>(bl.dim_block_x, bl.dim_block_y, bl.Bwidth_Z); // size of workgroup
	auto global_ndrange_z = range<3>(bl.Xmax, bl.Ymax, bl.Bwidth_Z);

	real_t *Vde_z[4] = {fdata.Vde[dwcx], fdata.Vde[dwcy], fdata.Vde[ducx], fdata.Vde[dvcy]};
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_z, local_ndrange_z), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0);
			int j = index.get_global_id(1);
    		int k0 = index.get_global_id(2) + 0;
			int k1 = index.get_global_id(2) + bl.Zmax - bl.Bwidth_Z;

			CenterDerivativeBCKernelZ(i, j, k0, bl, BC[4], Vde_z, 0, bl.Bwidth_Z, 1);
			CenterDerivativeBCKernelZ(i, j, k1, bl, BC[5], Vde_z, bl.Z_inner, bl.Zmax - bl.Bwidth_Z - 1, -1); }); });
#endif // DIM_Z
}

void GetWallViscousFluxes(sycl::queue &q, Block bl, FlowData &fdata)
{
	auto local_ndrange = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z); // size of workgroup
	real_t **Vde = fdata.Vde;
#if DIM_X
	real_t *V[3] = {fdata.u, fdata.v, fdata.w};
	auto global_ndrange_x = range<3>(bl.X_inner + 1, bl.Y_inner, bl.Z_inner);
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_x, local_ndrange), [=](sycl::nd_item<3> index)
							  { GetWallViscousFluxesKernelX(index, bl, V, Vde); }); });
	q.wait();
#endif // DIM_X
#if DIM_Y
#endif // DIM_Y
#if DIM_Z
#endif // DIM_Z
}
#endif // Visc
#ifdef Heat
#endif // Heat
#ifdef Diffu
#endif // Diffu
#ifdef React
void FluidODESolver(sycl::queue &q, Block bl, Thermal *thermal, FlowData &fdata, real_t *UI, Reaction *react, const real_t dt)
{
	auto local_ndrange = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z); // size of workgroup
	auto global_ndrange = range<3>(bl.Xmax, bl.Ymax, bl.Zmax);

	real_t *rho = fdata.rho;
	real_t *p = fdata.p;
	real_t *H = fdata.H;
	real_t *c = fdata.c;
	real_t *u = fdata.u;
	real_t *v = fdata.v;
	real_t *w = fdata.w;
	real_t *y = fdata.y;
	real_t *T = fdata.T;

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0);
    		int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			FluidODESolverKernel(i ,j ,k ,bl ,thermal ,react ,UI ,y ,rho , T, dt); }); });
	q.wait();
}
#endif // React
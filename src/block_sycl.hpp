#include "include/global_class.h"
#include "sycl_kernels.hpp"

using namespace std;
using namespace sycl;

void InitializeFluidStates(sycl::queue &q, Block bl, IniShape ini, MaterialProperty material, Thermal *thermal, FlowData &fdata, real_t *U, real_t *U1, real_t *LU,
						   real_t *FluxF, real_t *FluxG, real_t *FluxH, real_t *FluxFw, real_t *FluxGw, real_t *FluxHw)
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
	real_t *T = fdata.T;

	q.submit([&](sycl::handler &h)
			 {
		//auto out = sycl::stream(1024, 256, h);
		h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index)
					   {
			int i = index.get_global_id(0);
			int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			InitialStatesKernel(i, j, k, bl, ini, material, thermal, U, U1, LU, FluxF, FluxG, FluxH, FluxFw, FluxGw, FluxHw, u, v, w, rho, p, fdata.y, T, H, c); }); })
		.wait();
}

real_t GetDt(sycl::queue &q, Block bl, FlowData &fdata, real_t *uvw_c_max)
{
	real_t *rho = fdata.rho;
	real_t *c = fdata.c;
	real_t *u = fdata.u;
	real_t *v = fdata.v;
	real_t *w = fdata.w;

	auto local_ndrange = range<1>(16); // size of workgroup
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
					   [=](nd_item<1> index, auto &temp_max_x, auto &temp_max_y, auto &temp_max_z)
					   {
						   auto id = index.get_global_id();
						   temp_max_x.combine(u[id] + c[id]);
						   temp_max_y.combine(v[id] + c[id]);
						   temp_max_z.combine(w[id] + c[id]);
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
	real_t *T = fdata.T;
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index)
							  {
					int i = index.get_global_id(0);
					int j = index.get_global_id(1);
					int k = index.get_global_id(2);

			UpdateFuidStatesKernel(i, j, k, bl, thermal, UI, FluxF, FluxG, FluxH, rho, p, c, H, u, v, w, fdata.y, T, Gamma); }); })
		.wait();
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
			UpdateURK3rdKernel(i, j, k, bl, U, U1, LU, dt, flag); }); })
		.wait();
}

#ifdef Visc
void GetCellCenterDerivative(sycl::queue &q, Block bl, FlowData &fdata, BConditions BC[6])
{
	int range_x = DIM_X ? bl.X_inner + 4 : 1; // NOTE：这里的4是由求微分的算法确定的,内点网格向两边各延伸两个点
	int range_y = DIM_Y ? bl.Y_inner + 4 : 1;
	int range_z = DIM_Z ? bl.Z_inner + 4 : 1;
	int offset_x = DIM_X ? bl.Bwidth_X - 2 : 0; // NOTE: 这是计算第i(j/k)个点右边的那个半点，所以从=(+Bwidth-2) 开始到<(+inner+Bwidth+2)结束
	int offset_y = DIM_Y ? bl.Bwidth_Y - 2 : 0;
	int offset_z = DIM_Z ? bl.Bwidth_Z - 2 : 0;
	auto local_ndrange_ck = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z);
	auto global_ndrange_ck = range<3>(range_x, range_y, range_z);

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_ck, local_ndrange_ck), [=](sycl::nd_item<3> index)
							  {
			int i = index.get_global_id(0) + offset_x;
			int j = index.get_global_id(1) + offset_y;
			int k = index.get_global_id(2) + offset_z;
			GetInnerCellCenterDerivativeKernel(i, j, k, bl, fdata.u, fdata.v, fdata.w, fdata.Vde); }); })
		.wait();

#if DIM_X
	auto local_ndrange_x = range<3>(bl.Bwidth_X, bl.dim_block_y, bl.dim_block_z); // size of workgroup
	auto global_ndrange_x = range<3>(bl.Bwidth_X, bl.Ymax, bl.Zmax);

	BConditions BC0 = BC[0], BC1 = BC[1];
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_x, local_ndrange_x), [=](sycl::nd_item<3> index)
							  {
    		int i0 = index.get_global_id(0) + 0;
			int i1 = index.get_global_id(0) + bl.Xmax - bl.Bwidth_X;
			int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			CenterDerivativeBCKernelX(i0, j, k, bl, BC0, fdata.Vde, 0, bl.Bwidth_X, 1);
			CenterDerivativeBCKernelX(i1, j, k, bl, BC1, fdata.Vde, bl.X_inner, bl.Xmax-bl.Bwidth_X-1, -1); }); })
		.wait();
#endif // DIM_X

#if DIM_Y
	auto local_ndrange_y = range<3>(bl.dim_block_x, bl.Bwidth_Y, bl.dim_block_z); // size of workgroup
	auto global_ndrange_y = range<3>(bl.Xmax, bl.Bwidth_Y, bl.Zmax);

	BConditions BC2 = BC[2], BC3 = BC[3];
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_y, local_ndrange_y), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0);
			int j0 = index.get_global_id(1) + 0;
			int j1 = index.get_global_id(1) + bl.Ymax - bl.Bwidth_Y;
			int k = index.get_global_id(2);

			CenterDerivativeBCKernelY(i, j0, k, bl, BC2, fdata.Vde, 0, bl.Bwidth_Y, 1);
			CenterDerivativeBCKernelY(i, j1, k, bl, BC3, fdata.Vde, bl.Y_inner, bl.Ymax - bl.Bwidth_Y - 1, -1); }); })
		.wait();
#endif // DIM_Y

#if DIM_Z
	auto local_ndrange_z = range<3>(bl.dim_block_x, bl.dim_block_y, bl.Bwidth_Z); // size of workgroup
	auto global_ndrange_z = range<3>(bl.Xmax, bl.Ymax, bl.Bwidth_Z);

	BConditions BC4 = BC[4], BC5 = BC[5];
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_z, local_ndrange_z), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0);
			int j = index.get_global_id(1);
    		int k0 = index.get_global_id(2) + 0;
			int k1 = index.get_global_id(2) + bl.Zmax - bl.Bwidth_Z;

			CenterDerivativeBCKernelZ(i, j, k0, bl, BC4, fdata.Vde, 0, bl.Bwidth_Z, 1);
			CenterDerivativeBCKernelZ(i, j, k1, bl, BC5, fdata.Vde, bl.Z_inner, bl.Zmax - bl.Bwidth_Z - 1, -1); }); })
		.wait();
#endif // DIM_Z
}
#endif // Visc

void GetLU(sycl::queue &q, Block bl, BConditions BCs[6], Thermal *thermal, real_t *UI, real_t *LU,
		   real_t *FluxF, real_t *FluxG, real_t *FluxH, real_t *FluxFw, real_t *FluxGw, real_t *FluxHw,
		   real_t const Gamma, int const Mtrl_ind, FlowData &fdata, real_t *eigen_local)
{
	real_t *rho = fdata.rho;
	real_t *p = fdata.p;
	real_t *H = fdata.H;
	real_t *c = fdata.c;
	real_t *u = fdata.u;
	real_t *v = fdata.v;
	real_t *w = fdata.w;
	real_t *T = fdata.T;

	bool is_3d = DIM_X * DIM_Y * DIM_Z ? true : false;

	auto local_ndrange = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z);
	auto global_ndrange_max = range<3>(bl.Xmax, bl.Ymax, bl.Zmax);

#if DIM_X
	// proceed at x directiom and get F-flux terms at node wall
	auto global_ndrange_x = range<3>(bl.X_inner + local_ndrange[0], bl.Y_inner, bl.Z_inner);

	event ex = q.submit([&](sycl::handler &h)
						{ h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), [=](sycl::nd_item<3> index)
										 {
    		int i = index.get_global_id(0);
    		int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			GetLocalEigen(i, j, k, bl, 1.0, 0.0, 0.0, eigen_local, u, v, w, c); }); });

	q.submit([&](sycl::handler &h)
			 {
		h.depends_on(ex);
		h.parallel_for(sycl::nd_range<3>(global_ndrange_x, local_ndrange), [=](sycl::nd_item<3> index)
					   {
    		int i = index.get_global_id(0) + bl.Bwidth_X - 1;
			int j = index.get_global_id(1) + bl.Bwidth_Y;
			int k = index.get_global_id(2) + bl.Bwidth_Z;
			ReconstructFluxX(i, j, k, bl,thermal,Gamma, UI, FluxF, FluxFw, eigen_local, rho, u, v, w, fdata.y, T, H);
			 }); })
		.wait();
#endif

#if DIM_Y
	// proceed at y directiom and get G-flux terms at node wall
	auto global_ndrange_y = range<3>(bl.X_inner, bl.Y_inner + local_ndrange[1], bl.Z_inner);

	event ey = q.submit([&](sycl::handler &h)
						{ h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), [=](sycl::nd_item<3> index)
										 {
    		int i = index.get_global_id(0);
    		int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			GetLocalEigen(i, j, k, bl, 0.0, 1.0, 0.0, eigen_local, u, v, w, c); }); });

	q.submit([&](sycl::handler &h)
			 {
		h.depends_on(ey);
		h.parallel_for(sycl::nd_range<3>(global_ndrange_y, local_ndrange), [=](sycl::nd_item<3> index)
					   {
    		int i = index.get_global_id(0) + bl.Bwidth_X;
			int j = index.get_global_id(1) + bl.Bwidth_Y - 1;
			int k = index.get_global_id(2) + bl.Bwidth_Z;
			ReconstructFluxY(i, j, k, bl, thermal, Gamma, UI, FluxG, FluxGw, eigen_local, rho, u, v, w, fdata.y, T, H); }); })
		.wait();
#endif

#if DIM_Z
	// proceed at y directiom and get G-flux terms at node wall
	auto global_ndrange_z = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner + local_ndrange[2]);

	event ez = q.submit([&](sycl::handler &h)
						{ h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), [=](sycl::nd_item<3> index)
										 {
    		int i = index.get_global_id(0);
    		int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			GetLocalEigen(i, j, k, bl, 0.0, 0.0, 1.0, eigen_local, u, v, w, c); }); });

	q.submit([&](sycl::handler &h)
			 {
		h.depends_on(ez);
		h.parallel_for(sycl::nd_range<3>(global_ndrange_y, local_ndrange), [=](sycl::nd_item<3> index)
					   {
    		int i = index.get_global_id(0) + bl.Bwidth_X;
			int j = index.get_global_id(1) + bl.Bwidth_Y;
			int k = index.get_global_id(2) + bl.Bwidth_Z - 1;
			ReconstructFluxZ(i, j, k, bl, thermal, Gamma, UI, FluxH, FluxHw, eigen_local, rho, u, v, w, fdata.y, T, H); }); })
		.wait();
#endif
	q.wait();
// NOTE: add visc flux to Fluxw
#ifdef Visc
	/* Viscous LU including physical visc(切应力),Heat transfer(传热), mass Diffusion(质量扩散)
	 * Physical Visc must be included Heat is alternative and Diffu depends on compent
	 */
	real_t *V[3] = {fdata.u, fdata.v, fdata.w};
	real_t **Vde = fdata.Vde;
	real_t *va = fdata.viscosity_aver;
	real_t *tca = fdata.thermal_conduct_aver;
	real_t *Da = fdata.Dkm_aver;
	real_t *hi = fdata.hi;

	GetCellCenterDerivative(q, bl, fdata, BCs);
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0);
			int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			Gettransport_coeff_aver(i, j, k, bl, thermal, va, tca, Da, fdata.y, hi, rho, p, T); }); })
		.wait();
// NOTE: get wall flux
#if DIM_X
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_x, local_ndrange), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0) + bl.Bwidth_X - 1;
			int j = index.get_global_id(1) + bl.Bwidth_Y;
			int k = index.get_global_id(2) + bl.Bwidth_Z;
			GetWallViscousFluxX(i, j, k, bl, FluxFw, va, tca, Da, T, rho, hi, fdata.y, V, Vde); }); })
		.wait();
#endif // end DIM_X
#if DIM_Y
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_y, local_ndrange), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0) + bl.Bwidth_X;
			int j = index.get_global_id(1) + bl.Bwidth_Y - 1;
			int k = index.get_global_id(2) + bl.Bwidth_Z;
			GetWallViscousFluxY(i, j, k, bl, FluxGw, va, tca, Da, T, rho, hi, fdata.y, V, Vde); }); })
		.wait();
#endif // end DIM_Y
#if DIM_Z
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_z, local_ndrange), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0) + bl.Bwidth_X;
			int j = index.get_global_id(1) + bl.Bwidth_Y;
			int k = index.get_global_id(2) + bl.Bwidth_Z - 1;
			GetWallViscousFluxZ(i, j, k, bl, FluxHw, va, tca, Da, T, rho, hi, fdata.y, V, Vde); }); })
		.wait();
#endif // end DIM_Z
#endif // add Visc flux

// NOTE: update LU from cell-face fluxes
	auto global_ndrange_inner = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner);
#if NumFluid == 2
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

void FluidBoundaryCondition(sycl::queue &q, Setup setup, BConditions BCs[6], real_t *d_UI)
{
	Block bl = setup.BlSz;
#if USE_MPI
	MpiTrans Trans = *(setup.mpiTrans);
#endif // end USE_MPI
#if DIM_X
	auto local_ndrange_x = range<3>(bl.Bwidth_X, bl.dim_block_y, bl.dim_block_z); // size of workgroup
	auto global_ndrange_x = range<3>(bl.Bwidth_X, bl.Ymax, bl.Zmax);

#if USE_MPI
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_x, local_ndrange_x), [=](sycl::nd_item<3> index)
							  {
								  int i0 = index.get_global_id(0) + 0;
								  int i1 = index.get_global_id(0) + bl.Xmax - bl.Bwidth_X;
								  int j = index.get_global_id(1);
								  int k = index.get_global_id(2);

								  FluidMpiCopyKernelX(i0, j, k, bl, Trans.d_mpiData->TransBufSend_xmin, d_UI, 0, -bl.Bwidth_X, BorToBuf);					 // X_MIN
								  FluidMpiCopyKernelX(i1, j, k, bl, Trans.d_mpiData->TransBufSend_xmax, d_UI, bl.Xmax - bl.Bwidth_X, bl.Bwidth_X, BorToBuf); // X_MAX
							  }); })
		.wait();

	Trans.MpiTransBuf(q, XDIR);
#endif // USE_MPI
	BConditions BC0 = BCs[0], BC1 = BCs[1];
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(
				   sycl::nd_range<3>(global_ndrange_x, local_ndrange_x), [=](sycl::nd_item<3> index)
				   {
					   int i0 = index.get_global_id(0) + 0;
					   int i1 = index.get_global_id(0) + bl.Xmax - bl.Bwidth_X;
					   int j = index.get_global_id(1);
					   int k = index.get_global_id(2);
#if USE_MPI
					   if (Trans.neighborsBC[XMIN] == BC_COPY) // 将接收到的RecvBuf拷入Ghostcell
						   FluidMpiCopyKernelX(i0, j, k, bl, Trans.d_mpiData->TransBufRecv_xmin, d_UI, 0, -bl.Bwidth_X, BufToBC);
					   else
#endif // USE_MPI
						   FluidBCKernelX(i0, j, k, bl, BC0, d_UI, 0, bl.Bwidth_X, 1);
#ifdef USE_MPI
					   if (Trans.neighborsBC[XMAX] == BC_COPY)
						   FluidMpiCopyKernelX(i1, j, k, bl, Trans.d_mpiData->TransBufRecv_xmax, d_UI, bl.Xmax - bl.Bwidth_X, bl.Bwidth_X, BufToBC);
					   else
#endif // USE_MPI
						   FluidBCKernelX(i1, j, k, bl, BC1, d_UI, bl.X_inner, bl.Xmax - bl.Bwidth_X - 1, -1); }); })
		.wait();
#endif // end DIM_X

#if DIM_Y
	auto local_ndrange_y = range<3>(bl.dim_block_x, bl.Bwidth_Y, bl.dim_block_z); // size of workgroup
	auto global_ndrange_y = range<3>(bl.Xmax, bl.Bwidth_Y, bl.Zmax);

#if USE_MPI
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_y, local_ndrange_y), [=](sycl::nd_item<3> index)
							  {
								  int i = index.get_global_id(0);
								  int j0 = index.get_global_id(1) + 0;
								  int j1 = index.get_global_id(1) + bl.Ymax - bl.Bwidth_Y;
								  int k = index.get_global_id(2);

								  FluidMpiCopyKernelY(i, j0, k, bl, Trans.d_mpiData->TransBufSend_ymin, d_UI, 0, -bl.Bwidth_Y, BorToBuf);					 // X_MIN
								  FluidMpiCopyKernelY(i, j1, k, bl, Trans.d_mpiData->TransBufSend_ymax, d_UI, bl.Ymax - bl.Bwidth_Y, bl.Bwidth_Y, BorToBuf); // X_MAX
							  }); })
		.wait();

	Trans.MpiTransBuf(q, YDIR);
#endif // USE_MPI
	BConditions BC2 = BCs[2], BC3 = BCs[3];
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_y, local_ndrange_y), [=](sycl::nd_item<3> index)
							  {
								  int i = index.get_global_id(0);
								  int j0 = index.get_global_id(1) + 0;
								  int j1 = index.get_global_id(1) + bl.Ymax - bl.Bwidth_Y;
								  int k = index.get_global_id(2);
#if USE_MPI
								  if (Trans.neighborsBC[YMIN] == BC_COPY) // 将接收到的RecvBuf拷入Ghostcell
									  FluidMpiCopyKernelY(i, j0, k, bl, Trans.d_mpiData->TransBufRecv_ymin, d_UI, 0, -bl.Bwidth_Y, BufToBC);
								  else
#endif // USE_MPI
									  FluidBCKernelY(i, j0, k, bl, BC2, d_UI, 0, bl.Bwidth_Y, 1);
#ifdef USE_MPI
								  if (Trans.neighborsBC[YMAX] == BC_COPY)
									  FluidMpiCopyKernelX(i, j1, k, bl, Trans.d_mpiData->TransBufRecv_ymax, d_UI, bl.Ymax - bl.Bwidth_Y, bl.Bwidth_Y, BufToBC);
								  else
#endif																													  // USE_MPI
									  FluidBCKernelY(i, j1, k, bl, BC3, d_UI, bl.Y_inner, bl.Ymax - bl.Bwidth_Y - 1, -1); //
							  }); })
		.wait();
#endif // end DIM_Y

#if DIM_Z
	auto local_ndrange_z = range<3>(bl.dim_block_x, bl.dim_block_y, bl.Bwidth_Z); // size of workgroup
	auto global_ndrange_z = range<3>(bl.Xmax, bl.Ymax, bl.Bwidth_Z);

#if USE_MPI
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_z, local_ndrange_z), [=](sycl::nd_item<3> index)
							  {
								  int i = index.get_global_id(0);
								  int j = index.get_global_id(1);
								  int k0 = index.get_global_id(2) + 0;
								  int k1 = index.get_global_id(2) + bl.Zmax - bl.Bwidth_Z;

								  FluidMpiCopyKernelZ(i, j, k0, bl, Trans.d_mpiData->TransBufSend_zmin, d_UI, 0, -bl.Bwidth_Z, BorToBuf);					 // X_MIN
								  FluidMpiCopyKernelZ(i, j, k1, bl, Trans.d_mpiData->TransBufSend_zmax, d_UI, bl.Zmax - bl.Bwidth_Z, bl.Bwidth_Z, BorToBuf); // X_MAX
							  }); })
		.wait();

	Trans.MpiTransBuf(q, ZDIR);
#endif // USE_MPI
	BConditions BC4 = BCs[4], BC5 = BCs[5];
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_z, local_ndrange_z), [=](sycl::nd_item<3> index)
							  {
								  int i = index.get_global_id(0);
								  int j = index.get_global_id(1);
								  int k0 = index.get_global_id(2) + 0;
								  int k1 = index.get_global_id(2) + bl.Zmax - bl.Bwidth_Z;
#if USE_MPI
								  if (Trans.neighborsBC[ZMIN] == BC_COPY) // 将接收到的RecvBuf拷入Ghostcell
									  FluidMpiCopyKernelY(i, j, k0, bl, Trans.d_mpiData->TransBufRecv_zmin, d_UI, 0, -bl.Bwidth_Z, BufToBC);
								  else
#endif // USE_MPI
									  FluidBCKernelZ(i, j, k0, bl, BC4, d_UI, 0, bl.Bwidth_Z, 1);
#ifdef USE_MPI
								  if (Trans.neighborsBC[ZMAX] == BC_COPY)
									  FluidMpiCopyKernelX(i, j, k1, bl, Trans.d_mpiData->TransBufRecv_zmax, d_UI, bl.Zmax - bl.Bwidth_Z, bl.Bwidth_Z, BufToBC);
								  else
#endif
									  FluidBCKernelZ(i, j, k1, bl, BC5, d_UI, bl.Z_inner, bl.Zmax - bl.Bwidth_Z - 1, -1); //
							  }); })
		.wait();
#endif // end DIM_Z
}

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
	real_t *T = fdata.T;

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0);
    		int j = index.get_global_id(1);
			int k = index.get_global_id(2);
			FluidODESolverKernel( i, j, k, bl, thermal, react, UI, fdata.y, rho, T, dt); }); });
	q.wait();
}
#endif // React

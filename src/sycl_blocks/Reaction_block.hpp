#include "Utils_block.hpp"

void ZeroDimensionalFreelyFlameBlock(Setup &sep)
{
	ZeroDimensionalFreelyFlameKernel(sep, 0);
}

void ChemeODEQ2Solver(sycl::queue &q, Block bl, Thermal thermal, FlowData &fdata, real_t *UI, Reaction react, const real_t dt)
{
	auto local_ndrange = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z); // size of workgroup
	auto global_ndrange = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner);

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
								  int i = index.get_global_id(0) + bl.Bwidth_X;
								  int j = index.get_global_id(1) + bl.Bwidth_Y;
								  int k = index.get_global_id(2) + bl.Bwidth_Z;
								  ChemeODEQ2SolverKernel( i, j, k, bl, thermal, react, UI, fdata.y, rho, T, fdata.e, dt); }); })
		.wait();
}
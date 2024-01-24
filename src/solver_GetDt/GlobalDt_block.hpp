#pragma once

#include "global_setup.h"
#include "../read_ini/setupini.h"

real_t GetDt(sycl::queue &q, Block bl, Thermal &thermal, FlowData &fdata, real_t *uvw_c_max)
{
	real_t *c = fdata.c;
	real_t *u = fdata.u;
	real_t *v = fdata.v;
	real_t *w = fdata.w;
	real_t *T = fdata.T;
	real_t *yi = fdata.y;

	int meshSize = bl.Xmax * bl.Ymax * bl.Zmax;
	auto local_ndrange = range<1>(bl.BlockSize); // size of workgroup
	auto global_ndrange = range<1>(meshSize);

	real_t dtref = _DF(0.0);

	// NOTE: dt of inviscous flow
	// add uvw and c individually if need more resources
	for (int n = 0; n < 6; n++)
		uvw_c_max[n] = _DF(0.0);

	// define reduction objects for sum, min, max reduction
	// auto reduction_sum = reduction(sum, sycl::plus<real_t>());
	if (bl.DimX)
	{
		uvw_c_max[5] = sycl::max(uvw_c_max[5], bl._dx * bl._dx);
		q.submit([&](sycl::handler &h) { //
			auto reduction_max_x = sycl_reduction_max(uvw_c_max[0]);
			h.parallel_for(sycl::nd_range<1>(global_ndrange, local_ndrange), reduction_max_x, [=](nd_item<1> index, auto &temp_max_x) { //
				auto id = index.get_global_id();
				temp_max_x.combine(sycl::fabs(u[id]) + c[id]);
			});
		});
	}
	if (bl.DimY)
	{
		uvw_c_max[5] = sycl::max(uvw_c_max[5], bl._dy * bl._dy);
		q.submit([&](sycl::handler &h) {																								//
			auto reduction_max_y = sycl_reduction_max(uvw_c_max[1]);																	// reduction(&(uvw_c_max[1]), sycl::maximum<real_t>());
			h.parallel_for(sycl::nd_range<1>(global_ndrange, local_ndrange), reduction_max_y, [=](nd_item<1> index, auto &temp_max_y) { //
				auto id = index.get_global_id();
				temp_max_y.combine(sycl::fabs(v[id]) + c[id]);
			});
		});
	}
	if (bl.DimZ)
	{
		uvw_c_max[5] = sycl::max(uvw_c_max[5], bl._dz * bl._dz);
		q.submit([&](sycl::handler &h) {																								//
			auto reduction_max_z = sycl_reduction_max(uvw_c_max[2]);																	// reduction(&(uvw_c_max[2]), sycl::maximum<real_t>());
			h.parallel_for(sycl::nd_range<1>(global_ndrange, local_ndrange), reduction_max_z, [=](nd_item<1> index, auto &temp_max_z) { //
				auto id = index.get_global_id();
				temp_max_z.combine(sycl::fabs(w[id]) + c[id]);
			});
		});
	}

	q.wait();

	dtref = uvw_c_max[0] * bl._dx + uvw_c_max[1] * bl._dy + uvw_c_max[2] * bl._dz;

	// NOTE: dt of viscous flow
#if Visc
	// real_t *va = fdata.viscosity_aver;
	// real_t *tca = fdata.thermal_conduct_aver;
	// real_t *Da = fdata.Dkm_aver;
	// real_t *hi = fdata.hi;

	// auto global_ndrange_max = range<3>(bl.Xmax, bl.Ymax, bl.Zmax);
	// auto local_ndrange_max = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z);
	// q.submit([&](sycl::handler &h)
	// 		 { h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange_max), [=](sycl::nd_item<3> index)
	// 						  {
	// 		int i = index.get_global_id(0);
	// 		int j = index.get_global_id(1);
	// 		int k = index.get_global_id(2);
	// 		Gettransport_coeff_aver(i, j, k, bl, thermal, va, tca, Da, fdata.y, hi, fdata.rho, fdata.p, fdata.T, fdata.Ertemp1, fdata.Ertemp2); }); })
	// 	.wait();

	// // max viscosity
	// uvw_c_max[3] = _DF(0.0);
	// auto reduction_max_miu = reduction(&(uvw_c_max[3]), sycl::maximum<real_t>());
	// q.submit([&](sycl::handler &h)
	// 		 { h.parallel_for(sycl::nd_range<1>(global_ndrange, local_ndrange), reduction_max_miu, [=](nd_item<1> index, auto &temp_max_miu)
	// 						  {
	// 		auto id = index.get_global_id();
	// 		temp_max_miu.combine(va[id]); }); });
	// // max rho
	// uvw_c_max[4] = _DF(100.0);
	// auto reduction_max_rho = reduction(&(uvw_c_max[4]), sycl::minimum<real_t>());
	// q.submit([&](sycl::handler &h)
	// 		 { h.parallel_for(sycl::nd_range<1>(global_ndrange, local_ndrange), reduction_max_rho, [=](nd_item<1> index, auto &temp_max_rho)
	// 						  {
	// 		auto id = index.get_global_id();
	// 		temp_max_rho.combine(fdata.rho[id]); }); });
	// q.wait();

	// real_t temp_visc = _DF(_DF(14.0) / _DF(3.0)) * uvw_c_max[3] * uvw_c_max[5] / uvw_c_max[4];
	// dtref = sycl::max(dtref, temp_visc);
#endif // end get viscity

	return bl.CFLnumber / dtref;
}

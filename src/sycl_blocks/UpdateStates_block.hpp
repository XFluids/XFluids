#include "Utils_block.hpp"

bool UpdateFluidStateFlux(sycl::queue &q, Block bl, Thermal thermal, real_t *UI, FlowData &fdata, real_t *FluxF, real_t *FluxG, real_t *FluxH, real_t const Gamma, int &error_patched_times)
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

	// // update rho and yi
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index)
							  {
					int i = index.get_global_id(0);
					int j = index.get_global_id(1);
					int k = index.get_global_id(2);
					Updaterhoyi(i, j, k, bl, UI, rho, fdata.y); }); })
		.wait();

#ifdef ESTIM_NAN
	int *error_posyi;
	bool *error_org, *error_nan;
	error_posyi = middle::MallocShared<int>(error_posyi, 4 + NUM_SPECIES, q);
	error_org = middle::MallocShared<bool>(error_org, 1, q), error_nan = middle::MallocShared<bool>(error_nan, 1, q);
	*error_nan = false, *error_org = false;
	for (size_t i = 0; i < NUM_SPECIES + 3; i++)
		error_posyi[i] = _DF(0.0);
	auto Sum_Epts = sycl::reduction(&(error_posyi[NUM_SPECIES + 3]), sycl::plus<>()); // error_patch_times

	// // update estimate negative or nan yi and patch
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), Sum_Epts, [=](sycl::nd_item<3> index, auto &tEpts)
							  {
									int i = index.get_global_id(0) + bl.Bwidth_X;
									int j = index.get_global_id(1) + bl.Bwidth_Y;
									int k = index.get_global_id(2) + bl.Bwidth_Z;
									EstimateYiKernel(i, j, k, bl, error_posyi, error_org, error_nan, UI, rho, fdata.y); }); }) //, tEpts
		.wait();

	int offsetx = bl.OutBC ? 0 : bl.Bwidth_X;
	int offsety = bl.OutBC ? 0 : bl.Bwidth_Y;
	int offsetz = bl.OutBC ? 0 : bl.Bwidth_Z;

	if (*error_org)
		error_patched_times += 1; // error_posyi[NUM_SPECIES + 3];
	if (*error_nan)
	{
		error_patched_times++;
		std::cout << "Errors of Yi[";
		for (size_t ii = 0; ii < NUM_COP; ii++)
			std::cout << error_posyi[ii] << ", ";
		std::cout << error_posyi[NUM_COP] << "] located at (i, j, k)= (";
		std::cout << error_posyi[NUM_SPECIES] - offsetx << ", " << error_posyi[NUM_SPECIES + 1] - offsety << ", " << error_posyi[NUM_SPECIES + 2] - offsetz;
#ifdef ERROR_PATCH_YI
		std::cout << ") patched.\n";
#else
		std::cout << ") captured.\n";
		return true;
#endif // end ERROR_PATCH_YI
	}
#endif // end ESTIM_NAN

	q.submit([&](sycl::handler &h)
			 {         
				// sycl::stream stream_ct1(64 * 1024, 80, h);// for output error: sycl::stream decline running efficiency
				h.parallel_for(sycl::nd_range<3>(global_ndrange, local_ndrange), [=](sycl::nd_item<3> index)
							  {
					int i = index.get_global_id(0);
					int j = index.get_global_id(1);
					int k = index.get_global_id(2);
					UpdateFuidStatesKernel(i, j, k, bl, thermal, UI, FluxF, FluxG, FluxH, rho, p, c, H, u, v, w, fdata.y, fdata.gamma, T, fdata.e, Gamma); }); }) //, stream_ct1
		.wait();

#ifdef ESTIM_NAN
	int *error_pos;
	bool *error_yi, *error_nga;
	error_pos = middle::MallocShared<int>(error_pos, 6 + NUM_SPECIES, q);
	error_yi = middle::MallocShared<bool>(error_yi, 1, q), error_nga = middle::MallocShared<bool>(error_nga, 1, q);
	*error_nga = false, *error_yi = false;
	for (size_t n = 0; n < 6 + NUM_SPECIES; n++)
		error_pos[n] = 0;

	auto global_in_ndrange = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner);
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_in_ndrange, local_ndrange), [=](sycl::nd_item<3> index)
							  {
								int i = index.get_global_id(0) + bl.Bwidth_X;
								int j = index.get_global_id(1) + bl.Bwidth_Y;
								int k = index.get_global_id(2) + bl.Bwidth_Z;
								EstimatePrimitiveVarKernel(i, j, k, bl, thermal, error_pos, error_nga, error_yi,
														   UI, rho, u, v, w, p, T, fdata.y, H, fdata.e, fdata.gamma, c); }); })
		.wait();

	if (*error_nga)
	{
		std::cout << "Errors of Primitive variables[rho, T, P][";
		for (size_t ii = 0; ii < 2 + NUM_SPECIES; ii++)
			std::cout << error_pos[ii] << ", ";
		std::cout << error_pos[2 + NUM_SPECIES] << "] located at (i, j, k)= (";
		std::cout << error_pos[3 + NUM_SPECIES] - offsetx << ", " << error_pos[4 + NUM_SPECIES] - offsety << ", " << error_pos[5 + NUM_SPECIES] - offsetz;
#ifdef ERROR_PATCH
		std::cout << ") patched.\n";
#else
		std::cout << ") captured.\n";
		return true;
#endif // end ERROR_PATCH
	}

	// free
	{
		middle::Free(error_posyi, q);
		middle::Free(error_org, q);
		middle::Free(error_nan, q);
		middle::Free(error_pos, q);
		middle::Free(error_yi, q);
		middle::Free(error_nga, q);
	}
#endif // end ESTIM_NAN

	return false;
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
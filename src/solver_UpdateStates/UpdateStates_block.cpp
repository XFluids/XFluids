#include "timer/timer.h"

#include "Estimate_kernels.hpp"
#include "UpdateStates_block.h"
#include "kattribute/attribute.h"

std::pair<bool, std::vector<float>> UpdateFluidStateFlux(sycl::queue &q, Setup Ss, Thermal thermal, real_t *UI,
														 FlowData &fdata, real_t *FluxF, real_t *FluxG, real_t *FluxH,
														 real_t const Gamma, int &error_patched_times, const int rank)
{
#ifndef __REVERSE_NDRANGE__
	Block bl = Ss.BlSz;
	MeshSize ms = bl.Ms;
	real_t *rho = fdata.rho;
	real_t *p = fdata.p;
	real_t *H = fdata.H;
	real_t *c = fdata.c;
	real_t *u = fdata.u;
	real_t *v = fdata.v;
	real_t *w = fdata.w;
	real_t *e = fdata.e;
	real_t *g = fdata.gamma;
	real_t *T = fdata.T;
	real_t *Ri = fdata.Ri;
	real_t *Cp = fdata.Cp;

	std::vector<float> timer_UD;
	float runtime_emyi = 0.0f, runtime_empv = 0.0f;
	float runtime_rhoyi = 0.0f, runtime_states = 0.0f;
	std::chrono::high_resolution_clock::time_point runtime_ud_start;

	auto global_ndrange = sycl::range<3>(bl.Xmax, bl.Ymax, bl.Zmax);
	auto local_ndrange = sycl::range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z); // size of workgroup

	// // update rho and yi
	Assign temury(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "Updaterhoyi");
	runtime_ud_start = std::chrono::high_resolution_clock::now();
	q.submit([&](sycl::handler &h) {																						//
		 h.parallel_for(sycl::nd_range<3>(temury.global_nd(global_ndrange), temury.local_nd), [=](sycl::nd_item<3> index) { //
			 int i = index.get_global_id(0);
			 int j = index.get_global_id(1);
			 int k = index.get_global_id(2);
			 Updaterhoyi(i, j, k, ms, UI, rho, fdata.y);
		 });
	 })
		.wait();
	runtime_rhoyi = OutThisTime(runtime_ud_start);
	if (Setup::adv_push)
		Setup::adv_nd[Setup::adv_id].push_back(temury.Time(runtime_rhoyi));

#if ESTIM_NAN
	int *error_posyi;
	bool *error_org, *error_nan;
	error_org = middle::MallocShared<bool>(error_org, 1, q);
	error_nan = middle::MallocShared<bool>(error_nan, 1, q);
	error_posyi = middle::MallocShared<int>(error_posyi, 5 + NUM_SPECIES, q);
	*error_nan = false, *error_org = false;
	for (size_t i = 0; i < NUM_SPECIES + 4; i++)
		error_posyi[i] = _DF(0.0);

	// // update estimate negative or nan rho, yi
	Assign temey(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "EstimateYiKernel");
	runtime_ud_start = std::chrono::high_resolution_clock::now();
	q.submit([&](sycl::handler &h) {																					  //
		 h.parallel_for(sycl::nd_range<3>(temey.global_nd(global_ndrange), temey.local_nd), [=](sycl::nd_item<3> index) { //
			 int i = index.get_global_id(0) + ms.Bwidth_X;
			 int j = index.get_global_id(1) + ms.Bwidth_Y;
			 int k = index.get_global_id(2) + ms.Bwidth_Z;
			 EstimateYiKernel(i, j, k, bl, error_posyi, error_org, error_nan, UI, rho, fdata.y);
		 });
	 })
		.wait();
	runtime_emyi = OutThisTime(runtime_ud_start);
	if (Setup::adv_push)
		Setup::adv_nd[Setup::adv_id].push_back(temey.Time(runtime_emyi));

	int offsetx = OutBoundary ? 0 : bl.Bwidth_X;
	int offsety = OutBoundary ? 0 : bl.Bwidth_Y;
	int offsetz = OutBoundary ? 0 : bl.Bwidth_Z;

	if (*error_org)
		error_patched_times += 1;
	if (*error_nan)
	{
		error_patched_times++;
		std::cout << "\nErrors of rho/Yi[";
		std::cout << error_posyi[NUM_SPECIES + 1] << ", ";
		for (size_t ii = 0; ii < NUM_SPECIES - 1; ii++)
			std::cout << error_posyi[ii] << ", ";
		std::cout << error_posyi[NUM_SPECIES] << "] located at (i, j, k)= ("
				  << error_posyi[NUM_SPECIES + 2] - offsetx << ", "
				  << error_posyi[NUM_SPECIES + 3] - offsety << ", "
				  << error_posyi[NUM_SPECIES + 4] - offsetz << ") of rank: " << rank;
#ifdef ERROR_PATCH_YI
		std::cout << " patched.\n";
#else
		std::cout << " captured.\n";
		timer_UD.push_back(runtime_emyi);
		timer_UD.push_back(runtime_empv);
		timer_UD.push_back(runtime_rhoyi);
		timer_UD.push_back(runtime_states);
		return std::make_pair(true, timer_UD);
#endif // end ERROR_PATCH_YI
	}
#endif // end ESTIM_NAN

	// sycl::stream stream_ct1(64 * 1024, 80, h);// for output error: sycl::stream decline running
	Assign temud(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "UpdateFuidStatesKernel");
	runtime_ud_start = std::chrono::high_resolution_clock::now();
#if __VENDOR_SUBMIT__
	CheckGPUErrors(vendorSetDevice(Ss.DeviceSelect[2]));
	static bool dummy = (GetKernelAttributes((const void *)UpdateFuidStatesKernelVendorWrapper, "UpdateFuidStatesKernelVendorWrapper"), true); // call only once
	UpdateFuidStatesKernelVendorWrapper<<<temud.global_gd(global_ndrange), temud.local_blk>>>(ms, thermal, UI, FluxF, FluxG, FluxH, rho, p, u, v, w, c, g, e, H, T, fdata.y, Ri, Cp);
	CheckGPUErrors(vendorDeviceSynchronize());
#else
	q.submit([&](sycl::handler &h) {																					  //
		 h.parallel_for(sycl::nd_range<3>(temud.global_nd(global_ndrange), temud.local_nd), [=](sycl::nd_item<3> index) { //
			 int i = index.get_global_id(0);
			 int j = index.get_global_id(1);
			 int k = index.get_global_id(2);
			 UpdateFuidStatesKernel(i, j, k, ms, thermal, UI, FluxF, FluxG, FluxH, rho, p, u, v, w, c, g, e, H, T, fdata.y, Ri, Cp);
		 });
	 }) //, stream_ct1
		.wait();
#endif
	runtime_states = OutThisTime(runtime_ud_start);
	if (Setup::adv_push)
		Setup::adv_nd[Setup::adv_id].push_back(temud.Time(runtime_states));

#if ESTIM_NAN
	int *error_pos;
	bool *error_yi, *error_nga;
	error_pos = middle::MallocShared<int>(error_pos, 6 + NUM_SPECIES, q);
	error_yi = middle::MallocShared<bool>(error_yi, 1, q), error_nga = middle::MallocShared<bool>(error_nga, 1, q);
	*error_nga = false, *error_yi = false;
	for (size_t n = 0; n < 6 + NUM_SPECIES; n++)
		error_pos[n] = 0;

	Assign temep(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "EstimatePrimitiveVarKernel");
	runtime_ud_start = std::chrono::high_resolution_clock::now();
	q.submit([&](sycl::handler &h) {																										  //
		 h.parallel_for(sycl::nd_range<3>(temep.global_nd(ms.X_inner, ms.Y_inner, ms.Z_inner), temep.local_nd), [=](sycl::nd_item<3> index) { //
			 int i = index.get_global_id(0) + ms.Bwidth_X;
			 int j = index.get_global_id(1) + ms.Bwidth_Y;
			 int k = index.get_global_id(2) + ms.Bwidth_Z;
			 EstimatePrimitiveVarKernel(i, j, k, bl, thermal, error_pos, error_nga, error_yi,
										UI, rho, u, v, w, p, T, fdata.y, H, fdata.e, fdata.gamma, c);
		 });
	 })
		.wait();
	runtime_empv = OutThisTime(runtime_ud_start);
	if (Setup::adv_push)
		Setup::adv_nd[Setup::adv_id].push_back(temep.Time(runtime_empv));

	if (*error_nga)
	{
		std::cout << "\nErrors of Primitive variables[rho, T, P][";
		for (size_t ii = 0; ii < 2; ii++)
			std::cout << error_pos[ii] << ", ";
		std::cout << error_pos[2] << "] located at (i, j, k)= ("
				  << error_pos[3 + NUM_SPECIES] - offsetx << ", "
				  << error_pos[4 + NUM_SPECIES] - offsety << ", "
				  << error_pos[5 + NUM_SPECIES] - offsetz << ") of rank: " << rank;
#ifdef ERROR_PATCH
		std::cout << " patched.\n";
#else
		std::cout << " captured.\n";
		timer_UD.push_back(runtime_emyi);
		timer_UD.push_back(runtime_empv);
		timer_UD.push_back(runtime_rhoyi);
		timer_UD.push_back(runtime_states);
		return std::make_pair(true, timer_UD);
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

	timer_UD.push_back(runtime_emyi);
	timer_UD.push_back(runtime_empv);
	timer_UD.push_back(runtime_rhoyi);
	timer_UD.push_back(runtime_states);
	return std::make_pair(false, timer_UD);

#else // __REVERSE_NDRANGE__
    Block bl = Ss.BlSz;
	MeshSize ms = bl.Ms;
	real_t *rho = fdata.rho;
	real_t *p = fdata.p;
	real_t *H = fdata.H;
	real_t *c = fdata.c;
	real_t *u = fdata.u;
	real_t *v = fdata.v;
	real_t *w = fdata.w;
	real_t *e = fdata.e;
	real_t *g = fdata.gamma;
	real_t *T = fdata.T;
	real_t *Ri = fdata.Ri;
	real_t *Cp = fdata.Cp;

	std::vector<float> timer_UD;
	float runtime_emyi = 0.0f, runtime_empv = 0.0f;
	float runtime_rhoyi = 0.0f, runtime_states = 0.0f;
	std::chrono::high_resolution_clock::time_point runtime_ud_start;

    // --- MPI Timer Variables ---
    #ifdef USE_MPI_TIMER
        double mpitime_t0;
        float mpitime_rhoyi_kernel = 0.f;
        float mpitime_states_kernel = 0.f;
    #endif // end USE_MPI_TIMER

	auto global_ndrange = sycl::range<3>(bl.Zmax, bl.Ymax, bl.Xmax);
	auto local_ndrange = sycl::range<3>(bl.dim_block_z, bl.dim_block_y, bl.dim_block_x); // size of workgroup

	// // update rho and yi
	Assign temury(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "Updaterhoyi");
	runtime_ud_start = std::chrono::high_resolution_clock::now();
    #ifdef USE_MPI_TIMER
        mpitime_t0 = MPI_Wtime(); // Start MPI timer before kernel submission
    #endif
	q.submit([&](sycl::handler &h) {																						//
		 h.parallel_for(sycl::nd_range<3>(temury.global_nd(global_ndrange), temury.local_nd), [=](sycl::nd_item<3> index)
         [[sycl::reqd_sub_group_size(32)]]
         { //
			 int k = index.get_global_id(0);
			 int j = index.get_global_id(1);
			 int i = index.get_global_id(2);
			 Updaterhoyi(i, j, k, ms, UI, rho, fdata.y);
		 });
	 })
		.wait();
	runtime_rhoyi = OutThisTime(runtime_ud_start);
    #ifdef USE_MPI_TIMER
        mpitime_rhoyi_kernel = OutThisTime(mpitime_t0); // End MPI timer
    #endif
	if (Setup::adv_push)
		Setup::adv_nd[Setup::adv_id].push_back(temury.Time(runtime_rhoyi));

#if ESTIM_NAN
	int *error_posyi;
	bool *error_org, *error_nan;
	error_org = middle::MallocShared<bool>(error_org, 1, q);
	error_nan = middle::MallocShared<bool>(error_nan, 1, q);
	error_posyi = middle::MallocShared<int>(error_posyi, 5 + NUM_SPECIES, q);
	*error_nan = false, *error_org = false;
	for (size_t i = 0; i < NUM_SPECIES + 4; i++)
		error_posyi[i] = _DF(0.0);

	// // update estimate negative or nan rho, yi
	Assign temey(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "EstimateYiKernel");
	runtime_ud_start = std::chrono::high_resolution_clock::now();
	q.submit([&](sycl::handler &h) {																					  //
		 h.parallel_for(sycl::nd_range<3>(temey.global_nd(global_ndrange), temey.local_nd), [=](sycl::nd_item<3> index)
         [[sycl::reqd_sub_group_size(32)]]
         { //
			 int k = index.get_global_id(0) + ms.Bwidth_Z;
			 int j = index.get_global_id(1) + ms.Bwidth_Y;
			 int i = index.get_global_id(2) + ms.Bwidth_X;
			 EstimateYiKernel(i, j, k, bl, error_posyi, error_org, error_nan, UI, rho, fdata.y);
		 });
	 })
		.wait();
	runtime_emyi = OutThisTime(runtime_ud_start);
	if (Setup::adv_push)
		Setup::adv_nd[Setup::adv_id].push_back(temey.Time(runtime_emyi));

	int offsetx = OutBoundary ? 0 : bl.Bwidth_X;
	int offsety = OutBoundary ? 0 : bl.Bwidth_Y;
	int offsetz = OutBoundary ? 0 : bl.Bwidth_Z;

	if (*error_org)
		error_patched_times += 1;
	if (*error_nan)
	{
		error_patched_times++;
		std::cout << "\nErrors of rho/Yi[";
		std::cout << error_posyi[NUM_SPECIES + 1] << ", ";
		for (size_t ii = 0; ii < NUM_SPECIES - 1; ii++)
			std::cout << error_posyi[ii] << ", ";
		std::cout << error_posyi[NUM_SPECIES] << "] located at (i, j, k)= ("
				  << error_posyi[NUM_SPECIES + 2] - offsetx << ", "
				  << error_posyi[NUM_SPECIES + 3] - offsety << ", "
				  << error_posyi[NUM_SPECIES + 4] - offsetz << ") of rank: " << rank;
#ifdef ERROR_PATCH_YI
		std::cout << " patched.\n";
#else
		std::cout << " captured.\n";
		// --- MPI Timer Specific Output for Early Exit ---
        #ifdef USE_MPI_TIMER
            timer_UD.push_back(0.f); // runtime_emyi (not MPI_Wtime tracked)
            timer_UD.push_back(0.f); // runtime_empv (not MPI_Wtime tracked)
            timer_UD.push_back(mpitime_rhoyi_kernel);
            timer_UD.push_back(0.f); // runtime_states (not reached)
        #else
            timer_UD.push_back(runtime_emyi);
            timer_UD.push_back(runtime_empv);
            timer_UD.push_back(runtime_rhoyi);
            timer_UD.push_back(runtime_states);
        #endif
		return std::make_pair(true, timer_UD);
#endif // end ERROR_PATCH_YI
	}
#endif // end ESTIM_NAN

	// sycl::stream stream_ct1(64 * 1024, 80, h);// for output error: sycl::stream decline running
	Assign temud(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "UpdateFuidStatesKernel");
	runtime_ud_start = std::chrono::high_resolution_clock::now();
    #ifdef USE_MPI_TIMER
        mpitime_t0 = MPI_Wtime(); // Start MPI timer before kernel submission
    #endif
#if __VENDOR_SUBMIT__
	CheckGPUErrors(vendorSetDevice(Ss.DeviceSelect[2]));
	static bool dummy = (GetKernelAttributes((const void *)UpdateFuidStatesKernelVendorWrapper, "UpdateFuidStatesKernelVendorWrapper"), true); // call only once
	UpdateFuidStatesKernelVendorWrapper<<<temud.global_gd(global_ndrange), temud.local_blk>>>(ms, thermal, UI, FluxF, FluxG, FluxH, rho, p, u, v, w, c, g, e, H, T, fdata.y, Ri, Cp);
	CheckGPUErrors(vendorDeviceSynchronize());
#else // !__VENDOR_SUBMIT__
    #if COP
	q.submit([&](sycl::handler &h) {																					  //
		 h.parallel_for(sycl::nd_range<3>(temud.global_nd(global_ndrange), temud.local_nd), [=](sycl::nd_item<3> index)
         [[sycl::reqd_sub_group_size(32)]]
         { //
			 int k = index.get_global_id(0);
			 int j = index.get_global_id(1);
			 int i = index.get_global_id(2);
			 UpdateFuidStatesKernel(i, j, k, ms, thermal, UI, FluxF, FluxG, FluxH, rho, p, u, v, w, c, g, e, H, T, fdata.y, Ri, Cp);
		 });
	 }); //, stream_ct1
	#else // !COP
    q.submit([&](sycl::handler &h) {																					  //
        h.parallel_for(sycl::nd_range<3>(temud.global_nd(global_ndrange), temud.local_nd), [=](sycl::nd_item<3> index)
        [[sycl::reqd_sub_group_size(32)]]
        { //
            int k = index.get_global_id(0);
            int j = index.get_global_id(1);
            int i = index.get_global_id(2);
            UpdateFuidStatesSPKernel(i, j, k, ms, UI, FluxF, FluxG, FluxH, rho, p, u, v, w, c, H, NCOP_Gamma);
        });
    });
	#endif // end COP
    q.wait();
#endif // end __VENDOR_SUBMIT__
	runtime_states = OutThisTime(runtime_ud_start);
    #ifdef USE_MPI_TIMER
        mpitime_states_kernel = OutThisTime(mpitime_t0); // End MPI timer
    #endif
	if (Setup::adv_push)
		Setup::adv_nd[Setup::adv_id].push_back(temud.Time(runtime_states));

#if ESTIM_NAN
	int *error_pos;
	bool *error_yi, *error_nga;
	error_pos = middle::MallocShared<int>(error_pos, 6 + NUM_SPECIES, q);
	error_yi = middle::MallocShared<bool>(error_yi, 1, q), error_nga = middle::MallocShared<bool>(error_nga, 1, q);
	*error_nga = false, *error_yi = false;
	for (size_t n = 0; n < 6 + NUM_SPECIES; n++)
		error_pos[n] = 0;

	Assign temep(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "EstimatePrimitiveVarKernel");
	runtime_ud_start = std::chrono::high_resolution_clock::now();
    #ifdef USE_MPI_TIMER
        // mpitime_t0 = MPI_Wtime(); // Not tracking this kernel with MPI_Wtime as per request
    #endif
	q.submit([&](sycl::handler &h) {																										  //
		 h.parallel_for(sycl::nd_range<3>(temep.global_nd(ms.Z_inner, ms.Y_inner, ms.X_inner), temep.local_nd), [=](sycl::nd_item<3> index)
         [[sycl::reqd_sub_group_size(32)]]
         { //
			 int k = index.get_global_id(0) + ms.Bwidth_Z;
			 int j = index.get_global_id(1) + ms.Bwidth_Y;
			 int i = index.get_global_id(2) + ms.Bwidth_X;
			 EstimatePrimitiveVarKernel(i, j, k, bl, thermal, error_pos, error_nga, error_yi,
										UI, rho, u, v, w, p, T, fdata.y, H, fdata.e, fdata.gamma, c);
		 });
	 })
		.wait();
	runtime_empv = OutThisTime(runtime_ud_start);
	if (Setup::adv_push)
		Setup::adv_nd[Setup::adv_id].push_back(temep.Time(runtime_empv));

	if (*error_nga)
	{
		std::cout << "\nErrors of Primitive variables[rho, T, P][";
		for (size_t ii = 0; ii < 2; ii++)
			std::cout << error_pos[ii] << ", ";
		std::cout << error_pos[2] << "] located at (i, j, k)= ("
				  << error_pos[3 + NUM_SPECIES] - offsetx << ", "
				  << error_pos[4 + NUM_SPECIES] - offsety << ", "
				  << error_pos[5 + NUM_SPECIES] - offsetz << ") of rank: " << rank;
#ifdef ERROR_PATCH
		std::cout << " patched.\n";
#else
		std::cout << " captured.\n";
		// --- MPI Timer Specific Output for Early Exit ---
        #ifdef USE_MPI_TIMER
            timer_UD.push_back(0.f); // runtime_emyi
            timer_UD.push_back(0.f); // runtime_empv
            timer_UD.push_back(mpitime_rhoyi_kernel);
            timer_UD.push_back(mpitime_states_kernel);
        #else
            timer_UD.push_back(runtime_emyi);
            timer_UD.push_back(runtime_empv);
            timer_UD.push_back(runtime_rhoyi);
            timer_UD.push_back(runtime_states);
        #endif
		return std::make_pair(true, timer_UD);
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

    // --- Final timer_UD push_back logic ---
    #ifdef USE_MPI_TIMER
        timer_UD.push_back(0.f); // runtime_emyi
        timer_UD.push_back(0.f); // runtime_empv
        timer_UD.push_back(mpitime_rhoyi_kernel);
        timer_UD.push_back(mpitime_states_kernel);
    #else // Original chrono based
        timer_UD.push_back(runtime_emyi);
        timer_UD.push_back(runtime_empv);
        timer_UD.push_back(runtime_rhoyi);
        timer_UD.push_back(runtime_states);
    #endif

	return std::make_pair(false, timer_UD);
#endif // end __REVERSE_NDRANGE__
}

std::vector<float> UpdateURK3rd(sycl::queue &q, Block bl, real_t *U, real_t *U1, real_t *LU, real_t const dt, int flag, int rank)
{
    std::vector<float> timer_U;

    #ifdef USE_MPI_TIMER
        double mpitime_t0;
        float mpitime_uu_kernel = 0.f;
    #endif

#ifndef __REVERSE_NDRANGE__
    MeshSize ms = bl.Ms;
	auto global_ndrange = sycl::range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner);

	Assign temud(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "UpdateURK3rdKernel");
	std::chrono::high_resolution_clock::time_point runtime_ud_start = std::chrono::high_resolution_clock::now();

    #ifdef USE_MPI_TIMER
        mpitime_t0 = MPI_Wtime();
    #endif // end USE_MPI_TIMER

	q.submit([&](sycl::handler &h) {																					  //
		 h.parallel_for(sycl::nd_range<3>(temud.global_nd(global_ndrange), temud.local_nd), [=](sycl::nd_item<3> index) { //
			 int i = index.get_global_id(0) + ms.Bwidth_X;
			 int j = index.get_global_id(1) + ms.Bwidth_Y;
			 int k = index.get_global_id(2) + ms.Bwidth_Z;
			 UpdateURK3rdKernel(i, j, k, ms, U, U1, LU, dt, flag);
		 });
	 })
		.wait();

    #ifdef USE_MPI_TIMER
        mpitime_uu_kernel = OutThisTime(mpitime_t0);
    #endif // end USE_MPI_TIMER

	if (Setup::adv_push)
		Setup::adv_nd[Setup::adv_id].push_back(temud.Time(OutThisTime(runtime_ud_start)));
#else // __REVERSE_NDRANGE__
    MeshSize ms = bl.Ms;
	auto global_ndrange = sycl::range<3>(bl.Z_inner, bl.Y_inner, bl.X_inner);    

    Assign temud(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "UpdateURK3rdKernel");
    std::chrono::high_resolution_clock::time_point runtime_ud_start = std::chrono::high_resolution_clock::now();

    #ifdef USE_MPI_TIMER
        mpitime_t0 = MPI_Wtime();
    #endif // end USE_MPI_TIMER

    q.submit([&](sycl::handler &h) {																					  //
    	 h.parallel_for(sycl::nd_range<3>(temud.global_nd(global_ndrange), temud.local_nd), [=](sycl::nd_item<3> index)
         [[sycl::reqd_sub_group_size(32)]]
         { //
    		 int k = index.get_global_id(0) + ms.Bwidth_Z;
    		 int j = index.get_global_id(1) + ms.Bwidth_Y;
    		 int i = index.get_global_id(2) + ms.Bwidth_X;
    		 UpdateURK3rdKernel(i, j, k, ms, U, U1, LU, dt, flag);
    	 });
     })
    	.wait();

    #ifdef USE_MPI_TIMER
        mpitime_uu_kernel = OutThisTime(mpitime_t0);
    #endif // end USE_MPI_TIMER

    if (Setup::adv_push)
    	Setup::adv_nd[Setup::adv_id].push_back(temud.Time(OutThisTime(runtime_ud_start)));

#endif // END __REVERSE_NDRANGE__

    #ifdef USE_MPI_TIMER
        timer_U.push_back(mpitime_uu_kernel);
    #endif

    return timer_U;
}

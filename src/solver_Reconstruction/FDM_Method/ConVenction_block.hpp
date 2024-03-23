#pragma once

#include <vector>
#include "Eigen_value.hpp"
#include "Reconstruction_kernels.hpp"
#include "../viscosity/Visc_block.hpp"
#include "PositivityPreserving_kernels.hpp"
#include "../../solver_UpdateStates/UpdateStates_block.h"

std::vector<float> GetLU(sycl::queue &q, Setup &setup, Block bl, BConditions BCs[6], Thermal thermal, real_t *UI, real_t *LU,
						 real_t *FluxF, real_t *FluxG, real_t *FluxH, real_t *FluxFw, real_t *FluxGw, real_t *FluxHw,
						 real_t const Gamma, int const Mtrl_ind, FlowData &fdata,
						 real_t *eigen_local_x, real_t *eigen_local_y, real_t *eigen_local_z, real_t *eigen_l, real_t *eigen_r,
						 real_t *uvw_c_max, real_t *eigen_block_x, real_t *eigen_block_y, real_t *eigen_block_z,
						 real_t *yi_min, real_t *yi_max, real_t *Dim_min, real_t *Dim_max)
{
	MeshSize ms = bl.Ms;
	real_t *rho = fdata.rho;
	real_t *p = fdata.p;
	real_t *H = fdata.H;
	real_t *c = fdata.c;
	real_t *u = fdata.u;
	real_t *v = fdata.v;
	real_t *w = fdata.w;
	real_t *T = fdata.T;

	std::vector<float> timer_LU;
	std::chrono::high_resolution_clock::time_point runtime_lu_start, runtime_lu_astart;
	float runtime_velDeri = 0.0f, runtime_transport = 0.0f, runtime_updatelu = 0.0f;
	float runtime_ppx = 0.0f, runtime_ppy = 0.0f, runtime_ppz = 0.0f, runtime_pp = 0.0f;
	float runtime_viscx = 0.0f, runtime_viscy = 0.0f, runtime_viscz = 0.0f, runtime_visc = 0.0f;
	float runtime_fluxx = 0.0f, runtime_fluxy = 0.0f, runtime_fluxz = 0.0f, runtime_flux = 0.0f;
	float runtime_eigenx = 0.0f, runtime_eigeny = 0.0f, runtime_eigenz = 0.0f, runtime_eigen = 0.0f;
	float runtime_geigenx = 0.0f, runtime_geigeny = 0.0f, runtime_geigenz = 0.0f, runtime_geigen = 0.0f;

	auto global_ndrange_max = range<3>(bl.Xmax, bl.Ymax, bl.Zmax);
	auto local_ndrange = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z);
	auto global_ndrange_x = range<3>(bl.X_inner + 1, bl.Y_inner, bl.Z_inner);
	auto global_ndrange_y = range<3>(bl.X_inner, bl.Y_inner + 1, bl.Z_inner);
	auto global_ndrange_z = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner + 1);

	Assign temk(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "GetLocalEigen");
	{ // get local eigen
		runtime_lu_astart = std::chrono::high_resolution_clock::now();
		if (bl.DimX)
		{
			// #ifdef DEBUG
			// 	// std::cout << "  sleep before ReconstructFluxX\n";
			// 	// sleep(5);
			// #endif // end DEBUG
			// proceed at x directiom and get F-flux terms at node wall

			q.submit([&](sycl::handler &h) {																					   //
				h.parallel_for(sycl::nd_range<3>(temk.global_nd(global_ndrange_max), temk.local_nd), [=](sycl::nd_item<3> index) { //
					int i = index.get_global_id(0);
					int j = index.get_global_id(1);
					int k = index.get_global_id(2);
					GetLocalEigen(i, j, k, ms, _DF(1.0), _DF(0.0), _DF(0.0), eigen_local_x, u, v, w, c);
				});
			});
		}
#if __SYNC_TIMER_
		q.wait();
		runtime_eigenx = OutThisTime(runtime_lu_astart);
		runtime_lu_start = std::chrono::high_resolution_clock::now();
#endif // end __SYNC_TIMER_
		if (bl.DimY)
		{
			// #ifdef DEBUG
			// 	// std::cout << "  sleep before ReconstructFluxY\n";
			// 	// sleep(5);
			// #endif // end DEBUG
			// proceed at y directiom and get G-flux terms at node wall

			q.submit([&](sycl::handler &h) {																					   //
				h.parallel_for(sycl::nd_range<3>(temk.global_nd(global_ndrange_max), temk.local_nd), [=](sycl::nd_item<3> index) { //
					int i = index.get_global_id(0);
					int j = index.get_global_id(1);
					int k = index.get_global_id(2);
					GetLocalEigen(i, j, k, ms, _DF(0.0), _DF(1.0), _DF(0.0), eigen_local_y, u, v, w, c);
				});
			});
		}
#if __SYNC_TIMER_
		q.wait();
		runtime_eigeny = OutThisTime(runtime_lu_start);
		runtime_lu_start = std::chrono::high_resolution_clock::now();
#endif // end __SYNC_TIMER_
		if (bl.DimZ)
		{
			// #ifdef DEBUG
			// 	// std::cout << "  sleep before ReconstructFluxZ\n";
			// 	// sleep(5);
			// #endif // end DEBUG
			// proceed at y directiom and get G-flux terms at node wall

			q.submit([&](sycl::handler &h) {																					   //
				h.parallel_for(sycl::nd_range<3>(temk.global_nd(global_ndrange_max), temk.local_nd), [=](sycl::nd_item<3> index) { //
					int i = index.get_global_id(0);
					int j = index.get_global_id(1);
					int k = index.get_global_id(2);
					GetLocalEigen(i, j, k, ms, _DF(0.0), _DF(0.0), _DF(1.0), eigen_local_z, u, v, w, c);
				});
			});
		}
	}
	q.wait();
#if __SYNC_TIMER_
	runtime_eigenz = OutThisTime(runtime_lu_start);
#endif
	runtime_eigen = OutThisTime(runtime_lu_astart);
	if (Setup::adv_push)
		Setup::adv_nd[Setup::adv_id].push_back(temk.Time(runtime_eigen));

	{ // get global LF eigen
		runtime_lu_astart = std::chrono::high_resolution_clock::now();
		if (bl.DimX)
			for (size_t nn = 0; nn < Emax; nn++)
			{
				q.submit([&](sycl::handler &h) {																											//
					auto reduction_max_eigen = sycl_reduction_max(eigen_block_x[nn]);																		//
					h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), reduction_max_eigen, [=](nd_item<3> index, auto &temp_max_eigen) { //
						int i = index.get_global_id(0);
						int j = index.get_global_id(1);
						int k = index.get_global_id(2);
						int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
						temp_max_eigen.combine(sycl::fabs(eigen_local_x[Emax * id + nn]));
					});
				});
			}
#if __SYNC_TIMER_
		q.wait();
		runtime_geigenx = OutThisTime(runtime_lu_astart);
		runtime_lu_start = std::chrono::high_resolution_clock::now();
#endif // end __SYNC_TIMER_
		if (bl.DimY)
			for (size_t nn = 0; nn < Emax; nn++)
			{
				q.submit([&](sycl::handler &h) {																											//
					auto reduction_max_eigen = sycl_reduction_max(eigen_block_y[nn]);																		//
					h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), reduction_max_eigen, [=](nd_item<3> index, auto &temp_max_eigen) { //
						int i = index.get_global_id(0);
						int j = index.get_global_id(1);
						int k = index.get_global_id(2);
						int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
						temp_max_eigen.combine(sycl::fabs(eigen_local_y[Emax * id + nn]));
					});
				});
			}
#if __SYNC_TIMER_
		q.wait();
		runtime_geigeny = OutThisTime(runtime_lu_start);
		runtime_lu_start = std::chrono::high_resolution_clock::now();
#endif // end __SYNC_TIMER_
		if (bl.DimZ)
			for (size_t nn = 0; nn < Emax; nn++)
			{
				q.submit([&](sycl::handler &h) {																											//
					auto reduction_max_eigen = sycl_reduction_max(eigen_block_z[nn]);																		//
					h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), reduction_max_eigen, [=](nd_item<3> index, auto &temp_max_eigen) { //
						int i = index.get_global_id(0);
						int j = index.get_global_id(1);
						int k = index.get_global_id(2);
						int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
						temp_max_eigen.combine(sycl::fabs(eigen_local_z[Emax * id + nn]));
					});
				});
			}
		q.wait();
#if __SYNC_TIMER_
		runtime_geigenz = OutThisTime(runtime_lu_start);
#endif // end __SYNC_TIMER_
#ifdef USE_MPI
		runtime_lu_start = std::chrono::high_resolution_clock::now();
		if (bl.DimX)
			for (size_t nn = 0; nn < Emax; nn++)
			{
				real_t mpi_eigen_block_x = _DF(0.0);
				setup.mpiTrans->communicator->synchronize();
				setup.mpiTrans->communicator->allReduce(&(eigen_block_x[nn]), &(mpi_eigen_block_x), 1, setup.mpiTrans->data_type, mpiUtils::MpiComm::MAX);
				setup.mpiTrans->communicator->synchronize();
				eigen_block_x[nn] = mpi_eigen_block_x;
			}
#if __SYNC_TIMER_
		runtime_geigenx += OutThisTime(runtime_lu_start);
		runtime_lu_start = std::chrono::high_resolution_clock::now();
#endif // end __SYNC_TIMER_
		if (bl.DimY)
			for (size_t nn = 0; nn < Emax; nn++)
			{
				real_t mpi_eigen_block_y = _DF(0.0);
				setup.mpiTrans->communicator->synchronize();
				setup.mpiTrans->communicator->allReduce(&(eigen_block_y[nn]), &(mpi_eigen_block_y), 1, setup.mpiTrans->data_type, mpiUtils::MpiComm::MAX);
				setup.mpiTrans->communicator->synchronize();
				eigen_block_y[nn] = mpi_eigen_block_y;
			}
#if __SYNC_TIMER_
		runtime_geigeny += OutThisTime(runtime_lu_start);
		runtime_lu_start = std::chrono::high_resolution_clock::now();
#endif // end __SYNC_TIMER_
		if (bl.DimZ)
			for (size_t nn = 0; nn < Emax; nn++)
			{
				real_t mpi_eigen_block_z = _DF(0.0);
				setup.mpiTrans->communicator->synchronize();
				setup.mpiTrans->communicator->allReduce(&(eigen_block_z[nn]), &(mpi_eigen_block_z), 1, setup.mpiTrans->data_type, mpiUtils::MpiComm::MAX);
				setup.mpiTrans->communicator->synchronize();
				eigen_block_z[nn] = mpi_eigen_block_z;
			}
#if __SYNC_TIMER_
		runtime_geigenz += OutThisTime(runtime_lu_start);
#endif // end __SYNC_TIMER_
#endif // end MPI
		runtime_geigen = OutThisTime(runtime_lu_astart);
	}

	{ // // Reconstruction Physical Fluxes
		Assign temfwx(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "GetWallFluxX");
		runtime_lu_astart = std::chrono::high_resolution_clock::now();
		if (bl.DimX)
		{
#if __VENDOR_SUBMIT__
			CheckGPUErrors(vendorSetDevice(setup.DeviceSelect[2]));
			static bool dummx = (GetKernelAttributes((const void *)ReconstructFluxXVendorWrapper, "ReconstructFluxXVendorWrapper"), true); // call only once
			ReconstructFluxXVendorWrapper<<<temfwx.global_gd(global_ndrange_x), temfwx.local_blk>>>(bl.dx, ms, thermal, UI, FluxF, FluxFw, eigen_local_x, eigen_l, eigen_r, fdata.b1x,
																									fdata.b3x, fdata.c2x, fdata.zix, p, rho, u, v, w, fdata.y, T, H, eigen_block_x);
#else
			q.submit([&](sycl::handler &h) {																						 //
				h.parallel_for(sycl::nd_range<3>(temfwx.global_nd(global_ndrange_x), temfwx.local_nd), [=](sycl::nd_item<3> index) { //
					int i = index.get_global_id(0) + ms.Bwidth_X - 1;
					int j = index.get_global_id(1) + ms.Bwidth_Y;
					int k = index.get_global_id(2) + ms.Bwidth_Z;
					ReconstructFluxX(i, j, k, bl.dx, ms, thermal, UI, FluxF, FluxFw, eigen_local_x, eigen_l, eigen_r,
									 fdata.b1x, fdata.b3x, fdata.c2x, fdata.zix, p, rho, u, v, w, fdata.y, T, H, eigen_block_x);
				});
			});
#endif
		}

		if (Setup::adv_push || __SYNC_TIMER_)
		{
#if __VENDOR_SUBMIT__
			CheckGPUErrors(vendorDeviceSynchronize());
#else
			q.wait();
#endif
			runtime_fluxx = OutThisTime(runtime_lu_astart);
			if (Setup::adv_push)
				Setup::adv_nd[Setup::adv_id].push_back(temfwx.Time(runtime_fluxx));
			runtime_lu_start = std::chrono::high_resolution_clock::now();
		}

		Assign temfwy(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "GetWallFluxY");
		if (bl.DimY)
		{
#if __VENDOR_SUBMIT__
			CheckGPUErrors(vendorSetDevice(setup.DeviceSelect[2]));
			static bool dummy = (GetKernelAttributes((const void *)ReconstructFluxYVendorWrapper, "ReconstructFluxYVendorWrapper"), true); // call only once
			ReconstructFluxYVendorWrapper<<<temfwy.global_gd(global_ndrange_y), temfwy.local_blk>>>(bl.dy, ms, thermal, UI, FluxG, FluxGw, eigen_local_y, eigen_l, eigen_r, fdata.b1y,
																									fdata.b3y, fdata.c2y, fdata.ziy, p, rho, u, v, w, fdata.y, T, H, eigen_block_y);
#else
			q.submit([&](sycl::handler &h) {																						 //
				h.parallel_for(sycl::nd_range<3>(temfwy.global_nd(global_ndrange_y), temfwy.local_nd), [=](sycl::nd_item<3> index) { //
					int i = index.get_global_id(0) + ms.Bwidth_X;
					int j = index.get_global_id(1) + ms.Bwidth_Y - 1;
					int k = index.get_global_id(2) + ms.Bwidth_Z;
					ReconstructFluxY(i, j, k, bl.dy, ms, thermal, UI, FluxG, FluxGw, eigen_local_y, eigen_l, eigen_r,
									 fdata.b1y, fdata.b3y, fdata.c2y, fdata.ziy, p, rho, u, v, w, fdata.y, T, H, eigen_block_y);
				});
			});
#endif
		}

		if (Setup::adv_push || __SYNC_TIMER_)
		{
#if __VENDOR_SUBMIT__
			CheckGPUErrors(vendorDeviceSynchronize());
#else
			q.wait();
#endif
			runtime_fluxy = OutThisTime(runtime_lu_start);
			if (Setup::adv_push)
				Setup::adv_nd[Setup::adv_id].push_back(temfwy.Time(runtime_fluxy));
			runtime_lu_start = std::chrono::high_resolution_clock::now();
		}

		Assign temfwz(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "GetWallFluxZ");
		if (bl.DimZ)
		{
#if __VENDOR_SUBMIT__
			CheckGPUErrors(vendorSetDevice(setup.DeviceSelect[2]));
			static bool dummz = (GetKernelAttributes((const void *)ReconstructFluxZVendorWrapper, "ReconstructFluxZVendorWrapper"), true); // call only once
			ReconstructFluxZVendorWrapper<<<temfwz.global_gd(global_ndrange_z), temfwz.local_blk>>>(bl.dz, ms, thermal, UI, FluxH, FluxHw, eigen_local_z, eigen_l, eigen_r, fdata.b1z,
																									fdata.b3z, fdata.c2z, fdata.ziz, p, rho, u, v, w, fdata.y, T, H, eigen_block_z);
#else
			q.submit([&](sycl::handler &h) {																						 //
				h.parallel_for(sycl::nd_range<3>(temfwz.global_nd(global_ndrange_z), temfwz.local_nd), [=](sycl::nd_item<3> index) { //
					int i = index.get_global_id(0) + ms.Bwidth_X;
					int j = index.get_global_id(1) + ms.Bwidth_Y;
					int k = index.get_global_id(2) + ms.Bwidth_Z - 1;
					ReconstructFluxZ(i, j, k, bl.dz, ms, thermal, UI, FluxH, FluxHw, eigen_local_z, eigen_l, eigen_r,
									 fdata.b1z, fdata.b3z, fdata.c2z, fdata.ziz, p, rho, u, v, w, fdata.y, T, H, eigen_block_z);
				});
			});
#endif
		}

#if __VENDOR_SUBMIT__
		CheckGPUErrors(vendorDeviceSynchronize());
#else
		q.wait();
#endif
		if (Setup::adv_push || __SYNC_TIMER_)
		{
			runtime_fluxz = OutThisTime(runtime_lu_start);
			if (Setup::adv_push)
				Setup::adv_nd[Setup::adv_id].push_back(temfwz.Time(runtime_fluxz));
		}
	}
	runtime_flux = OutThisTime(runtime_lu_astart);

	// 	// 	int cellsize = bl.Xmax * bl.Ymax * bl.Zmax * sizeof(real_t) * NUM_SPECIES;
	// 	// 	q.memcpy(fdata.preFwx, FluxFw, cellsize);
	// 	// 	q.memcpy(fdata.preFwy, FluxGw, cellsize);
	// 	// 	q.memcpy(fdata.preFwz, FluxHw, cellsize);
	// 	// 	q.wait();

	// NOTE: positive preserving
	auto global_ndrange_inner = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner);
	if (PositivityPreserving)
	{
		real_t lambda_x0 = uvw_c_max[0], lambda_y0 = uvw_c_max[1], lambda_z0 = uvw_c_max[2];
		real_t lambda_x = bl.CFLnumber / lambda_x0, lambda_y = bl.CFLnumber / lambda_y0, lambda_z = bl.CFLnumber / lambda_z0;
		real_t *epsilon = static_cast<real_t *>(sycl::malloc_shared((NUM_SPECIES + 2) * sizeof(real_t), q));
		epsilon[0] = _DF(1.0e-13), epsilon[1] = _DF(1.0e-13); // 0 for rho and 1 for T and P
		for (size_t ii = 2; ii < NUM_SPECIES + 2; ii++)		  // for Yi
			epsilon[ii] = _DF(0.0);							  // Ini epsilon for y1-yN(N species)

		Assign temppx(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "PositivityPreservingKernelX");
		runtime_lu_astart = std::chrono::high_resolution_clock::now();
		if (bl.DimX)
		{																											 // sycl::stream error_out(1024 * 1024, 1024, h);
			q.submit([&](sycl::handler &h) {																		 //
				h.parallel_for(sycl::nd_range<3>(global_ndrange_inner, local_ndrange), [=](sycl::nd_item<3> index) { //
					int i = index.get_global_id(0) + ms.Bwidth_X;
					int j = index.get_global_id(1) + ms.Bwidth_Y;
					int k = index.get_global_id(2) + ms.Bwidth_Z;
					int id_l = (ms.Xmax * ms.Ymax * k + ms.Xmax * j + i);
					int id_r = (ms.Xmax * ms.Ymax * k + ms.Xmax * j + i + 1);
					PositivityPreservingKernel(i, j, k, id_l, id_r, ms, thermal, UI, FluxF, FluxFw, T, lambda_x0, lambda_x, epsilon);
				});
			});
		}

		if (Setup::adv_push || __SYNC_TIMER_)
		{
			q.wait();
			runtime_ppx = OutThisTime(runtime_lu_astart);
			if (Setup::adv_push)
				Setup::adv_nd[Setup::adv_id].push_back(temppx.Time(runtime_ppx));
			runtime_lu_start = std::chrono::high_resolution_clock::now();
		}

		Assign temppy(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "PositivityPreservingKernelY");
		if (bl.DimY)
		{																											 // sycl::stream error_out(1024 * 1024, 1024, h);
			q.submit([&](sycl::handler &h) {																		 //
				h.parallel_for(sycl::nd_range<3>(global_ndrange_inner, local_ndrange), [=](sycl::nd_item<3> index) { //
					int i = index.get_global_id(0) + ms.Bwidth_X;
					int j = index.get_global_id(1) + ms.Bwidth_Y;
					int k = index.get_global_id(2) + ms.Bwidth_Z;
					int id_l = (ms.Xmax * ms.Ymax * k + ms.Xmax * j + i);
					int id_r = (ms.Xmax * ms.Ymax * k + ms.Xmax * (j + 1) + i);
					PositivityPreservingKernel(i, j, k, id_l, id_r, ms, thermal, UI, FluxG, FluxGw, T, lambda_y0, lambda_y, epsilon);
				});
			});
		}

		if (Setup::adv_push || __SYNC_TIMER_)
		{
			q.wait();
			runtime_ppy = OutThisTime(runtime_lu_start);
			if (Setup::adv_push)
				Setup::adv_nd[Setup::adv_id].push_back(temppy.Time(runtime_ppy));
			runtime_lu_start = std::chrono::high_resolution_clock::now();
		}

		Assign temppz(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "PositivityPreservingKernelZ");
		if (bl.DimZ)
		{
			q.submit([&](sycl::handler &h) {																		 //
				h.parallel_for(sycl::nd_range<3>(global_ndrange_inner, local_ndrange), [=](sycl::nd_item<3> index) { //
					int i = index.get_global_id(0) + ms.Bwidth_X;
					int j = index.get_global_id(1) + ms.Bwidth_Y;
					int k = index.get_global_id(2) + ms.Bwidth_Z;
					int id_l = (ms.Xmax * ms.Ymax * k + ms.Xmax * j + i);
					int id_r = (ms.Xmax * ms.Ymax * (k + 1) + ms.Xmax * j + i);
					PositivityPreservingKernel(i, j, k, id_l, id_r, ms, thermal, UI, FluxH, FluxHw, T, lambda_z0, lambda_z, epsilon);
				});
			});
		}
		q.wait();
		if (Setup::adv_push || __SYNC_TIMER_)
		{
			runtime_ppz = OutThisTime(runtime_lu_start);
			if (Setup::adv_push)
				Setup::adv_nd[Setup::adv_id].push_back(temppz.Time(runtime_ppz));
		}
		runtime_pp = OutThisTime(runtime_lu_astart);
	}

	// 	// 	q.wait();
	// 	// 	int cellsize = bl.Xmax * bl.Ymax * bl.Zmax * sizeof(real_t) * NUM_SPECIES;
	// 	// 	q.memcpy(fdata.preFwx, FluxFw, cellsize);
	// 	// 	q.memcpy(fdata.preFwy, FluxGw, cellsize);
	// 	// 	q.memcpy(fdata.preFwz, FluxHw, cellsize);
	// 	// 	q.wait();

	/**
	 * NOTE: calculate and add viscous wall Flux to physical convection Flux
	 * Viscous LU including physical visc(切应力),Visc_Heat transfer(传热), mass Diffusion(质量扩散)
	 * Physical Visc must be included, Visc_Heat is alternative, Visc_Diffu depends on compent
	 */
	if (Visc)
	{
		runtime_lu_astart = std::chrono::high_resolution_clock::now();
		GetCellCenterDerivative(q, bl, fdata, BCs); // get Vortex
		runtime_velDeri = OutThisTime(runtime_lu_astart);

		real_t *va = fdata.viscosity_aver;
		real_t *tca = fdata.thermal_conduct_aver;
		real_t *Da = fdata.Dkm_aver;
		real_t *hi = fdata.hi;

		Assign temcoeff(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "Gettransport_coeff_aver");
		runtime_lu_astart = std::chrono::high_resolution_clock::now();
#if __VENDOR_SUBMIT__
		CheckGPUErrors(vendorSetDevice(setup.DeviceSelect[2]));
		static bool dummv = (GetKernelAttributes((const void *)Gettransport_coeff_averVendorWrapper, "Gettransport_coeff_averVendorWrapper"), true); // call only once
		Gettransport_coeff_averVendorWrapper<<<temcoeff.global_gd(global_ndrange_max), temcoeff.local_blk>>>(bl, thermal, va, tca, Da, fdata.y, hi, rho, p, T, fdata.Ertemp1, fdata.Ertemp2);
		CheckGPUErrors(vendorDeviceSynchronize());
#else
		q.submit([&](sycl::handler &h) {																								//
			 h.parallel_for(sycl::nd_range<3>(temcoeff.global_nd(global_ndrange_max), temcoeff.local_nd), [=](sycl::nd_item<3> index) { //
				 int i = index.get_global_id(0);
				 int j = index.get_global_id(1);
				 int k = index.get_global_id(2);
				 Gettransport_coeff_aver(i, j, k, bl, thermal, va, tca, Da, fdata.y, hi, rho, p, T, fdata.Ertemp1, fdata.Ertemp2);
			 });
		 })
			.wait();
#endif
		runtime_transport = OutThisTime(runtime_lu_astart);
		if (Setup::adv_push)
			Setup::adv_nd[Setup::adv_id].push_back(temcoeff.Time(runtime_transport));

		// // get visc robust limiter
		if (Visc_Diffu)
		{
			for (size_t nn = 0; nn < NUM_SPECIES; nn++)
			{
				yi_min[nn] = _DF(0.0), yi_max[nn] = _DF(0.0);
				Dim_min[nn] = _DF(0.0), Dim_max[nn] = _DF(0.0);
				auto Yi_min = sycl_reduction_min(yi_min[nn]);
				auto Yi_max = sycl_reduction_max(yi_max[nn]);
				auto Dkm_min = sycl_reduction_min(Dim_min[nn]);
				auto Dkm_max = sycl_reduction_max(Dim_max[nn]);

				Assign temDim(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "GetYiDimMaxMin[" + std::to_string(nn) + "]");
				runtime_lu_astart = std::chrono::high_resolution_clock::now();
				q.submit([&](sycl::handler &h) { //
					 h.parallel_for(sycl::nd_range<3>(global_ndrange_inner, local_ndrange), Yi_min, Yi_max, Dkm_min, Dkm_max,
									[=](nd_item<3> index, auto &temp_Ymin, auto &temp_Ymax, auto &temp_Dmin, auto &temp_Dmax) { //
										int i = index.get_global_id(0) + ms.Bwidth_X;
										int j = index.get_global_id(1) + ms.Bwidth_Y;
										int k = index.get_global_id(2) + ms.Bwidth_Z;
										int id = ms.Xmax * ms.Ymax * k + ms.Xmax * j + i;
										real_t *yi = &(fdata.y[NUM_SPECIES * id]);
										temp_Ymin.combine(yi[nn]), temp_Ymax.combine(yi[nn]);
										// real_t *Dkm = &(Da[NUM_SPECIES * id]);
										// temp_Dmin.combine(Dkm[nn]), temp_Dmax.combine(Dkm[nn]);
										//
									});
				 })
					.wait();
				if (Setup::adv_push)
					Setup::adv_nd[Setup::adv_id].push_back(temDim.Time(OutThisTime(runtime_lu_astart)));

#ifdef USE_MPI
				real_t mpi_Ymin = _DF(0.0), mpi_Ymax = _DF(0.0), mpi_Dmin = _DF(0.0), mpi_Dmax = _DF(0.0);
				setup.mpiTrans->communicator->synchronize();
				setup.mpiTrans->communicator->allReduce(&(yi_min[nn]), &(mpi_Ymin), 1, setup.mpiTrans->data_type, mpiUtils::MpiComm::MIN);
				setup.mpiTrans->communicator->allReduce(&(yi_max[nn]), &(mpi_Ymax), 1, setup.mpiTrans->data_type, mpiUtils::MpiComm::MAX);
				// setup.mpiTrans->communicator->allReduce(&(Dim_min[nn]), &(mpi_Dmin), 1, setup.mpiTrans->data_type, mpiUtils::MpiComm::MIN);
				// setup.mpiTrans->communicator->allReduce(&(Dim_max[nn]), &(mpi_Dmax), 1, setup.mpiTrans->data_type, mpiUtils::MpiComm::MAX);
				setup.mpiTrans->communicator->synchronize();
				yi_min[nn] = sycl::max(mpi_Ymin, _DF(0.0)), yi_max[nn] = sycl::min(mpi_Ymax, _DF(1.0));
				Dim_min[nn] = _DF(1.0), Dim_max[nn] = _DF(1.0); // Dim_min[nn] = mpi_Dmin, Dim_max[nn] = mpi_Dmax;
#endif															// end USE_MPI
				yi_max[nn] -= yi_min[nn];
				yi_max[nn] *= setup.BlSz.Yil_limiter;				// // yil limiter
				Dim_max[nn] *= setup.BlSz.Dim_limiter * yi_max[nn]; // // Diffu_limiter=Yil_limiter*Dim_limiter
			}
		}

		// // calculate viscous Fluxes
		Assign temvFwx(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "GetWallViscousFluxX");
		runtime_lu_astart = std::chrono::high_resolution_clock::now();
		if (bl.DimX)
		{
			q.submit([&](sycl::handler &h) {																									   //
				h.parallel_for(sycl::nd_range<3>(temvFwx.global_nd(global_ndrange_x, local_ndrange), local_ndrange), [=](sycl::nd_item<3> index) { //
					int i = index.get_global_id(0) + bl.Bwidth_X - 1;
					int j = index.get_global_id(1) + bl.Bwidth_Y;
					int k = index.get_global_id(2) + bl.Bwidth_Z;
					GetWallViscousFluxX(i, j, k, bl, FluxFw, va, tca, Da, T, rho, hi, fdata.y, u, v, w, fdata.Vde,
										yi_max, Dim_max, fdata.visFwx, fdata.Dim_wallx, fdata.hi_wallx, fdata.Yi_wallx, fdata.Yil_wallx);
				});
			}); //.wait()
		}

		if (Setup::adv_push || __SYNC_TIMER_)
		{
			q.wait();
			runtime_viscx = OutThisTime(runtime_lu_astart);
			if (Setup::adv_push)
				Setup::adv_nd[Setup::adv_id].push_back(temvFwx.Time(runtime_viscx));
			runtime_lu_start = std::chrono::high_resolution_clock::now();
		}

		Assign temvFwy(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "GetWallViscousFluxY");
		if (bl.DimY)
		{
			q.submit([&](sycl::handler &h) {																									   //
				h.parallel_for(sycl::nd_range<3>(temvFwy.global_nd(global_ndrange_y, local_ndrange), local_ndrange), [=](sycl::nd_item<3> index) { //
					int i = index.get_global_id(0) + bl.Bwidth_X;
					int j = index.get_global_id(1) + bl.Bwidth_Y - 1;
					int k = index.get_global_id(2) + bl.Bwidth_Z;
					GetWallViscousFluxY(i, j, k, bl, FluxGw, va, tca, Da, T, rho, hi, fdata.y, u, v, w, fdata.Vde,
										yi_max, Dim_max, fdata.visFwy, fdata.Dim_wally, fdata.hi_wally, fdata.Yi_wally, fdata.Yil_wally);
				});
			}); //.wait()
		}

		if (Setup::adv_push || __SYNC_TIMER_)
		{
			q.wait();
			runtime_viscy = OutThisTime(runtime_lu_start);
			if (Setup::adv_push)
				Setup::adv_nd[Setup::adv_id].push_back(temvFwy.Time(runtime_viscy));
			runtime_lu_start = std::chrono::high_resolution_clock::now();
		}

		Assign temvFwz(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "GetWallViscousFluxZ");
		if (bl.DimZ)
		{
			q.submit([&](sycl::handler &h) {																									   //
				h.parallel_for(sycl::nd_range<3>(temvFwz.global_nd(global_ndrange_z, local_ndrange), local_ndrange), [=](sycl::nd_item<3> index) { //
					int i = index.get_global_id(0) + bl.Bwidth_X;
					int j = index.get_global_id(1) + bl.Bwidth_Y;
					int k = index.get_global_id(2) + bl.Bwidth_Z - 1;
					GetWallViscousFluxZ(i, j, k, bl, FluxHw, va, tca, Da, T, rho, hi, fdata.y, u, v, w, fdata.Vde,
										yi_max, Dim_max, fdata.visFwz, fdata.Dim_wallz, fdata.hi_wallz, fdata.Yi_wallz, fdata.Yil_wallz);
				});
			}); //.wait()
		}
		q.wait();
		if (Setup::adv_push || __SYNC_TIMER_)
		{
			runtime_viscz = OutThisTime(runtime_lu_start);
			if (Setup::adv_push)
				Setup::adv_nd[Setup::adv_id].push_back(temvFwz.Time(runtime_viscz));
		}
		runtime_visc = OutThisTime(runtime_lu_astart);
	}

	Assign temulu(Setup::adv_nd[Setup::adv_id][Setup::sbm_id++].local_nd, "UpdateFluidLU");
	runtime_lu_start = std::chrono::high_resolution_clock::now();
	q.submit([&](sycl::handler &h) {																							  //
		 h.parallel_for(sycl::nd_range<3>(temulu.global_nd(global_ndrange_inner), temulu.local_nd), [=](sycl::nd_item<3> index) { //
			 int i = index.get_global_id(0) + bl.Bwidth_X;
			 int j = index.get_global_id(1) + bl.Bwidth_Y;
			 int k = index.get_global_id(2) + bl.Bwidth_Z;
			 UpdateFluidLU(i, j, k, bl, LU, FluxFw, FluxGw, FluxHw);
		 });
	 })
		.wait();
	runtime_updatelu = OutThisTime(runtime_lu_start);
	if (Setup::adv_push)
		Setup::adv_nd[Setup::adv_id].push_back(temulu.Time(runtime_updatelu));

	timer_LU.push_back(runtime_eigenx);
	timer_LU.push_back(runtime_eigeny);
	timer_LU.push_back(runtime_eigenz);
	timer_LU.push_back(runtime_eigen);
	timer_LU.push_back(runtime_geigenx);
	timer_LU.push_back(runtime_geigeny);
	timer_LU.push_back(runtime_geigenz);
	timer_LU.push_back(runtime_geigen);
	timer_LU.push_back(runtime_fluxx);
	timer_LU.push_back(runtime_fluxy);
	timer_LU.push_back(runtime_fluxz);
	timer_LU.push_back(runtime_flux);
	timer_LU.push_back(runtime_ppx);
	timer_LU.push_back(runtime_ppy);
	timer_LU.push_back(runtime_ppz);
	timer_LU.push_back(runtime_pp);
	timer_LU.push_back(runtime_velDeri);
	timer_LU.push_back(runtime_transport);
	timer_LU.push_back(runtime_viscx);
	timer_LU.push_back(runtime_viscy);
	timer_LU.push_back(runtime_viscz);
	timer_LU.push_back(runtime_visc);
	timer_LU.push_back(runtime_updatelu);

	return timer_LU;
}

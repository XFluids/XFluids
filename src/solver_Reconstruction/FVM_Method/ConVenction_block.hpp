#pragma once

#include "global_setup.h"
#include "marcos/marco_global.h"
#include "../read_ini/setupini.h"
#include "../include/sycl_devices.hpp"

#include "Reconstruction_kernels.hpp"
#include "PositivityPreserving_kernels.hpp"
#include "../viscosity/Visc_block.hpp"
#include "../../solver_UpdateStates/UpdateStates_block.hpp"

void GetLU(sycl::queue &q, Setup &setup, Block bl, BConditions BCs[6], Thermal thermal, real_t *UI, real_t *LU,
		   real_t *FluxFw, real_t *FluxGw, real_t *FluxHw, real_t const Gamma, FlowData &fdata, real_t *face_vector)
{
	real_t *rho = fdata.rho;
	real_t *p = fdata.p;
	real_t *H = fdata.H;
	real_t *c = fdata.c;
	real_t *u = fdata.u;
	real_t *v = fdata.v;
	real_t *w = fdata.w;
	real_t *T = fdata.T;

	auto local_ndrange = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z);
	auto global_ndrange_max = range<3>(bl.Xmax, bl.Ymax, bl.Zmax);

	if (bl.DimX)
	{
		q.submit([&](sycl::handler &h)
				 { h.parallel_for(sycl::nd_range<3>(global_ndrange_x, local_ndrange), [=](sycl::nd_item<3> index)
								  {
	    		int i = index.get_global_id(0) + bl.Bwidth_X - 1;
				int j = index.get_global_id(1) + bl.Bwidth_Y;
				int k = index.get_global_id(2) + bl.Bwidth_Z;
				int id = Xmax * Ymax * k + Xmax * j + i;
				int left_idx[5] = {id - 3, id - 2, id - 1, id, id + 1};
				int right_idx[5] = {id + 2, id + 1, id, id - 1, id - 2};
				ReconstructFlux(left_idx, right_idx, id, 2, bl, thermal, FluxFw, p, rho, u, v, w, fdata.y, T, face_vector); }); });
	}
	if (bl.DimY)
	{
		q.submit([&](sycl::handler &h)
				 { h.parallel_for(
					   sycl::nd_range<3>(global_ndrange_y, local_ndrange), [=](sycl::nd_item<3> index)
					   {
				int i = index.get_global_id(0) + bl.Bwidth_X;
				int j = index.get_global_id(1) + bl.Bwidth_Y - 1;
				int k = index.get_global_id(2) + bl.Bwidth_Z;
				int id = Xmax * Ymax * k + Xmax * j + i;
				int left_idy[5] = {id - 3 * Xmax, id - 2 * Xmax, id - 1 * Xmax, id, id + 1 * Xmax};
				int right_idy[5] = {id + 2 * Xmax, id + 1 * Xmax, id, id - 1 * Xmax, id - 2 * Xmax};
				ReconstructFlux(left_idy, right_idy, id, 3, bl, thermal, FluxGw, p, rho, u, v, w, fdata.y, T, face_vector); }); });
	}
	if (bl.DimZ)
	{
		q.submit([&](sycl::handler &h)
				 { h.parallel_for(
					   sycl::nd_range<3>(global_ndrange_z, local_ndrange), [=](sycl::nd_item<3> index)
					   {
				int i = index.get_global_id(0) + bl.Bwidth_X;
				int j = index.get_global_id(1) + bl.Bwidth_Y;
				int k = index.get_global_id(2) + bl.Bwidth_Z - 1;
				int id = Xmax * Ymax * k + Xmax * j + i;
				int left_idz[5] = {id - 3 * Xmax * Ymax, id - 2 * Xmax * Ymax, id - 1 * Xmax * Ymax, id, id + 1 * Xmax * Ymax};
				int right_idz[5] = {id + 2 * Xmax * Ymax, id + 1 * Xmax * Ymax, id, id - 1 * Xmax * Ymax, id - 2 * Xmax * Ymax};
				ReconstructFlux(left_idz, right_idz, id, 4, bl, thermal, FluxHw, p, rho, u, v, w, fdata.y, T, face_vector); }); });
	}

	q.wait();

	// 	// 	// 	int cellsize = bl.Xmax * bl.Ymax * bl.Zmax * sizeof(real_t) * NUM_SPECIES;
	// 	// 	// 	q.memcpy(fdata.preFwx, FluxFw, cellsize);
	// 	// 	// 	q.memcpy(fdata.preFwy, FluxGw, cellsize);
	// 	// 	// 	q.memcpy(fdata.preFwz, FluxHw, cellsize);
	// 	// 	// 	q.wait();

	// 	// NOTE: positive preserving
	auto global_ndrange_inner = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner);
	// 	real_t lambda_x0 = uvw_c_max[0], lambda_y0 = uvw_c_max[1], lambda_z0 = uvw_c_max[2];
	// 	real_t lambda_x = bl.CFLnumber / lambda_x0, lambda_y = bl.CFLnumber / lambda_y0, lambda_z = bl.CFLnumber / lambda_z0;
	// 	real_t *epsilon = static_cast<real_t *>(sycl::malloc_shared((NUM_SPECIES + 2) * sizeof(real_t), q));
	// 	epsilon[0] = _DF(1.0e-13), epsilon[1] = _DF(1.0e-13); // 0 for rho and 1 for T and P
	// 	// real_t epsilon[NUM_SPECIES + 2] = {_DF(1.0e-13), _DF(1.0e-13)};
	// 	for (size_t ii = 2; ii < NUM_SPECIES + 2; ii++) // for Yi
	// 		epsilon[ii] = _DF(0.0);						// Ini epsilon for y1-yN(N species)

	// sycl::stream error_out(1024 * 1024, 1024, h);
	// 	q.submit([&](sycl::handler &h)
	// 			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_inner, local_ndrange), [=](sycl::nd_item<3> index)
	// 							  {
	// 					int i = index.get_global_id(0) + bl.Bwidth_X;
	// 					int j = index.get_global_id(1) + bl.Bwidth_Y;
	// 					int k = index.get_global_id(2) + bl.Bwidth_Z;
	// 					int id_l = (bl.Xmax * bl.Ymax * k + bl.Xmax * j + i);
	// 					int id_r = (bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 1);
	// 					PositivityPreservingKernel(i, j, k, id_l, id_r, bl, thermal, UI, FluxF, FluxFw, T, lambda_x0, lambda_x, epsilon); }); });
	// sycl::stream error_out(1024 * 1024, 1024, h);
	// 	q.submit([&](sycl::handler &h)
	// 			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_inner, local_ndrange), [=](sycl::nd_item<3> index)
	// 							  {
	// 	   				int i = index.get_global_id(0) + bl.Bwidth_X;
	// 	   				int j = index.get_global_id(1) + bl.Bwidth_Y;
	// 	   				int k = index.get_global_id(2) + bl.Bwidth_Z;
	// 	   				int id_l = (bl.Xmax * bl.Ymax * k + bl.Xmax * j + i);
	// 	   				int id_r = (bl.Xmax * bl.Ymax * k + bl.Xmax * (j + 1) + i);
	// 	   				PositivityPreservingKernel(i, j, k, id_l, id_r, bl, thermal, UI, FluxG, FluxGw, T, lambda_y0, lambda_y, epsilon); }); });
	// 	q.submit([&](sycl::handler &h)
	// 			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_inner, local_ndrange), [=](sycl::nd_item<3> index)
	// 							  {
	// 	   				int i = index.get_global_id(0) + bl.Bwidth_X;
	// 	   				int j = index.get_global_id(1) + bl.Bwidth_Y;
	// 	   				int k = index.get_global_id(2) + bl.Bwidth_Z;
	// 	   				int id_l = (bl.Xmax * bl.Ymax * k + bl.Xmax * j + i);
	// 	   				int id_r = (bl.Xmax * bl.Ymax * (k + 1) + bl.Xmax * j + i);
	// 	   				PositivityPreservingKernel(i, j, k, id_l, id_r, bl, thermal, UI, FluxH, FluxHw, T, lambda_z0, lambda_z, epsilon); }); });

	// 	// 	// 	q.wait();

	// 	// 	// 	int cellsize = bl.Xmax * bl.Ymax * bl.Zmax * sizeof(real_t) * NUM_SPECIES;
	// 	// 	// 	q.memcpy(fdata.preFwx, FluxFw, cellsize);
	// 	// 	// 	q.memcpy(fdata.preFwy, FluxGw, cellsize);
	// 	// 	// 	q.memcpy(fdata.preFwz, FluxHw, cellsize);
	// 	// 	// 	q.wait();

	GetCellCenterDerivative(q, bl, fdata, BCs); // get Vortex

#if Visc // NOTE: calculate and add viscous wall Flux to physical convection Flux
	/**
	 * Viscous LU including physical visc(切应力),Visc_Heat transfer(传热), mass Diffusion(质量扩散)
	 * Physical Visc must be included, Visc_Heat is alternative, Visc_Diffu depends on compent
	 */
	real_t *va = fdata.viscosity_aver;
	real_t *tca = fdata.thermal_conduct_aver;
	real_t *Da = fdata.Dkm_aver;
	real_t *hi = fdata.hi;

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), [=](sycl::nd_item<3> index)
							  {
					int i = index.get_global_id(0);
					int j = index.get_global_id(1);
					int k = index.get_global_id(2);
					Gettransport_coeff_aver(i, j, k, bl, thermal, va, tca, Da, fdata.y, hi, rho, p, T, fdata.Ertemp1, fdata.Ertemp2); }); })
		.wait();

	if (bl.DimX)
	{
		q.submit([&](sycl::handler &h)
				 { h.parallel_for(sycl::nd_range<3>(global_ndrange_x, local_ndrange), [=](sycl::nd_item<3> index)
								  {
					int i = index.get_global_id(0) + bl.Bwidth_X - 1;
					int j = index.get_global_id(1) + bl.Bwidth_Y;
					int k = index.get_global_id(2) + bl.Bwidth_Z;
					GetWallViscousFluxX(i, j, k, bl, FluxFw, va, tca, Da, T, rho, hi, fdata.y, u, v, w, fdata.Vde, fdata.visFwx, fdata.Dim_wallx, fdata.hi_wallx, fdata.Yi_wallx, fdata.Yil_wallx); }); }); //.wait()
	}
	if (bl.DimY)
	{
		q.submit([&](sycl::handler &h)
				 { h.parallel_for(sycl::nd_range<3>(global_ndrange_y, local_ndrange), [=](sycl::nd_item<3> index)
								  {
					int i = index.get_global_id(0) + bl.Bwidth_X;
					int j = index.get_global_id(1) + bl.Bwidth_Y - 1;
					int k = index.get_global_id(2) + bl.Bwidth_Z;
					GetWallViscousFluxY(i, j, k, bl, FluxGw, va, tca, Da, T, rho, hi, fdata.y, u, v, w, fdata.Vde, fdata.visFwy, fdata.Dim_wally, fdata.hi_wally, fdata.Yi_wally, fdata.Yil_wally); }); }); //.wait()
	}
	if (bl.DimZ)
	{
		q.submit([&](sycl::handler &h)
				 { h.parallel_for(sycl::nd_range<3>(global_ndrange_z, local_ndrange), [=](sycl::nd_item<3> index)
								  {
					int i = index.get_global_id(0) + bl.Bwidth_X;
					int j = index.get_global_id(1) + bl.Bwidth_Y;
					int k = index.get_global_id(2) + bl.Bwidth_Z - 1;
					GetWallViscousFluxZ(i, j, k, bl, FluxHw, va, tca, Da, T, rho, hi, fdata.y, u, v, w, fdata.Vde, fdata.visFwz, fdata.Dim_wallz, fdata.hi_wallz, fdata.Yi_wallz, fdata.Yil_wallz); }); }); //.wait()
	}

#endif // end Visc
	q.wait();

	// NOTE: update LU from cell-face fluxes
	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_inner, local_ndrange), [=](sycl::nd_item<3> index)
							  {
					int i = index.get_global_id(0) + bl.Bwidth_X;
					int j = index.get_global_id(1) + bl.Bwidth_Y;
					int k = index.get_global_id(2) + bl.Bwidth_Z;
					UpdateFluidLU(i, j, k, bl, LU, FluxFw, FluxGw, FluxHw); }); })
		.wait();

	// // // free
	// // {
	// // 	middle::Free(epsilon, q);
	// // }
}

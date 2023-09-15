#pragma once

#include "Eigen_value.hpp"
#include "Reconstruction_kernels.hpp"
#include "../viscosity/Visc_block.hpp"
#include "PositivityPreserving_kernels.hpp"
#include "../../solver_UpdateStates/UpdateStates_block.hpp"

void GetLU(sycl::queue &q, Setup &setup, Block bl, BConditions BCs[6], Thermal thermal, real_t *UI, real_t *LU,
		   real_t *FluxF, real_t *FluxG, real_t *FluxH, real_t *FluxFw, real_t *FluxGw, real_t *FluxHw,
		   real_t const Gamma, int const Mtrl_ind, FlowData &fdata, real_t *eigen_local_x, real_t *eigen_local_y, real_t *eigen_local_z,
		   real_t *eigen_l, real_t *eigen_r, real_t *uvw_c_max, real_t *eigen_block_x, real_t *eigen_block_y, real_t *eigen_block_z,
		   real_t *yi_min, real_t *yi_max, real_t *Dim_min, real_t *Dim_max)
{
	real_t *rho = fdata.rho;
	real_t *p = fdata.p;
	real_t *H = fdata.H;
	real_t *c = fdata.c;
	real_t *u = fdata.u;
	real_t *v = fdata.v;
	real_t *w = fdata.w;
	real_t *T = fdata.T;

	auto global_ndrange_max = range<3>(bl.Xmax, bl.Ymax, bl.Zmax);
	auto local_ndrange = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z);
	auto global_ndrange_x = range<3>(bl.X_inner + local_ndrange[0], bl.Y_inner, bl.Z_inner);
	auto global_ndrange_y = range<3>(bl.X_inner, bl.Y_inner + local_ndrange[1], bl.Z_inner);
	auto global_ndrange_z = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner + local_ndrange[2]);

	if (bl.DimX)
	{
		// #ifdef DEBUG
		// 	// std::cout << "  sleep before ReconstructFluxX\n";
		// 	// sleep(5);
		// #endif // end DEBUG
		// proceed at x directiom and get F-flux terms at node wall

		q.submit([&](sycl::handler &h)
				 { h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), [=](sycl::nd_item<3> index)
								  {
	    		int i = index.get_global_id(0);
	    		int j = index.get_global_id(1);
				int k = index.get_global_id(2);
				GetLocalEigen(i, j, k, bl, _DF(1.0), _DF(0.0), _DF(0.0), eigen_local_x, u, v, w, c); }); })
			.wait();

		for (size_t nn = 0; nn < Emax; nn++)
		{
			q.submit([&](sycl::handler &h)
					 {	auto reduction_max_eigen = sycl_reduction_max(eigen_block_x[nn]);//reduction(&(eigen_block_x[nn]), sycl::maximum<real_t>());
					h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), reduction_max_eigen, [=](nd_item<3> index, auto &temp_max_eigen)
								{
					int i = index.get_global_id(0);
					int j = index.get_global_id(1);
					int k = index.get_global_id(2);
					int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
					temp_max_eigen.combine(sycl::fabs(eigen_local_x[Emax*id+nn])); }); });
		}
		q.wait();

#ifdef USE_MPI
	for (size_t nn = 0; nn < Emax; nn++)
	{
		real_t mpi_eigen_block_x = _DF(0.0);
		setup.mpiTrans->communicator->synchronize();
		setup.mpiTrans->communicator->allReduce(&(eigen_block_x[nn]), &(mpi_eigen_block_x), 1, setup.mpiTrans->data_type, mpiUtils::MpiComm::MAX);
		setup.mpiTrans->communicator->synchronize();
		eigen_block_x[nn] = mpi_eigen_block_x;
	}
#endif // end USE_MPI

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(
				   sycl::nd_range<3>(global_ndrange_x, local_ndrange), [=](sycl::nd_item<3> index)
				   {
	    		int i = index.get_global_id(0) + bl.Bwidth_X - 1;
				int j = index.get_global_id(1) + bl.Bwidth_Y;
				int k = index.get_global_id(2) + bl.Bwidth_Z;
				ReconstructFluxX(i, j, k, bl, thermal, UI, FluxF, FluxFw, eigen_local_x, eigen_l, eigen_r, fdata.b1x, fdata.b3x, fdata.c2x, fdata.zix, p, rho, u, v, w, fdata.y, T, H, eigen_block_x); }); });
	// real_t *eb1, real_t *eb3, real_t *ec2, real_t *ezi,
	// #ifdef DEBUG
	// 	// std::cout << "  sleep after ReconstructFluxX\n";
	// 	// sleep(5);
	// #endif // end DEBUG
	}

	if (bl.DimY)
	{
		// #ifdef DEBUG
		// 	// std::cout << "  sleep before ReconstructFluxY\n";
		// 	// sleep(5);
		// #endif // end DEBUG
		// proceed at y directiom and get G-flux terms at node wall

		q.submit([&](sycl::handler &h)
				 { h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), [=](sycl::nd_item<3> index)
								  {
		    		int i = index.get_global_id(0);
		    		int j = index.get_global_id(1);
					int k = index.get_global_id(2);
					GetLocalEigen(i, j, k, bl, _DF(0.0), _DF(1.0), _DF(0.0), eigen_local_y, u, v, w, c); }); })
			.wait();

		for (size_t nn = 0; nn < Emax; nn++)
		{
			q.submit([&](sycl::handler &h)
					 {	auto reduction_max_eigen = sycl_reduction_max(eigen_block_y[nn]);//reduction(&(eigen_block_y[nn]), sycl::maximum<real_t>());
						h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), reduction_max_eigen, [=](nd_item<3> index, auto &temp_max_eigen)
								   {
						int i = index.get_global_id(0);
						int j = index.get_global_id(1);
						int k = index.get_global_id(2);
						int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
						temp_max_eigen.combine(sycl::fabs(eigen_local_y[Emax*id+nn])); }); });
		}
		q.wait();

#ifdef USE_MPI
	for (size_t nn = 0; nn < Emax; nn++)
	{
		real_t mpi_eigen_block_y = _DF(0.0);
		setup.mpiTrans->communicator->synchronize();
		setup.mpiTrans->communicator->allReduce(&(eigen_block_y[nn]), &(mpi_eigen_block_y), 1, setup.mpiTrans->data_type, mpiUtils::MpiComm::MAX);
		setup.mpiTrans->communicator->synchronize();
		eigen_block_y[nn] = mpi_eigen_block_y;
	}
#endif // end USE_MPI

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(sycl::nd_range<3>(global_ndrange_y, local_ndrange), [=](sycl::nd_item<3> index)
							  {
		    		int i = index.get_global_id(0) + bl.Bwidth_X;
					int j = index.get_global_id(1) + bl.Bwidth_Y - 1;
					int k = index.get_global_id(2) + bl.Bwidth_Z;
					ReconstructFluxY(i, j, k, bl, thermal, UI, FluxG, FluxGw, eigen_local_y, eigen_l, eigen_r, fdata.b1y, fdata.b3y, fdata.c2y, fdata.ziy, p, rho, u, v, w, fdata.y, T, H, eigen_block_y); }); });
	//.wait()
	// #ifdef DEBUG
	// 	// std::cout << "  sleep after ReconstructFluxY\n";
	// 	// sleep(5);
	// #endif // end DEBUG
	}

	if (bl.DimZ)
	{
		// #ifdef DEBUG
		// 	// std::cout << "  sleep before ReconstructFluxZ\n";
		// 	// sleep(5);
		// #endif // end DEBUG
		// proceed at y directiom and get G-flux terms at node wall

		q.submit([&](sycl::handler &h)
				 { h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), [=](sycl::nd_item<3> index)
								  {
					int i = index.get_global_id(0);
					int j = index.get_global_id(1);
					int k = index.get_global_id(2);
					GetLocalEigen(i, j, k, bl, _DF(0.0), _DF(0.0), _DF(1.0), eigen_local_z, u, v, w, c); }); })
			.wait();

		for (size_t nn = 0; nn < Emax; nn++)
		{
			q.submit([&](sycl::handler &h)
					 {	auto reduction_max_eigen = sycl_reduction_max(eigen_block_z[nn]);//reduction(&(eigen_block_z[nn]), sycl::maximum<real_t>());
						h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), reduction_max_eigen, [=](nd_item<3> index, auto &temp_max_eigen)
									   {
						int i = index.get_global_id(0);
						int j = index.get_global_id(1);
						int k = index.get_global_id(2);
						int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
						temp_max_eigen.combine(sycl::fabs(eigen_local_z[Emax*id+nn])); }); });
		}
		q.wait();

#ifdef USE_MPI
	for (size_t nn = 0; nn < Emax; nn++)
	{
		real_t mpi_eigen_block_z = _DF(0.0);
		setup.mpiTrans->communicator->synchronize();
		setup.mpiTrans->communicator->allReduce(&(eigen_block_z[nn]), &(mpi_eigen_block_z), 1, setup.mpiTrans->data_type, mpiUtils::MpiComm::MAX);
		setup.mpiTrans->communicator->synchronize();
		eigen_block_z[nn] = mpi_eigen_block_z;
	}
#endif // end USE_MPI

	q.submit([&](sycl::handler &h)
			 { h.parallel_for(
				   sycl::nd_range<3>(global_ndrange_z, local_ndrange), [=](sycl::nd_item<3> index)
				   {
					int i = index.get_global_id(0) + bl.Bwidth_X;
					int j = index.get_global_id(1) + bl.Bwidth_Y;
					int k = index.get_global_id(2) + bl.Bwidth_Z - 1;
					ReconstructFluxZ(i, j, k, bl, thermal, UI, FluxH, FluxHw, eigen_local_z, eigen_l, eigen_r, fdata.b1z, fdata.b3z, fdata.c2z, fdata.ziz, p, rho, u, v, w, fdata.y, T, H, eigen_block_z); }); });
	//.wait()
	// #ifdef DEBUG
	// 	// std::cout << "  sleep after ReconstructFluxZ\n";
	// 	// sleep(5);
	// #endif // end DEBUG
	}

	q.wait();

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
		// real_t epsilon[NUM_SPECIES + 2] = {_DF(1.0e-13), _DF(1.0e-13)};
		real_t *epsilon = static_cast<real_t *>(sycl::malloc_shared((NUM_SPECIES + 2) * sizeof(real_t), q));
		epsilon[0] = _DF(1.0e-13), epsilon[1] = _DF(1.0e-13); // 0 for rho and 1 for T and P
		for (size_t ii = 2; ii < NUM_SPECIES + 2; ii++)		  // for Yi
			epsilon[ii] = _DF(0.0);							  // Ini epsilon for y1-yN(N species)

		if (bl.DimX)
		{
			// sycl::stream error_out(1024 * 1024, 1024, h);
			q.submit([&](sycl::handler &h)
					 { h.parallel_for(sycl::nd_range<3>(global_ndrange_inner, local_ndrange), [=](sycl::nd_item<3> index)
									  {
					int i = index.get_global_id(0) + bl.Bwidth_X;
					int j = index.get_global_id(1) + bl.Bwidth_Y;
					int k = index.get_global_id(2) + bl.Bwidth_Z;
					int id_l = (bl.Xmax * bl.Ymax * k + bl.Xmax * j + i);
					int id_r = (bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 1);
					PositivityPreservingKernel(i, j, k, id_l, id_r, bl, thermal, UI, FluxF, FluxFw, T, lambda_x0, lambda_x, epsilon); }); });
		}

		if (bl.DimY)
		{ // sycl::stream error_out(1024 * 1024, 1024, h);
			q.submit([&](sycl::handler &h)
					 { h.parallel_for(sycl::nd_range<3>(global_ndrange_inner, local_ndrange), [=](sycl::nd_item<3> index)
									  {
	   				int i = index.get_global_id(0) + bl.Bwidth_X;
	   				int j = index.get_global_id(1) + bl.Bwidth_Y;
	   				int k = index.get_global_id(2) + bl.Bwidth_Z;
	   				int id_l = (bl.Xmax * bl.Ymax * k + bl.Xmax * j + i);
	   				int id_r = (bl.Xmax * bl.Ymax * k + bl.Xmax * (j + 1) + i);
	   				PositivityPreservingKernel(i, j, k, id_l, id_r, bl, thermal, UI, FluxG, FluxGw, T, lambda_y0, lambda_y, epsilon); }); });
		}

		if (bl.DimZ)
		{
			q.submit([&](sycl::handler &h)
					 { h.parallel_for(sycl::nd_range<3>(global_ndrange_inner, local_ndrange), [=](sycl::nd_item<3> index)
									  {
	   				int i = index.get_global_id(0) + bl.Bwidth_X;
	   				int j = index.get_global_id(1) + bl.Bwidth_Y;
	   				int k = index.get_global_id(2) + bl.Bwidth_Z;
	   				int id_l = (bl.Xmax * bl.Ymax * k + bl.Xmax * j + i);
	   				int id_r = (bl.Xmax * bl.Ymax * (k + 1) + bl.Xmax * j + i);
	   				PositivityPreservingKernel(i, j, k, id_l, id_r, bl, thermal, UI, FluxH, FluxHw, T, lambda_z0, lambda_z, epsilon); }); });
		}
	}

	// 	// 	q.wait();
	// 	// 	int cellsize = bl.Xmax * bl.Ymax * bl.Zmax * sizeof(real_t) * NUM_SPECIES;
	// 	// 	q.memcpy(fdata.preFwx, FluxFw, cellsize);
	// 	// 	q.memcpy(fdata.preFwy, FluxGw, cellsize);
	// 	// 	q.memcpy(fdata.preFwz, FluxHw, cellsize);
	// 	// 	q.wait();

	GetCellCenterDerivative(q, bl, fdata, BCs); // get Vortex

#if Visc // NOTE: calculate and add viscous wall Flux to physical convection Flux
	/* Viscous LU including physical visc(切应力),Visc_Heat transfer(传热), mass Diffusion(质量扩散)
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

	for (size_t nn = 0; nn < NUM_SPECIES; nn++)
	{
		yi_min[nn] = _DF(0.0), yi_max[nn] = _DF(0.0);
		Dim_min[nn] = _DF(0.0), Dim_max[nn] = _DF(0.0);
		auto Yi_min = sycl_reduction_min(yi_min[nn]);
		auto Yi_max = sycl_reduction_max(yi_max[nn]);
		auto Dkm_min = sycl_reduction_min(Dim_min[nn]);
		auto Dkm_max = sycl_reduction_max(Dim_max[nn]);
		q.submit([&](sycl::handler &h)
				 { h.parallel_for(sycl::nd_range<3>(global_ndrange_inner, local_ndrange), Yi_min, Yi_max, Dkm_min, Dkm_max,
								  [=](nd_item<3> index, auto &temp_Ymin, auto &temp_Ymax, auto &temp_Dmin, auto &temp_Dmax)
								  {
						int i = index.get_global_id(0) + bl.Bwidth_X;
						int j = index.get_global_id(1) + bl.Bwidth_Y;
						int k = index.get_global_id(2) + bl.Bwidth_Z;
						int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
						real_t *yi = &(fdata.y[NUM_SPECIES * id]);
						temp_Ymin.combine(yi[nn]), temp_Ymax.combine(yi[nn]);
						real_t *Dkm = &(Da[NUM_SPECIES * id]);
						temp_Dmin.combine(Dkm[nn]), temp_Dmax.combine(Dkm[nn]); }); })
			.wait();

#ifdef USE_MPI
		real_t mpi_Ymin = _DF(0.0), mpi_Ymax = _DF(0.0), mpi_Dmin = _DF(0.0), mpi_Dmax = _DF(0.0);
		setup.mpiTrans->communicator->synchronize();
		setup.mpiTrans->communicator->allReduce(&(yi_min[nn]), &(mpi_Ymin), 1, setup.mpiTrans->data_type, mpiUtils::MpiComm::MIN);
		setup.mpiTrans->communicator->allReduce(&(yi_max[nn]), &(mpi_Ymax), 1, setup.mpiTrans->data_type, mpiUtils::MpiComm::MAX);
		setup.mpiTrans->communicator->allReduce(&(Dim_min[nn]), &(mpi_Dmin), 1, setup.mpiTrans->data_type, mpiUtils::MpiComm::MIN);
		setup.mpiTrans->communicator->allReduce(&(Dim_max[nn]), &(mpi_Dmax), 1, setup.mpiTrans->data_type, mpiUtils::MpiComm::MAX);
		setup.mpiTrans->communicator->synchronize();
		yi_min[nn] = mpi_Ymin, yi_max[nn] = mpi_Ymin, Dim_min[nn] = mpi_Dmin, Dim_max[nn] = mpi_Dmax;
#endif // end USE_MPI
		yi_max[nn] -= yi_min[nn];
		yi_max[nn] *= setup.BlSz.Yil_limiter;				// // yil limiter
		Dim_max[nn] *= setup.BlSz.Dim_limiter * yi_max[nn]; // // Diffu_limiter=Yil_limiter*Dim_limiter
	}

	// // calculate viscous Fluxes
	if (bl.DimX)
	{
		q.submit([&](sycl::handler &h)
				 { h.parallel_for(sycl::nd_range<3>(global_ndrange_x, local_ndrange), [=](sycl::nd_item<3> index)
								  {
					int i = index.get_global_id(0) + bl.Bwidth_X - 1;
					int j = index.get_global_id(1) + bl.Bwidth_Y;
					int k = index.get_global_id(2) + bl.Bwidth_Z;
					GetWallViscousFluxX(i, j, k, bl, FluxFw, va, tca, Da, T, rho, hi, fdata.y, u, v, w, fdata.Vde, yi_max, Dim_max, fdata.visFwx, fdata.Dim_wallx, fdata.hi_wallx, fdata.Yi_wallx, fdata.Yil_wallx); }); }); //.wait()
	}
	if (bl.DimY)
	{
		q.submit([&](sycl::handler &h)
				 { h.parallel_for(sycl::nd_range<3>(global_ndrange_y, local_ndrange), [=](sycl::nd_item<3> index)
								  {
					int i = index.get_global_id(0) + bl.Bwidth_X;
					int j = index.get_global_id(1) + bl.Bwidth_Y - 1;
					int k = index.get_global_id(2) + bl.Bwidth_Z;
					GetWallViscousFluxY(i, j, k, bl, FluxGw, va, tca, Da, T, rho, hi, fdata.y, u, v, w, fdata.Vde, yi_max, Dim_max, fdata.visFwy, fdata.Dim_wally, fdata.hi_wally, fdata.Yi_wally, fdata.Yil_wally); }); }); //.wait()
	}
	if (bl.DimZ)
	{
		q.submit([&](sycl::handler &h)
				 { h.parallel_for(sycl::nd_range<3>(global_ndrange_z, local_ndrange), [=](sycl::nd_item<3> index)
								  {
					int i = index.get_global_id(0) + bl.Bwidth_X;
					int j = index.get_global_id(1) + bl.Bwidth_Y;
					int k = index.get_global_id(2) + bl.Bwidth_Z - 1;
					GetWallViscousFluxZ(i, j, k, bl, FluxHw, va, tca, Da, T, rho, hi, fdata.y, u, v, w, fdata.Vde, yi_max, Dim_max, fdata.visFwz, fdata.Dim_wallz, fdata.hi_wallz, fdata.Yi_wallz, fdata.Yil_wallz); }); }); //.wait()
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
}

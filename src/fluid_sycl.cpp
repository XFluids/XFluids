#include "global_class.h"
#include "block_sycl.hpp"

void FluidSYCL::initialize(int n)
{
	Fluid_name = Fs.fname[n]; // give a name to the fluid
	// type of material, 0: gamma gas, 1: water, 2: stiff gas
	material_property.Mtrl_ind = Fs.material_kind[n];
	// fluid indicator and EOS Parameters
	material_property.Rgn_ind = Fs.material_props[n][0];
	// gamma, A, B, rho0, mu_0, R_0, lambda_0
	material_property.Gamma = Fs.material_props[n][1];
	material_property.A = Fs.material_props[n][2];
	material_property.B = Fs.material_props[n][3];
	material_property.rho0 = Fs.material_props[n][4];
	material_property.R_0 = Fs.material_props[n][5];
	material_property.lambda_0 = Fs.material_props[n][6];
}

void FluidSYCL::AllocateFluidMemory(sycl::queue &q)
{
	int bytes = Fs.bytes;
	int cellbytes = Fs.cellbytes;
	// 主机内存
	h_fstate.rho = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.p = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.c = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.H = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.u = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.v = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.w = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.T = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.gamma = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.y = static_cast<real_t *>(sycl::malloc_host(bytes * NUM_SPECIES, q));
#ifdef ESTIM_NAN
	h_U = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
	h_U1 = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
	h_LU = static_cast<real_t *>(sycl::malloc_host(cellbytes, q));
#if DIM_X
	h_fstate.b1x = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.b3x = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.c2x = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.zix = static_cast<real_t *>(sycl::malloc_host(bytes * NUM_COP, q));
#endif
#if DIM_Y
	h_fstate.b1y = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.b3y = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.c2y = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.ziy = static_cast<real_t *>(sycl::malloc_host(bytes * NUM_COP, q));
#endif
#if DIM_Z
	h_fstate.b1z = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.b3z = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.c2z = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	h_fstate.ziz = static_cast<real_t *>(sycl::malloc_host(bytes * NUM_COP, q));
#endif
#endif // ESTIM_NAN

	// 设备内存
	d_U = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_U1 = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_LU = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	Ubak = static_cast<real_t *>(sycl::malloc_shared(cellbytes, q));
	d_eigen_local = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_fstate.rho = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.p = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.c = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.H = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.u = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.v = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.w = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.T = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.gamma = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.y = static_cast<real_t *>(sycl::malloc_device(bytes * NUM_SPECIES, q));
	long double MemMbSize = (5.0 * (double(cellbytes) / 1024.0) + (9.0 + NUM_SPECIES) * (double(bytes) / 1024.0)) / 1024.0; // shared memory may inside device
#ifdef ESTIM_NAN
#if DIM_X
	d_fstate.b1x = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.b3x = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.c2x = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.zix = static_cast<real_t *>(sycl::malloc_device(bytes * NUM_COP, q));
#endif
#if DIM_Y
	d_fstate.b1y = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.b3y = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.c2y = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.ziy = static_cast<real_t *>(sycl::malloc_device(bytes * NUM_COP, q));
#endif
#if DIM_Z
	d_fstate.b1z = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.b3z = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.c2z = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.ziz = static_cast<real_t *>(sycl::malloc_device(bytes * NUM_COP, q));
#endif
	MemMbSize += ((double(bytes * (NUM_COP + 3) * (DIM_X + DIM_Y + DIM_Z)) / 1024.0)) / 1024.0;
#endif // ESTIM_NAN
#if 2 == EIGEN_ALLOC
	d_eigen_l = static_cast<real_t *>(sycl::malloc_device(cellbytes * Emax, q));
	d_eigen_r = static_cast<real_t *>(sycl::malloc_device(cellbytes * Emax, q));
	MemMbSize += ((double(cellbytes) / 1024.0) * 2 * Emax) / 1024.0;
#endif // end EIGEN_ALLOC
#ifdef Visc
	d_fstate.viscosity_aver = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	MemMbSize += bytes / 1024.0 / 1024.0;
#ifdef Heat
	d_fstate.thermal_conduct_aver = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	MemMbSize += bytes / 1024.0 / 1024.0;
#endif // end Heat
#ifdef Diffu
	d_fstate.hi = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
	d_fstate.Dkm_aver = static_cast<real_t *>(sycl::malloc_device(NUM_SPECIES * bytes, q));
	MemMbSize += bytes / 1024.0 / 1024.0 * 2.0;
#endif // end Diffu
	for (size_t i = 0; i < 9; i++)
		d_fstate.Vde[i] = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	MemMbSize += bytes / 1024.0 / 1024.0 * 9.0;
#endif // end Visc
	d_FluxF = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_FluxG = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_FluxH = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_wallFluxF = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_wallFluxG = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_wallFluxH = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	MemMbSize += cellbytes / 1024.0 / 1024.0 * 6.0;
	// shared memory
	uvw_c_max = static_cast<real_t *>(sycl::malloc_shared(3 * sizeof(real_t), q));

#if USE_MPI
	MemMbSize += Fs.mpiTrans->AllocMemory(q, Fs.BlSz, Emax);
	if (0 == Fs.mpiTrans->myRank)
#endif // end USE_MPI
	{
		std::cout << "Memory Usage: " << MemMbSize / 1024.0 << " GB\n";
	}
}

void FluidSYCL::InitialU(sycl::queue &q)
{
	InitializeFluidStates(q, Fs.BlSz, Fs.ini, material_property, Fs.d_thermal, d_fstate, d_U, d_U1, d_LU, d_FluxF, d_FluxG, d_FluxH, d_wallFluxF, d_wallFluxG, d_wallFluxH);
}

real_t FluidSYCL::GetFluidDt(sycl::queue &q)
{
	return GetDt(q, Fs.BlSz, d_fstate, uvw_c_max);
}

void FluidSYCL::BoundaryCondition(sycl::queue &q, BConditions BCs[6], int flag)
{
	if (flag == 0)
		FluidBoundaryCondition(q, Fs, BCs, d_U);
	else
	{
		FluidBoundaryCondition(q, Fs, BCs, d_U1);
	}
}

void FluidSYCL::UpdateFluidStates(sycl::queue &q, int flag)
{
	if (flag == 0)
		UpdateFluidStateFlux(q, Fs.BlSz, Fs.d_thermal, d_U, d_fstate, d_FluxF, d_FluxG, d_FluxH, material_property.Gamma);
	else
		UpdateFluidStateFlux(q, Fs.BlSz, Fs.d_thermal, d_U1, d_fstate, d_FluxF, d_FluxG, d_FluxH, material_property.Gamma);
}

void FluidSYCL::UpdateFluidURK3(sycl::queue &q, int flag, real_t const dt)
{
	UpdateURK3rd(q, Fs.BlSz, d_U, d_U1, d_LU, dt, flag);
}

void FluidSYCL::ComputeFluidLU(sycl::queue &q, int flag)
{
	// // SYCL kernel cannot call through a function pointer
	// void (*RoeAverageLeftX[Emax])(int n, real_t *eigen_l, real_t &eigen_value, real_t *z, const real_t *yi, real_t const c2,
	// 							  real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
	// 							  real_t const b1, real_t const b3, real_t Gamma);

	// void (*RoeAverageLeftY[Emax])(int n, real_t *eigen_l, real_t &eigen_value, real_t *z, const real_t *yi, real_t const c2,
	// 							  real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
	// 							  real_t const b1, real_t const b3, real_t Gamma);

	// void (*RoeAverageLeftZ[Emax])(int n, real_t *eigen_l, real_t &eigen_value, real_t *z, const real_t *yi, real_t const c2,
	// 							  real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
	// 							  real_t const b1, real_t const b3, real_t Gamma);

	// void (*RoeAverageRightX[Emax])(int n, real_t *eigen_r, real_t *z, const real_t *yi, real_t const c2,
	// 							   real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
	// 							   real_t const b1, real_t const b3, real_t Gamma);

	// void (*RoeAverageRightY[Emax])(int n, real_t *eigen_r, real_t *z, const real_t *yi, real_t const c2,
	// 							   real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
	// 							   real_t const b1, real_t const b3, real_t Gamma);

	// void (*RoeAverageRightZ[Emax])(int n, real_t *eigen_r, real_t *z, const real_t *yi, real_t const c2,
	// 							   real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
	// 							   real_t const b1, real_t const b3, real_t Gamma);

	// RoeAverageLeftX[0] = RoeAverageLeft_x_0;
	// RoeAverageLeftX[1] = RoeAverageLeft_x_1;
	// RoeAverageLeftX[2] = RoeAverageLeft_x_2;
	// RoeAverageLeftX[3] = RoeAverageLeft_x_3;
	// RoeAverageLeftX[Emax - 1] = RoeAverageLeft_x_end;

	// RoeAverageLeftY[0] = RoeAverageLeft_y_0;
	// RoeAverageLeftY[1] = RoeAverageLeft_y_1;
	// RoeAverageLeftY[2] = RoeAverageLeft_y_2;
	// RoeAverageLeftY[3] = RoeAverageLeft_y_3;
	// RoeAverageLeftY[Emax - 1] = RoeAverageLeft_y_end;

	// RoeAverageLeftZ[0] = RoeAverageLeft_z_0;
	// RoeAverageLeftZ[1] = RoeAverageLeft_z_1;
	// RoeAverageLeftZ[2] = RoeAverageLeft_z_2;
	// RoeAverageLeftZ[3] = RoeAverageLeft_z_3;
	// RoeAverageLeftZ[Emax - 1] = RoeAverageLeft_z_end;

	// RoeAverageRightX[0] = RoeAverageRight_x_0;
	// RoeAverageRightX[1] = RoeAverageRight_x_1;
	// RoeAverageRightX[2] = RoeAverageRight_x_2;
	// RoeAverageRightX[3] = RoeAverageRight_x_3;
	// RoeAverageRightX[Emax - 1] = RoeAverageRight_x_end;

	// RoeAverageRightY[0] = RoeAverageRight_y_0;
	// RoeAverageRightY[1] = RoeAverageRight_y_1;
	// RoeAverageRightY[2] = RoeAverageRight_y_2;
	// RoeAverageRightY[3] = RoeAverageRight_y_3;
	// RoeAverageRightY[Emax - 1] = RoeAverageRight_y_end;

	// RoeAverageRightZ[0] = RoeAverageRight_z_0;
	// RoeAverageRightZ[1] = RoeAverageRight_z_1;
	// RoeAverageRightZ[2] = RoeAverageRight_z_2;
	// RoeAverageRightZ[3] = RoeAverageRight_z_3;
	// RoeAverageRightZ[Emax - 1] = RoeAverageRight_z_end;

	// #ifdef COP
	// 	for (size_t i = 4; i < Emax - 1; i++)
	// 	{
	// 		RoeAverageLeftX[i] = RoeAverageLeft_x_cop;
	// 		RoeAverageLeftY[i] = RoeAverageLeft_y_cop;
	// 		RoeAverageLeftZ[i] = RoeAverageLeft_z_cop;
	// 		RoeAverageRightX[i] = RoeAverageRight_x_cop;
	// 		RoeAverageRightY[i] = RoeAverageRight_y_cop;
	// 		RoeAverageRightZ[i] = RoeAverageRight_z_cop;
	// 	}
	// #endif // end COP

	if (flag == 0)
		GetLU(q, Fs.BlSz, Fs.Boundarys, Fs.d_thermal, d_U, d_LU, d_FluxF, d_FluxG, d_FluxH, d_wallFluxF, d_wallFluxG, d_wallFluxH,
			  material_property.Gamma, material_property.Mtrl_ind, d_fstate, d_eigen_local, d_eigen_l, d_eigen_r);
	else
		GetLU(q, Fs.BlSz, Fs.Boundarys, Fs.d_thermal, d_U1, d_LU, d_FluxF, d_FluxG, d_FluxH, d_wallFluxF, d_wallFluxG, d_wallFluxH,
			  material_property.Gamma, material_property.Mtrl_ind, d_fstate, d_eigen_local, d_eigen_l, d_eigen_r);
}

bool FluidSYCL::EstimateFluidNAN(sycl::queue &q, int flag)
{
	Block bl = Fs.BlSz;
	real_t *UI, *LU = d_LU;
	switch (flag)
	{
	case 1:
		UI = d_U1;
		break;

	case 2:
		UI = d_U1;
		break;

	case 3:
		UI = d_U;
		break;
	}

	auto local_ndrange = range<3>(bl.dim_block_x, bl.dim_block_y, bl.dim_block_z);
	auto global_ndrange_max = range<3>(bl.X_inner, bl.Y_inner, bl.Z_inner);

	int x_offset = Fs.OutBoundary ? 0 : bl.Bwidth_X;
	int y_offset = Fs.OutBoundary ? 0 : bl.Bwidth_Y;
	int z_offset = Fs.OutBoundary ? 0 : bl.Bwidth_Z;

	bool *h_error, *d_error;
	int *error_pos;
	real_t *error_value;
	h_error = middle::MallocHost<bool>(h_error, 1, q);
	d_error = middle::MallocDevice<bool>(d_error, 1, q);
	error_pos = sycl::malloc_shared<int>(Emax + 3, q);
	for (size_t n = 0; n < Emax + 3; n++)
		error_pos[n] = 0;
	*h_error = false;

	middle::MemCpy<bool>(d_error, h_error, 1, q);
	q.submit([&](sycl::handler &h)
			 { 
			 	sycl::stream error_out(64 * 1024, 10, h);
				h.parallel_for(sycl::nd_range<3>(global_ndrange_max, local_ndrange), [=](sycl::nd_item<3> index)
							  {
    		int i = index.get_global_id(0) + x_offset;
			int j = index.get_global_id(1) + y_offset;
			int k = index.get_global_id(2) + z_offset;
			EstimateFluidNANKernel(i, j, k, x_offset, y_offset, z_offset, bl, error_pos, UI, LU, d_error, error_out); }); }) //
		.wait();
	middle::MemCpy<bool>(h_error, d_error, 1, q);

	if (*h_error)
	{
		std::cout << "Errors of UI[";
		for (size_t ii = 0; ii < Emax - 1; ii++)
		{
			std::cout << error_pos[ii] << ", ";
		}
		std::cout << error_pos[Emax - 1] << "] inside the step " << flag << " of RungeKutta located(i, j, k = "
				  << error_pos[Emax] << ", " << error_pos[Emax + 1] << ", " << error_pos[Emax + 2] << ")";
#ifdef ERROR_PATCH
		error_patched_times += 1;
		std::cout << " patched.\n";
#else
		std::cout << " captured.\n";
#endif // end ERROR_PATCH
		return true;
	}
	return false;
}

#ifdef COP_CHEME
void FluidSYCL::ODESolver(sycl::queue &q, real_t Time)
{
#if 0 == CHEME_SOLVER
	ChemeODEQ2Solver(q, Fs.BlSz, Fs.d_thermal, d_fstate, d_U, Fs.d_react, Time);
//#else 1 == CHEME_SOLVER // CVODE from LLNL to SYCL only support Intel GPUs
#endif // end CHEME_SOLVER
}
#endif // end COP_CHEME

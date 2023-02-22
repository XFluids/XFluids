#include "include/global_class.h"
#include "global_function.hpp"
#include "block_sycl.hpp"

using namespace std;
using namespace sycl;

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
#ifdef COP
	for (size_t n = 0; n < NUM_SPECIES; n++)
	{
		h_fstate.y[n] = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	}
	h_fstate.T = static_cast<real_t *>(sycl::malloc_host(bytes, q));
#endif // COP
#ifdef Visc
	for (size_t i = 0; i < 9; i++)
	{
		h_fstate.Vde[i] = static_cast<real_t *>(sycl::malloc_host(bytes, q));
	}
#endif // Visc
	//  设备内存
	d_U = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_U1 = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_LU = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_eigen_local = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_fstate.rho = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.p = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.c = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.H = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.u = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.v = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	d_fstate.w = static_cast<real_t *>(sycl::malloc_device(bytes, q));
#ifdef COP
	for (size_t n = 0; n < NUM_SPECIES; n++)
	{
		d_fstate.y[n] = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	}
	d_fstate.T = static_cast<real_t *>(sycl::malloc_device(bytes, q));
#endif // COP
#ifdef Visc
	for (size_t i = 0; i < 9; i++)
	{
		d_fstate.Vde[i] = static_cast<real_t *>(sycl::malloc_device(bytes, q));
	}
#endif // Visc
	d_FluxF = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_FluxG = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_FluxH = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_wallFluxF = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_wallFluxG = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	d_wallFluxH = static_cast<real_t *>(sycl::malloc_device(cellbytes, q));
	// shared memory
	uvw_c_max = static_cast<real_t *>(sycl::malloc_shared(3 * sizeof(real_t), q));

	q.wait();

	cout << "Memory Usage: " << (real_t)((long)10 * cellbytes / real_t(1024 * 1024) + (long)(8 + NUM_SPECIES) * bytes / real_t(1024 * 1024)) / (real_t)(1024) << " GB\n";
}

void FluidSYCL::InitialU(sycl::queue &q)
{
	InitializeFluidStates(q, Fs.BlSz, Fs.ini, material_property, Fs.d_thermal, d_fstate, d_U, d_U1, d_LU, d_FluxF, d_FluxG, d_FluxH, d_wallFluxF, d_wallFluxG, d_wallFluxH);
	q.memcpy(d_U1, d_U, Fs.cellbytes).wait();
}

real_t FluidSYCL::GetFluidDt(sycl::queue &q)
{
	return GetDt(q, Fs.BlSz, d_fstate, uvw_c_max);
}

void FluidSYCL::BoundaryCondition(sycl::queue &q, BConditions BCs[6], int flag)
{
	if (flag == 0)
		FluidBoundaryCondition(q, Fs.BlSz, BCs, d_U);
	else
		FluidBoundaryCondition(q, Fs.BlSz, BCs, d_U1);
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
	if (flag == 0)
		GetLU(q, Fs.BlSz, Fs.d_thermal, d_U, d_LU, d_FluxF, d_FluxG, d_FluxH, d_wallFluxF, d_wallFluxG, d_wallFluxH,
			  material_property.Gamma, material_property.Mtrl_ind, d_fstate, d_eigen_local);
	else
		GetLU(q, Fs.BlSz, Fs.d_thermal, d_U1, d_LU, d_FluxF, d_FluxG, d_FluxH, d_wallFluxF, d_wallFluxG, d_wallFluxH,
			  material_property.Gamma, material_property.Mtrl_ind, d_fstate, d_eigen_local);
}
#ifdef Visc
void FluidSYCL::PhysicVisc(sycl::queue &q, Block bl, FlowData &fdata)
{
	GetCellCenterDerivative(q, Fs.BlSz, d_fstate, Fs.Boundarys);
}
#endif
#ifdef Heat
void FluidSYCL::HeatVisc(sycl::queue &q)
{
}
#endif
#ifdef Diffu
void FluidSYCL::DiffuVisc(sycl::queue &q)
{
} // add viscity of mass diffusion
#endif
#ifdef React
void FluidSYCL::ODESolver(sycl::queue &q, real_t Time)
{
	FluidODESolver(q, Fs.BlSz, Fs.d_thermal, d_fstate, d_U, Fs.d_react, Time);
}
#endif // React
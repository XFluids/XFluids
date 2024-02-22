#pragma once

#include "global_setup.h"
#include "../read_ini/setupini.h"

#include "Reaction_device.hpp"

extern SYCL_KERNEL void ChemeODEQ2SolverKernel(int i, int j, int k, MeshSize bl, Thermal thermal, Reaction react, real_t *UI, real_t *y, real_t *rho, real_t *T, real_t *e, const real_t dt)
{
	MARCO_DOMAIN();
	if (i >= Xmax - bl.Bwidth_X)
		return;
	if (j >= Ymax - bl.Bwidth_Y)
		return;
	if (k >= Zmax - bl.Bwidth_Z)
		return;

	int id = Xmax * Ymax * k + Xmax * j + i;

	real_t Kf[NUM_REA], Kb[NUM_REA], *yi = &(y[NUM_SPECIES * id]);
	get_KbKf(Kf, Kb, react.Rargus, thermal._Wi, thermal.Hia, thermal.Hib, react.Nu_d_, T[id]); // get_e
	// for (size_t n = 0; n < Emax - NUM_COP; n++)
	// {
	//         U[n] = UI[Emax * id + n];
	// }
	// real_t rho1 = _DF(1.0) / U[0];
	// real_t u = U[1] * rho1;
	// real_t v = U[2] * rho1;
	// real_t w = U[3] * rho1;
	// real_t e = U[4] * rho1 - _DF(0.5) * (u * u + v * v + w * w);
	Chemeq2(id, thermal, Kf, Kb, react.React_ThirdCoef, react.Rargus, react.Nu_b_, react.Nu_f_, react.Nu_d_, react.third_ind,
			react.reaction_list, react.reactant_list, react.product_list, react.rns, react.rts, react.pls, yi, dt, T[id], rho[id], e[id]);
	// update partial density according to C0
	for (int n = 0; n < NUM_COP; n++)
	{
		UI[Emax * id + n + 5] = yi[n] * rho[id];
	}
}

#if __VENDOR_SUBMIT__
_VENDOR_KERNEL_LB_(__LBMt, 1)
void ChemeODEQ2SolverKernelVendorWrapper(MeshSize bl, Thermal thermal, Reaction react, real_t *UI, real_t *y, real_t *rho, real_t *T, real_t *e, const real_t dt)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + bl.Bwidth_X;
	int j = blockIdx.y * blockDim.y + threadIdx.y + bl.Bwidth_Y;
	int k = blockIdx.z * blockDim.z + threadIdx.z + bl.Bwidth_Z;

	ChemeODEQ2SolverKernel(i, j, k, bl, thermal, react, UI, y, rho, T, e, dt);
}

#endif

#pragma once

#include "global_setup.h"
#include "Visc_device.h"
#include "Flux_discrete.h"

extern void CenterDerivativeBCKernelX(int i, int j, int k, Block bl, BConditions const BC, real_t *const *Vde, int const mirror_offset, int const index_inner, int const sign)
{
	MARCO_DOMAIN_GHOST();
	if (j >= Ymax)
		return;
	if (k >= Zmax)
		return;

	int id = Xmax * Ymax * k + Xmax * j + i;
	real_t *Vde_x[] = {Vde[ducy], Vde[ducz], Vde[dvcy], Vde[dwcz]};

	switch (BC)
	{
	case Symmetry:
	{
		int offset = 2 * (Bwidth_X + mirror_offset) - 1;
		int target_id = Xmax * Ymax * k + Xmax * j + (offset - i);
		for (int n = 0; n < 4; n++)
			Vde_x[n][id] = Vde_x[n][target_id];
	}
	break;

	case Periodic:
	{
		int target_id = Xmax * Ymax * k + Xmax * j + (i + sign * X_inner);
		for (int n = 0; n < 4; n++)
			Vde_x[n][id] = Vde_x[n][target_id];
	}
	break;

	case Inflow:
		for (int n = 0; n < 4; n++)
			Vde_x[n][id] = real_t(0.0f);
		break;

	case Outflow:
	{
		int target_id = Xmax * Ymax * k + Xmax * j + index_inner;
		for (int n = 0; n < 4; n++)
			Vde_x[n][id] = Vde_x[n][target_id];
	}
	break;

	case nslipWall:
	{
		int offset = 2 * (Bwidth_X + mirror_offset) - 1;
		int target_id = Xmax * Ymax * k + Xmax * j + (offset - i);
		for (int n = 0; n < 4; n++)
			Vde_x[n][id] = -Vde_x[n][target_id];
	}
	break;

	case viscWall:
		break;

	case slipWall:
		break;

	case innerBlock:
		break;

#ifdef USE_MPI
	case BC_COPY:
		break;
	case BC_UNDEFINED:
		break;
#endif
	}
}

extern void CenterDerivativeBCKernelY(int i, int j, int k, Block bl, BConditions const BC, real_t *const *Vde, int const mirror_offset, int const index_inner, int const sign)
{
	MARCO_DOMAIN_GHOST();
	if (i >= Xmax)
		return;
	if (k >= Zmax)
		return;

	int id = Xmax * Ymax * k + Xmax * j + i;
	real_t *Vde_y[4] = {Vde[dvcx], Vde[dvcz], Vde[ducx], Vde[dwcz]};

	switch (BC)
	{
	case Symmetry:
	{
		int offset = 2 * (Bwidth_Y + mirror_offset) - 1;
		int target_id = Xmax * Ymax * k + Xmax * (offset - j) + i;
		for (int n = 0; n < 4; n++)
			Vde_y[n][id] = Vde_y[n][target_id];
	}
	break;

	case Periodic:
	{
		int target_id = Xmax * Ymax * k + Xmax * (j + sign * Y_inner) + i;
		for (int n = 0; n < 4; n++)
			Vde_y[n][id] = Vde_y[n][target_id];
	}
	break;

	case Inflow:
		for (int n = 0; n < 4; n++)
			Vde_y[n][id] = real_t(0.0f);
		break;

	case Outflow:
	{
		int target_id = Xmax * Ymax * k + Xmax * index_inner + i;
		for (int n = 0; n < 4; n++)
			Vde_y[n][id] = Vde_y[n][target_id];
	}
	break;

	case nslipWall:
	{
		int offset = 2 * (Bwidth_Y + mirror_offset) - 1;
		int target_id = Xmax * Ymax * k + Xmax * (offset - j) + i;
		for (int n = 0; n < 4; n++)
			Vde_y[n][id] = -Vde_y[n][target_id];
	}
	break;

	case viscWall:
		break;

	case slipWall:
		break;

	case innerBlock:
		break;

#ifdef USE_MPI
	case BC_COPY:
		break;
	case BC_UNDEFINED:
		break;
#endif
	}
}

extern void CenterDerivativeBCKernelZ(int i, int j, int k, Block bl, BConditions const BC, real_t *const *Vde, int const mirror_offset, int const index_inner, int const sign)
{
	MARCO_DOMAIN_GHOST();
	if (i >= Xmax)
		return;
	if (j >= Ymax)
		return;

	int id = Xmax * Ymax * k + Xmax * j + i;
	real_t *Vde_z[4] = {Vde[dwcx], Vde[dwcy], Vde[ducx], Vde[dvcy]};

	switch (BC)
	{
	case Symmetry:
	{
		int offset = 2 * (Bwidth_Z + mirror_offset) - 1;
		int target_id = Xmax * Ymax * (offset - k) + Xmax * j + i;
		for (int n = 0; n < 4; n++)
			Vde_z[n][id] = Vde_z[n][target_id];
	}
	break;

	case Periodic:
	{
		int target_id = Xmax * Ymax * (k + sign * Z_inner) + Xmax * j + i;
		for (int n = 0; n < 4; n++)
			Vde_z[n][id] = Vde_z[n][target_id];
	}
	break;

	case Inflow:
		for (int n = 0; n < 4; n++)
			Vde_z[n][id] = real_t(0.0f);
		break;

	case Outflow:
	{
		int target_id = Xmax * Ymax * index_inner + Xmax * j + i;
		for (int n = 0; n < 4; n++)
			Vde_z[n][id] = Vde_z[n][target_id];
	}
	break;

	case nslipWall:
	{
		int offset = 2 * (Bwidth_Z + mirror_offset) - 1;
		int target_id = Xmax * Ymax * (k - offset) + Xmax * j + i;
		for (int n = 0; n < 4; n++)
			Vde_z[n][id] = -Vde_z[n][target_id];
	}
	break;

	case viscWall:
		break;

	case slipWall:
		break;

	case innerBlock:
		break;

#ifdef USE_MPI
	case BC_COPY:
		break;
	case BC_UNDEFINED:
		break;
#endif
	}
}

extern SYCL_KERNEL void Gettransport_coeff_aver(int i, int j, int k, Block bl, Thermal thermal, real_t *viscosity_aver, real_t *thermal_conduct_aver,
												real_t *Dkm_aver, real_t *y, real_t *hi, real_t *rho, real_t *p, real_t *T, real_t *Ertemp1, real_t *Ertemp2)
{
	if (i >= bl.Xmax)
		return;
	if (j >= bl.Ymax)
		return;
	if (k >= bl.Zmax)
		return;

	int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
	// get mole fraction of each specie
	real_t X[MAX_SPECIES] = {_DF(0.0)}; //, yi[NUM_SPECIES] = {_DF(0.0)};
#if Visc_Diffu
	for (size_t ii = 0; ii < NUM_SPECIES; ii++)
		hi[ii + NUM_SPECIES * id] = get_Enthalpy(thermal.Hia, thermal.Hib, T[id], thermal.Ri[ii], ii);
#endif									 // end Visc_Diffu
	real_t *yi = &(y[NUM_SPECIES * id]); // get_yi(y, yi, id);
	real_t C_total = get_xi(X, yi, thermal._Wi, rho[id]);
	//  real_t *temp = &(Dkm_aver[NUM_SPECIES * id]);
	//  real_t *temp = &(hi[NUM_SPECIES * id]);
	Get_transport_coeff_aver(i, j, k, thermal, &(Dkm_aver[NUM_SPECIES * id]), viscosity_aver[id], thermal_conduct_aver[id],
							 X, rho[id], p[id], T[id], C_total, &(Ertemp1[NUM_SPECIES * id]), &(Ertemp2[NUM_SPECIES * id]));
}

#if __VENDOR_SUBMIT__
_VENDOR_KERNEL_LB_(__LBMt, 1)
void Gettransport_coeff_averVendorWrapper(Block bl, Thermal thermal, real_t *viscosity_aver, real_t *thermal_conduct_aver, real_t *Dkm_aver,
										  real_t *y, real_t *hi, real_t *rho, real_t *p, real_t *T, real_t *Ertemp1, real_t *Ertemp2)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	Gettransport_coeff_aver(i, j, k, bl, thermal, viscosity_aver, thermal_conduct_aver, Dkm_aver, y, hi, rho, p, T, Ertemp1, Ertemp2);
}
#endif
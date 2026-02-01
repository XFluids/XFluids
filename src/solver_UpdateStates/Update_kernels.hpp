#pragma once

#include "Update_device.hpp"

extern SYCL_KERNEL void UpdateFuidStatesSPKernel(int i, int j, int k, MeshSize bl, real_t *UI, real_t *FluxF, real_t *FluxG, real_t *FluxH,
												 real_t *rho, real_t *p, real_t *u, real_t *v, real_t *w, real_t *c, real_t *H, real_t const gamma);

extern inline void Updaterhoyi(int i, int j, int k, MeshSize bl, real_t *UI, real_t *rho, real_t *_y)
{
	if (i >= bl.Xmax)
		return;
	if (j >= bl.Ymax)
		return;
	if (k >= bl.Zmax)
		return;

	int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
	real_t *U = &(UI[Emax * id]), *yi = &(_y[NUM_SPECIES * id]);

	Getrhoyi(U, rho[id], yi);
}

extern SYCL_KERNEL inline void UpdateFuidStatesKernel(int i, int j, int k, MeshSize bl, Thermal thermal, real_t *UI, real_t *FluxF, real_t *FluxG, real_t *FluxH,
											   real_t *rho, real_t *p, real_t *u, real_t *v, real_t *w, real_t *c, real_t *gamma, real_t *e, real_t *H,
											   real_t *T, real_t *_y, real_t *Ri, real_t *Cp)
{
	MARCO_DOMAIN_GHOST();
	if (i >= Xmax)
		return;
	if (j >= Ymax)
		return;
	if (k >= Zmax)
		return;

	int id = Xmax * Ymax * k + Xmax * j + i;
	real_t *U = &(UI[Emax * id]), *yi = &(_y[NUM_SPECIES * id]);

	// Getrhoyi(U, rho[id], yi);
	GetStates(U, rho[id], u[id], v[id], w[id], p[id], H[id], c[id], gamma[id], T[id], e[id], Cp[id], Ri[id], thermal, yi);

	real_t *Fx = &(FluxF[Emax * id]);
	real_t *Fy = &(FluxG[Emax * id]);
	real_t *Fz = &(FluxH[Emax * id]);

	// GetPhysFlux(U, yi, Fx, Fy, Fz, rho[id], u[id], v[id], w[id], p[id], H[id], c[id]);
	Fx[0] = U[1];
	Fx[1] = U[1] * u[id] + p[id];
	Fx[2] = U[1] * v[id];
	Fx[3] = U[1] * w[id];
	Fx[4] = (U[4] + p[id]) * u[id];

	Fy[0] = U[2];
	Fy[1] = U[2] * u[id];
	Fy[2] = U[2] * v[id] + p[id];
	Fy[3] = U[2] * w[id];
	Fy[4] = (U[4] + p[id]) * v[id];

	Fz[0] = U[3];
	Fz[1] = U[3] * u[id];
	Fz[2] = U[3] * v[id];
	Fz[3] = U[3] * w[id] + p[id];
	Fz[4] = (U[4] + p[id]) * w[id];

	// // real_t de_fx[Emax], de_fy[Emax], de_fz[Emax];
	// // get_Array(FluxF, de_fx, Emax, id);
	// // get_Array(FluxG, de_fy, Emax, id);
	// // get_Array(FluxH, de_fz, Emax, id);
}

#if __VENDOR_SUBMIT__
_VENDOR_KERNEL_LB_(__LBMt, 1)
void UpdateFuidStatesKernelVendorWrapper(MeshSize bl, Thermal thermal, real_t *UI, real_t *FluxF, real_t *FluxG, real_t *FluxH,
										 real_t *rho, real_t *p, real_t *u, real_t *v, real_t *w, real_t *c, real_t *gamma,
										 real_t *e, real_t *H, real_t *T, real_t *_y, real_t *Ri, real_t *Cp)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	int k = blockIdx.z * blockDim.z + threadIdx.z;

	UpdateFuidStatesKernel(i, j, k, bl, thermal, UI, FluxF, FluxG, FluxH, rho, p, u, v, w, c, gamma, e, H, T, _y, Ri, Cp);
}
#endif

extern inline void UpdateURK3rdKernel(int i, int j, int k, MeshSize bl, real_t *U, real_t *U1, real_t *LU, real_t const dt, int flag)
{
	MARCO_DOMAIN();
	if (i >= bl.Xmax)
		return;
	if (j >= bl.Ymax)
		return;
	if (k >= bl.Zmax)
		return;
	int id = Xmax * Ymax * k + Xmax * j + i;

	// real_t de_U[Emax], de_U1[Emax], de_LU[Emax];
	switch (flag)
	{
	case 1:
		for (int n = 0; n < Emax; n++)
			U1[Emax * id + n] = U[Emax * id + n] + dt * LU[Emax * id + n];
		break;
	case 2:
		for (int n = 0; n < Emax; n++)
			U1[Emax * id + n] = _DF(0.75) * U[Emax * id + n] + _DF(0.25) * U1[Emax * id + n] + _DF(0.25) * dt * LU[Emax * id + n];
		break;
	case 3:
		for (int n = 0; n < Emax; n++)
			U[Emax * id + n] = (U[Emax * id + n] + _DF(2.0) * U1[Emax * id + n] + _DF(2.0) * dt * LU[Emax * id + n]) * _OT;
		break;
	}
	// get_Array(U, de_U, Emax, id);
	// get_Array(U1, de_U1, Emax, id);
	// get_Array(LU, de_LU, Emax, id);
}

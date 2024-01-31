#pragma once

#include "Update_device.hpp"

extern void Updaterhoyi(int i, int j, int k, Block bl, real_t *UI, real_t *rho, real_t *_y)
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

extern void UpdateFuidStatesKernel(int i, int j, int k, Block bl, Thermal thermal, real_t *UI, real_t *FluxF, real_t *FluxG, real_t *FluxH,
								   real_t *rho, real_t *p, real_t *c, real_t *H, real_t *u, real_t *v, real_t *w, real_t *_y,
								   real_t *gamma, real_t *T, real_t *e, real_t const Gamma) //, const sycl::stream &stream_ct1
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
	GetStates(U, rho[id], u[id], v[id], w[id], p[id], H[id], c[id], gamma[id], T[id], e[id], thermal, yi);

	real_t *Fx = &(FluxF[Emax * id]);
	real_t *Fy = &(FluxG[Emax * id]);
	real_t *Fz = &(FluxH[Emax * id]);

	GetPhysFlux(U, yi, Fx, Fy, Fz, rho[id], u[id], v[id], w[id], p[id], H[id], c[id]);

	// // real_t de_fx[Emax], de_fy[Emax], de_fz[Emax];
	// // get_Array(FluxF, de_fx, Emax, id);
	// // get_Array(FluxG, de_fy, Emax, id);
	// // get_Array(FluxH, de_fz, Emax, id);
}

extern void UpdateURK3rdKernel(int i, int j, int k, Block bl, real_t *U, real_t *U1, real_t *LU, real_t const dt, int flag)
{
	MARCO_DOMAIN();
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

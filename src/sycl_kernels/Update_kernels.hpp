#pragma once

#include "Utils_kernels.hpp"

extern SYCL_EXTERNAL void Updaterhoyi(int i, int j, int k, Block bl, real_t *UI, real_t *rho, real_t *_y)
{
#if DIM_X
	if (i >= bl.Xmax)
		return;
#endif
#if DIM_Y
	if (j >= bl.Ymax)
		return;
#endif
#if DIM_Z
	if (k >= bl.Zmax)
		return;
#endif

	int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
	real_t *U = &(UI[Emax * id]), *yi = &(_y[NUM_SPECIES * id]);

	Getrhoyi(U, rho[id], yi);
}

extern SYCL_EXTERNAL void UpdateFuidStatesKernel(int i, int j, int k, Block bl, Thermal thermal, real_t *UI, real_t *FluxF, real_t *FluxG, real_t *FluxH,
												 real_t *rho, real_t *p, real_t *c, real_t *H, real_t *u, real_t *v, real_t *w, real_t *_y,
												 real_t *gamma, real_t *T, real_t *e, real_t const Gamma) //, const sycl::stream &stream_ct1
{
	MARCO_DOMAIN_GHOST();
	int id = Xmax * Ymax * k + Xmax * j + i;

#if DIM_X
	if (i >= Xmax)
		return;
#endif
#if DIM_Y
	if (j >= Ymax)
		return;
#endif
#if DIM_Z
	if (k >= Zmax)
		return;
#endif

	real_t *U = &(UI[Emax * id]), *yi = &(_y[NUM_SPECIES * id]);

	// Getrhoyi(U, rho[id], yi);

	GetStates(U, rho[id], u[id], v[id], w[id], p[id], H[id], c[id], gamma[id], T[id], e[id], thermal, yi);

	// // real_t x = DIM_X ? (i - Bwidth_X + bl.myMpiPos_x * X_inner + _DF(0.5)) * bl.dx + bl.Domain_xmin : _DF(0.0);
	// // real_t y = DIM_Y ? (j - Bwidth_Y + bl.myMpiPos_y * Y_inner + _DF(0.5)) * bl.dy + bl.Domain_ymin : _DF(0.0);
	// // real_t z = DIM_Z ? (k - Bwidth_Z + bl.myMpiPos_z * Z_inner + _DF(0.5)) * bl.dz + bl.Domain_zmin : _DF(0.0);

	real_t *Fx = &(FluxF[Emax * id]);
	real_t *Fy = &(FluxG[Emax * id]);
	real_t *Fz = &(FluxH[Emax * id]);

	GetPhysFlux(U, yi, Fx, Fy, Fz, rho[id], u[id], v[id], w[id], p[id], H[id], c[id]);

	// // for (size_t n = 0; n < NUM_SPECIES; n++)
	// //     _y[n][id] = yi[n];

	// // real_t de_fx[Emax], de_fy[Emax], de_fz[Emax];
	// // get_Array(FluxF, de_fx, Emax, id);
	// // get_Array(FluxG, de_fy, Emax, id);
	// // get_Array(FluxH, de_fz, Emax, id);
}

extern SYCL_EXTERNAL void UpdateURK3rdKernel(int i, int j, int k, Block bl, real_t *U, real_t *U1, real_t *LU, real_t const dt, int flag)
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

extern SYCL_EXTERNAL void UpdateFluidLU(int i, int j, int k, Block bl, real_t *LU, real_t *FluxFw, real_t *FluxGw, real_t *FluxHw)
{
	MARCO_DOMAIN();
	int id = Xmax * Ymax * k + Xmax * j + i;

#if DIM_X
	int id_im = Xmax * Ymax * k + Xmax * j + i - 1;
#endif
#if DIM_Y
	int id_jm = Xmax * Ymax * k + Xmax * (j - 1) + i;
#endif
#if DIM_Z
	int id_km = Xmax * Ymax * (k - 1) + Xmax * j + i;
#endif

	for (int n = 0; n < Emax; n++)
	{
		real_t LU0 = _DF(0.0);
#if DIM_X
		LU0 += (FluxFw[Emax * id_im + n] - FluxFw[Emax * id + n]) * bl._dx;
#endif
#if DIM_Y
		LU0 += (FluxGw[Emax * id_jm + n] - FluxGw[Emax * id + n]) * bl._dy;
#endif
#if DIM_Z
		LU0 += (FluxHw[Emax * id_km + n] - FluxHw[Emax * id + n]) * bl._dz;
#endif
		LU[Emax * id + n] = LU0;
	}

	// real_t de_LU[Emax];
	// get_Array(LU, de_LU, Emax, id);
}
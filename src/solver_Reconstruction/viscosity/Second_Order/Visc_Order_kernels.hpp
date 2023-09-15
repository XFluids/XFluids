#pragma once

#include "Flux_discrete.h"

/**
 * GetInnerCellCenterDerivativeKernel: to calculate velocity derivative at each grid point.
 * This is a Second-Order viscosity Flux discretization implementation.
 * @param i,j,k is the index of grid point
 * @param bl is the block size arguments of block
 * @param u,v,w is the velocity variables of the fluid flow field
 * @param Vde is where the velocity derivative calculated to be stored
 * @param Voxs is where the vorticity vector elements to be stored
 * @param Vox is the magnitude squre of vorticity vector
 */
extern void GetInnerCellCenterDerivativeKernel(int i, int j, int k, Block bl, real_t *u, real_t *v, real_t *w, real_t *const *Vde, real_t *const *Voxs, real_t *Vox)
{
	if (i > bl.Xmax - bl.Bwidth_X + 1)
		return;
	if (j > bl.Ymax - bl.Bwidth_Y + 1)
		return;
	if (k > bl.Zmax - bl.Bwidth_Z + 1)
		return;

	int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
	real_t Dmp[9] = {_DF(0.0), _DF(0.0), _DF(0.0), _DF(0.0), _DF(0.0), _DF(0.0), _DF(0.0), _DF(0.0), _DF(0.0)};
#if DIM_X
	real_t _dx = bl._dx;
	int id_m1_x = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i - 1;
	int id_p1_x = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 1;
	// int id_m2_x = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i - 2;
	// int id_p2_x = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 2;

	Dmp[ducx] = ((u[id_p1_x] - u[id_m1_x])) * _dx * _DF(0.5);
#if DIM_Y
	Dmp[dvcx] = ((v[id_p1_x] - v[id_m1_x])) * _dx * _DF(0.5);
#endif // DIM_Y
#if DIM_Z
	Dmp[dwcx] = ((w[id_p1_x] - w[id_m1_x])) * _dx * _DF(0.5);
#endif // DIM_Z
#endif // end DIM_X

#if DIM_Y
	real_t _dy = bl._dy;
	int id_m1_y = bl.Xmax * bl.Ymax * k + bl.Xmax * (j - 1) + i;
	int id_p1_y = bl.Xmax * bl.Ymax * k + bl.Xmax * (j + 1) + i;
	// int id_m2_y = bl.Xmax * bl.Ymax * k + bl.Xmax * (j - 2) + i;
	// int id_p2_y = bl.Xmax * bl.Ymax * k + bl.Xmax * (j + 2) + i;

#if DIM_X
	Dmp[ducy] = ((u[id_p1_y] - u[id_m1_y])) * _dy * _DF(0.5);
#endif // DIM_X
	Dmp[dvcy] = ((v[id_p1_y] - v[id_m1_y])) * _dy * _DF(0.5);
#if DIM_Z
	Dmp[dwcy] = ((w[id_p1_y] - w[id_m1_y])) * _dy * _DF(0.5);
#endif // DIM_Z

#endif // end DIM_Y

#if DIM_Z
	real_t _dz = bl._dz;
	int id_m1_z = bl.Xmax * bl.Ymax * (k - 1) + bl.Xmax * j + i;
	int id_p1_z = bl.Xmax * bl.Ymax * (k + 1) + bl.Xmax * j + i;
	// int id_m2_z = bl.Xmax * bl.Ymax * (k - 2) + bl.Xmax * j + i;
	// int id_p2_z = bl.Xmax * bl.Ymax * (k + 2) + bl.Xmax * j + i;

#if DIM_X
	Dmp[ducz] = ((u[id_p1_z] - u[id_m1_z])) * _dz * _DF(0.5);
#endif // DIM_X
#if DIM_Y
	Dmp[dvcz] = ((v[id_p1_z] - v[id_m1_z])) * _dz * _DF(0.5);
#endif // DIM_Y
	Dmp[dwcz] = ((w[id_p1_z] - w[id_m1_z])) * _dz * _DF(0.5);

#endif // end DIM_Z

	Vde[ducx][id] = Dmp[ducx];
	Vde[dvcx][id] = Dmp[dvcx];
	Vde[dwcx][id] = Dmp[dwcx];
	Vde[ducy][id] = Dmp[ducy];
	Vde[dvcy][id] = Dmp[dvcy];
	Vde[dwcy][id] = Dmp[dwcy];
	Vde[ducz][id] = Dmp[ducz];
	Vde[dvcz][id] = Dmp[dvcz];
	Vde[dwcz][id] = Dmp[dwcz];

	real_t wx = Dmp[dwcy] - Dmp[dvcz], wy = Dmp[ducz] - Dmp[dwcx], wz = Dmp[dvcx] - Dmp[ducy]; // vorticity w=wx*i+wy*j+wz*k;
	Voxs[0][id] = wx, Voxs[1][id] = wy, Voxs[2][id] = wz;
	Vox[id] = wx * wx + wy * wy + wz * wz; // |w|, magnitude or the vorticity, w^2 used later, sqrt while output
}

/**
 * GetWallViscousFluxX: to calculate each half-point viscosity Flux.
 * This is a Second-Order viscosity Flux discretization implementation.
 * @param i,j,k is the index of grid point
 * @param bl is the block size arguments of block
 * @param Flux_wall is the solution half-point viscosity Flux.
 * @param viscosity_aver is the viscosity coefficient of Mixture.
 * @param thermal_conduct_aver is the Fourier heat transfer coeffcient of Mixture.
 * @param Dkm_aver is the mass diffusion of the ith species to Mixture.
 * @param Vde is where the velocity derivative.
 * @param T,rho,hi,Yi,u,v,w is the variables of the fluid flow field.
 * @param Er-* are the Error to be Output in Flux reconstruction.
 */
extern void GetWallViscousFluxX(int i, int j, int k, Block bl, real_t *Flux_wall, real_t *viscosity_aver, real_t *thermal_conduct_aver, real_t *Dkm_aver,
								real_t *T, real_t *rho, real_t *hi, real_t *Yi, real_t *u, real_t *v, real_t *w, real_t *const *Vde,
								real_t *Yil_limiter, real_t *Diffu_limiter,
								real_t *ErvisFw, real_t *ErDimw, real_t *Erhiw, real_t *ErYiw, real_t *ErYilw)
{ // compute Physical、Visc_Heat、Visc_Diffu viscity in this function
	if (i >= bl.X_inner + bl.Bwidth_X)
		return;
	if (j >= bl.Y_inner + bl.Bwidth_Y)
		return;
	if (k >= bl.Z_inner + bl.Bwidth_Z)
		return;

	int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
	int id_m1 = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i - 1;
	int id_m2 = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i - 2;
	int id_p1 = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 1;
	int id_p2 = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 2;
	real_t *Ducy = Vde[ducy];
	real_t *Ducz = Vde[ducz];
	real_t *Dvcy = Vde[dvcy];
	real_t *Dwcz = Vde[dwcz];
	real_t _dl = bl._dx;

	MARCO_PREVISCFLUX();

	f_x = (_DF(2.0) * mue + lamada) * ((u[id_p1] - u[id])) * _dl;
	f_x += lamada * ((Dvcy[id_p1] + Dvcy[id]) + (Dwcz[id_p1] + Dwcz[id])) * _DF(0.5);
#if DIM_Y
	f_y = mue * ((v[id_p1] - v[id])) * _dl;
	f_y += mue * ((Ducy[id_p1] + Ducy[id])) * _DF(0.5);
#else
	f_y = _DF(0.0);
#endif // DIM_Y
#if DIM_Z
	f_z = mue * ((w[id_p1] - w[id])) * _dl;
	f_z += mue * ((Ducz[id_p1] + Ducz[id])) * _DF(0.5);
#else
	f_z = _DF(0.0);
#endif // DIM_Z

	u_hlf = ((u[id_p1] + u[id])) * _DF(0.5);
#if DIM_Y
	v_hlf = ((v[id_p1] + v[id])) * _DF(0.5);
#else
	v_hlf = _DF(0.0);
#endif // DIM_Y
#if DIM_Z
	w_hlf = ((w[id_p1] + w[id])) * _DF(0.5);
#else
	w_hlf = _DF(0.0);
#endif // DIM_Z

	MARCO_VISCFLUX();
}

extern void GetWallViscousFluxY(int i, int j, int k, Block bl, real_t *Flux_wall, real_t *viscosity_aver, real_t *thermal_conduct_aver, real_t *Dkm_aver,
								real_t *T, real_t *rho, real_t *hi, real_t *Yi, real_t *u, real_t *v, real_t *w, real_t *const *Vde,
								real_t *Yil_limiter, real_t *Diffu_limiter,
								real_t *ErvisFw, real_t *ErDimw, real_t *Erhiw, real_t *ErYiw, real_t *ErYilw)
{ // compute Physical、Visc_Heat、Visc_Diffu viscity in this function
	if (i >= bl.X_inner + bl.Bwidth_X)
		return;
	if (j >= bl.Y_inner + bl.Bwidth_Y)
		return;
	if (k >= bl.Z_inner + bl.Bwidth_Z)
		return;

	int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
	int id_m1 = bl.Xmax * bl.Ymax * k + bl.Xmax * (j - 1) + i;
	int id_m2 = bl.Xmax * bl.Ymax * k + bl.Xmax * (j - 2) + i;
	int id_p1 = bl.Xmax * bl.Ymax * k + bl.Xmax * (j + 1) + i;
	int id_p2 = bl.Xmax * bl.Ymax * k + bl.Xmax * (j + 2) + i;
	real_t *Dvcx = Vde[dvcx];
	real_t *Dvcz = Vde[dvcz];
	real_t *Ducx = Vde[ducx];
	real_t *Dwcz = Vde[dwcz];
	real_t _dl = bl._dy;

	MARCO_PREVISCFLUX();

#if DIM_X
	f_x = mue * ((u[id_p1] - u[id])) * _dl;
	f_x += mue * ((Dvcx[id_p1] + Dvcx[id])) * _DF(0.5);
#else
	f_x = _DF(0.0);
#endif // DIM_X
	f_y = (_DF(2.0) * mue + lamada) * ((v[id_p1] - v[id])) * _dl;
	f_y += lamada * ((Ducx[id_p1] + Ducx[id]) + (Dwcz[id_p1] + Dwcz[id])) * _DF(0.5);
#if DIM_Z
	f_z = mue * ((w[id_p1] - w[id])) * _dl;
	f_z += mue * ((Dvcz[id_p1] + Dvcz[id])) * _DF(0.5);
#else
	f_z = _DF(0.0);
#endif // DIM_Z

#if DIM_X
	u_hlf = ((u[id_p1] + u[id])) * _DF(0.5);
#else
	u_hlf = _DF(0.0);
#endif // DIM_X
	v_hlf = ((v[id_p1] + v[id])) * _DF(0.5);
#if DIM_Z
	w_hlf = ((w[id_p1] + w[id])) * _DF(0.5);
#else
	w_hlf = _DF(0.0);
#endif // DIMZ

	MARCO_VISCFLUX();
}

extern void GetWallViscousFluxZ(int i, int j, int k, Block bl, real_t *Flux_wall, real_t *viscosity_aver, real_t *thermal_conduct_aver, real_t *Dkm_aver,
								real_t *T, real_t *rho, real_t *hi, real_t *Yi, real_t *u, real_t *v, real_t *w, real_t *const *Vde,
								real_t *Yil_limiter, real_t *Diffu_limiter,
								real_t *ErvisFw, real_t *ErDimw, real_t *Erhiw, real_t *ErYiw, real_t *ErYilw)
{ // compute Physical、Visc_Heat、Visc_Diffu viscity in this function
	if (i >= bl.X_inner + bl.Bwidth_X)
		return;
	if (j >= bl.Y_inner + bl.Bwidth_Y)
		return;
	if (k >= bl.Z_inner + bl.Bwidth_Z)
		return;

	int id = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i;
	int id_m1 = bl.Xmax * bl.Ymax * (k - 1) + bl.Xmax * j + i;
	int id_m2 = bl.Xmax * bl.Ymax * (k - 2) + bl.Xmax * j + i;
	int id_p1 = bl.Xmax * bl.Ymax * (k + 1) + bl.Xmax * j + i;
	int id_p2 = bl.Xmax * bl.Ymax * (k + 2) + bl.Xmax * j + i;
	real_t *Dwcx = Vde[dwcx];
	real_t *Dwcy = Vde[dwcy];
	real_t *Ducx = Vde[ducx];
	real_t *Dvcy = Vde[dvcy];
	real_t _dl = bl._dz;

	MARCO_PREVISCFLUX();

#if DIM_X
	f_x = mue * ((u[id_p1] - u[id])) * _dl;
	f_x += mue * ((Dwcx[id_p1] + Dwcx[id])) * _DF(0.5);
#else
	f_x = _DF(0.0);
#endif // DIM_X
#if DIM_Y
	f_y = mue * ((v[id_p1] - v[id])) * _dl;
	f_y += mue * ((Dwcy[id_p1] + Dwcy[id])) * _DF(0.5);
#else
	f_y = _DF(0.0);
#endif
	f_z = (_DF(2.0) * mue + lamada) * ((w[id_p1] - w[id])) * _dl;
	f_z += lamada * ((Ducx[id_p1] + Ducx[id]) + (Dvcy[id_p1] + Dvcy[id])) * _DF(0.5);

#if DIM_X
	u_hlf = ((u[id_p1] + u[id])) * _DF(0.5);
#else
	u_hlf = _DF(0.0);
#endif // DIM_X
#if DIM_Y
	v_hlf = ((v[id_p1] + v[id])) * _DF(0.5);
#else
	v_hlf = _DF(0.0);
#endif // DIM_Y
	w_hlf = ((w[id_p1] + w[id])) * _DF(0.5);

	MARCO_VISCFLUX();
}

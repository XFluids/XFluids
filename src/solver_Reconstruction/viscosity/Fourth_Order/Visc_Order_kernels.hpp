#pragma once

#include "Flux_discrete.h"

/**
 * GetInnerCellCenterDerivativeKernel: to calculate velocity derivative at each grid point.
 * This is a Fourth-Order viscosity Flux discretization implementation.
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
	int id_m2_x = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i - 2;
	int id_p1_x = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 1;
	int id_p2_x = bl.Xmax * bl.Ymax * k + bl.Xmax * j + i + 2;

	Dmp[ducx] = (_DF(8.0) * (u[id_p1_x] - u[id_m1_x]) - (u[id_p2_x] - u[id_m2_x])) * _dx * _twle;
#if DIM_Y
	Dmp[dvcx] = (_DF(8.0) * (v[id_p1_x] - v[id_m1_x]) - (v[id_p2_x] - v[id_m2_x])) * _dx * _twle;
#endif // DIM_Y
#if DIM_Z
	Dmp[dwcx] = (_DF(8.0) * (w[id_p1_x] - w[id_m1_x]) - (w[id_p2_x] - w[id_m2_x])) * _dx * _twle;
#endif // DIM_Z
#endif // end DIM_X

#if DIM_Y
	real_t _dy = bl._dy;
	int id_m1_y = bl.Xmax * bl.Ymax * k + bl.Xmax * (j - 1) + i;
	int id_m2_y = bl.Xmax * bl.Ymax * k + bl.Xmax * (j - 2) + i;
	int id_p1_y = bl.Xmax * bl.Ymax * k + bl.Xmax * (j + 1) + i;
	int id_p2_y = bl.Xmax * bl.Ymax * k + bl.Xmax * (j + 2) + i;

#if DIM_X
	Dmp[ducy] = (_DF(8.0) * (u[id_p1_y] - u[id_m1_y]) - (u[id_p2_y] - u[id_m2_y])) * _dy * _twle;
#endif // DIM_X
	Dmp[dvcy] = (_DF(8.0) * (v[id_p1_y] - v[id_m1_y]) - (v[id_p2_y] - v[id_m2_y])) * _dy * _twle;
#if DIM_Z
	Dmp[dwcy] = (_DF(8.0) * (w[id_p1_y] - w[id_m1_y]) - (w[id_p2_y] - w[id_m2_y])) * _dy * _twle;
#endif // DIM_Z

#endif // end DIM_Y

#if DIM_Z
	real_t _dz = bl._dz;
	int id_m1_z = bl.Xmax * bl.Ymax * (k - 1) + bl.Xmax * j + i;
	int id_m2_z = bl.Xmax * bl.Ymax * (k - 2) + bl.Xmax * j + i;
	int id_p1_z = bl.Xmax * bl.Ymax * (k + 1) + bl.Xmax * j + i;
	int id_p2_z = bl.Xmax * bl.Ymax * (k + 2) + bl.Xmax * j + i;

#if DIM_X
	Dmp[ducz] = (_DF(8.0) * (u[id_p1_z] - u[id_m1_z]) - (u[id_p2_z] - u[id_m2_z])) * _dz * _twle;
#endif // DIM_X
#if DIM_Y
	Dmp[dvcz] = (_DF(8.0) * (v[id_p1_z] - v[id_m1_z]) - (v[id_p2_z] - v[id_m2_z])) * _dz * _twle;
#endif // DIM_Y
	Dmp[dwcz] = (_DF(8.0) * (w[id_p1_z] - w[id_m1_z]) - (w[id_p2_z] - w[id_m2_z])) * _dz * _twle;

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
	// real_t F_x_wall_v[Emax];
	// real_t mue = (_DF(9.0) * (viscosity_aver[id_p1] + viscosity_aver[id]) - (viscosity_aver[id_p2] + viscosity_aver[id_m1])) / _DF(16.0);
	// real_t lamada = -_DF(2.0) * _OT * mue;
	// real_t f_x, f_y, f_z;
	// real_t u_hlf, v_hlf, w_hlf;

	f_x = (_DF(2.0) * mue + lamada) * (_DF(27.0) * (u[id_p1] - u[id]) - (u[id_p2] - u[id_m1])) * _dl * _twfr;
	f_x += lamada * (_DF(9.0) * (Dvcy[id_p1] + Dvcy[id]) - (Dvcy[id_p2] + Dvcy[id_m1]) + _DF(9.0) * (Dwcz[id_p1] + Dwcz[id]) - (Dwcz[id_p2] + Dwcz[id_m1])) * _sxtn;
#if DIM_Y
	f_y = mue * (_DF(27.0) * (v[id_p1] - v[id]) - (v[id_p2] - v[id_m1])) * _dl * _twfr;
	f_y += mue * (_DF(9.0) * (Ducy[id_p1] + Ducy[id]) - (Ducy[id_p2] + Ducy[id_m1])) * _sxtn;
#else
	f_y = _DF(0.0);
#endif // DIM_Y
#if DIM_Z
	f_z = mue * (_DF(27.0) * (w[id_p1] - w[id]) - (w[id_p2] - w[id_m1])) * _dl * _twfr;
	f_z += mue * (_DF(9.0) * (Ducz[id_p1] + Ducz[id]) - (Ducz[id_p2] + Ducz[id_m1])) * _sxtn;
#else
	f_z = _DF(0.0);
#endif // DIM_Z

	u_hlf = (_DF(9.0) * (u[id_p1] + u[id]) - (u[id_p2] + u[id_m1])) * _sxtn;
#if DIM_Y
	v_hlf = (_DF(9.0) * (v[id_p1] + v[id]) - (v[id_p2] + v[id_m1])) * _sxtn;
#else
	v_hlf = _DF(0.0);
#endif // DIM_Y
#if DIM_Z
	w_hlf = (_DF(9.0) * (w[id_p1] + w[id]) - (w[id_p2] + w[id_m1])) * _sxtn;
#else
	w_hlf = _DF(0.0);
#endif // DIM_Z

	MARCO_VISCFLUX();

	//     F_x_wall_v[0] = _DF(0.0);
	//     F_x_wall_v[1] = f_x;
	//     F_x_wall_v[2] = f_y;
	//     F_x_wall_v[3] = f_z;
	//     F_x_wall_v[4] = f_x * u_hlf + f_y * v_hlf + f_z * w_hlf;

	// #if Visc_Heat // Fourier thermal conductivity; // thermal conductivity at wall
	//     real_t kk = (_DF(9.0) * (thermal_conduct_aver[id_p1] + thermal_conduct_aver[id]) - (thermal_conduct_aver[id_p2] + thermal_conduct_aver[id_m1])) / _DF(16.0);
	//     kk *= (_DF(27.0) * (T[id_p1] - T[id]) - (T[id_p2] - T[id_m1])) / dx / _DF(24.0);                                                // temperature gradient at wall
	//     F_x_wall_v[4] += kk;                                                                                                            // Equation (32) or Equation (10)
	// #endif                                                                                                                              // end Visc_Heat
	// #if Visc_Diffu                                                                                                                        // energy fiffusion depends on mass diffusion
	//     real_t rho_wall = (_DF(9.0) * (rho[id_p1] + rho[id]) - (rho[id_p2] + rho[id_m1])) / _DF(16.0);
	//     real_t hi_wall[NUM_SPECIES], Dim_wall[NUM_SPECIES], Yix_wall[NUM_SPECIES], Yi_wall[NUM_SPECIES];
	//     for (int l = 0; l < NUM_SPECIES; l++)
	//     {
	//             hi_wall[l] = (_DF(9.0) * (hi[l + NUM_SPECIES * id_p1] + hi[l + NUM_SPECIES * id]) - (hi[l + NUM_SPECIES * id_p2] + hi[l + NUM_SPECIES * id_m1])) / _DF(16.0);
	//             Dim_wall[l] = (_DF(9.0) * (Dkm_aver[l + NUM_SPECIES * id_p1] + Dkm_aver[l + NUM_SPECIES * id]) - (Dkm_aver[l + NUM_SPECIES * id_p2] + Dkm_aver[l + NUM_SPECIES * id_m1])) / _DF(16.0);
	// #ifdef COP
	//             Yi_wall[l] = (_DF(9.0) * (Yi[l][id_p1] + Yi[l][id]) - (Yi[l][id_p2] + Yi[l][id_m1])) / _DF(16.0);
	//             Yix_wall[l] = (_DF(27.0) * (Yi[l][id_p1] - Yi[l][id]) - (Yi[l][id_p2] - Yi[l][id_m1])) / dx / _DF(24.0); // temperature gradient at wall
	// #else
	//             Yix_wall[l] = _DF(0.0);
	// #endif // end COP
	//     }
	// #if Visc_Heat
	//     for (int l = 0; l < NUM_SPECIES; l++)
	//             F_x_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yix_wall[l];
	// #endif     // end Visc_Heat
	// #ifdef COP // visc flux for cop equations
	//     real_t CorrectTermX = _DF(0.0);
	//     for (int l = 0; l < NUM_SPECIES; l++)
	//             CorrectTermX += Dim_wall[l] * Yix_wall[l];
	//     CorrectTermX *= rho_wall;
	//     // ADD Correction Term in X-direction
	//     for (int p = 5; p < Emax; p++)
	//             F_x_wall_v[p] = rho_wall * Dim_wall[p - 5] * Yix_wall[p - 5] - Yi_wall[p - 5] * CorrectTermX;
	// #endif // end COP
	// #endif // end Visc_Diffu
	//     for (size_t n = 0; n < Emax; n++)
	//     { // add viscous flux to fluxwall
	//             FluxFw[n + Emax * id] -= F_x_wall_v[n];
	//     }
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
	// // mue at wall
	// real_t F_y_wall_v[Emax];
	// real_t mue = (_DF(9.0) * (viscosity_aver[id_p1] + viscosity_aver[id]) - (viscosity_aver[id_p2] + viscosity_aver[id_m1])) / _DF(16.0);
	// real_t lamada = -_DF(2.0) * _OT * mue;
	// real_t f_x, f_y, f_z;
	// real_t u_hlf, v_hlf, w_hlf;

#if DIM_X
	f_x = mue * (_DF(27.0) * (u[id_p1] - u[id]) - (u[id_p2] - u[id_m1])) * _dl * _twfr;
	f_x += mue * (_DF(9.0) * (Dvcx[id_p1] + Dvcx[id]) - (Dvcx[id_p2] + Dvcx[id_m1])) * _sxtn;
#else
	f_x = _DF(0.0);
#endif // DIM_X
	f_y = (_DF(2.0) * mue + lamada) * (_DF(27.0) * (v[id_p1] - v[id]) - (v[id_p2] - v[id_m1])) * _dl * _twfr;
	f_y += lamada * (_DF(9.0) * (Ducx[id_p1] + Ducx[id]) - (Ducx[id_p2] + Ducx[id_m1]) + _DF(9.0) * (Dwcz[id_p1] + Dwcz[id]) - (Dwcz[id_p2] + Dwcz[id_m1])) * _sxtn;
#if DIM_Z
	f_z = mue * (_DF(27.0) * (w[id_p1] - w[id]) - (w[id_p2] - w[id_m1])) * _dl * _twfr;
	f_z += mue * (_DF(9.0) * (Dvcz[id_p1] + Dvcz[id]) - (Dvcz[id_p2] + Dvcz[id_m1])) * _sxtn;
#else
	f_z = _DF(0.0);
#endif // DIM_Z

#if DIM_X
	u_hlf = (_DF(9.0) * (u[id_p1] + u[id]) - (u[id_p2] + u[id_m1])) * _sxtn;
#else
	u_hlf = _DF(0.0);
#endif // DIM_X
	v_hlf = (_DF(9.0) * (v[id_p1] + v[id]) - (v[id_p2] + v[id_m1])) * _sxtn;
#if DIM_Z
	w_hlf = (_DF(9.0) * (w[id_p1] + w[id]) - (w[id_p2] + w[id_m1])) * _sxtn;
#else
	w_hlf = _DF(0.0);
#endif // DIMZ
	MARCO_VISCFLUX();

	//     F_y_wall_v[0] = _DF(0.0);
	//     F_y_wall_v[1] = f_x;
	//     F_y_wall_v[2] = f_y;
	//     F_y_wall_v[3] = f_z;
	//     F_y_wall_v[4] = f_x * u_hlf + f_y * v_hlf + f_z * w_hlf;

	// #if Visc_Heat    // Fourier thermal conductivity
	//     real_t kk = (_DF(9.0) * (thermal_conduct_aver[id_p1] + thermal_conduct_aver[id]) - (thermal_conduct_aver[id_p2] + thermal_conduct_aver[id_m1])) / _DF(16.0); // thermal conductivity at wall
	//     kk *= (_DF(27.0) * (T[id_p1] - T[id]) - (T[id_p2] - T[id_m1])) / dy / _DF(24.0);                                                // temperature gradient at wall
	//     F_y_wall_v[4] += kk;                                                                                                            // Equation (32) or Equation (10)
	// #endif                                                                                                                              // end Visc_Heat
	// #if Visc_Diffu                                                                                                                        // energy fiffusion depends on mass diffusion
	//     real_t rho_wall = (_DF(9.0) * (rho[id_p1] + rho[id]) - (rho[id_p2] + rho[id_m1])) / _DF(16.0);
	//     real_t hi_wall[NUM_SPECIES], Dim_wall[NUM_SPECIES], Yiy_wall[NUM_SPECIES], Yi_wall[NUM_SPECIES];
	//     for (int l = 0; l < NUM_SPECIES; l++)
	//     {
	//             hi_wall[l] = (_DF(9.0) * (hi[l + NUM_SPECIES * id_p1] + hi[l + NUM_SPECIES * id]) - (hi[l + NUM_SPECIES * id_p2] + hi[l + NUM_SPECIES * id_m1])) / _DF(16.0);
	//             Dim_wall[l] = (_DF(9.0) * (Dkm_aver[l + NUM_SPECIES * id_p1] + Dkm_aver[l + NUM_SPECIES * id]) - (Dkm_aver[l + NUM_SPECIES * id_p2] + Dkm_aver[l + NUM_SPECIES * id_m1])) / _DF(16.0);
	// #ifdef COP
	//             Yi_wall[l] = (_DF(9.0) * (Yi[l][id_p1] + Yi[l][id]) - (Yi[l][id_p2] + Yi[l][id_m1])) / _DF(16.0);
	//             Yiy_wall[l] = (_DF(27.0) * (Yi[l][id_p1] - Yi[l][id]) - (Yi[l][id_p2] - Yi[l][id_m1])) / dy / _DF(24.0); // temperature gradient at wal
	// #else
	//             Yiy_wall[l] = _DF(0.0);
	// #endif // end COP
	//     }
	// #if Visc_Heat
	//     for (int l = 0; l < NUM_SPECIES; l++)
	//             F_y_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yiy_wall[l];
	// #endif     // end Visc_Heat
	// #ifdef COP // visc flux for cop equations
	//     real_t CorrectTermY = _DF(0.0);
	//     for (int l = 0; l < NUM_SPECIES; l++)
	//             CorrectTermY += Dim_wall[l] * Yiy_wall[l];
	//     CorrectTermY *= rho_wall;
	//     // ADD Correction Term in X-direction
	//     for (int p = 5; p < Emax; p++)
	//             F_y_wall_v[p] = rho_wall * Dim_wall[p - 5] * Yiy_wall[p - 5] - Yi_wall[p - 5] * CorrectTermY;
	// #endif // end COP
	// #endif // end Visc_Diffu
	//     for (size_t n = 0; n < Emax; n++)
	//     { // add viscous flux to fluxwall
	//             FluxGw[n + Emax * id] -= F_y_wall_v[n];
	//     }
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
	// real_t F_z_wall_v[Emax];
	// real_t mue = (_DF(9.0) * (viscosity_aver[id_p1] + viscosity_aver[id]) - (viscosity_aver[id_p2] + viscosity_aver[id_m1])) / _DF(16.0);
	// real_t lamada = -_DF(2.0) * _OT * mue;
	// real_t f_x, f_y, f_z;
	// real_t u_hlf, v_hlf, w_hlf;

#if DIM_X
	f_x = mue * (_DF(27.0) * (u[id_p1] - u[id]) - (u[id_p2] - u[id_m1])) * _dl * _twfr;
	f_x += mue * (_DF(9.0) * (Dwcx[id_p1] + Dwcx[id]) - (Dwcx[id_p2] + Dwcx[id_m1])) * _sxtn;
#else
	f_x = _DF(0.0);
#endif // DIM_X
#if DIM_Y
	f_y = mue * (_DF(27.0) * (v[id_p1] - v[id]) - (v[id_p2] - v[id_m1])) * _dl * _twfr;
	f_y += mue * (_DF(9.0) * (Dwcy[id_p1] + Dwcy[id]) - (Dwcy[id_p2] + Dwcy[id_m1])) * _sxtn;
#else
	f_y = _DF(0.0);
#endif
	f_z = (_DF(2.0) * mue + lamada) * (_DF(27.0) * (w[id_p1] - w[id]) - (w[id_p2] - w[id_m1])) * _dl * _twfr;
	f_z += lamada * (_DF(9.0) * (Ducx[id_p1] + Ducx[id]) - (Ducx[id_p2] + Ducx[id_m1]) + _DF(9.0) * (Dvcy[id_p1] + Dvcy[id]) - (Dvcy[id_p2] + Dvcy[id_m1])) * _sxtn;

#if DIM_X
	u_hlf = (_DF(9.0) * (u[id_p1] + u[id]) - (u[id_p2] + u[id_m1])) * _sxtn;
#else
	u_hlf = _DF(0.0);
#endif // DIM_X
#if DIM_Y
	v_hlf = (_DF(9.0) * (v[id_p1] + v[id]) - (v[id_p2] + v[id_m1])) * _sxtn;
#else
	v_hlf = _DF(0.0);
#endif // DIM_Y
	w_hlf = (_DF(9.0) * (w[id_p1] + w[id]) - (w[id_p2] + w[id_m1])) * _sxtn;

	MARCO_VISCFLUX();

	//     F_z_wall_v[0] = _DF(0.0);
	//     F_z_wall_v[1] = f_x;
	//     F_z_wall_v[2] = f_y;
	//     F_z_wall_v[3] = f_z;
	//     F_z_wall_v[4] = f_x * u_hlf + f_y * v_hlf + f_z * w_hlf;

	// #if Visc_Heat    // Fourier thermal conductivity
	//     real_t kk = (_DF(9.0) * (thermal_conduct_aver[id_p1] + thermal_conduct_aver[id]) - (thermal_conduct_aver[id_p2] + thermal_conduct_aver[id_m1])) / _DF(16.0); // thermal conductivity at wall
	//     kk *= (_DF(27.0) * (T[id_p1] - T[id]) - (T[id_p2] - T[id_m1])) / dz / _DF(24.0);                                                // temperature gradient at wall
	//     F_z_wall_v[4] += kk;                                                                                                            // Equation (32) or Equation (10)
	// #endif                                                                                                                              // end Visc_Heat
	// #if Visc_Diffu                                                                                                                        // energy fiffusion depends on mass diffusion
	//     real_t rho_wall = (_DF(9.0) * (rho[id_p1] + rho[id]) - (rho[id_p2] + rho[id_m1])) / _DF(16.0);
	//     real_t hi_wall[NUM_SPECIES], Dim_wall[NUM_SPECIES], Yiz_wall[NUM_SPECIES], Yi_wall[NUM_SPECIES];
	//     for (int l = 0; l < NUM_SPECIES; l++)
	//     {
	//             hi_wall[l] = (_DF(9.0) * (hi[l + NUM_SPECIES * id_p1] + hi[l + NUM_SPECIES * id]) - (hi[l + NUM_SPECIES * id_p2] + hi[l + NUM_SPECIES * id_m1])) / _DF(16.0);
	//             Dim_wall[l] = (_DF(9.0) * (Dkm_aver[l + NUM_SPECIES * id_p1] + Dkm_aver[l + NUM_SPECIES * id]) - (Dkm_aver[l + NUM_SPECIES * id_p2] + Dkm_aver[l + NUM_SPECIES * id_m1])) / _DF(16.0);
	// #ifdef COP
	//             Yi_wall[l] = (_DF(9.0) * (Yi[l][id_p1] + Yi[l][id]) - (Yi[l][id_p2] + Yi[l][id_m1])) / _DF(16.0);
	//             Yiz_wall[l] = (_DF(27.0) * (Yi[l][id_p1] - Yi[l][id]) - (Yi[l][id_p2] - Yi[l][id_m1])) / dz / _DF(24.0); // temperature gradient at wall
	// #else
	//             Yiz_wall[l] = _DF(0.0);
	// #endif // end COP
	//     }
	// #if Visc_Heat
	//     for (int l = 0; l < NUM_SPECIES; l++)
	//             F_z_wall_v[4] += rho_wall * hi_wall[l] * Dim_wall[l] * Yiz_wall[l];
	// #endif     // end Visc_Heat
	// #ifdef COP // visc flux for cop equations
	//     real_t CorrectTermZ = _DF(0.0);
	//     for (int l = 0; l < NUM_SPECIES; l++)
	//             CorrectTermZ += Dim_wall[l] * Yiz_wall[l];
	//     CorrectTermZ *= rho_wall;
	//     // ADD Correction Term in X-direction
	//     for (int p = 5; p < Emax; p++)
	//             F_z_wall_v[p] = rho_wall * Dim_wall[p - 5] * Yiz_wall[p - 5] - Yi_wall[p - 5] * CorrectTermZ;
	// #endif // end COP
	// #endif // end Visc_Diffu
	//     for (size_t n = 0; n < Emax; n++)
	//     { // add viscous flux to fluxwall
	//             FluxHw[n + Emax * id] -= F_z_wall_v[n];
	//     }
}

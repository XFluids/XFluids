#pragma once

#include "global_setup.h"
#include "../read_ini/setupini.h"
#include "../include/sycl_devices.hpp"

#define MARCO_COPC2()                                                                                                                     \
	real_t _yi[MAX_SPECIES], z[MAX_SPECIES] = {_DF(0.0)}, b1 = _DF(0.0), b3 = _DF(0.0), _k = _DF(0.0), _ht = _DF(0.0), Gamma0 = _DF(1.4); \
	real_t c2 = ReconstructSoundSpeed(thermal, id_l, id_r, D, D1, _rho, _P, rho, u, v, w, y, p, T, H,                                     \
									  _yi, z, b1, b3, _k, _ht, Gamma0);                                                                   \
	real_t _c = sycl::sqrt(c2);

// #define MARCO_COPC2()                                                                                                                                                                                                                              \
// 	real_t _yi[MAX_SPECIES], z[MAX_SPECIES] = {_DF(0.0)}, b1 = _DF(0.0), b3 = _DF(0.0); /*_hi[NUM_SPECIES],*/                                                                                                                                      \
// 	real_t hi_l[MAX_SPECIES], hi_r[MAX_SPECIES];                                                                                                                                                                                                   \
// 	for (size_t n = 0; n < NUM_SPECIES; n++)                                                                                                                                                                                                       \
// 	{                                                                                                                                                                                                                                              \
// 		hi_l[n] = get_Enthalpy(thermal.Hia, thermal.Hib, T[id_l], thermal.Ri[n], n);                                                                                                                                                               \
// 		hi_r[n] = get_Enthalpy(thermal.Hia, thermal.Hib, T[id_r], thermal.Ri[n], n);                                                                                                                                                               \
// 	}                                                                                                                                                                                                                                              \
// 	real_t *yi_l = &(y[NUM_SPECIES * id_l]), *yi_r = &(y[NUM_SPECIES * id_r]);                                                                                                                                                                     \
// 	for (size_t ii = 0; ii < NUM_SPECIES; ii++)                                                                                                                                                                                                    \
// 	{                                                                                                                                                                                                                                              \
// 		_yi[ii] = (yi_l[ii] + D * yi_r[ii]) * D1;                                                                                                                                                                                                  \
// 		/*_hi[ii] = (hi_l[ii] + D * hi_r[ii]) * D1;*/                                                                                                                                                                                              \
// 	}                                                                                                                                                                                                                                              \
// 	real_t _dpdrhoi[NUM_COP], drhoi[NUM_COP];                                                                                                                                                                                                      \
// 	real_t gamma_l = get_CopGamma(thermal, yi_l, T[id_l]);                                                                                                                                                                                         \
// 	real_t gamma_r = get_CopGamma(thermal, yi_r, T[id_r]);                                                                                                                                                                                         \
// 	real_t Gamma0 = get_RoeAverage(gamma_l, gamma_r, D, D1);                                                                                                                                                                                       \
// 	real_t q2_l = u[id_l] * u[id_l] + v[id_l] * v[id_l] + w[id_l] * w[id_l];                                                                                                                                                                       \
// 	real_t q2_r = u[id_r] * u[id_r] + v[id_r] * v[id_r] + w[id_r] * w[id_r];                                                                                                                                                                       \
// 	real_t e_l = H[id_l] - _DF(0.5) * q2_l - p[id_l] / rho[id_l];                                                                                                                                                                                  \
// 	real_t e_r = H[id_r] - _DF(0.5) * q2_r - p[id_r] / rho[id_r];                                                                                                                                                                                  \
// 	real_t Cp_l = get_CopCp(thermal, yi_l, T[id_l]);                                                                                                                                                                                               \
// 	real_t Cp_r = get_CopCp(thermal, yi_r, T[id_r]);                                                                                                                                                                                               \
// 	real_t R_l = get_CopR(thermal._Wi, yi_l);                                                                                                                                                                                                      \
// 	real_t R_r = get_CopR(thermal._Wi, yi_r);                                                                                                                                                                                                      \
// 	real_t _dpdrho = get_RoeAverage(get_DpDrho(hi_l[NUM_COP], thermal.Ri[NUM_COP], q2_l, Cp_l, R_l, T[id_l], e_l, gamma_l),                                                                                                                        \
// 									get_DpDrho(hi_r[NUM_COP], thermal.Ri[NUM_COP], q2_r, Cp_r, R_r, T[id_r], e_r, gamma_r), D, D1);                                                                                                                \
// 	for (size_t nn = 0; nn < NUM_COP; nn++)                                                                                                                                                                                                        \
// 	{                                                                                                                                                                                                                                              \
// 		_dpdrhoi[nn] = get_RoeAverage(get_DpDrhoi(hi_l[nn], thermal.Ri[nn], hi_l[NUM_COP], thermal.Ri[NUM_COP], T[id_l], Cp_l, R_l, gamma_l),                                                                                                      \
// 									  get_DpDrhoi(hi_r[nn], thermal.Ri[nn], hi_r[NUM_COP], thermal.Ri[NUM_COP], T[id_r], Cp_r, R_r, gamma_r), D, D1);                                                                                              \
// 		drhoi[nn] = rho[id_r] * yi_r[nn] - rho[id_l] * yi_l[nn];                                                                                                                                                                                   \
// 	}                                                                                                                                                                                                                                              \
// 	real_t _prho = get_RoeAverage(p[id_l] / rho[id_l], p[id_r] / rho[id_r], D, D1) + _DF(0.5) * D * D1 * D1 * ((u[id_r] - u[id_l]) * (u[id_r] - u[id_l]) + (v[id_r] - v[id_l]) * (v[id_r] - v[id_l]) + (w[id_r] - w[id_l]) * (w[id_r] - w[id_l])); \
// 	real_t _dpdE = get_RoeAverage(gamma_l - _DF(1.0), gamma_r - _DF(1.0), D, D1);                                                                                                                                                                  \
// 	real_t _dpde = get_RoeAverage((gamma_l - _DF(1.0)) * rho[id_l], (gamma_r - _DF(1.0)) * rho[id_r], D, D1);                                                                                                                                      \
// 	real_t c2 = SoundSpeedMultiSpecies(z, b1, b3, _yi, _dpdrhoi, drhoi, _dpdrho, _dpde, _dpdE, _prho, p[id_r] - p[id_l], rho[id_r] - rho[id_l], e_r - e_l, _rho);                                                                                  \
// 	/*add support while c2<0 use c2 Refed in https://doi.org/10.1006/jcph.1996.5622 */                                                                                                                                                             \
// 	real_t c2w = sycl::step(c2, _DF(0.0)); /*sycl::step(a, b)： return 0 while a>b，return 1 while a<=b*/                                                                                                                                        \
// 	c2 = Gamma0 * _P * _rho * c2w + (_DF(1.0) - c2w) * c2;                                                                                                                                                                                         \
// 	MARCO_ERROR_OUT();

// =======================================================
//    get c2 #ifdef COP inside Reconstructflux
#define MARCO_COPC2_ROB()                                                                                                                       \
	real_t _yi[NUM_SPECIES], Ri[NUM_SPECIES], _hi[NUM_SPECIES], hi_l[NUM_SPECIES], hi_r[NUM_SPECIES], z[NUM_COP], b1 = _DF(0.0), b3 = _DF(0.0); \
	for (size_t n = 0; n < NUM_SPECIES; n++)                                                                                                    \
	{                                                                                                                                           \
		hi_l[n] = get_Enthalpy(thermal.Hia, thermal.Hib, T[id_l], thermal.Ri[n], n);                                                            \
		hi_r[n] = get_Enthalpy(thermal.Hia, thermal.Hib, T[id_r], thermal.Ri[n], n);                                                            \
		Ri[n] = Ru * thermal._Wi[n];                                                                                                            \
	}                                                                                                                                           \
	real_t *yi_l = &(y[NUM_SPECIES * id_l]), *yi_r = &(y[NUM_SPECIES * id_r]);                                                                  \
	for (size_t ii = 0; ii < NUM_SPECIES; ii++)                                                                                                 \
	{                                                                                                                                           \
		_yi[ii] = (yi_l[ii] + D * yi_r[ii]) * D1;                                                                                               \
		_hi[ii] = (hi_l[ii] + D * hi_r[ii]) * D1;                                                                                               \
	}                                                                                                                                           \
	real_t gamma_l = get_CopGamma(thermal, yi_l, T[id_l]);                                                                                      \
	real_t gamma_r = get_CopGamma(thermal, yi_r, T[id_r]);                                                                                      \
	real_t Cp_l = get_CopCp(thermal, yi_l, T[id_l]);                                                                                            \
	real_t Cp_r = get_CopCp(thermal, yi_r, T[id_r]);                                                                                            \
	real_t R_l = get_CopR(thermal._Wi, yi_l);                                                                                                   \
	real_t R_r = get_CopR(thermal._Wi, yi_r);                                                                                                   \
	real_t Gamma0 = get_RoeAverage(gamma_l, gamma_r, D, D1);                                                                                    \
	real_t _R = get_RoeAverage(R_l, R_r, D, D1);                                                                                                \
	real_t _Cp = get_RoeAverage(Cp_l, Cp_r, D, D1);                                                                                             \
	real_t _T = get_RoeAverage(T[id_l], T[id_r], D, D1);                                                                                        \
	real_t c2 = get_CopC2(z, b1, b3, Ri, _yi, _hi, Gamma0, _R, _Cp, _T);                                                                        \
	MARCO_ERROR_OUT();

// =======================================================
//    get c2 #else COP
#define MARCO_NOCOPC2()                                                                                            \
	real_t _yi[NUM_SPECIES] = {_DF(1.0)}, b3 = _DF(0.0), z[] = {_DF(0.0)};                                         \
	real_t Gamma0 = NCOP_Gamma;                                                                                    \
	real_t c2 = Gamma0 * _P / _rho; /*(_H - _DF(0.5) * (_u * _u + _v * _v + _w * _w)); // out from RoeAverage_x */ \
	real_t b1 = (Gamma0 - _DF(1.0)) / c2;
// #define MARCO_NOCOPC2()                                                                                                  \
//     real_t yi_l[NUM_SPECIES] = {_DF(1.0)}, yi_r[NUM_SPECIES] = {_DF(1.0)}, _yi[] = {_DF(1.0)}, b3 = _DF(0.0), z[] = {0}; \
//     real_t gamma_l = get_CopGamma(thermal, yi_l, T[id_l]);                                                               \
//     real_t gamma_r = get_CopGamma(thermal, yi_r, T[id_r]);                                                               \
//     real_t Gamma0 = get_RoeAverage(gamma_l, gamma_r, D, D1);                                                             \
//     real_t c2 = Gamma0 * _P / _rho; /*(_H - _DF(0.5) * (_u * _u + _v * _v + _w * _w)); // out from RoeAverage_x */       \
//     real_t b1 = (Gamma0 - _DF(1.0)) / c2;

// =======================================================
//    Pre get eigen_martix
#define MARCO_PREEIGEN()                      \
	real_t q2 = _u * _u + _v * _v + _w * _w;  \
	real_t _c = sycl::sqrt(c2);               \
	real_t b2 = _DF(1.0) + b1 * q2 - b1 * _H; \
	real_t _c1 = _DF(1.0) / _c;

// =======================================================
//    Caculate flux_wall
#if 0 == EIGEN_ALLOC
// RoeAverage_Left and RoeAverage_Right for each DIR
#define MARCO_ROEAVERAGE_LEFTX \
	RoeAverageLeft_x(n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);
#define MARCO_ROEAVERAGE_LEFTY \
	RoeAverageLeft_y(n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);
#define MARCO_ROEAVERAGE_LEFTZ \
	RoeAverageLeft_z(n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);

#define MARCO_ROEAVERAGE_RIGHTX \
	RoeAverageRight_x(n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);
#define MARCO_ROEAVERAGE_RIGHTY \
	RoeAverageRight_y(n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);
#define MARCO_ROEAVERAGE_RIGHTZ \
	RoeAverageRight_z(n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);

// WENO 7 // used by MARCO_FLUXWALL_WENO7(i + m, j, k, i + m - stencil_P, j, k); in x
#define MARCO_FLUXWALL_WENO7(MARCO_ROE_LEFT, MARCO_ROE_RIGHT, _i_1, _j_1, _k_1, _i_2, _j_2, _k_2)                                       \
	real_t uf[10], ff[10], pp[10], mm[10], _p[Emax][Emax], f_flux, eigen_lr[Emax], eigen_value, artificial_viscosity;                   \
	for (int n = 0; n < Emax; n++)                                                                                                      \
	{                                                                                                                                   \
		real_t eigen_local_max = _DF(0.0);                                                                                              \
		MARCO_ROE_LEFT; /* eigen_r actually */                                                                                          \
		/* RoeAverageLeft_x(n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); get eigen_l */                 \
		eigen_local_max = eigen_value;                                                                                                  \
		real_t lambda_l = eigen_local[Emax * id_l + n];                                                                                 \
		real_t lambda_r = eigen_local[Emax * id_r + n];                                                                                 \
		if (lambda_l * lambda_r < _DF(0.0))                                                                                             \
		{                                                                                                                               \
			for (int m = -stencil_P; m < stencil_size - stencil_P; m++)                                                                 \
			{ /* int _i_1 = i + m, _j_1 = j, _k_1 = k; Xmax * Ymax * k + Xmax * j + i + m */                                            \
				int id_local_1 = Xmax * Ymax * (_k_1) + Xmax * (_j_1) + (_i_1);                                                         \
				eigen_local_max = sycl::max(eigen_local_max, sycl::fabs(eigen_local[Emax * id_local_1 + n])); /* local lax-friedrichs*/ \
			}                                                                                                                           \
		}                                                                                                                               \
		artificial_viscosity = Roe_type * eigen_value + LLF_type * eigen_local_max + GLF_type * eigen_block[n];                         \
		for (size_t m = 0; m < stencil_size; m++)                                                                                       \
		{ /* int _i_2 = i + m, _j_2 = j, _k_2 = k; Xmax * Ymax * k + Xmax * j + m + i - stencil_P */                                    \
			int id_local_2 = Xmax * Ymax * (_k_2) + Xmax * (_j_2) + (_i_2);                                                             \
			uf[m] = _DF(0.0);                                                                                                           \
			ff[m] = _DF(0.0);                                                                                                           \
			for (int n1 = 0; n1 < Emax; n1++)                                                                                           \
			{                                                                                                                           \
				uf[m] = uf[m] + UI[Emax * id_local_2 + n1] * eigen_lr[n1];                                                              \
				ff[m] = ff[m] + Fl[Emax * id_local_2 + n1] * eigen_lr[n1];                                                              \
			} /* for local speed*/                                                                                                      \
			pp[m] = _DF(0.5) * (ff[m] + artificial_viscosity * uf[m]);                                                                  \
			mm[m] = _DF(0.5) * (ff[m] - artificial_viscosity * uf[m]);                                                                  \
		}                                                                                                                               \
		/* calculate the scalar numerical flux at x direction*/                                                                         \
		f_flux = weno7_P(&pp[stencil_P], dl) + weno7_M(&mm[stencil_P], dl); /* get Fp*/                                                 \
		MARCO_ROE_RIGHT;													/* eigen_r actually */                                      \
		/* RoeAverageRight_x(n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); get eigen_r */                             \
		for (int n1 = 0; n1 < Emax; n1++)                                                                                               \
		{                                                                                                                               \
			_p[n][n1] = f_flux * eigen_lr[n1];                                                                                          \
		}                                                                                                                               \
	}                                                                                                                                   \
	for (int n = 0; n < Emax; n++)                                                                                                      \
	{ /* reconstruction the F-flux terms*/                                                                                              \
		real_t fluxl = _DF(0.0);                                                                                                        \
		for (int n1 = 0; n1 < Emax; n1++)                                                                                               \
		{                                                                                                                               \
			fluxl += _p[n1][n];                                                                                                         \
		}                                                                                                                               \
		Fwall[Emax * id_l + n] = fluxl;                                                                                                 \
	}

// WENO 5 //used by: MARCO_FLUXWALL_WENO5(i + m, j, k, i + m, j, k);
#define MARCO_FLUXWALL_WENO5(MARCO_ROE_LEFT, MARCO_ROE_RIGHT, _i_1, _j_1, _k_1, _i_2, _j_2, _k_2)                                   \
	real_t uf[10], ff[10], pp[10], mm[10], f_flux, _p[Emax][Emax], eigen_lr[Emax], eigen_value, artificial_viscosity;               \
	for (int n = 0; n < Emax; n++)                                                                                                  \
	{                                                                                                                               \
		real_t eigen_local_max = _DF(0.0);                                                                                          \
		MARCO_ROE_LEFT;                                                                                                             \
		/* RoeAverageLeft_x(n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); get eigen_l*/              \
		/* RoeAverageLeft_y(n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);*/                          \
		/* RoeAverageLeft_z(n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);*/                          \
		for (int m = -stencil_P; m < stencil_size - stencil_P; m++)                                                                 \
		{ /*int _i_1 = i + m, _j_1 = j, _k_1 = k; Xmax * Ymax * k + Xmax * j + i + m*/                                              \
			int id_local_1 = Xmax * Ymax * (_k_1) + Xmax * (_j_1) + (_i_1);                                                         \
			eigen_local_max = sycl::max(eigen_local_max, sycl::fabs(eigen_local[Emax * id_local_1 + n])); /* local lax-friedrichs*/ \
		}                                                                                                                           \
		artificial_viscosity = Roe_type * eigen_value + LLF_type * eigen_local_max + GLF_type * eigen_block[n];                     \
		for (int m = -3; m <= 4; m++)                                                                                               \
		{                                                                                                                           \
			/* int _i_2 = i + m, _j_2 = j, _k_2 = k; Xmax * Ymax * k + Xmax * j + m + i; 3rd oder and can be modified */            \
			int id_local = Xmax * Ymax * (_k_2) + Xmax * (_j_2) + (_i_2);                                                           \
			uf[m + 3] = _DF(0.0);                                                                                                   \
			ff[m + 3] = _DF(0.0);                                                                                                   \
			for (int n1 = 0; n1 < Emax; n1++)                                                                                       \
			{                                                                                                                       \
				uf[m + 3] = uf[m + 3] + UI[Emax * id_local + n1] * eigen_lr[n1]; /* eigen_l actually */                             \
				ff[m + 3] = ff[m + 3] + Fl[Emax * id_local + n1] * eigen_lr[n1];                                                    \
			} /*  for local speed*/                                                                                                 \
			pp[m + 3] = _DF(0.5) * (ff[m + 3] + artificial_viscosity * uf[m + 3]);                                                  \
			mm[m + 3] = _DF(0.5) * (ff[m + 3] - artificial_viscosity * uf[m + 3]);                                                  \
		} /* calculate the scalar numerical flux at x direction*/                                                                   \
		f_flux = WENO_GPU;                                                                                                          \
		/* WENOCU6_GPU(&pp[3], &mm[3], dl) WENO_GPU WENOCU6_P(&pp[3], dl) + WENOCU6_P(&mm[3], dl);*/                                \
		/*(weno5old_P(&pp[3], dl) + weno5old_M(&mm[3], dl)) / _DF(6.0);*/                                                           \
		/* f_flux = (linear_5th_P(&pp[3], dx) + linear_5th_M(&mm[3], dx))/60.0;*/                                                   \
		/* f_flux = weno_P(&pp[3], dx) + weno_M(&mm[3], dx);*/                                                                      \
		MARCO_ROE_RIGHT;                                                                                                            \
		/* RoeAverageRight_x(n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); get eigen_r */                         \
		/* RoeAverageRight_y(n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);*/                                      \
		/* RoeAverageRight_z(n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);*/                                      \
		for (int n1 = 0; n1 < Emax; n1++)                                                                                           \
		{									   /* get Fp */                                                                         \
			_p[n][n1] = f_flux * eigen_lr[n1]; /* eigen_r actually */                                                               \
		}                                                                                                                           \
	}                                                                                                                               \
	for (int n = 0; n < Emax; n++)                                                                                                  \
	{ /* reconstruction the F-flux terms*/                                                                                          \
		real_t fluxl = _DF(0.0);                                                                                                    \
		for (int n1 = 0; n1 < Emax; n1++)                                                                                           \
		{                                                                                                                           \
			fluxl += _p[n1][n];                                                                                                     \
		}                                                                                                                           \
		Fwall[Emax * id_l + n] = fluxl;                                                                                             \
	}

#elif 1 == EIGEN_ALLOC

// RoeAverage for each DIR
#define MARCO_ROEAVERAGE_X \
	RoeAverage_x(eigen_l, eigen_r, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);
#define MARCO_ROEAVERAGE_Y \
	RoeAverage_y(eigen_l, eigen_r, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);
#define MARCO_ROEAVERAGE_Z \
	RoeAverage_z(eigen_l, eigen_r, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);

// WENO 7 // used by MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_X, i + m, j, k, i + m - stencil_P, j, k); in x
#define MARCO_FLUXWALL_WENO7(ROE_AVERAGE, _i_1, _j_1, _k_1, _i_2, _j_2, _k_2)                                                           \
	ROE_AVERAGE;                                                                                                                        \
	real_t uf[10], ff[10], pp[10], mm[10], _p[Emax][Emax], f_flux;                                                                      \
	for (int n = 0; n < Emax; n++)                                                                                                      \
	{                                                                                                                                   \
		real_t eigen_local_max = _DF(0.0);                                                                                              \
		eigen_local_max = eigen_value[n];                                                                                               \
		real_t lambda_l = eigen_local[Emax * id_l + n];                                                                                 \
		real_t lambda_r = eigen_local[Emax * id_r + n];                                                                                 \
		if (lambda_l * lambda_r < _DF(0.0))                                                                                             \
		{                                                                                                                               \
			for (int m = -stencil_P; m < stencil_size - stencil_P; m++)                                                                 \
			{ /* int _i_1 = i + m, _j_1 = j, _k_1 = k; Xmax * Ymax * k + Xmax * j + i + m */                                            \
				int id_local_1 = Xmax * Ymax * (_k_1) + Xmax * (_j_1) + (_i_1);                                                         \
				eigen_local_max = sycl::max(eigen_local_max, sycl::fabs(eigen_local[Emax * id_local_1 + n])); /* local lax-friedrichs*/ \
			}                                                                                                                           \
		}                                                                                                                               \
		for (size_t m = 0; m < stencil_size; m++)                                                                                       \
		{ /* int _i_2 = i + m, _j_2 = j, _k_2 = k; Xmax * Ymax * k + Xmax * j + m + i - stencil_P */                                    \
			int id_local_2 = Xmax * Ymax * (_k_2) + Xmax * (_j_2) + (_i_2);                                                             \
			uf[m] = _DF(0.0);                                                                                                           \
			ff[m] = _DF(0.0);                                                                                                           \
			for (int n1 = 0; n1 < Emax; n1++)                                                                                           \
			{                                                                                                                           \
				uf[m] = uf[m] + UI[Emax * id_local_2 + n1] * eigen_l[n][n1];                                                            \
				ff[m] = ff[m] + Fl[Emax * id_local_2 + n1] * eigen_l[n][n1];                                                            \
			}                                                                                                                           \
			/* for local speed*/                                                                                                        \
			pp[m] = _DF(0.5) * (ff[m] + eigen_local_max * uf[m]);                                                                       \
			mm[m] = _DF(0.5) * (ff[m] - eigen_local_max * uf[m]);                                                                       \
		}                                                                                                                               \
		/* calculate the scalar numerical flux at x direction*/                                                                         \
		f_flux = weno7_P(&pp[stencil_P], dl) + weno7_M(&mm[stencil_P], dl);                                                             \
		/* WENO_GPU;*/                                                                                                                  \
		/* weno7_P(&pp[stencil_P], dl) + weno7_M(&mm[stencil_P], dl);*/                                                                 \
		/* get Fp*/                                                                                                                     \
		for (int n1 = 0; n1 < Emax; n1++)                                                                                               \
		{                                                                                                                               \
			_p[n][n1] = f_flux * eigen_r[n1][n];                                                                                        \
		}                                                                                                                               \
	}                                                                                                                                   \
	for (int n = 0; n < Emax; n++)                                                                                                      \
	{ /* reconstruction the F-flux terms*/                                                                                              \
		real_t fluxl = _DF(0.0);                                                                                                        \
		for (int n1 = 0; n1 < Emax; n1++)                                                                                               \
		{                                                                                                                               \
			fluxl += _p[n1][n];                                                                                                         \
		}                                                                                                                               \
		Fwall[Emax * id_l + n] = fluxl;                                                                                                 \
	}

// WENO 5 //used by: MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_X, i + m, j, k, i + m, j, k);
#define MARCO_FLUXWALL_WENO5(ROE_AVERAGE, _i_1, _j_1, _k_1, _i_2, _j_2, _k_2)                                                       \
	ROE_AVERAGE;                                                                                                                    \
	real_t uf[10], ff[10], pp[10], mm[10], f_flux, _p[Emax][Emax];                                                                  \
	for (int n = 0; n < Emax; n++)                                                                                                  \
	{                                                                                                                               \
		real_t eigen_local_max = _DF(0.0);                                                                                          \
		for (int m = -stencil_P; m < stencil_size - stencil_P; m++)                                                                 \
		{ /*int _i_1 = i + m, _j_1 = j, _k_1 = k; Xmax * Ymax * k + Xmax * j + i + m*/                                              \
			int id_local_1 = Xmax * Ymax * (_k_1) + Xmax * (_j_1) + (_i_1);                                                         \
			eigen_local_max = sycl::max(eigen_local_max, sycl::fabs(eigen_local[Emax * id_local_1 + n])); /* local lax-friedrichs*/ \
		}                                                                                                                           \
		for (int m = -3; m <= 4; m++)                                                                                               \
		{ /* int _i_2 = i + m, _j_2 = j, _k_2 = k; 3rd oder and can be modified*/ /* 3rd oder and can be modified*/                 \
			int id_local = Xmax * Ymax * (_k_2) + Xmax * (_j_2) + (_i_2);                                                           \
			uf[m + 3] = _DF(0.0);                                                                                                   \
			ff[m + 3] = _DF(0.0);                                                                                                   \
			for (int n1 = 0; n1 < Emax; n1++)                                                                                       \
			{                                                                                                                       \
				uf[m + 3] = uf[m + 3] + UI[Emax * id_local + n1] * eigen_l[n][n1];                                                  \
				ff[m + 3] = ff[m + 3] + Fl[Emax * id_local + n1] * eigen_l[n][n1];                                                  \
			} /*  for local speed*/                                                                                                 \
			pp[m + 3] = _DF(0.5) * (ff[m + 3] + eigen_local_max * uf[m + 3]);                                                       \
			mm[m + 3] = _DF(0.5) * (ff[m + 3] - eigen_local_max * uf[m + 3]);                                                       \
		}                                                                                                                           \
		/* calculate the scalar numerical flux at x direction*/                                                                     \
		f_flux = WENO_GPU;                                                                                                          \
		/* (weno5old_P(&pp[3], dl) + weno5old_M(&mm[3], dl));*/                                                                     \
		/* f_flux = (linear_5th_P(&pp[3], dx) + linear_5th_M(&mm[3], dx))/60.0;*/                                                   \
		/* f_flux = weno_P(&pp[3], dx) + weno_M(&mm[3], dx);*/                                                                      \
		for (int n1 = 0; n1 < Emax; n1++)                                                                                           \
		{ /* get Fp*/                                                                                                               \
			_p[n][n1] = f_flux * eigen_r[n1][n];                                                                                    \
		}                                                                                                                           \
	}                                                                                                                               \
	for (int n = 0; n < Emax; n++)                                                                                                  \
	{ /* reconstruction the F-flux terms*/                                                                                          \
		real_t fluxl = _DF(0.0);                                                                                                    \
		for (int n1 = 0; n1 < Emax; n1++)                                                                                           \
		{                                                                                                                           \
			fluxl += _p[n1][n];                                                                                                     \
		}                                                                                                                           \
		Fwall[Emax * id_l + n] = fluxl;                                                                                             \
	}
#endif // end EIGEN_ALLOC

#pragma once

#include "global_setup.h"
#include "marcos/marco_global.h"
#include "../read_ini/setupini.h"
#include "../include/sycl_devices.hpp"

// =======================================================
//    Caculate flux_wall
// RoeAverage_Left and RoeAverage_Right for each DIR
#define MARCO_ROEAVERAGE_LEFTX \
	RoeAverageLeft_x(n, eigen_lr, eigen_value, z, _yi, _c, _u, _v, _w, _k, b1, b3, b3 - b1 * _ht);
#define MARCO_ROEAVERAGE_LEFTY \
	RoeAverageLeft_y(n, eigen_lr, eigen_value, z, _yi, _c, _u, _v, _w, _k, b1, b3, b3 - b1 * _ht);
#define MARCO_ROEAVERAGE_LEFTZ \
	RoeAverageLeft_z(n, eigen_lr, eigen_value, z, _yi, _c, _u, _v, _w, _k, b1, b3, b3 - b1 * _ht);

#define MARCO_ROEAVERAGE_RIGHTX \
	RoeAverageRight_x(n, eigen_lr, z, _yi, _c, _u, _v, _w, _k, _ht);
#define MARCO_ROEAVERAGE_RIGHTY \
	RoeAverageRight_y(n, eigen_lr, z, _yi, _c, _u, _v, _w, _k, _ht);
#define MARCO_ROEAVERAGE_RIGHTZ \
	RoeAverageRight_z(n, eigen_lr, z, _yi, _c, _u, _v, _w, _k, _ht);

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
#define MARCO_FLUXWALL_WENO5(MARCO_ROE_LEFT, MARCO_ROE_RIGHT, _i_1, _j_1, _k_1, _i_2, _j_2, _k_2)                                       \
	real_t uf[10], ff[10], pp[10], mm[10], f_flux, _p[Emax][Emax], eigen_lr[Emax], eigen_value, artificial_viscosity;                   \
	for (int n = 0; n < Emax; n++)                                                                                                      \
	{                                                                                                                                   \
		real_t eigen_local_max = _DF(0.0);                                                                                              \
		MARCO_ROE_LEFT; /* RoeAverageLeft_x(n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); get eigen_l */ \
		for (int m = -stencil_P; m < stencil_size - stencil_P; m++)                                                                     \
		{ /*int _i_1 = i + m, _j_1 = j, _k_1 = k; Xmax * Ymax * k + Xmax * j + i + m*/                                                  \
			int id_local_1 = Xmax * Ymax * (_k_1) + Xmax * (_j_1) + (_i_1);                                                             \
			eigen_local_max = sycl::max(eigen_local_max, sycl::fabs(eigen_local[Emax * id_local_1 + n])); /* local lax-friedrichs*/     \
		}                                                                                                                               \
		artificial_viscosity = Roe_type * eigen_value + LLF_type * eigen_local_max + GLF_type * eigen_block[n];                         \
		for (int m = -3; m <= 4; m++)                                                                                                   \
		{                                                                                                                               \
			/* int _i_2 = i + m, _j_2 = j, _k_2 = k; Xmax * Ymax * k + Xmax * j + m + i; 3rd oder and can be modified */                \
			int id_local = Xmax * Ymax * (_k_2) + Xmax * (_j_2) + (_i_2);                                                               \
			uf[m + 3] = _DF(0.0);                                                                                                       \
			ff[m + 3] = _DF(0.0);                                                                                                       \
			for (int n1 = 0; n1 < Emax; n1++)                                                                                           \
			{                                                                                                                           \
				uf[m + 3] = uf[m + 3] + UI[Emax * id_local + n1] * eigen_lr[n1]; /* eigen_l actually */                                 \
				ff[m + 3] = ff[m + 3] + Fl[Emax * id_local + n1] * eigen_lr[n1];                                                        \
			} /*  for local speed*/                                                                                                     \
			pp[m + 3] = _DF(0.5) * (ff[m + 3] + artificial_viscosity * uf[m + 3]);                                                      \
			mm[m + 3] = _DF(0.5) * (ff[m + 3] - artificial_viscosity * uf[m + 3]);                                                      \
		} /* calculate the scalar numerical flux at x direction*/                                                                       \
		f_flux = WENO_GPU;                                                                                                              \
		/* WENOCU6_GPU(&pp[3], &mm[3], dl) WENO_GPU WENOCU6_P(&pp[3], dl) + WENOCU6_P(&mm[3], dl);*/                                    \
		/*(weno5old_P(&pp[3], dl) + weno5old_M(&mm[3], dl)) / _DF(6.0);*/                                                               \
		/* f_flux = (linear_5th_P(&pp[3], dx) + linear_5th_M(&mm[3], dx))/60.0;*/                                                       \
		/* f_flux = weno_P(&pp[3], dx) + weno_M(&mm[3], dx);*/                                                                          \
		MARCO_ROE_RIGHT; /* RoeAverageRight_x(n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); get eigen_r */            \
		for (int n1 = 0; n1 < Emax; n1++)                                                                                               \
		{									   /* get Fp */                                                                             \
			_p[n][n1] = f_flux * eigen_lr[n1]; /* eigen_r actually */                                                                   \
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

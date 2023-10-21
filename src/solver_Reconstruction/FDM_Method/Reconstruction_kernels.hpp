#pragma once

#include "Utils_kernels.hpp"
#include "Eigen_value.hpp"
#include "Eigen_callback.h"

extern void ReconstructFluxX(int i, int j, int k, Block bl, Thermal thermal, real_t *UI, real_t *Fl, real_t *Fwall,
										   real_t *eigen_local, real_t *eigen_lt, real_t *eigen_rt, real_t *eb1, real_t *eb3, real_t *ec2, real_t *ezi,
										   real_t *p, real_t *rho, real_t *u, real_t *v, real_t *w, real_t *y, real_t *T, real_t *H, real_t *eigen_block)
{
	MARCO_DOMAIN_GHOST();
	if (i >= X_inner + Bwidth_X)
		return;
	if (j >= Y_inner + Bwidth_Y)
		return;
	if (k >= Z_inner + Bwidth_Z)
		return;

	int id_l = Xmax * Ymax * k + Xmax * j + i;
	int id_r = Xmax * Ymax * k + Xmax * j + i + 1;
	real_t dl = bl.dx;

	// preparing some interval value for roe average
	MARCO_ROE();

	MARCO_GETC2();

#if 1 == EIGEN_ALLOC
	real_t eigen_l[Emax][Emax], eigen_r[Emax][Emax], eigen_value[Emax];
#elif 2 == EIGEN_ALLOC
	real_t *eigen_l[Emax], *eigen_r[Emax], eigen_value[Emax];
	for (size_t n = 0; n < Emax; n++)
	{
		eigen_l[n] = &(eigen_lt[Emax * Emax * id_l + n * Emax]);
		eigen_r[n] = &(eigen_rt[Emax * Emax * id_l + n * Emax]);
	}
#endif // end EIGEN_ALLOC
//     // RoeAverage_x(eigen_l, eigen_r, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);

//     // construct the right value & the left value scalar equations by characteristic reduction
//     // at i+1/2 in x direction
#if 0 == EIGEN_ALLOC
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
    MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_LEFTX, MARCO_ROEAVERAGE_RIGHTX, i + m, j, k, i + m - stencil_P, j, k);
#elif SCHEME_ORDER <= 6
    MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_LEFTX, MARCO_ROEAVERAGE_RIGHTX, i + m, j, k, i + m, j, k);
#endif

// #if SCHEME_ORDER == 7
//     MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_LEFTNX, MARCO_ROEAVERAGE_RIGHTNX, i + m, j, k, i + m - stencil_P, j, k);
// #elif SCHEME_ORDER <= 6
//     MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_LEFTNX, MARCO_ROEAVERAGE_RIGHTNX, i + m, j, k, i + m, j, k);
// #endif

//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#elif 1 == EIGEN_ALLOC || 2 == EIGEN_ALLOC
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
	MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_X, i + m, j, k, i + m - stencil_P, j, k);
#elif SCHEME_ORDER <= 6
	MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_X, i + m, j, k, i + m, j, k);
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif // end EIGEN_ALLOC

	//     // real_t uf[10], ff[10], pp[10], mm[10], f_flux, _p[Emax][Emax], eigen_lr[Emax], eigen_value;
	//     // for (int n = 0; n < Emax; n++)
	//     // {
	//     //     real_t eigen_local_max = _DF(0.0);
	//     //     RoeAverageLeft_x(n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); /*get eigen_l*/
	//     //     for (int m = -stencil_P; m < stencil_size - stencil_P; m++)
	//     //     {
	//     //         int _i_1 = i + m, _j_1 = j, _k_1 = k;
	//     //         int id_local_1 = Xmax * Ymax * (_k_1) + Xmax * (_j_1) + (_i_1);                                       /*int _i_1 = i + m, _j_1 = j, _k_1 = k; Xmax * Ymax * k + Xmax * j + i + m*/
	//     //         eigen_local_max = sycl::max(eigen_local_max, sycl::fabs(eigen_local[Emax * id_local_1 + n])); /* local lax-friedrichs*/
	//     //     }
	//     //     for (int m = -3; m <= 4; m++)
	//     //     {
	//     //         int _i_2 = i + m, _j_2 = j, _k_2 = k;                         /* int _i_2 = i + m, _j_2 = j, _k_2 = k; 3rd oder and can be modified*/
	//     //         int id_local = Xmax * Ymax * (_k_2) + Xmax * (_j_2) + (_i_2); /*Xmax * Ymax * k + Xmax * j + m + i;*/
	//     //         uf[m + 3] = _DF(0.0);
	//     //         ff[m + 3] = _DF(0.0);
	//     //         for (int n1 = 0; n1 < Emax; n1++)
	//     //         {
	//     //             uf[m + 3] = uf[m + 3] + UI[Emax * id_local + n1] * eigen_lr[n1]; /* eigen_l actually */
	//     //             ff[m + 3] = ff[m + 3] + Fl[Emax * id_local + n1] * eigen_lr[n1];
	//     //         } /*  for local speed*/
	//     //         pp[m + 3] = _DF(0.5) * (ff[m + 3] + eigen_local_max * uf[m + 3]);
	//     //         mm[m + 3] = _DF(0.5) * (ff[m + 3] - eigen_local_max * uf[m + 3]);
	//     //     }                                                                                                                                     /* calculate the scalar numerical flux at x direction*/
	//     //     f_flux = (weno5old_P(&pp[3], dl) + weno5old_M(&mm[3], dl)); /* f_flux = (linear_5th_P(&pp[3], dx) + linear_5th_M(&mm[3], dx))/60.0;*/ /* f_flux = weno_P(&pp[3], dx) + weno_M(&mm[3], dx);*/
	//     //     RoeAverageRight_x(n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);                                                     /* get eigen_r */
	//     //     for (int n1 = 0; n1 < Emax; n1++)
	//     //     {                                      /* get Fp */
	//     //         _p[n][n1] = f_flux * eigen_lr[n1]; /* eigen_r actually */
	//     //     }
	//     // }
	//     // for (int n = 0; n < Emax; n++)
	//     // { /* reconstruction the F-flux terms*/
	//     //     Fwall[Emax * id_l + n] = _DF(0.0);
	//     //     for (int n1 = 0; n1 < Emax; n1++)
	//     //     {
	//     //         Fwall[Emax * id_l + n] += _p[n1][n];
	//     //     }
	//     // }

	//     // real_t uf[10], ff[10], pp[10], mm[10], _p[Emax][Emax], f_flux, eigen_lr[Emax], eigen_value;
	//     // for (int n = 0; n < Emax; n++)
	//     // {
	//     //     real_t eigen_local_max = _DF(0.0);
	//     //     MARCO_ROEAVERAGE_LEFTX; // MARCO_ROE_LEFT; /* eigen_r actually */ /* RoeAverageLeft_x(n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); get eigen_l */
	//     //     eigen_local_max = eigen_value;
	//     //     real_t lambda_l = eigen_local[Emax * id_l + n];
	//     //     real_t lambda_r = eigen_local[Emax * id_r + n];
	//     //     if (lambda_l * lambda_r < 0.0)
	//     //     {
	//     //         for (int m = -stencil_P; m < stencil_size - stencil_P; m++)
	//     //         {
	//     //             int _i_1 = i + m, _j_1 = j, _k_1 = k;
	//     //             int id_local_1 = Xmax * Ymax * (_k_1) + Xmax * (_j_1) + (_i_1);                                       /* int _i_1 = i + m, _j_1 = j, _k_1 = k; Xmax * Ymax * k + Xmax * j + i + m */
	//     //             eigen_local_max = sycl::max(eigen_local_max, sycl::fabs(eigen_local[Emax * id_local_1 + n])); /* local lax-friedrichs*/
	//     //         }
	//     //     }
	//     //     for (size_t m = 0; m < stencil_size; m++)
	//     //     {
	//     //         int _i_2 = i + m - stencil_P, _j_2 = j, _k_2 = k;
	//     //         int id_local_2 = Xmax * Ymax * (_k_2) + Xmax * (_j_2) + (_i_2); /* int _i_2 = i + m, _j_2 = j, _k_2 = k; Xmax * Ymax * k + Xmax * j + m + i - stencil_P */
	//     //         uf[m] = _DF(0.0);
	//     //         ff[m] = _DF(0.0);
	//     //         for (int n1 = 0; n1 < Emax; n1++)
	//     //         {
	//     //             uf[m] = uf[m] + UI[Emax * id_local_2 + n1] * eigen_lr[n1];
	//     //             ff[m] = ff[m] + Fl[Emax * id_local_2 + n1] * eigen_lr[n1];
	//     //         } /* for local speed*/
	//     //         pp[m] = _DF(0.5) * (ff[m] + eigen_local_max * uf[m]);
	//     //         mm[m] = _DF(0.5) * (ff[m] - eigen_local_max * uf[m]);
	//     //     }                                                                   /* calculate the scalar numerical flux at x direction*/
	//     //     f_flux = weno7_P(&pp[stencil_P], dl) + weno7_M(&mm[stencil_P], dl); /* get Fp*/
	//     //     MARCO_ROEAVERAGE_RIGHTX;                                            // MARCO_ROE_RIGHT; /* eigen_r actually */                                                    /* RoeAverageRight_x(n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); get eigen_r */
	//     //     for (int n1 = 0; n1 < Emax; n1++)
	//     //     {
	//     //         _p[n][n1] = f_flux * eigen_lr[n1];
	//     //     }
	//     // } /* reconstruction the F-flux terms*/
	//     // for (int n = 0; n < Emax; n++)
	//     // {
	//     //     real_t temp_flux = _DF(0.0);
	//     //     for (int n1 = 0; n1 < Emax; n1++)
	//     //     {
	//     //         temp_flux += _p[n1][n];
	//     //     }
	//     //     Fwall[Emax * id_l + n] = temp_flux;
	//     // }

	//     // real_t de_fw[Emax];
	//     // get_Array(Fwall, de_fw, Emax, id_l);
	//     // real_t de_fx[Emax];
}

extern void ReconstructFluxY(int i, int j, int k, Block bl, Thermal thermal, real_t *UI, real_t *Fl, real_t *Fwall,
										   real_t *eigen_local, real_t *eigen_lt, real_t *eigen_rt, real_t *eb1, real_t *eb3, real_t *ec2, real_t *ezi,
										   real_t *p, real_t *rho, real_t *u, real_t *v, real_t *w, real_t *y, real_t *T, real_t *H, real_t *eigen_block)
{
	MARCO_DOMAIN_GHOST();
	if (i >= X_inner + Bwidth_X)
		return;
	if (j >= Y_inner + Bwidth_Y)
		return;
	if (k >= Z_inner + Bwidth_Z)
		return;

	int id_l = Xmax * Ymax * k + Xmax * j + i;
	int id_r = Xmax * Ymax * k + Xmax * (j + 1) + i;
	real_t dl = bl.dy;

	// preparing some interval value for roe average
	MARCO_ROE();

	MARCO_GETC2();

#if 1 == EIGEN_ALLOC
	real_t eigen_l[Emax][Emax], eigen_r[Emax][Emax], eigen_value[Emax];
#elif 2 == EIGEN_ALLOC
	real_t *eigen_l[Emax], *eigen_r[Emax], eigen_value[Emax];
	for (size_t n = 0; n < Emax; n++)
	{
		eigen_l[n] = &(eigen_lt[Emax * Emax * id_l + n * Emax]);
		eigen_r[n] = &(eigen_rt[Emax * Emax * id_l + n * Emax]);
	}
#endif // end EIGEN_ALLOC
	   //     // RoeAverage_y(eigen_l, eigen_r, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);

	//     // // construct the right value & the left value scalar equations by characteristic reduction
	//     // // at i+1/2 in x direction

#if 0 == EIGEN_ALLOC
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
    MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_LEFTY, MARCO_ROEAVERAGE_RIGHTY, i, j + m, k, i, j + m - stencil_P, k);
#elif SCHEME_ORDER <= 6
    MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_LEFTY, MARCO_ROEAVERAGE_RIGHTY, i, j + m, k, i, j + m, k);
#endif

// #if SCHEME_ORDER == 7
//     MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_LEFTNY, MARCO_ROEAVERAGE_RIGHTNY, i, j + m, k, i, j + m - stencil_P, k);
// #elif SCHEME_ORDER <= 6
//     MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_LEFTNY, MARCO_ROEAVERAGE_RIGHTNY, i, j + m, k, i, j + m, k);
// #endif
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#elif 1 == EIGEN_ALLOC || 2 == EIGEN_ALLOC
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
	MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_Y, i, j + m, k, i, j + m - stencil_P, k);
#elif SCHEME_ORDER <= 6
	MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_Y, i, j + m, k, i, j + m, k);
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif // end EIGEN_ALLOC

	//     // real_t de_fw[Emax];
	//     // get_Array(Fwall, de_fw, Emax, id_l);
	//     // real_t de_fx[Emax];
}

extern void ReconstructFluxZ(int i, int j, int k, Block bl, Thermal thermal, real_t *UI, real_t *Fl, real_t *Fwall,
										   real_t *eigen_local, real_t *eigen_lt, real_t *eigen_rt, real_t *eb1, real_t *eb3, real_t *ec2, real_t *ezi,
										   real_t *p, real_t *rho, real_t *u, real_t *v, real_t *w, real_t *y, real_t *T, real_t *H, real_t *eigen_block)
{
	MARCO_DOMAIN_GHOST();
	if (i >= X_inner + Bwidth_X)
		return;
	if (j >= Y_inner + Bwidth_Y)
		return;
	if (k >= Z_inner + Bwidth_Z)
		return;

	int id_l = Xmax * Ymax * k + Xmax * j + i;
	int id_r = Xmax * Ymax * (k + 1) + Xmax * j + i;
	real_t dl = bl.dz;

	// preparing some interval value for roe average
	MARCO_ROE();

	MARCO_GETC2();

#if 1 == EIGEN_ALLOC
	real_t eigen_l[Emax][Emax], eigen_r[Emax][Emax], eigen_value[Emax];
#elif 2 == EIGEN_ALLOC
	real_t *eigen_l[Emax], *eigen_r[Emax], eigen_value[Emax];
	for (size_t n = 0; n < Emax; n++)
	{
		eigen_l[n] = &(eigen_lt[Emax * Emax * id_l + n * Emax]);
		eigen_r[n] = &(eigen_rt[Emax * Emax * id_l + n * Emax]);
	}
#endif // end EIGEN_ALLOC
	   //     // RoeAverage_z(eigen_l, eigen_r, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0);

	//     // // construct the right value & the left value scalar equations by characteristic reduction
	//     // // at i+1/2 in x direction

#if 0 == EIGEN_ALLOC
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
    MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_LEFTZ, MARCO_ROEAVERAGE_RIGHTZ, i, j, k + m, i, j, k + m - stencil_P);
#elif SCHEME_ORDER <= 6
    MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_LEFTZ, MARCO_ROEAVERAGE_RIGHTZ, i, j, k + m, i, j, k + m);
#endif

// #if SCHEME_ORDER == 7
//     MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_LEFTNZ, MARCO_ROEAVERAGE_RIGHTNZ, i, j, k + m, i, j, k + m - stencil_P);
// #elif SCHEME_ORDER <= 6
//     MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_LEFTNZ, MARCO_ROEAVERAGE_RIGHTNZ, i, j, k + m, i, j, k + m);
// #endif
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#elif 1 == EIGEN_ALLOC || 2 == EIGEN_ALLOC
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
	MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_Z, i, j, k + m, i, j, k + m - stencil_P);
#elif SCHEME_ORDER <= 6
	MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_Z, i, j, k + m, i, j, k + m);
#endif
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#endif // end EIGEN_ALLOC

	//     // real_t uf[10], ff[10], pp[10], mm[10], _p[Emax][Emax], f_flux, eigen_lr[Emax], eigen_value, eigen_l[Emax][Emax], eigen_r[Emax][Emax];
	//     // for (int n = 0; n < Emax; n++)
	//     // {
	//     //     real_t eigen_local_max = _DF(0.0);
	//     //     MARCO_ROEAVERAGE_LEFTZ; /* eigen_r actually */ /* RoeAverageLeft_x(n, eigen_lr, eigen_value, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); get eigen_l */
	//     //     eigen_local_max = eigen_value;
	//     //     for (int n1 = 0; n1 < Emax; n1++)
	//     //     {
	//     //         eigen_l[n][n1] = eigen_lr[n1];
	//     //     }
	//     //     real_t lambda_l = eigen_local[Emax * id_l + n];
	//     //     real_t lambda_r = eigen_local[Emax * id_r + n];
	//     //     if (lambda_l * lambda_r < 0.0)
	//     //     {
	//     //         for (int m = -stencil_P; m < stencil_size - stencil_P; m++)
	//     //         {
	//     //             int _i_1 = i, _j_1 = j, _k_1 = k + m;
	//     //             int id_local_1 = Xmax * Ymax * (_k_1) + Xmax * (_j_1) + (_i_1);                                       /* int _i_1 = i + m, _j_1 = j, _k_1 = k; Xmax * Ymax * k + Xmax * j + i + m */
	//     //             eigen_local_max = sycl::max(eigen_local_max, sycl::fabs(eigen_local[Emax * id_local_1 + n])); /* local lax-friedrichs*/
	//     //         }
	//     //     }
	//     //     for (size_t m = 0; m < stencil_size; m++)
	//     //     {
	//     //         int _i_2 = i, _j_2 = j, _k_2 = k + m - stencil_P;
	//     //         int id_local_2 = Xmax * Ymax * (_k_2) + Xmax * (_j_2) + (_i_2); /* int _i_2 = i + m, _j_2 = j, _k_2 = k; Xmax * Ymax * k + Xmax * j + m + i - stencil_P */
	//     //         uf[m] = _DF(0.0);
	//     //         ff[m] = _DF(0.0);
	//     //         for (int n1 = 0; n1 < Emax; n1++)
	//     //         {
	//     //             uf[m] = uf[m] + UI[Emax * id_local_2 + n1] * eigen_lr[n1];
	//     //             ff[m] = ff[m] + Fl[Emax * id_local_2 + n1] * eigen_lr[n1];
	//     //         } /* for local speed*/
	//     //         pp[m] = _DF(0.5) * (ff[m] + eigen_local_max * uf[m]);
	//     //         mm[m] = _DF(0.5) * (ff[m] - eigen_local_max * uf[m]);
	//     //     }                                                                   /* calculate the scalar numerical flux at x direction*/
	//     //     f_flux = weno7_P(&pp[stencil_P], dl) + weno7_M(&mm[stencil_P], dl); /* get Fp*/
	//     //     MARCO_ROEAVERAGE_RIGHTZ; /* eigen_r actually */                     /* RoeAverageRight_x(n, eigen_lr, z, _yi, c2, _rho, _u, _v, _w, _H, b1, b3, Gamma0); get eigen_r */
	//     //     for (int n1 = 0; n1 < Emax; n1++)
	//     //     {
	//     //         eigen_r[n1][n] = eigen_lr[n1];
	//     //     }
	//     //     for (int n1 = 0; n1 < Emax; n1++)
	//     //     {
	//     //         _p[n][n1] = f_flux * eigen_lr[n1];
	//     //     }
	//     // }
	//     // for (int n = 0; n < Emax; n++)
	//     // { /* reconstruction the F-flux terms*/
	//     //     Fwall[Emax * id_l + n] = _DF(0.0);
	//     //     for (int n1 = 0; n1 < Emax; n1++)
	//     //     {
	//     //         Fwall[Emax * id_l + n] += _p[n1][n];
	//     //     }
	//     // }

	//     // real_t de_fw[Emax];
	//     // get_Array(Fwall, de_fw, Emax, id_l);
	//     // real_t de_fx[Emax];
}

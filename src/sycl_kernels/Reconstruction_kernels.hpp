#pragma once

#include "Utils_kernels.hpp"

extern SYCL_EXTERNAL void GetLocalEigen(int i, int j, int k, Block bl, real_t AA, real_t BB, real_t CC, real_t *eigen_local, real_t *u, real_t *v, real_t *w, real_t *c)
{
	MARCO_DOMAIN();
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
#if SCHEME_ORDER <= 6
	real_t uu = AA * u[id] + BB * v[id] + CC * w[id];
	real_t uuPc = uu + c[id];
	real_t uuMc = uu - c[id];

	// local eigen values
	eigen_local[Emax * id + 0] = uuMc;
	for (size_t ii = 1; ii < Emax - 1; ii++)
	{
		eigen_local[Emax * id + ii] = uu;
	}
	eigen_local[Emax * id + Emax - 1] = uuPc;
#elif SCHEME_ORDER == 7
	for (size_t ii = 0; ii < Emax; ii++)
		eigen_local[Emax * id + ii] = 0.0;
#endif // end FLUX_method

	// real_t de_fw[Emax];
	// get_Array(eigen_local, de_fw, Emax, id);
	// real_t de_fx[Emax];
}

#if DIM_X
extern SYCL_EXTERNAL void ReconstructFluxX(int i, int j, int k, Block bl, Thermal thermal, real_t *UI, real_t *Fl, real_t *Fwall,
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

	MARCO_GETC2()

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
	//     //         eigen_local_max = sycl::max(eigen_local_max, sycl::fabs<real_t>(eigen_local[Emax * id_local_1 + n])); /* local lax-friedrichs*/
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
	//     //             eigen_local_max = sycl::max(eigen_local_max, sycl::fabs<real_t>(eigen_local[Emax * id_local_1 + n])); /* local lax-friedrichs*/
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
#endif // end DIM_X

#if DIM_Y
extern SYCL_EXTERNAL void ReconstructFluxY(int i, int j, int k, Block bl, Thermal thermal, real_t *UI, real_t *Fl, real_t *Fwall,
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

	MARCO_GETC2()

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
#endif // end DIM_Y

#if DIM_Z
extern SYCL_EXTERNAL void ReconstructFluxZ(int i, int j, int k, Block bl, Thermal thermal, real_t *UI, real_t *Fl, real_t *Fwall,
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

	MARCO_GETC2()

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
	//     //             eigen_local_max = sycl::max(eigen_local_max, sycl::fabs<real_t>(eigen_local[Emax * id_local_1 + n])); /* local lax-friedrichs*/
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
#endif // end DIM_Z

extern SYCL_EXTERNAL void PositivityPreservingKernel(int i, int j, int k, int id_l, int id_r, Block bl, Thermal thermal,
													 real_t *UI, real_t *Fl, real_t *Fwall, real_t *T,
													 const real_t lambda_0, const real_t lambda, const real_t *epsilon) // , sycl::stream stream epsilon[NUM_SPECIES+2]={rho, e, y(0), ..., y(n)}
{
#if DIM_X
	if (i >= bl.Xmax - bl.Bwidth_X)
		return;
#endif
#if DIM_Y
	if (j >= bl.Ymax - bl.Bwidth_Y)
		return;
#endif
#if DIM_Z
	if (k >= bl.Zmax - bl.Bwidth_Z)
		return;
#endif
	real_t T_l = T[id_l], T_r = T[id_r];
	id_l *= Emax, id_r *= Emax;
	// stream << "eps: " << epsilon[0] << " " << epsilon[3] << " " << epsilon[5] << " " << epsilon[NUM_SPECIES + 1] << "\n";
	// int id_l = (Xmax * Ymax * k + Xmax * j + i) * Emax;
	// int id_r = (Xmax * Ymax * k + Xmax * j + i + 1) * Emax;
	// Positivity preserving flux limiter form Dr.Hu.https://www.sciencedirect.com/science/article/pii/S0021999113000557, expand to multicomponent, need positivity Initial value
	real_t rho_min, theta, theta_u, theta_p, F_LF[Emax], FF_LF[Emax], FF[Emax], *UU = &(UI[id_l]), *UP = &(UI[id_r]); // UU[Emax], UP[Emax],
	for (int n = 0; n < Emax; n++)
	{
		F_LF[n] = _DF(0.5) * (Fl[n + id_l] + Fl[n + id_r] + lambda_0 * (UI[n + id_l] - UI[n + id_r])); // FF_LF == F_(i+1/2) of Lax-Friedrichs
		FF_LF[n] = _DF(2.0) * lambda * F_LF[n];														   // U_i^+ == (U_i^n-FF_LF)
		FF[n] = _DF(2.0) * lambda * Fwall[n + id_l];												   // FF from original high accuracy scheme
	}

	// // correct for positive density
	theta_u = _DF(1.0), theta_p = _DF(1.0);
	rho_min = sycl::min<real_t>(UU[0], epsilon[0]);
	if (UU[0] - FF[0] < rho_min)
		theta_u = (UU[0] - FF_LF[0] - rho_min) / (FF[0] - FF_LF[0]);
	rho_min = sycl::min<real_t>(UP[0], epsilon[0]);
	if (UP[0] + FF[0] < rho_min)
		theta_p = (UP[0] + FF_LF[0] - rho_min) / (FF_LF[0] - FF[0]);
	theta = sycl::min<real_t>(theta_u, theta_p);
	for (int n = 0; n < Emax; n++)
	{
		FF[n] = (_DF(1.0) - theta) * FF_LF[n] + theta * FF[n];
		Fwall[n + id_l] = (_DF(1.0) - theta) * F_LF[n] + theta * Fwall[n + id_l];
	}

	// // correct for yi // need Ini of yi > 0(1.0e-10 set)
	real_t yi_q[NUM_SPECIES], yi_u[NUM_SPECIES], yi_qp[NUM_SPECIES], yi_up[NUM_SPECIES], _rhoq, _rhou, _rhoqp, _rhoup;
	_rhoq = _DF(1.0) / (UU[0] - FF[0]), _rhou = _DF(1.0) / (UU[0] - FF_LF[0]), yi_q[NUM_COP] = _DF(1.0), yi_u[NUM_COP] = _DF(1.0);
	_rhoqp = _DF(1.0) / (UP[0] + FF[0]), _rhoup = _DF(1.0) / (UP[0] + FF_LF[0]), yi_qp[NUM_COP] = _DF(1.0), yi_up[NUM_COP] = _DF(1.0);
	for (size_t n = 0; n < NUM_COP; n++)
	{
		int tid = n + 5;
		yi_q[n] = (UU[tid] - FF[tid]) * _rhoq, yi_q[NUM_COP] -= yi_q[n];
		yi_u[n] = (UU[tid] - FF_LF[tid]) * _rhou, yi_u[NUM_COP] -= yi_u[n];
		yi_qp[n] = (UP[tid] + FF[tid]) * _rhoqp, yi_qp[NUM_COP] -= yi_qp[n];
		yi_up[n] = (UP[tid] + FF_LF[tid]) * _rhoup, yi_up[NUM_COP] -= yi_up[n];
		theta_u = _DF(1.0), theta_p = _DF(1.0);
		real_t temp = epsilon[n + 2];
		if (yi_q[n] < temp)
		{
			real_t yi_min = sycl::min<real_t>(yi_u[n], temp);
			theta_u = (yi_u[n] - yi_min + _DF(1.0e-100)) / (yi_u[n] - yi_q[n] + _DF(1.0e-100));
		}
		if (yi_qp[n] < temp)
		{
			real_t yi_min = sycl::min<real_t>(yi_up[n], temp);
			theta_p = (yi_up[n] - yi_min + _DF(1.0e-100)) / (yi_up[n] - yi_qp[n] + _DF(1.0e-100));
		}
		theta = sycl::min<real_t>(theta_u, theta_p);
		for (int nn = 0; nn < Emax; nn++)
			Fwall[nn + id_l] = (_DF(1.0) - theta) * F_LF[nn] + theta * Fwall[nn + id_l];
	}
	// // // correct for yn
	// real_t temp = epsilon[NUM_SPECIES + 1];
	// if (yi_q[NUM_COP] < temp)
	// {
	//     real_t yi_min = sycl::min<real_t>(yi_u[NUM_COP], temp);
	//     theta_u = (yi_u[NUM_COP] - yi_min) / (yi_u[NUM_COP] - yi_q[NUM_COP]);
	// }
	// if (yi_qp[NUM_COP] < temp)
	// {
	//     real_t yi_min = sycl::min<real_t>(yi_up[NUM_COP], temp);
	//     theta_p = (yi_up[NUM_COP] - yi_min) / (yi_up[NUM_COP] - yi_qp[NUM_COP]);
	// }
	// theta = sycl::min<real_t>(theta_u, theta_p);
	// for (int nn = 0; nn < Emax; nn++)
	//     Fwall[nn + id_l] = (_DF(1.0) - theta) * F_LF[nn] + theta * Fwall[nn + id_l];

	// size_t begin = 0, end = NUM_COP - 1;
	// #ifdef COP_CHEME
	//     for (size_t n = 0; n < 2; n++) // NUM_SPECIES - 2
	//     {
	//         theta_u = _DF(1.0), theta_p = _DF(1.0);
	//         real_t temp = epsilon[n + 2];
	//         if (yi_q[n] < temp)
	//         {
	//             real_t yi_min = sycl::min<real_t>(yi_u[n], temp);
	//             theta_u = (yi_u[n] - yi_min + 1.0e-100) / (yi_u[n] - yi_q[n] + 1.0e-100);
	//         }
	//         if (yi_qp[n] < temp)
	//         {
	//             real_t yi_min = sycl::min<real_t>(yi_up[n], temp);
	//             theta_p = (yi_up[n] - yi_min + 1.0e-100) / (yi_up[n] - yi_qp[n] + 1.0e-100);
	//         }
	//         theta = sycl::min<real_t>(theta_u, theta_p);
	//         for (int nn = 0; nn < Emax; nn++)
	//             Fwall[nn + id_l] = (_DF(1.0) - theta) * F_LF[nn] + theta * Fwall[nn + id_l];
	//     }
	// #endif // end COP_CHEME

	// for (size_t n = begin; n < end; n++) // NUM_SPECIES - 2
	// {
	//     theta_u = _DF(1.0), theta_p = _DF(1.0);
	//     real_t temp = epsilon[n + 2];
	//     if (yi_q[n] < temp)
	//     {
	//         real_t yi_min = sycl::min<real_t>(yi_u[n], temp);
	//         theta_u = (yi_u[n] - yi_min + 1.0e-100) / (yi_u[n] - yi_q[n] + 1.0e-100);
	//     }
	//     if (yi_qp[n] < temp)
	//     {
	//         real_t yi_min = sycl::min<real_t>(yi_up[n], temp);
	//         theta_p = (yi_up[n] - yi_min + 1.0e-100) / (yi_up[n] - yi_qp[n] + 1.0e-100);
	//     }
	//     theta = sycl::min<real_t>(theta_u, theta_p);
	//     for (int nn = 0; nn < Emax; nn++)
	//         Fwall[nn + id_l] = (_DF(1.0) - theta) * F_LF[nn] + theta * Fwall[nn + id_l];
	// }

	// // // correct for positive p, method to get p for multicomponent theory:
	// // // e = UI[4]*_rho-_DF(0.5)*_rho*_rho*(UI[1]*UI[1]+UI[2]*UI[2]+UI[3]*UI[3]);
	// // // R = get_CopR(thermal._Wi, yi); T = get_T(thermal, yi, e, T); p = rho * R * T;
	// // // known that rho and yi has been preserved to be positive, only need to preserve positive T
	// real_t e_q, T_q, P_q, theta_pu = 1.0, theta_pp = 1.0;
	// theta_u = _DF(1.0), theta_p = _DF(1.0);
	// e_q = (UU[4] - FF[4] - _DF(0.5) * ((UU[1] - FF[1]) * (UU[1] - FF[1]) + (UU[2] - FF[2]) * (UU[2] - FF[2]) + (UU[3] - FF[3]) * (UU[3] - FF[3])) * _rhoq) * _rhoq;
	// T_q = get_T(thermal, yi_q, e_q, T_l);
	// if (T_q < epsilon[1])
	// {
	//     real_t e_u = (UU[4] - FF_LF[4] - _DF(0.5) * ((UU[1] - FF_LF[1]) * (UU[1] - FF_LF[1]) + (UU[2] - FF_LF[2]) * (UU[2] - FF_LF[2]) + (UU[3] - FF_LF[3]) * (UU[3] - FF_LF[3])) * _rhou) * _rhou;
	//     real_t T_u = get_T(thermal, yi_u, e_u, T_l);
	//     real_t T_min = sycl::min<real_t>(T_u, epsilon[1]);
	//     theta_u = (T_u - T_min + 1.0e-100) / (T_u - T_q + 1.0e-100);
	// }
	// // P_q = T_q * get_CopR(thermal._Wi, yi_q);
	// // if (P_q < epsilon[1])
	// // {
	// //     real_t e_u = (UU[4] - FF_LF[4] - _DF(0.5) * ((UU[1] - FF_LF[1]) * (UU[1] - FF_LF[1]) + (UU[2] - FF_LF[2]) * (UU[2] - FF_LF[2]) + (UU[3] - FF_LF[3]) * (UU[3] - FF_LF[3])) * _rhou) * _rhou;
	// //     real_t P_u = get_T(thermal, yi_u, e_u, T_l) * get_CopR(thermal._Wi, yi_u);
	// //     real_t P_min = sycl::min<real_t>(P_u, epsilon[1]);
	// //     theta_pu = (P_u - P_min) / (P_u - P_q);
	// // }

	// e_q = (UP[4] + FF[4] - _DF(0.5) * ((UP[1] + FF[1]) * (UP[1] + FF[1]) + (UP[2] + FF[2]) * (UP[2] + FF[2]) + (UP[3] + FF[3]) * (UP[3] + FF[3])) * _rhoqp) * _rhoqp;
	// T_q = get_T(thermal, yi_qp, e_q, T_r);
	// if (T_q < epsilon[1])
	// {
	//     real_t e_p = (UP[4] + FF_LF[4] - _DF(0.5) * ((UP[1] + FF_LF[1]) * (UP[1] + FF_LF[1]) + (UP[2] + FF_LF[2]) * (UP[2] + FF_LF[2]) + (UP[3] + FF_LF[3]) * (UP[3] + FF_LF[3])) * _rhoup) * _rhoup;
	//     real_t T_p = get_T(thermal, yi_up, e_p, T_r);
	//     real_t T_min = sycl::min<real_t>(T_p, epsilon[1]);
	//     theta_p = (T_p - T_min + 1.0e-100) / (T_p - T_q + 1.0e-100);
	// }
	// // P_q = T_q * get_CopR(thermal._Wi, yi_qp);
	// // if (P_q < epsilon[1])
	// // {
	// //     real_t e_p = (UP[4] + FF_LF[4] - _DF(0.5) * ((UP[1] + FF_LF[1]) * (UP[1] + FF_LF[1]) + (UP[2] + FF_LF[2]) * (UP[2] + FF_LF[2]) + (UP[3] + FF_LF[3]) * (UP[3] + FF_LF[3])) * _rhoup) * _rhoup;
	// //     real_t P_p = get_T(thermal, yi_up, e_p, T_r) * get_CopR(thermal._Wi, yi_qp);
	// //     real_t P_min = sycl::min<real_t>(P_p, epsilon[1]);
	// //     theta_pp = (P_p - P_min) / (P_p - P_q);
	// // }
	// theta = sycl::min<real_t>(theta_u, theta_p);
	// for (int n = 0; n < Emax; n++)
	//     Fwall[n + id_l] = (_DF(1.0) - theta) * F_LF[n] + theta * Fwall[n + id_l];
	// // theta = sycl::min<real_t>(theta_pu, theta_pp);
	// // for (int n = 0; n < Emax; n++)
	// //     Fwall[n + id_l] = (_DF(1.0) - theta) * F_LF[n] + theta * Fwall[n + id_l];
}
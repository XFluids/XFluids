#pragma once

#include "Eigen_matrix.hpp"
#include "Utils_device.hpp"
#include "../Recon_device.hpp"
#include "../schemes/schemes_device.hpp"

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

	size_t id_l = Xmax * Ymax * k + Xmax * j + i;
	size_t id_r = Xmax * Ymax * k + Xmax * j + i + 1;
	real_t dl = bl.dx;

	// preparing some interval value for roe average
	MARCO_ROE();

	// // MARCO_GETC2();
	// /_hi[NUM_SPECIES],*/
	real_t _yi[MAX_SPECIES], z[MAX_SPECIES] = {_DF(0.0)}, b1 = _DF(0.0), b3 = _DF(0.0), _k = _DF(0.0), _ht = _DF(0.0), Gamma0 = _DF(1.4);
	real_t c2 = ReconstructSoundSpeed(thermal, id_l, id_r, D, D1, _rho, _P, rho, u, v, w, y, p, T, H,
									  _yi, z, b1, b3, _k, _ht, Gamma0);
	real_t _c = sycl::sqrt(c2);
	MARCO_ERROR_OUT();

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

//     // construct the right value & the left value scalar equations by characteristic reduction
//     // at i+1/2 in x direction
#if 0 == EIGEN_ALLOC
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
    MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_LEFTX, MARCO_ROEAVERAGE_RIGHTX, i + m, j, k, i + m - stencil_P, j, k);
#elif SCHEME_ORDER <= 6
    MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_LEFTX, MARCO_ROEAVERAGE_RIGHTX, i + m, j, k, i + m, j, k);
#endif

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

	size_t id_l = Xmax * Ymax * k + Xmax * j + i;
	size_t id_r = Xmax * Ymax * k + Xmax * (j + 1) + i;
	real_t dl = bl.dy;

	// preparing some interval value for roe average
	MARCO_ROE();

	// // MARCO_GETC2();
	// /_hi[NUM_SPECIES],*/
	real_t _yi[MAX_SPECIES], z[MAX_SPECIES] = {_DF(0.0)}, b1 = _DF(0.0), b3 = _DF(0.0), _k = _DF(0.0), _ht = _DF(0.0), Gamma0 = _DF(1.4);
	real_t c2 = ReconstructSoundSpeed(thermal, id_l, id_r, D, D1, _rho, _P, rho, u, v, w, y, p, T, H,
									  _yi, z, b1, b3, _k, _ht, Gamma0);
	real_t _c = sycl::sqrt(c2);
	MARCO_ERROR_OUT();

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

	//     // // construct the right value & the left value scalar equations by characteristic reduction
	//     // // at i+1/2 in x direction
#if 0 == EIGEN_ALLOC
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
    MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_LEFTY, MARCO_ROEAVERAGE_RIGHTY, i, j + m, k, i, j + m - stencil_P, k);
#elif SCHEME_ORDER <= 6
    MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_LEFTY, MARCO_ROEAVERAGE_RIGHTY, i, j + m, k, i, j + m, k);
#endif

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

	size_t id_l = Xmax * Ymax * k + Xmax * j + i;
	size_t id_r = Xmax * Ymax * (k + 1) + Xmax * j + i;
	real_t dl = bl.dz;

	// preparing some interval value for roe average
	MARCO_ROE();

	// // MARCO_GETC2();
	// /_hi[NUM_SPECIES],*/
	real_t _yi[MAX_SPECIES], z[MAX_SPECIES] = {_DF(0.0)}, b1 = _DF(0.0), b3 = _DF(0.0), _k = _DF(0.0), _ht = _DF(0.0), Gamma0 = _DF(1.4);
	real_t c2 = ReconstructSoundSpeed(thermal, id_l, id_r, D, D1, _rho, _P, rho, u, v, w, y, p, T, H,
									  _yi, z, b1, b3, _k, _ht, Gamma0);
	real_t _c = sycl::sqrt(c2);
	MARCO_ERROR_OUT();

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

	//     // // construct the right value & the left value scalar equations by characteristic reduction
	//     // // at i+1/2 in x direction
#if 0 == EIGEN_ALLOC
//////////////////////////////////////////////////////////////////////////////////////////////////////////////
#if SCHEME_ORDER == 7
    MARCO_FLUXWALL_WENO7(MARCO_ROEAVERAGE_LEFTZ, MARCO_ROEAVERAGE_RIGHTZ, i, j, k + m, i, j, k + m - stencil_P);
#elif SCHEME_ORDER <= 6
    MARCO_FLUXWALL_WENO5(MARCO_ROEAVERAGE_LEFTZ, MARCO_ROEAVERAGE_RIGHTZ, i, j, k + m, i, j, k + m);
#endif

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

	//     // real_t de_fw[Emax];
	//     // get_Array(Fwall, de_fw, Emax, id_l);
	//     // real_t de_fx[Emax];
}

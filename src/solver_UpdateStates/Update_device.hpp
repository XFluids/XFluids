#pragma once

#include "../read_ini/setupini.h"
#include "../solver_Ini/Mixing_device.h"
#include "../solver_Ini/Mixing_device.h"

SYCL_DEVICE inline void Getrhoyi(real_t *UI, real_t &rho, real_t *yi)
{
	rho = UI[0];
	real_t rho1 = _DF(1.0) / rho;
	// /** ceil(m): get an real_t value >= m and < m+1
	//  * step(a,b): return 1 while  a <= b
	//  */
#ifdef COP
#if defined(GhostSpecies)
	yi[NUM_COP] = _DF(0.0);
	real_t sum_yi = _DF(0.0);
	for (size_t ii = 0; ii < NUM_COP; ii++) // calculate yi
		yi[ii] = UI[ii + 5] * rho1, sum_yi += yi[ii];
	sum_yi = _DF(1.0) / sum_yi;
	for (size_t ii = 0; ii < NUM_COP; ii++)
		yi[ii] *= sum_yi, UI[ii + 5] = rho * yi[ii];
#else
	yi[NUM_COP] = _DF(1.0);
	for (size_t ii = 5; ii < Emax; ii++) // calculate yi
		yi[ii - 5] = UI[ii] * rho1, yi[NUM_COP] += -yi[ii - 5];
#endif // end GhostSpecies
#endif // end COP
}
/**
 * @brief Obtain state at a grid point
 */
SYCL_DEVICE inline void GetStates(real_t *UI, real_t &rho, real_t &u, real_t &v, real_t &w, real_t &p, real_t &H, real_t &c,
						   real_t &gamma, real_t &T, real_t &e, real_t &Cp, real_t &R, Thermal thermal, real_t *yi)
{
	// rho = UI[0];
	real_t rho1 = _DF(1.0) / rho;
	u = UI[1] * rho1, v = UI[2] * rho1, w = UI[3] * rho1;
	real_t tme = UI[4] * rho1 - _DF(0.5) * (u * u + v * v + w * w);

#ifdef COP
	real_t R_ = get_CopR(thermal._Wi, yi);
	T = get_T(thermal, yi, tme, T);
	p = rho * R_ * T, R = R_; // 对所有气体都适用
	Cp = get_CopCp(thermal, yi, T);
	gamma = get_CopGamma(thermal, yi, Cp, T);
#else
	gamma = NCOP_Gamma;
	p = (NCOP_Gamma - _DF(1.0)) * rho * tme; //(UI[4] - _DF(0.5) * rho * (u * u + v * v + w * w));
#endif // end COP
	H = (UI[4] + p) * rho1;
    // [DEBUG] CPU getlu_error
	// c = sycl::sqrt(gamma * p * rho1);
    real_t pres = sycl::fmax(p, _DF(1.0e-30));
    real_t dens = sycl::fmax(rho, _DF(1.0e-30));
    c = sycl::sqrt(gamma * pres / dens);
	e = tme;
}

SYCL_DEVICE inline void GetStatesSP(real_t *UI, real_t &rho, real_t &u, real_t &v, real_t &w, real_t &p, real_t &H, real_t &c, real_t const gamma)
{
	rho = UI[0];
	real_t rho1 = _DF(1.0) / rho;
	u = UI[1] * rho1, v = UI[2] * rho1, w = UI[3] * rho1;

	p = (gamma - _DF(1.0)) * (UI[4] - _DF(0.5) * rho * (u * u + v * v + w * w));
	H = (UI[4] + p) * rho1;
    c = sycl::sqrt(gamma * p * rho1);
}

SYCL_DEVICE inline void ReGetStates(Thermal thermal, real_t *yi, real_t *U, real_t &rho, real_t &u, real_t &v, real_t &w,
							 real_t &p, real_t &T, real_t &H, real_t &c, real_t &e, real_t &gamma)
{
	// real_t h = get_Coph(thermal, yi, T);
	// U[4] = rho * (h + _DF(0.5) * (u * u + v * v + w * w)) - p;

	real_t R = get_CopR(thermal._Wi, yi), rho1 = _DF(1.0) / rho;
	e = U[4] * rho1 - _DF(0.5) * (u * u + v * v + w * w);
	T = get_T(thermal, yi, e, T);
	p = rho * R * T; // 对所有气体都适用
	gamma = get_CopGamma(thermal, yi, T);
	H = (U[4] + p) * rho1;

    // [DEBUG] CPU getlu_error
	// c = sycl::sqrt(gamma * p * rho1);
    real_t pres = sycl::fmax(p, _DF(1.0e-30));
    real_t dens = sycl::fmax(rho, _DF(1.0e-30));
    c = sycl::sqrt(gamma * pres / dens);

	// U[1] = rho * u;
	// U[2] = rho * v;
	// U[3] = rho * w;

	for (size_t nn = 0; nn < NUM_COP; nn++)
		U[nn + 5] = U[0] * yi[nn];
}

/**
 * @brief  Obtain fluxes at a grid point
 */
SYCL_DEVICE inline void GetPhysFlux(real_t *UI, real_t const *yi, real_t *FluxF, real_t *FluxG, real_t *FluxH, real_t const rho,
							 real_t const u, real_t const v, real_t const w, real_t const p, real_t const H, real_t const c)
{
	FluxF[0] = UI[1];
	FluxF[1] = UI[1] * u + p;
	FluxF[2] = UI[1] * v;
	FluxF[3] = UI[1] * w;
	FluxF[4] = (UI[4] + p) * u;

	FluxG[0] = UI[2];
	FluxG[1] = UI[2] * u;
	FluxG[2] = UI[2] * v + p;
	FluxG[3] = UI[2] * w;
	FluxG[4] = (UI[4] + p) * v;

	FluxH[0] = UI[3];
	FluxH[1] = UI[3] * u;
	FluxH[2] = UI[3] * v;
	FluxH[3] = UI[3] * w + p;
	FluxH[4] = (UI[4] + p) * w;

#ifdef COP
	for (size_t ii = 5; ii < Emax; ii++)
	{
		FluxF[ii] = UI[1] * yi[ii - 5];
		FluxG[ii] = UI[2] * yi[ii - 5];
		FluxH[ii] = UI[3] * yi[ii - 5];
	}
#endif
}

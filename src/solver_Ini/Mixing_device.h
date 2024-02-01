#pragma once

#include "Thermo_device.h"
/**
 *@brief debug for array
 */
SYCL_DEVICE inline void get_Array(real_t *Ori, real_t *Out, const int Length, const int id)
{
	for (size_t i = 0; i < Length; i++)
	{
		Out[i] = Ori[Length * id + i];
	}
}

/**
 *@brief calculate yi : mass fraction from xi : mole fraction.
 */
SYCL_DEVICE inline void get_yi(real_t *xi, real_t const *Wi)
{
	real_t W_mix = _DF(0.0);
	for (size_t i = 0; i < NUM_SPECIES; i++)
		W_mix += xi[i] * Wi[i];
	real_t _W_mix = _DF(1.0) / W_mix;
	for (size_t n = 0; n < NUM_SPECIES; n++) // Ri=Ru/Wi
		xi[n] = xi[n] * Wi[n] * _W_mix;
}

/**
 *@brief calculate xi : mole fraction
 */
SYCL_DEVICE inline real_t get_xi(real_t *xi, real_t const *yi, real_t const *_Wi, const real_t rho)
{
	real_t C_total = _DF(0.0);
	for (int i = 0; i < NUM_SPECIES; i++)
	{
		xi[i] = yi[i] * _Wi[i] * _DF(1e-3) * rho;
		C_total = C_total + xi[i];
	}
	// get mole fraction of each specie
	real_t _C_total = _DF(1.0) / C_total;
	for (int i = 0; i < NUM_SPECIES; i++)
		xi[i] = xi[i] * _C_total;
	return C_total;
}

/**
 *@brief calculate R for every cell
 */
SYCL_DEVICE inline real_t get_CopR(const real_t *_Wi, const real_t *yi)
{
	real_t R = _DF(0.0);
	for (size_t n = 0; n < NUM_SPECIES; n++)
		R += yi[n] * _Wi[n];

	return R * Ru;
}

/**
 * @brief Compute the Cp of the mixture at given point unit:J/kg/K
 */
SYCL_DEVICE inline real_t get_CopCp(Thermal thermal, const real_t *yi, const real_t T)
{
	real_t _CopCp = _DF(0.0);
	for (size_t ii = 0; ii < NUM_SPECIES; ii++)
		_CopCp += yi[ii] * HeatCapacity(thermal.Hia, T, thermal.Ri[ii], ii);

	return _CopCp;
}

/**
 * @brief calculate W of the mixture at given point
 */
SYCL_DEVICE inline real_t get_CopW(Thermal thermal, const real_t *yi)
{
	real_t _W = _DF(0.0);
	for (size_t ii = 0; ii < NUM_SPECIES; ii++)
		_W += yi[ii] * thermal._Wi[ii]; // Wi
	return _DF(1.0) / _W;
}

/**
 * @brief calculate Gamma of the mixture at given point
 */
SYCL_DEVICE inline real_t get_CopGamma(Thermal thermal, const real_t *yi, const real_t T)
{
	real_t Cp = get_CopCp(thermal, yi, T);
	real_t CopW = get_CopW(thermal, yi);
	real_t _CopGamma = Cp / (Cp - Ru / CopW);
	if (_CopGamma > _DF(1.0))
	{
		return _CopGamma;
	}
	else
	{
		return -1;
	}
}

/**
 * @brief calculate Gamma of the mixture at given point
 */
SYCL_DEVICE inline real_t get_CopGamma(Thermal thermal, const real_t *yi, const real_t Cp, const real_t T)
{
	real_t CopW = get_CopW(thermal, yi);
	real_t _CopGamma = Cp / (Cp - Ru / CopW);
	if (_CopGamma > _DF(1.0))
	{
		return _CopGamma;
	}
	else
	{
		return -1;
	}
}

/**
 * @brief calculate Hi of Mixture at given point	unit:J/kg/K
 */
SYCL_DEVICE inline real_t get_Coph(Thermal thermal, const real_t *yi, const real_t T)
{
	real_t h = _DF(0.0);
	for (size_t i = 0; i < NUM_SPECIES; i++)
	{
		real_t hi = get_Enthalpy(thermal.Hia, thermal.Hib, T, thermal.Ri[i], i);
		h += hi * yi[i];
	}
	return h;
}

/**
 *@brief sub_function_Steps of update T
 */
SYCL_DEVICE inline void sub_FuncT(real_t &func_T, real_t &dfunc_T, Thermal thermal, const real_t *yi, const real_t e, const real_t T)
{
	real_t h = get_Coph(thermal, yi, T);   // J/kg/K
	real_t R = get_CopR(thermal._Wi, yi);  // J/kg/K
	real_t Cp = get_CopCp(thermal, yi, T); // J/kg/K
	func_T = h - R * T - e;				   // unit:J/kg/K
	dfunc_T = Cp - R;					   // unit:J/kg/K
}

/**
 *@brief update T through Newtonian dynasty
 */
SYCL_DEVICE inline real_t get_T(Thermal thermal, const real_t *yi, const real_t e, const real_t T0)
{
	real_t T = T0;
	real_t tol = _DF(1.0e-6), T_dBdr = _DF(100.0), T_uBdr = _DF(1.0e4), x_eps = _DF(1.0e-3);
	// tol /= Tref, T_dBdr /= Tref, T_uBdr /= Tref, x_eps /= Tref;
	real_t rt_bis, f, f_mid;
	real_t func_T = _DF(0.0), dfunc_T = _DF(0.0);

	for (int i = 1; i < 101; i++)
	{
		sub_FuncT(func_T, dfunc_T, thermal, yi, e, T);
		// NOTE: T<0 makes a majority of NAN erors, three methods:
		// 1. add df limiter
		// 2. add visFlux limiter
		// 3. return origin T while update T<0
		real_t df = sycl::min(func_T / (dfunc_T + _DF(1.0e-30)), _DF(1e-3) * T);
		df = sycl::max(df, -_DF(1e-2) * T);
		T = T - df;
		if (sycl::fabs(df) <= tol)
			break;
		// if (i == 100)
		// {
		// 	// TODO printf("Temperature: Newton_Ramphson iteration failured, try Bisection Metho...d\n");
		// 	sub_FuncT(func_T, dfunc_T, thermal, yi, e, T);
		// 	f_mid = func_T;
		// 	sub_FuncT(func_T, dfunc_T, thermal, yi, e, T);
		// 	f = func_T;
		// 	if (f * f_mid > 0.0)
		// 	{
		// 		// printf("root must be bracketed in rtbis \n");
		// 	}
		// 	if (f < 0.0)
		// 	{
		// 		rt_bis = T_dBdr;
		// 		df = T_uBdr - T_dBdr;
		// 	}
		// 	else
		// 	{
		// 		rt_bis = T_uBdr;
		// 		df = T_dBdr - T_uBdr;
		// 	}
		// 	for (int j = 1; j <= 150; j++)
		// 	{
		// 		df = 0.5 * df;
		// 		T = rt_bis + df;
		// 		sub_FuncT(func_T, dfunc_T, thermal, yi, e, T);
		// 		f_mid = func_T;
		// 		if (f_mid <= 0.0)
		// 			rt_bis = T;
		// 		if (sycl::abs(df) <= x_eps || f_mid == 0.0)
		// 			break;
		// 		if (j == 100)
		// 		{
		// 			// printf("Temperature: Bisect also failured \n");
		// 		}
		// 	}
		// 	break;
		//}
	}
	return T;
}

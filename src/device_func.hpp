#pragma once

#include "global_class.h"

/**
 *@brief debug for array
 */

void get_Array(real_t *Ori, real_t *Out, const int Length, const int id)
{
	for (size_t i = 0; i < Length; i++)
	{
		Out[i] = Ori[Length * id + i];
	}
}

/**
 *@brief calculate yi from y
 */
void get_yi(real_t *const *y, real_t yi[NUM_SPECIES], const int id)
{
#ifdef COP
	for (size_t i = 0; i < NUM_SPECIES; i++)
	{
		yi[i] = y[i][id];
	}
#else
	yi[NUM_COP] = _DF(1.0);
#endif
}

/**
 *@brief calculate xi : mole fraction
 */
real_t get_xi(real_t xi[NUM_SPECIES], real_t const yi[NUM_SPECIES], real_t const *Wi, const real_t rho)
{
	real_t C[NUM_SPECIES] = {_DF(0.0)}, C_total = _DF(0.0);
	for (int i = 0; i < NUM_SPECIES; i++)
	{
		C[i] = yi[i] / Wi[i] * 1e-3 * rho; // Wi==6
		C_total = C_total + C[i];
	}
	// get mole fraction of each specie
	for (int i = 0; i < NUM_SPECIES; i++)
		xi[i] = C[i] / C_total;
	return C_total;
}

/**
 *@brief calculate R for every cell
 */
real_t get_CopR(real_t *species_chara, const real_t yi[NUM_SPECIES])
{
	real_t R = _DF(0.0);
	for (size_t n = 0; n < NUM_SPECIES; n++)
	{
		R += yi[n] * Ru / (species_chara[n * SPCH_Sz + 6]);
	}
	return R;
}

/**
 * @brief calculate Cpi of the specie at given point
 * unit: J/kg/K
 */
real_t get_Cpi(real_t *Hia, const real_t T0, const real_t Ri, const int n)
{
	real_t T = T0, Cpi = _DF(0.0);
	if (T < (_DF(200.0) / Tref))
		T = _DF(200.0) / Tref;
#if Thermo
	if (T >= (1000.0 / Tref) && T < (6000.0 / Tref))
		Cpi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 1] / T / T + Hia[n * 7 * 3 + 1 * 3 + 1] / T + Hia[n * 7 * 3 + 2 * 3 + 1] + Hia[n * 7 * 3 + 3 * 3 + 1] * T + Hia[n * 7 * 3 + 4 * 3 + 1] * T * T + Hia[n * 7 * 3 + 5 * 3 + 1] * T * T * T + Hia[n * 7 * 3 + 6 * 3 + 1] * T * T * T * T);
	else if (T < (1000.0 / Tref))
		Cpi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 0] / T / T + Hia[n * 7 * 3 + 1 * 3 + 0] / T + Hia[n * 7 * 3 + 2 * 3 + 0] + Hia[n * 7 * 3 + 3 * 3 + 0] * T + Hia[n * 7 * 3 + 4 * 3 + 0] * T * T + Hia[n * 7 * 3 + 5 * 3 + 0] * T * T * T + Hia[n * 7 * 3 + 6 * 3 + 0] * T * T * T * T);
	else if (T >= (6000.0 / Tref) && T < (15000.0 / Tref))
	{
		Cpi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 2] / T / T + Hia[n * 7 * 3 + 1 * 3 + 2] / T + Hia[n * 7 * 3 + 2 * 3 + 2] + Hia[n * 7 * 3 + 3 * 3 + 2] * T + Hia[n * 7 * 3 + 4 * 3 + 2] * T * T + Hia[n * 7 * 3 + 5 * 3 + 2] * T * T * T + Hia[n * 7 * 3 + 6 * 3 + 2] * T * T * T * T);
	}
	else
	{
		// TODO printf("T=%lf , Cpi=%lf , T > 15000 K,please check!!!NO Cpi[n] for T>15000 K \n", T, Cpi);
	}
#else
	// Cpi[n)/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
	if (T > (_DF(1000.0) / Tref))
		Cpi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 0] + Hia[n * 7 * 3 + 1 * 3 + 0] * T + Hia[n * 7 * 3 + 2 * 3 + 0] * T * T + Hia[n * 7 * 3 + 3 * 3 + 0] * T * T * T + Hia[n * 7 * 3 + 4 * 3 + 0] * T * T * T * T);
	else
		Cpi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 1] + Hia[n * 7 * 3 + 1 * 3 + 1] * T + Hia[n * 7 * 3 + 2 * 3 + 1] * T * T + Hia[n * 7 * 3 + 3 * 3 + 1] * T * T * T + Hia[n * 7 * 3 + 4 * 3 + 1] * T * T * T * T);
#endif
	return Cpi;
}

/**
 * @brief Compute the Cp of the mixture at given point unit:J/kg/K
 */
real_t get_CopCp(Thermal *material, const real_t yi[NUM_SPECIES], const real_t T)
{
	real_t _CopCp = _DF(0.0);
	for (size_t ii = 0; ii < NUM_SPECIES; ii++)
		_CopCp += yi[ii] * get_Cpi(material->Hia, T, material->Ri[ii], ii); // real_t Cpi = get_Cpi(material->Hia, T, Ri, ii) ;
	// printf("Cpi=%lf , %lf , yi=%lf , %lf , _CopCp=%lf \n", Cpi[0], Cpi[1], yi[0], yi[1], _CopCp);
	return _CopCp;
}

/**
 * @brief calculate W of the mixture at given point
 */
real_t get_CopW(Thermal *material, const real_t yi[NUM_SPECIES])
{
	real_t _W = _DF(0.0);
	for (size_t ii = 0; ii < NUM_SPECIES; ii++)
	{
		_W += yi[ii] / (material->species_chara[ii * SPCH_Sz + 6]); // Wi
	}
	// printf("W=%lf \n", 1.0 / _W);
	return _DF(1.0) / _W;
}

/**
 * @brief calculate Gamma of the mixture at given point
 */
real_t get_CopGamma(Thermal *material, const real_t yi[NUM_SPECIES], const real_t T)
{
	real_t Cp = get_CopCp(material, yi, T);
	real_t CopW = get_CopW(material, yi);
	real_t _CopGamma = Cp / (Cp - Ru / CopW);
	if (_CopGamma > 1)
	{
		return _CopGamma;
	}
	else
	{
		return 0;
	}
}

/**
 * @brief calculate Hi of every compoent at given point	unit:J/kg/K // get_hi
 */
real_t get_Enthalpy(real_t *Hia, real_t *Hib, const real_t T0, const real_t Ri, const int n)
{
	real_t T = T0;
	real_t TT = 30000.0 / Tref;
	real_t hi = _DF(0.0);
#if Thermo
	// NOTE：Non_dim of Hia && Hib*3+only for h&Cp not for S ATTENTATION
	//  200K~1000K
	Hia[n * 7 * 3 + 0 * 3 + 0] = Hia[n * 7 * 3 + 0 * 3 + 0] / sycl::pow<real_t>(Tref, 2);
	Hia[n * 7 * 3 + 1 * 3 + 0] = Hia[n * 7 * 3 + 1 * 3 + 0] / Tref;

	Hia[n * 7 * 3 + 3 * 3 + 0] = Hia[n * 7 * 3 + 3 * 3 + 0] * Tref;
	Hia[n * 7 * 3 + 4 * 3 + 0] = Hia[n * 7 * 3 + 4 * 3 + 0] * sycl::pow<real_t>(Tref, 2);
	Hia[n * 7 * 3 + 5 * 3 + 0] = Hia[n * 7 * 3 + 5 * 3 + 0] * sycl::pow<real_t>(Tref, 3);
	Hia[n * 7 * 3 + 6 * 3 + 0] = Hia[n * 7 * 3 + 6 * 3 + 0] * sycl::pow<real_t>(Tref, 4);
	Hib[n * 2 * 3 + 0 * 3 + 0] = Hib[n * 2 * 3 + 0 * 3 + 0] / Tref + Hia[n * 7 * 3 + 1 * 3 + 0] * sycl::log(Tref);
	// 1000K~6000K
	Hia[n * 7 * 3 + 0 * 3 + 1] = Hia[n * 7 * 3 + 0 * 3 + 1] / sycl::pow<real_t>(Tref, 2);
	Hia[n * 7 * 3 + 1 * 3 + 1] = Hia[n * 7 * 3 + 1 * 3 + 1] / Tref;

	Hia[n * 7 * 3 + 3 * 3 + 1] = Hia[n * 7 * 3 + 3 * 3 + 1] * Tref;
	Hia[n * 7 * 3 + 4 * 3 + 1] = Hia[n * 7 * 3 + 4 * 3 + 1] * sycl::pow<real_t>(Tref, 2);
	Hia[n * 7 * 3 + 5 * 3 + 1] = Hia[n * 7 * 3 + 5 * 3 + 1] * sycl::pow<real_t>(Tref, 3);
	Hia[n * 7 * 3 + 6 * 3 + 1] = Hia[n * 7 * 3 + 6 * 3 + 1] * sycl::pow<real_t>(Tref, 4);
	Hib[n * 2 * 3 + 0 * 3 + 1] = Hib[n * 2 * 3 + 0 * 3 + 1] / Tref + Hia[n * 7 * 3 + 1 * 3 + 1] * sycl::log(Tref);
	// 6000K~15000K
	Hia[n * 7 * 3 + 0 * 3 + 2] = Hia[n * 7 * 3 + 0 * 3 + 2] / sycl::pow<real_t>(Tref, 2);
	Hia[n * 7 * 3 + 1 * 3 + 2] = Hia[n * 7 * 3 + 1 * 3 + 2] / Tref;

	Hia[n * 7 * 3 + 3 * 3 + 2] = Hia[n * 7 * 3 + 3 * 3 + 2] * Tref;
	Hia[n * 7 * 3 + 4 * 3 + 2] = Hia[n * 7 * 3 + 4 * 3 + 2] * sycl::pow<real_t>(Tref, 2);
	Hia[n * 7 * 3 + 5 * 3 + 2] = Hia[n * 7 * 3 + 5 * 3 + 2] * sycl::pow<real_t>(Tref, 3);
	Hia[n * 7 * 3 + 6 * 3 + 2] = Hia[n * 7 * 3 + 6 * 3 + 2] * sycl::pow<real_t>(Tref, 4);
	Hib[n * 2 * 3 + 0 * 3 + 2] = Hib[n * 2 * 3 + 0 * 3 + 2] / Tref + Hia[n * 7 * 3 + 1 * 3 + 2] * sycl::log(Tref);
#else
	Hia[n * 7 * 3 + 1 * 3 + 0] = Hia[n * 7 * 3 + 1 * 3 + 0] * Tref;
	Hia[n * 7 * 3 + 2 * 3 + 0] = Hia[n * 7 * 3 + 2 * 3 + 0] * sycl::pow<real_t>(Tref, 2);
	Hia[n * 7 * 3 + 3 * 3 + 0] = Hia[n * 7 * 3 + 3 * 3 + 0] * sycl::pow<real_t>(Tref, 3);
	Hia[n * 7 * 3 + 4 * 3 + 0] = Hia[n * 7 * 3 + 4 * 3 + 0] * sycl::pow<real_t>(Tref, 4);
	Hia[n * 7 * 3 + 5 * 3 + 0] = Hia[n * 7 * 3 + 5 * 3 + 0] / Tref;

	Hia[n * 7 * 3 + 1 * 3 + 1] = Hia[n * 7 * 3 + 1 * 3 + 1] * Tref;
	Hia[n * 7 * 3 + 2 * 3 + 1] = Hia[n * 7 * 3 + 2 * 3 + 1] * sycl::pow<real_t>(Tref, 2);
	Hia[n * 7 * 3 + 3 * 3 + 1] = Hia[n * 7 * 3 + 3 * 3 + 1] * sycl::pow<real_t>(Tref, 3);
	Hia[n * 7 * 3 + 4 * 3 + 1] = Hia[n * 7 * 3 + 4 * 3 + 1] * sycl::pow<real_t>(Tref, 4);
	Hia[n * 7 * 3 + 5 * 3 + 1] = Hia[n * 7 * 3 + 5 * 3 + 1] / Tref;
#endif
	if (T < 200.0 / Tref)
	{
		TT = T;
		T = 200.0 / Tref;
	}
#if Thermo
	if (T >= (1000.0 / Tref) && T < (6000.0 / Tref))
		hi = Ri * (-Hia[n * 7 * 3 + 0 * 3 + 1] * 1.0 / T + Hia[n * 7 * 3 + 1 * 3 + 1] * sycl::log(T) + Hia[n * 7 * 3 + 2 * 3 + 1] * T + 0.5 * Hia[n * 7 * 3 + 3 * 3 + 1] * T * T + Hia[n * 7 * 3 + 4 * 3 + 1] * sycl::pow<real_t>(T, 3) / 3.0 + Hia[n * 7 * 3 + 5 * 3 + 1] * sycl::pow<real_t>(T, 4) / 4.0 + Hia[n * 7 * 3 + 6 * 3 + 1] * sycl::pow<real_t>(T, 5) / 5.0 + Hib[n * 2 * 3 + 0 * 3 + 1]);
	else if (T < (1000.0 / Tref))
	{
		hi = Ri * (-Hia[n * 7 * 3 + 0 * 3 + 0] * 1.0 / T + Hia[n * 7 * 3 + 1 * 3 + 0] * sycl::log(T) + Hia[n * 7 * 3 + 2 * 3 + 0] * T + 0.5 * Hia[n * 7 * 3 + 3 * 3 + 0] * T * T + Hia[n * 7 * 3 + 4 * 3 + 0] * sycl::pow<real_t>(T, 3) / 3.0 + Hia[n * 7 * 3 + 5 * 3 + 0] * sycl::pow<real_t>(T, 4) / 4.0 + Hia[n * 7 * 3 + 6 * 3 + 0] * sycl::pow<real_t>(T, 5) / 5.0 + Hib[n * 2 * 3 + 0 * 3 + 0]);
	}
	else if (T >= (6000.0 / Tref) && T < (15000.0 / Tref))
	{
		hi = Ri * (-Hia[n * 7 * 3 + 0 * 3 + 2] * 1.0 / T + Hia[n * 7 * 3 + 1 * 3 + 2] * sycl::log(T) + Hia[n * 7 * 3 + 2 * 3 + 2] * T + 0.5 * Hia[n * 7 * 3 + 3 * 3 + 2] * T * T + Hia[n * 7 * 3 + 4 * 3 + 2] * sycl::pow<real_t>(T, 3) / 3.0 + Hia[n * 7 * 3 + 5 * 3 + 2] * sycl::pow<real_t>(T, 4) / 4.0 + Hia[n * 7 * 3 + 6 * 3 + 2] * sycl::pow<real_t>(T, 5) / 5.0 + Hib[n * 2 * 3 + 0 * 3 + 2]);
	}
	else
	{
		// TODO printf("T=%lf,T > 15000 K,please check!!!NO h for T>15000 K. \n", T * Tref);
		return 0;
	}
#else
	// H/RT = a1 + a2/2*T + a3/3*T^2 + a4/4*T^3 + a5/5*T^4 + a6/T
	if (T > (1000.0 / Tref))
		hi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 0] * T + Hia[n * 7 * 3 + 1 * 3 + 0] * T * T / 2.0 + Hia[n * 7 * 3 + 2 * 3 + 0] * sycl::pow<real_t>(T, 3) / 3.0 + Hia[n * 7 * 3 + 3 * 3 + 0] * sycl::pow<real_t>(T, 4) / 4.0 + Hia[n * 7 * 3 + 4 * 3 + 0] * sycl::pow<real_t>(T, 5) / 5.0 + Hia[n * 7 * 3 + 5 * 3 + 0]);
	else
		hi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 1] * T + Hia[n * 7 * 3 + 1 * 3 + 1] * T * T / 2.0 + Hia[n * 7 * 3 + 2 * 3 + 1] * sycl::pow<real_t>(T, 3) / 3.0 + Hia[n * 7 * 3 + 3 * 3 + 1] * sycl::pow<real_t>(T, 4) / 4.0 + Hia[n * 7 * 3 + 4 * 3 + 1] * sycl::pow<real_t>(T, 5) / 5.0 + Hia[n * 7 * 3 + 5 * 3 + 1]);
#endif
	// printf("hi[n] of get_hi=%lf \n", hi[n]);
	// get_hi at T>200
	if (TT < 200.0 / Tref)
	{
		real_t Cpi = get_Cpi(Hia, 200.0 / Tref, Ri, n); // get_Cpi(real_t *Hia, const real_t T0, const real_t Ri, const int n)
		hi += Cpi * (TT - 200.0 / Tref);
		// printf("hi[%d] = %lf , Cpi[%d]=%lf \n", ii, hi[ii], ii, Cpi[ii]);
	}
	return hi;
}

/**
 * @brief calculate Hi of Mixture at given point	unit:J/kg/K
 */
real_t get_Coph(Thermal *material, const real_t yi[NUM_SPECIES], const real_t T)
{
	real_t H = 0.0, hi[NUM_SPECIES];
	for (size_t i = 0; i < NUM_SPECIES; i++)
	{
		real_t hi = get_Enthalpy(material->Hia, material->Hib, T, material->Ri[i], i);
		H += hi * yi[i];
	}
	return H;
}

/**
 *@brief sub_function_Steps of update T
 */
void sub_FuncT(real_t &func_T, real_t &dfunc_T, Thermal *thermal, const real_t yi[NUM_SPECIES], const real_t e, const real_t T)
{
	real_t h = get_Coph(thermal, yi, T);			 // J/kg/K
	real_t R = get_CopR(thermal->species_chara, yi); // J/kg/K
	real_t Cp = get_CopCp(thermal, yi, T);			 // J/kg/K
	func_T = h - R * T - e;							 // unit:J/kg/K
	dfunc_T = Cp - R;								 // unit:J/kg/K
}

/**
 *@brief update T through Newtonian dynasty
 */
real_t get_T(Thermal *thermal, const real_t yi[NUM_SPECIES], const real_t e, const real_t T0)
{
	real_t T = T0;
	real_t tol = 1.0e-6, T_dBdr = 100.0, T_uBdr = 1.0e4, x_eps = 1.0e-3;
	tol /= Tref, T_dBdr /= Tref, T_uBdr /= Tref, x_eps /= Tref;
	real_t rt_bis, f, f_mid;
	real_t func_T = 0, dfunc_T = 0;

	for (int i = 1; i <= 150; i++)
	{
		sub_FuncT(func_T, dfunc_T, thermal, yi, e, T);
		real_t df = func_T / dfunc_T;
		T = T - df;
		if (std::abs(df) <= tol)
			break;
		if (i == 100)
		{
			// TODO printf("Temperature: Newton_Ramphson iteration failured, try Bisection Metho...d\n");
			sub_FuncT(func_T, dfunc_T, thermal, yi, e, T);
			f_mid = func_T;
			sub_FuncT(func_T, dfunc_T, thermal, yi, e, T);
			f = func_T;
			if (f * f_mid > 0.0)
			{
				// printf("root must be bracketed in rtbis \n");
			}
			if (f < 0.0)
			{
				rt_bis = T_dBdr;
				df = T_uBdr - T_dBdr;
			}
			else
			{
				rt_bis = T_uBdr;
				df = T_dBdr - T_uBdr;
			}
			for (int j = 1; j <= 150; j++)
			{
				df = 0.5 * df;
				T = rt_bis + df;
				sub_FuncT(func_T, dfunc_T, thermal, yi, e, T);
				f_mid = func_T;
				if (f_mid <= 0.0)
					rt_bis = T;
				if (std::abs(df) <= x_eps || f_mid == 0.0)
					break;
				if (j == 100)
				{
					// printf("Temperature: Bisect also failured \n");
				}
			}
			break;
		}
	}
	return T;
}

/**
 * @brief Obtain state at a grid point
 */
void GetStates(real_t UI[Emax], real_t &rho, real_t &u, real_t &v, real_t &w, real_t &p, real_t &H, real_t &c,
			   real_t &T, Thermal *thermal, const real_t yi[NUM_SPECIES])
{
	rho = UI[0];
	real_t rho1 = _DF(1.0) / rho;
	u = UI[1] * rho1;
	v = UI[2] * rho1;
	w = UI[3] * rho1;

	real_t e = UI[4] * rho1 - _DF(0.5) * (u * u + v * v + w * w);
	T = get_T(thermal, yi, e, T);
	real_t R = get_CopR(thermal->species_chara, yi);
	p = rho * R * T; // 对所有气体都适用
	real_t Gamma = get_CopGamma(thermal, yi, T);
	H = (UI[4] + p) * rho1;
	c = sqrt(Gamma * p * rho1);
}

/**
 * @brief  Obtain fluxes at a grid point
 */
void GetPhysFlux(real_t UI[Emax], real_t const yi[NUM_COP], real_t *FluxF, real_t *FluxG, real_t *FluxH,
				 real_t const rho, real_t const u, real_t const v, real_t const w, real_t const p, real_t const H, real_t const c)
{
	FluxF[0] = UI[1];
	FluxF[1] = UI[1]*u + p;
	FluxF[2] = UI[1]*v;
	FluxF[3] = UI[1]*w;
	FluxF[4] = (UI[4] + p)*u;

	FluxG[0] = UI[2];
	FluxG[1] = UI[2]*u;
	FluxG[2] = UI[2]*v + p;
	FluxG[3] = UI[2]*w;
	FluxG[4] = (UI[4] + p)*v;

	FluxH[0] = UI[3];
	FluxH[1] = UI[3]*u;
	FluxH[2] = UI[3]*v;
	FluxH[3] = UI[3]*w + p;
	FluxH[4] = (UI[4] + p)*w;

#ifdef COP
	for (size_t ii = 5; ii < Emax; ii++)
	{
		FluxF[ii] = UI[1] * yi[ii - 5];
		FluxG[ii] = UI[2] * yi[ii - 5];
		FluxH[ii] = UI[3] * yi[ii - 5];
	}
#endif
}

inline void RoeAverage_x(real_t eigen_l[Emax][Emax], real_t eigen_r[Emax][Emax], real_t z[NUM_SPECIES], const real_t yi[NUM_SPECIES], real_t const c2,
						 real_t const _rho, real_t const _u, real_t const _v, real_t const _w,
						 real_t const _H, real_t const D, real_t const D1, real_t Gamma)
{

	real_t _Gamma = Gamma - _DF(1.0);
	real_t q2 = _u * _u + _v * _v + _w * _w;
	// real_t c2 = _Gamma * (_H - _DF(0.5) * q2);
	real_t _c = sqrt(c2);
	real_t c21_Gamma = _Gamma / c2;

	real_t b1 = c21_Gamma;
	real_t b2 = _DF(1.0) + c21_Gamma * q2 - c21_Gamma * _H;
	real_t b3 = _DF(0.0);
	for (size_t i = 0; i < NUM_COP; i++) // NOTE: related with yi
		b3 += yi[i] * z[i];
	b3 *= b1;
	real_t _c1 = _DF(1.0) / _c;

	eigen_l[0][0] = _DF(0.5) * (b2 + _u / _c + b3);
	eigen_l[0][1] = -_DF(0.5) * (b1 * _u + _c1);
	eigen_l[0][2] = -_DF(0.5) * (b1 * _v);
	eigen_l[0][3] = -_DF(0.5) * (b1 * _w);
	eigen_l[0][4] = _DF(0.5) * b1;

	eigen_l[1][0] = (_DF(1.0) - b2 - b3) / b1;
	eigen_l[1][1] = _u;
	eigen_l[1][2] = _v;
	eigen_l[1][3] = _w;
	eigen_l[1][4] = -_DF(1.0);

	eigen_l[2][0] = _v;
	eigen_l[2][1] = _DF(0.0);
	eigen_l[2][2] = -_DF(1.0);
	eigen_l[2][3] = _DF(0.0);
	eigen_l[2][4] = _DF(0.0);

	eigen_l[3][0] = -_w;
	eigen_l[3][1] = _DF(0.0);
	eigen_l[3][2] = _DF(0.0);
	eigen_l[3][3] = _DF(1.0);
	eigen_l[3][4] = _DF(0.0);

	eigen_l[Emax - 1][0] = _DF(0.5) * (b2 - _u / _c + b3);
	eigen_l[Emax - 1][1] = _DF(0.5) * (-b1 * _u + _c1);
	eigen_l[Emax - 1][2] = _DF(0.5) * (-b1 * _v);
	eigen_l[Emax - 1][3] = _DF(0.5) * (-b1 * _w);
	eigen_l[Emax - 1][4] = _DF(0.5) * b1;

	// right eigen vectors
	eigen_r[0][0] = _DF(1.0);
	eigen_r[0][1] = b1;
	eigen_r[0][2] = _DF(0.0);
	eigen_r[0][3] = _DF(0.0);
	eigen_r[0][Emax - 1] = _DF(1.0);

	eigen_r[1][0] = _u - _c;
	eigen_r[1][1] = _u * b1;
	eigen_r[1][2] = _DF(0.0);
	eigen_r[1][3] = _DF(0.0);
	eigen_r[1][Emax - 1] = _u + _c;

	eigen_r[2][0] = _v;
	eigen_r[2][1] = _v * b1;
	eigen_r[2][2] = -_DF(1.0);
	eigen_r[2][3] = _DF(0.0);
	eigen_r[2][Emax - 1] = _v;

	eigen_r[3][0] = _w;
	eigen_r[3][1] = _w * b1;
	eigen_r[3][2] = _DF(0.0);
	eigen_r[3][3] = _DF(1.0);
	eigen_r[3][Emax - 1] = _w;

	eigen_r[4][0] = _H - _u * _c;
	eigen_r[4][1] = _H * b1 - _DF(1.0);
	eigen_r[4][2] = -_v;
	eigen_r[4][3] = _w;
	eigen_r[4][Emax - 1] = _H + _u * _c;

#ifdef COP
	// left eigen vectors
	for (int n = 0; n < NUM_COP; n++)
	{
		eigen_l[0][n + Emax - NUM_COP] = -_DF(0.5) * b1 * z[n]; // NOTE: related with yi eigen values
		eigen_l[1][n + Emax - NUM_COP] = z[n];
		eigen_l[2][n + Emax - NUM_COP] = _DF(0.0);
		eigen_l[3][n + Emax - NUM_COP] = _DF(0.0);
		eigen_l[Emax - 1][n + Emax - NUM_COP] = -_DF(0.5) * b1 * z[n];
	}
	for (int m = 0; m < NUM_COP; m++)
	{
		eigen_l[m + Emax - NUM_SPECIES][0] = -yi[m];
		eigen_l[m + Emax - NUM_SPECIES][1] = _DF(0.0);
		eigen_l[m + Emax - NUM_SPECIES][2] = _DF(0.0);
		eigen_l[m + Emax - NUM_SPECIES][3] = _DF(0.0);
		eigen_l[m + Emax - NUM_SPECIES][4] = _DF(0.0);
		for (int n = 0; n < NUM_COP; n++)
			eigen_l[m + Emax - NUM_SPECIES][n + Emax - NUM_COP] = (m == n) ? _DF(1.0) : _DF(0.0);
	}
	// right eigen vectors
	for (int n = 0; n < NUM_COP; n++)
	{
		eigen_r[0][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[1][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[2][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[3][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[4][Emax - NUM_COP + n - 1] = z[n];
	}
	for (int m = 0; m < NUM_COP; m++)
	{
		eigen_r[m + Emax - NUM_COP][0] = yi[m];
		eigen_r[m + Emax - NUM_COP][1] = b1 * yi[m];
		eigen_r[m + Emax - NUM_COP][2] = _DF(0.0);
		eigen_r[m + Emax - NUM_COP][3] = _DF(0.0);
		eigen_r[m + Emax - NUM_COP][Emax - 1] = yi[m];
		for (int n = 0; n < NUM_COP; n++)
			eigen_r[m + Emax - NUM_COP][n + Emax - NUM_SPECIES] = (m == n) ? _DF(1.0) : _DF(0.0);
	}
#endif // COP
}

inline void RoeAverage_y(real_t eigen_l[Emax][Emax], real_t eigen_r[Emax][Emax], real_t z[NUM_SPECIES], const real_t yi[NUM_SPECIES], real_t const c2,
						 real_t const _rho, real_t const _u, real_t const _v, real_t const _w,
						 real_t const _H, real_t const D, real_t const D1, real_t const Gamma)
{

	real_t _Gamma = Gamma - _DF(1.0);
	real_t q2 = _u*_u + _v*_v + _w*_w;
	// real_t c2 = _Gamma*(_H - _DF(0.5)*q2);
	real_t _c = sqrt(c2);
	real_t c21_Gamma = _Gamma / c2;

	real_t b1 = c21_Gamma;
	real_t b2 = _DF(1.0) + c21_Gamma * q2 - c21_Gamma * _H;
	real_t b3 = _DF(0.0);
	for (size_t i = 0; i < NUM_COP; i++)
		b3 += yi[i] * z[i];
	b3 *= b1;
	real_t _c1 = _DF(1.0) / _c;

	eigen_l[0][0] = _DF(0.5) * (b2 + _v / _c + b3);
	eigen_l[0][1] = -_DF(0.5) * (b1 * _u);
	eigen_l[0][2] = -_DF(0.5) * (b1 * _v + _c1);
	eigen_l[0][3] = -_DF(0.5) * (b1 * _w);
	eigen_l[0][4] = _DF(0.5) * b1;

	eigen_l[1][0] = -_u;
	eigen_l[1][1] = _DF(1.0);
	eigen_l[1][2] = _DF(0.0);
	eigen_l[1][3] = _DF(0.0);
	eigen_l[1][4] = _DF(0.0);

	eigen_l[2][0] = (_DF(1.0) - b2 - b3) / b1;
	eigen_l[2][1] = _u;
	eigen_l[2][2] = _v;
	eigen_l[2][3] = _w;
	eigen_l[2][4] = -_DF(1.0);

	eigen_l[3][0] = _w;
	eigen_l[3][1] = _DF(0.0);
	eigen_l[3][2] = _DF(0.0);
	eigen_l[3][3] = -_DF(1.0);
	eigen_l[3][4] = _DF(0.0);

	eigen_l[Emax - 1][0] = _DF(0.5) * (b2 - _v / _c + b3);
	eigen_l[Emax - 1][1] = _DF(0.5) * (-b1 * _u);
	eigen_l[Emax - 1][2] = _DF(0.5) * (-b1 * _v + _c1);
	eigen_l[Emax - 1][3] = _DF(0.5) * (-b1 * _w);
	eigen_l[Emax - 1][4] = _DF(0.5) * b1;

	// right eigen vectors
	eigen_r[0][0] = _DF(1.0);
	eigen_r[0][1] = _DF(0.0);
	eigen_r[0][2] = b1;
	eigen_r[0][3] = _DF(0.0);
	eigen_r[0][Emax - 1] = _DF(1.0);

	eigen_r[1][0] = _u;
	eigen_r[1][1] = _DF(1.0);
	eigen_r[1][2] = _u * b1;
	eigen_r[1][3] = _DF(0.0);
	eigen_r[1][Emax - 1] = _u;

	eigen_r[2][0] = _v - _c;
	eigen_r[2][1] = _DF(0.0);
	eigen_r[2][2] = _v * b1;
	eigen_r[2][3] = _DF(0.0);
	eigen_r[2][Emax - 1] = _v + _c;

	eigen_r[3][0] = _w;
	eigen_r[3][1] = _DF(0.0);
	eigen_r[3][2] = _w * b1;
	eigen_r[3][3] = -_DF(1.0);
	eigen_r[3][Emax - 1] = _w;

	eigen_r[4][0] = _H - _v * _c;
	eigen_r[4][1] = _u;
	eigen_r[4][2] = _H * b1 - _DF(1.0);
	eigen_r[4][3] = -_w;
	eigen_r[4][Emax - 1] = _H + _v * _c;

#ifdef COP
	// left eigen vectors
	for (int n = 0; n < NUM_COP; n++)
	{
		eigen_l[0][n + Emax - NUM_COP] = -_DF(0.5) * b1 * z[n];
		eigen_l[1][n + Emax - NUM_COP] = _DF(0.0);
		eigen_l[2][n + Emax - NUM_COP] = z[n];
		eigen_l[3][n + Emax - NUM_COP] = _DF(0.0);
		eigen_l[Emax - 1][n + Emax - NUM_COP] = -_DF(0.5) * b1 * z[n];
	}
	for (int m = 0; m < NUM_COP; m++)
	{
		eigen_l[m + Emax - NUM_SPECIES][0] = -yi[m];
		eigen_l[m + Emax - NUM_SPECIES][1] = _DF(0.0);
		eigen_l[m + Emax - NUM_SPECIES][2] = _DF(0.0);
		eigen_l[m + Emax - NUM_SPECIES][3] = _DF(0.0);
		eigen_l[m + Emax - NUM_SPECIES][4] = _DF(0.0);
		for (int n = 0; n < NUM_COP; n++)
			eigen_l[m + Emax - NUM_SPECIES][n + Emax - NUM_COP] = (m == n) ? _DF(1.0) : _DF(0.0);
	}
	// right eigen vectors
	for (int n = 0; n < NUM_COP; n++)
	{
		eigen_r[0][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[1][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[2][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[3][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[4][Emax - NUM_COP + n - 1] = z[n];
	}
	for (int m = 0; m < NUM_COP; m++)
	{
		eigen_r[m + Emax - NUM_COP][0] = yi[m];
		eigen_r[m + Emax - NUM_COP][1] = _DF(0.0);
		eigen_r[m + Emax - NUM_COP][2] = b1 * yi[m];
		eigen_r[m + Emax - NUM_COP][3] = _DF(0.0);
		eigen_r[m + Emax - NUM_COP][Emax - 1] = yi[m];
		for (int n = 0; n < NUM_COP; n++)
			eigen_r[m + Emax - NUM_COP][n + Emax - NUM_SPECIES] = (m == n) ? _DF(1.0) : _DF(0.0);
	}
#endif // COP
}

inline void RoeAverage_z(real_t eigen_l[Emax][Emax], real_t eigen_r[Emax][Emax], real_t z[NUM_SPECIES], const real_t yi[NUM_SPECIES], real_t const c2,
						 real_t const _rho, real_t const _u, real_t const _v, real_t const _w,
						 real_t const _H, real_t const D, real_t const D1, real_t const Gamma)
{
	// preparing some interval value
	real_t _Gamma = Gamma - _DF(1.0);
	real_t q2 = _u * _u + _v * _v + _w * _w;
	// real_t c2 = _Gamma*(_H - _DF(0.5)*q2);
	real_t _c = sqrt(c2);
	real_t c21_Gamma = _Gamma / c2;

	real_t b1 = c21_Gamma;
	real_t b2 = _DF(1.0) + c21_Gamma * q2 - c21_Gamma * _H;
	real_t b3 = _DF(0.0);
	for (size_t i = 0; i < NUM_COP; i++)
		b3 += yi[i] * z[i];
	b3 *= b1;
	real_t _c1 = _DF(1.0) / _c;

	// left eigen vectors
	eigen_l[0][0] = _DF(0.5) * (b2 + _w / _c + b3);
	eigen_l[0][1] = -_DF(0.5) * (b1 * _u);
	eigen_l[0][2] = -_DF(0.5) * (b1 * _v);
	eigen_l[0][3] = -_DF(0.5) * (b1 * _w + _c1);
	eigen_l[0][4] = _DF(0.5) * b1;

	eigen_l[1][0] = _u;
	eigen_l[1][1] = -_DF(1.0);
	eigen_l[1][2] = _DF(0.0);
	eigen_l[1][3] = _DF(0.0);
	eigen_l[1][4] = _DF(0.0);

	eigen_l[2][0] = -_v;
	eigen_l[2][1] = _DF(0.0);
	eigen_l[2][2] = _DF(1.0);
	eigen_l[2][3] = _DF(0.0);
	eigen_l[2][4] = _DF(0.0);

	eigen_l[3][0] = (_DF(1.0) - b2 - b3) / b1; //-q2 + _H;
	eigen_l[3][1] = _u;
	eigen_l[3][2] = _v;
	eigen_l[3][3] = _w;
	eigen_l[3][4] = -_DF(1.0);

	eigen_l[Emax - 1][0] = _DF(0.5) * (b2 - _w / _c + b3);
	eigen_l[Emax - 1][1] = _DF(0.5) * (-b1 * _u);
	eigen_l[Emax - 1][2] = _DF(0.5) * (-b1 * _v);
	eigen_l[Emax - 1][3] = _DF(0.5) * (-b1 * _w + _c1);
	eigen_l[Emax - 1][4] = _DF(0.5) * b1;

	// right eigen vectors
	eigen_r[0][0] = _DF(1.0);
	eigen_r[0][1] = _DF(0.0);
	eigen_r[0][2] = _DF(0.0);
	eigen_r[0][3] = b1;
	eigen_r[0][Emax - 1] = _DF(1.0);

	eigen_r[1][0] = _u;
	eigen_r[1][1] = -_DF(1.0);
	eigen_r[1][2] = _DF(0.0);
	eigen_r[1][3] = b1 * _u;
	eigen_r[1][Emax - 1] = _u;

	eigen_r[2][0] = _v;
	eigen_r[2][1] = _DF(0.0);
	eigen_r[2][2] = _DF(1.0);
	eigen_r[2][3] = b1 * _v;
	eigen_r[2][Emax - 1] = _v;

	eigen_r[3][0] = _w - _c;
	eigen_r[3][1] = _DF(0.0);
	eigen_r[3][2] = _DF(0.0);
	eigen_r[3][3] = b1 * _w;
	eigen_r[3][Emax - 1] = _w + _c;

	eigen_r[4][0] = _H - _w * _c;
	eigen_r[4][1] = -_u;
	eigen_r[4][2] = _v;
	eigen_r[4][3] = _H * b1 - _DF(1.0);
	eigen_r[4][Emax - 1] = _H + _w * _c;

#ifdef COP
	// left eigen vectors
	for (int n = 0; n < NUM_COP; n++)
	{
		eigen_l[0][n + Emax - NUM_COP] = -_DF(0.5) * b1 * z[n];
		eigen_l[1][n + Emax - NUM_COP] = _DF(0.0);
		eigen_l[2][n + Emax - NUM_COP] = _DF(0.0);
		eigen_l[3][n + Emax - NUM_COP] = z[n];
		eigen_l[Emax - 1][n + Emax - NUM_COP] = -_DF(0.5) * b1 * z[n];
	}
	for (int m = 0; m < NUM_COP; m++)
	{
		eigen_l[m + Emax - NUM_SPECIES][0] = -yi[m];
		eigen_l[m + Emax - NUM_SPECIES][1] = _DF(0.0);
		eigen_l[m + Emax - NUM_SPECIES][2] = _DF(0.0);
		eigen_l[m + Emax - NUM_SPECIES][3] = _DF(0.0);
		eigen_l[m + Emax - NUM_SPECIES][4] = _DF(0.0);
		for (int n = 0; n < NUM_COP; n++)
			eigen_l[m + Emax - NUM_SPECIES][n + Emax - NUM_COP] = (m == n) ? _DF(1.0) : _DF(0.0);
	}
	// right eigen vectors
	for (int n = 0; n < NUM_COP; n++)
	{
		eigen_r[0][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[1][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[2][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[3][Emax - NUM_COP + n - 1] = _DF(0.0);
		eigen_r[4][Emax - NUM_COP + n - 1] = z[n];
	}
	for (int m = 0; m < NUM_COP; m++)
	{
		eigen_r[m + Emax - NUM_COP][0] = yi[m];
		eigen_r[m + Emax - NUM_COP][1] = _DF(0.0);
		eigen_r[m + Emax - NUM_COP][2] = _DF(0.0);
		eigen_r[m + Emax - NUM_COP][3] = b1 * yi[m];
		eigen_r[m + Emax - NUM_COP][Emax - 1] = yi[m];
		for (int n = 0; n < NUM_COP; n++)
			eigen_r[m + Emax - NUM_COP][n + Emax - NUM_SPECIES] = (m == n) ? _DF(1.0) : _DF(0.0);
	}
#endif // COP
}

const double epsilon_weno = 1.0e-6;

/**
 * @brief the 5th WENO Scheme
 * 
 * @param f 
 * @param delta 
 * @return real_t 
 */
real_t weno5old_P(real_t *f, real_t delta)
{
	int k;
	real_t v1, v2, v3, v4, v5;
	real_t a1, a2, a3;
	real_t w1, w2, w3;

	//assign value to v1, v2,...
	k = 0;
	v1 = *(f + k - 2);
	v2 = *(f + k - 1);
	v3 = *(f + k);
	v4 = *(f + k + 1); 
	v5 = *(f + k + 2);

	// smoothness indicator
	real_t s1 = _DF(13.0) * (v1 - _DF(2.0) * v2 + v3) * (v1 - _DF(2.0) * v2 + v3) + _DF(3.0) * (v1 - _DF(4.0) * v2 + _DF(3.0) * v3) * (v1 - _DF(4.0) * v2 + _DF(3.0) * v3);
	s1 /= _DF(12.0);
	real_t s2 = _DF(13.0) * (v2 - _DF(2.0) * v3 + v4) * (v2 - _DF(2.0) * v3 + v4) + _DF(3.0) * (v2 - v4) * (v2 - v4);
	s2 /= _DF(12.0);
	real_t s3 = _DF(13.0) * (v3 - _DF(2.0) * v4 + v5) * (v3 - _DF(2.0) * v4 + v5) + _DF(3.0) * (_DF(3.0) * v3 - _DF(4.0) * v4 + v5) * (_DF(3.0) * v3 - _DF(4.0) * v4 + v5);
	s3 /= _DF(12.0);

	// weights
	a1 = _DF(0.1) / ((epsilon_weno + s1) * (epsilon_weno + s1));
	a2 = _DF(0.6) / ((epsilon_weno + s2) * (epsilon_weno + s2));
	a3 = _DF(0.3) / ((epsilon_weno + s3) * (epsilon_weno + s3));
	real_t tw1 = _DF(1.0) / (a1 + a2 + a3);
	w1 = a1 * tw1;
	w2 = a2 * tw1;
	w3 = a3 * tw1;
	real_t temp = w1 * (_DF(2.0) * v1 - _DF(7.0) * v2 + _DF(11.0) * v3) / _DF(6.0) + w2 * (-v2 + _DF(5.0) * v3 + _DF(2.0) * v4) / _DF(6.0) + w3 * (_DF(2.0) * v3 + _DF(5.0) * v4 - v5) / _DF(6.0);
	// return weighted average
	return temp;
}
/**
 * @brief the 5th WENO Scheme
 * 
 * @param f 
 * @param delta 
 * @return real_t 
 */
real_t weno5old_M(real_t *f, real_t delta)
{
	int k;
	real_t v1, v2, v3, v4, v5;
	real_t a1, a2, a3;
	real_t w1, w2, w3;

	//assign value to v1, v2,...
	k = 1;
	v1 = *(f + k + 2);
	v2 = *(f + k + 1);
	v3 = *(f + k);
	v4 = *(f + k - 1); 
	v5 = *(f + k - 2);

	// smoothness indicator
	double s1 = _DF(13.0) * (v1 - _DF(2.0) * v2 + v3) * (v1 - _DF(2.0) * v2 + v3) + _DF(3.0) * (v1 - _DF(4.0) * v2 + _DF(3.0) * v3) * (v1 - _DF(4.0) * v2 + _DF(3.0) * v3);
	s1 /= _DF(12.0);
	double s2 = _DF(13.0) * (v2 - _DF(2.0) * v3 + v4) * (v2 - _DF(2.0) * v3 + v4) + _DF(3.0) * (v2 - v4) * (v2 - v4);
	s2 /= _DF(12.0);
	double s3 = _DF(13.0) * (v3 - _DF(2.0) * v4 + v5) * (v3 - _DF(2.0) * v4 + v5) + _DF(3.0) * (_DF(3.0) * v3 - _DF(4.0) * v4 + v5) * (_DF(3.0) * v3 - _DF(4.0) * v4 + v5);
	s3 /= _DF(12.0);

	// weights
	a1 = _DF(0.1) / ((epsilon_weno + s1) * (epsilon_weno + s1));
	a2 = _DF(0.6) / ((epsilon_weno + s2) * (epsilon_weno + s2));
	a3 = _DF(0.3) / ((epsilon_weno + s3) * (epsilon_weno + s3));
	double tw1 = _DF(1.0) / (a1 + a2 + a3);
	w1 = a1 * tw1;
	w2 = a2 * tw1;
	w3 = a3 * tw1;

	real_t temp = w1 * (_DF(2.0) * v1 - _DF(7.0) * v2 + _DF(11.0) * v3) / _DF(6.0) + w2 * (-v2 + _DF(5.0) * v3 + _DF(2.0) * v4) / _DF(6.0) + w3 * (_DF(2.0) * v3 + _DF(5.0) * v4 - v5) / _DF(6.0);
	// return weighted average
	return temp;
}

#ifdef React
/**
 * @brief get_Kf
 */
real_t get_Kf_ArrheniusLaw(const real_t A, const real_t B, const real_t E, const real_t T)
{
	return A * sycl::pow<real_t>(T, B) * sycl::exp(-E * 4.184 / Ru / T);
}

/**
 * @brief get_Entropy //S
 */
real_t get_Entropy(real_t *__restrict__ Hia, real_t *__restrict__ Hib, const real_t Ri, const real_t T, const int n)
{
	real_t S = _DF(0.0);
#if Thermo
	if (T > 1000) // Hia[n * 7 * 3 + 0 * 3 + 1]//Hib[n * 2 * 3 + 0 * 3 + 1]
		S = Ri * (-_DF(0.5) * Hia[n * 7 * 3 + 0 * 3 + 1] / T / T - Hia[n * 7 * 3 + 1 * 3 + 1] / T + Hia[n * 7 * 3 + 2 * 3 + 1] * sycl::log(T) + Hia[n * 7 * 3 + 3 * 3 + 1] * T + _DF(0.5) * Hia[n * 7 * 3 + 4 * 3 + 1] * T * T + Hia[n * 7 * 3 + 5 * 3 + 1] * sycl::pow<real_t>(T, 3) / real_t(3.0) + Hia[n * 7 * 3 + 6 * 3 + 1] * sycl::pow<real_t>(T, 4) / real_t(4.0) + Hib[n * 2 * 3 + 1 * 3 + 1]);
	else
		S = Ri * (-_DF(0.5) * Hia[n * 7 * 3 + 0 * 3 + 0] / T / T - Hia[n * 7 * 3 + 1 * 3 + 0] / T + Hia[n * 7 * 3 + 2 * 3 + 0] * sycl::log(T) + Hia[n * 7 * 3 + 3 * 3 + 0] * T + _DF(0.5) * Hia[n * 7 * 3 + 4 * 3 + 0] * T * T + Hia[n * 7 * 3 + 5 * 3 + 0] * sycl::pow<real_t>(T, 3) / real_t(3.0) + Hia[n * 7 * 3 + 6 * 3 + 0] * sycl::pow<real_t>(T, 4) / real_t(4.0) + Hib[n * 2 * 3 + 1 * 3 + 0]);
#else
	if (T > 1000)
		S = Ri * (Hia[n * 7 * 3 + 0 * 3 + 0] * sycl::log(T) + Hia[n * 7 * 3 + 1 * 3 + 0] * T + _DF(0.5) * Hia[n * 7 * 3 + 2 * 3 + 0] * T * T + Hia[n * 7 * 3 + 3 * 3 + 0] / _DF(3.0) * T * T * T + Hia[n * 7 * 3 + 4 * 3 + 0] / _DF(4.0) * T * T * T * T + Hia[n * 7 * 3 + 6 * 3 + 0]);
	else
		S = Ri * (Hia[n * 7 * 3 + 0 * 3 + 1] * sycl::log(T) + Hia[n * 7 * 3 + 1 * 3 + 1] * T + _DF(0.5) * Hia[n * 7 * 3 + 2 * 3 + 1] * T * T + Hia[n * 7 * 3 + 3 * 3 + 1] / _DF(3.0) * T * T * T + Hia[n * 7 * 3 + 4 * 3 + 1] / _DF(4.0) * T * T * T * T + Hia[n * 7 * 3 + 6 * 3 + 1]);
#endif // Thermo
	return S;
}

/**
 * @brief get_Gibson
 */
real_t get_Gibson(real_t *__restrict__ Hia, real_t *__restrict__ Hib, const real_t T, const real_t Ri, const int n)
{
	return get_Entropy(Hia, Hib, Ri, T, n) / Ri - get_Enthalpy(Hia, Hib, T, Ri, n) / Ri / T;
}

/**
 * @brief get_Kc
 */
real_t get_Kc(real_t *__restrict__ species_chara, real_t *__restrict__ Hia, real_t *__restrict__ Hib, int *__restrict__ Nu_d_, const real_t T, const int m)
{
	real_t Kck = _DF(0.0), Nu_sum = _DF(0.0);
	for (size_t n = 0; n < NUM_SPECIES; n++)
	{
		real_t Ri = Ru / species_chara[n * SPCH_Sz + 6];
		real_t S = get_Gibson(Hia, Hib, T, Ri, n);
		Kck += Nu_d_[m * NUM_SPECIES + n] * S;
		Nu_sum += Nu_d_[m * NUM_SPECIES + n];
	}
	Kck = sycl::exp(Kck);
	Kck *= sycl::pow<real_t>(p_atm / Ru / T * 1e-6, Nu_sum); // 1e-6: m^-3 -> cm^-3
	return Kck;
}

/**
 * @brief get_KbKf
 */
void get_KbKf(real_t *Kf, real_t *Kb, real_t *Rargus, real_t *species_chara, real_t *Hia, real_t *Hib, int *Nu_d_, const real_t T)
{
	for (size_t m = 0; m < NUM_REA; m++)
	{
		real_t A = Rargus[m * 6 + 0], B = Rargus[m * 6 + 1], E = Rargus[m * 6 + 2];
#if CJ
		Kf[m] = sycl::min<real_t>((20 * _DF(1.0)), A * sycl::pow<real_t>(T, B) * sycl::exp(-E / T));
		Kb[m] = _DF(0.0);
#else
		Kf[m] = get_Kf_ArrheniusLaw(A, B, E, T);
		real_t Kck = get_Kc(species_chara, Hia, Hib, Nu_d_, T, m);
		Kb[m] = Kf[m] / Kck;
#endif // CJ
	}
}

/**
 * @brief QSSAFun
 */
void QSSAFun(real_t *q, real_t *d, real_t *Kf, real_t *Kb, const real_t yi[NUM_SPECIES], real_t *species_chara, real_t *React_ThirdCoef,
			 int **reaction_list, int **reactant_list, int **product_list, int *rns, int *rts, int *pls,
			 int *Nu_b_, int *Nu_f_, int *third_ind, const real_t rho)
{
	real_t C[NUM_SPECIES] = {_DF(0.0)};
	for (int n = 0; n < NUM_SPECIES; n++)
		C[n] = rho * yi[n] / species_chara[n * SPCH_Sz + 6] * 1e-6;

	for (int n = 0; n < NUM_SPECIES; n++)
	{
		q[n] = _DF(0.0);
		d[n] = _DF(0.0);
		for (int iter = 0; iter < rns[n]; iter++)
		{
			int react_id = reaction_list[n][iter];
			// third-body collision effect
			real_t tb = _DF(0.0);
			if (1 == third_ind[react_id])
			{
				for (int it = 0; it < NUM_SPECIES; it++)
					tb += React_ThirdCoef[react_id * NUM_SPECIES + it] * C[it];
			}
			else
				tb = 1.0;
			double RPf = Kf[react_id], RPb = Kb[react_id];
			// forward
			for (int it = 0; it < rts[react_id]; it++)
			{
				int specie_id = reactant_list[react_id][it];
				int nu_f = Nu_f_[react_id * NUM_SPECIES + specie_id];
				RPf *= sycl::pow<real_t>(C[specie_id], nu_f);
			}
			// backward
			for (int it = 0; it < pls[react_id]; it++)
			{
				int specie_id = product_list[react_id][it];
				int nu_b = Nu_b_[react_id * NUM_SPECIES + specie_id];
				RPb *= sycl::pow<real_t>(C[specie_id], nu_b);
			}
			q[n] += Nu_b_[react_id * NUM_SPECIES + n] * tb * RPf + Nu_f_[react_id * NUM_SPECIES + n] * tb * RPb;
			d[n] += Nu_b_[react_id * NUM_SPECIES + n] * tb * RPb + Nu_f_[react_id * NUM_SPECIES + n] * tb * RPf;
		}
		q[n] *= species_chara[n * SPCH_Sz + 6] / rho * 1e6;
		d[n] *= species_chara[n * SPCH_Sz + 6] / rho * 1e6;
	}
}

/**
 * @brief sign for one argus
 */
real_t sign(real_t a)
{
	if (a > 0)
		return _DF(1.0);
	else if (0 == a)
		return _DF(0.0);
	else
		return -_DF(1.0);
}

/**
 * @brief sign for two argus
 */
real_t sign(real_t a, real_t b)
{
	return sign(b) * abs(a);
}

/**
 * @brief Chemeq2
 */
void Chemeq2(Thermal *material, real_t *Kf, real_t *Kb, real_t *React_ThirdCoef, real_t *Rargus, int *Nu_b_, int *Nu_f_, int *Nu_d_,
			 int *third_ind, int **reaction_list, int **reactant_list, int **product_list, int *rns, int *rts, int *pls,
			 real_t y[NUM_SPECIES], const real_t dtg, real_t &TT, const real_t rho, const real_t e)
{
	// parameter
	int itermax = 1;
	real_t epscl = _DF(100.0);	// 1/epsmin, intermediate variable used to avoid repeated divisions
	real_t tfd = _DF(1.000008); // round-off parameter used to determine when integration is complete
	real_t dtmin = _DF(1.0e-20);
	real_t sqreps = _DF(0.5); // 5*sqrt(\eps, parameter used to calculate initial timestep in Eq.(52) and (53),
							  // || \delta y_i^{c(Nc-1)} ||/||\delta y_i^{c(Nc)} ||
	real_t epsmax = _DF(10.0), epsmin = _DF(1.0e-4);
	real_t ymin = _DF(1.0e-20); // minimum concentration allowed for species i
	real_t scrtch = _DF(1e-25);
	real_t eps;
	real_t rhoi[NUM_SPECIES];
	real_t qs[NUM_SPECIES];
	real_t ys[NUM_SPECIES]; // y_i^0 in Eq. (35) and {36}
	real_t y0[NUM_SPECIES]; // y_i^0 in Eq (2), intial concentrations for the global timestep passed to Chemeq
	real_t y1[NUM_SPECIES]; // y_i^p, predicted value from Eq. (35)
	real_t rtau[NUM_SPECIES], rtaus[NUM_SPECIES], scrarray[NUM_SPECIES];
	real_t q[NUM_SPECIES] = {_DF(0.0)}, d[NUM_SPECIES] = {_DF(0.0)}; // production and loss rate
	int gcount = 0, rcount = 0;
	int iter;
	real_t dt = _DF(0.0);
	real_t tn = _DF(0.0);	// t-t^0, current value of the independent variable relative to the start of the global timestep
	real_t ts;				// independent variable at the start of the global timestep
	real_t TTn = TT;
	real_t TTs, TT0;
	// save the initial inputs
	TT0 = TTn;
	for (int i = 0; i < NUM_SPECIES; i++)
	{
		y0[i] = y[i];
		y[i] = sycl::max(y[i], ymin);
		rhoi[i] = y[i] * rho;
	}
	real_t *species_chara = material->species_chara, *Hia = material->Hia, *Hib = material->Hib;
	//=========================================================
	// to initilize the first 'dt', q, d
	get_KbKf(Kf, Kb, Rargus, species_chara, Hia, Hib, Nu_d_, TTn);
	QSSAFun(q, d, Kf, Kb, y, species_chara, React_ThirdCoef, reaction_list, reactant_list, product_list, rns, rts, pls, Nu_b_, Nu_f_, third_ind, rho);
	gcount++;

	real_t ascr = _DF(0.0), scr1 = _DF(0.0), scr2 = _DF(0.0); // scratch (temporary) variable
	for (int i = 0; i < NUM_SPECIES; i++)
	{
		ascr = sycl::abs(q[i]);
		scr2 = sign(_DF(1.0) / y[i], _DF(.1) * epsmin * ascr - d[i]);
		scr1 = scr2 * d[i];
		scrtch = sycl::max(scr1, scrtch);
		scrtch = sycl::max(scrtch, -sycl::abs(ascr - d[i]) * scr2);
	}
	dt = sycl::min(sqreps / scrtch, dtg);

//==========================================================
flag1:
	int num_iter = 0;

	ts = tn;
	TTs = TTn;
	for (int i = 0; i < NUM_SPECIES; i++)
	{
		rtau[i] = dt * d[i] / y[i];
		// store the 0-subscript state using s
		ys[i] = y[i];
		qs[i] = q[i];
		rtaus[i] = rtau[i];
	}

flag2:
	num_iter++;
	// prediction
	for (int i = 0; i < NUM_SPECIES; i++)
	{
		real_t alpha = (real_t(180.0) + rtau[i] * (real_t(60.0) + rtau[i] * (real_t(11.0) + rtau[i]))) / (real_t(360.0) + rtau[i] * (real_t(60.0) + rtau[i] * (real_t(12.0) + rtau[i])));
		scrarray[i] = (q[i] - d[i]) / (_DF(1.0) + alpha * rtau[i]);
	}
	iter = 1;
	while (iter <= itermax)
	{
		for (int i = 0; i < NUM_SPECIES; i++)
		{
			y[i] = sycl::max(ys[i] + dt * scrarray[i], ymin); // predicted y
			rhoi[i] = y[i] * rho;
		}
		TTn = get_T(material, y, e, TTs); // UpdateTemperature(-1, rhoi, rho, e, TTs); // predicted T
		// GetKfKb(TTn);
		if (iter == 1)
		{
			tn = ts + dt;
			for (int i = 0; i < NUM_SPECIES; i++)
				y1[i] = y[i]; // prediction results stored by y1
		}
		QSSAFun(q, d, Kf, Kb, y, species_chara, React_ThirdCoef, reaction_list, reactant_list, product_list, rns, rts, pls, Nu_b_, Nu_f_, third_ind, rho);
		gcount++;
		eps = 1.0e-10;
		for (int i = 0; i < NUM_SPECIES; i++)
		{
			real_t rtaub = 0.5 * (rtaus[i] + dt * d[i] / y[i]);
			real_t alpha = (180.0 + rtaub * (60.0 + rtaub * (11.0 + rtaub))) / (360.0 + rtaub * (60.0 + rtaub * (12.0 + rtaub)));
			real_t qt = (1.0 - alpha) * qs[i] + alpha * q[i];
			real_t pb = rtaub / dt;
			scrarray[i] = (qt - ys[i] * pb) / (1.0 + alpha * rtaub); // to get the correction y
		}
		iter++;
	}
	// get new dt & check convergence
	for (int i = 0; i < NUM_SPECIES; i++)
	{
		scr2 = sycl::max(ys[i] + dt * scrarray[i], _DF(0.0));
		scr1 = sycl::abs(scr2 - y1[i]);
		y[i] = sycl::max(scr2, ymin); // new y
		// rhoi[i] = y[i]*species[i].Wi*1e3;
		rhoi[i] = y[i] * rho;
		if (_DF(0.5) * _DF(0.5) * (ys[i] + y[i]) > ymin)
		{
			scr1 = scr1 / y[i];
			eps = sycl::max(_DF(0.5) * (scr1 + sycl::min(sycl::abs(q[i] - d[i]) / (q[i] + d[i] + real_t(1.0e-30)), scr1)), eps);
		}
	}
	eps = eps * epscl;
	if (eps < epsmax)
	{
		if (dtg < (tn * tfd))
		{
			for (int i = 0; i < NUM_SPECIES; i++)
				rhoi[i] = y[i] * rho;
			TT = get_T(material, y, e, TTs); //  UpdateTemperature(-1, rhoi, rho, e, TTs); // final T
											 // GetKfKb(TT);
			return;
		}
	}
	else
	{
		tn = ts;
	}
	real_t rteps = 0.5 * (eps + 1.0);
	rteps = 0.5 * (rteps + eps / rteps);
	rteps = 0.5 * (rteps + eps / rteps);
	real_t dto = dt;
	dt = sycl::min(dt * (1.0 / rteps + real_t(0.005)), tfd * (dtg - tn)); // new dt
	if (eps > epsmax)
	{
		rcount++;
		dto = dt / dto;
		for (int i = 0; i < NUM_SPECIES; i++)
			rtaus[i] = rtaus[i] * dto;
		goto flag2;
	}

flag3:
	for (int i = 0; i < NUM_SPECIES; i++)
		rhoi[i] = y[i] * rho;
	TTn = get_T(material, y, e, TTs); // UpdateTemperature(-1, rhoi, rho, e, TTs); // new T
	// GetKfKb(TTn);
	QSSAFun(q, d, Kf, Kb, y, species_chara, React_ThirdCoef, reaction_list, reactant_list, product_list, rns, rts, pls, Nu_b_, Nu_f_, third_ind, rho);
	gcount++;
	goto flag1;
}
#endif // React
#ifdef Visc
/**
 * @brief get viscosity at temperature T(unit:K)(fit)
 * @return double,unit: Pa.s=kg/(m.s)
 */
real_t Viscosity(real_t fitted_coefficients_visc[order_polynominal_fitted], const double T0)
{
	real_t Tref = Reference_params[3], visref = Reference_params[5];
	real_t T = T0 * Tref; // nondimension==>dimension
	real_t viscosity = fitted_coefficients_visc[order_polynominal_fitted - 1];
	for (int i = order_polynominal_fitted - 2; i >= 0; i--)
		viscosity = viscosity * sycl::log(T) + fitted_coefficients_visc[i];
	real_t temp = sycl::exp(viscosity) / visref;
	return temp; // dimension==>nondimension
}

/**
 * @brief get viscosity at temperature T(unit:K)
 * @return double,unit: Pa.s=kg/(m.s)
 */
real_t PHI(real_t *specie_k, real_t *specie_j, real_t *fcv[NUM_SPECIES], const real_t T)
{
	real_t phi = _DF(0.0);
	phi = sycl::pow<real_t>(specie_j[Wi] / specie_k[Wi], _DF(0.25)) * sycl::pow<real_t>(Viscosity(fcv[int(specie_k[SID])], T) / Viscosity(fcv[int(specie_j[SID])], T), _DF(0.5));
	phi = (phi + _DF(1.0)) * (phi + _DF(1.0)) * _DF(0.5) / sycl::sqrt(_DF(2.0));
	phi = phi * sycl::pow<real_t>(_DF(1.0) + specie_k[Wi] / specie_j[Wi], -_DF(0.5));
	return phi;
}

/**
 * @brief get thermal conductivity at temperature T(unit:K)
 * @return double,unit: W/(m.K)
 */
real_t Thermal_conductivity(real_t fitted_coefficients_therm[order_polynominal_fitted], const real_t T0)
{
	real_t rhoref = Reference_params[1], pref = Reference_params[2];
	real_t Tref = Reference_params[3], W0ref = Reference_params[6], visref = Reference_params[7];
	real_t kref = visref * (pref / rhoref) / Tref;
	real_t T = T0 * Tref; // nondimension==>dimension
	real_t thermal_conductivity = fitted_coefficients_therm[order_polynominal_fitted - 1];
	for (int i = order_polynominal_fitted - 2; i >= 0; i--)
		thermal_conductivity = thermal_conductivity * sycl::log(T) + fitted_coefficients_therm[i];
	real_t temp = sycl::exp(thermal_conductivity) / kref;
	return temp; // dimension==>nondimension
}

/**
 * @brief get Dkj:the binary difffusion coefficient of specie-k to specie-j via equation 5-37
 * @para TT temperature unit:K
 * @para PP pressure unit:Pa
 */
real_t GetDkj(real_t *specie_k, real_t *specie_j, real_t **Dkj_matrix, const real_t T0, const real_t P0)
{
	real_t rhoref = Reference_params[1], pref = Reference_params[2];
	real_t Tref = Reference_params[3], W0ref = Reference_params[6], visref = Reference_params[7];
	real_t Dref = visref / rhoref;
	real_t TT = T0 * Tref; // nondimension==>dimension
	real_t PP = P0 * pref; // nondimension==>dimension
	real_t Dkj = Dkj_matrix[int(specie_k[SID]) * NUM_SPECIES + int(specie_j[SID])][order_polynominal_fitted - 1];
	for (int i = order_polynominal_fitted - 2; i >= 0; i--)
		Dkj = Dkj * sycl::log(TT) + Dkj_matrix[int(specie_k[SID]) * NUM_SPECIES + int(specie_j[SID])][i];
	real_t temp = (sycl::exp(Dkj) / PP) / Dref;
	return temp; // unit:cm*cm/s　//dimension==>nondimension
}

/**
 * @brief get average transport coefficient
 * @param chemi is set to get species information
 */
void Get_transport_coeff_aver(Thermal *thermal, real_t *Dkm_aver_id, real_t &viscosity_aver, real_t &thermal_conduct_aver, real_t const X[NUM_SPECIES],
							  const real_t rho, const real_t p, const real_t T, const real_t C_total)
{
	real_t **fcv = thermal->fitted_coefficients_visc;
	real_t **fct = thermal->fitted_coefficients_therm;
	real_t **Dkj = thermal->Dkj_matrix;
	viscosity_aver = _DF(0.0);
#ifdef Heat
	thermal_conduct_aver = _DF(0.0);
#endif
	real_t denominator = _DF(0.0);
	real_t *specie[NUM_SPECIES];
	for (size_t ii = 0; ii < NUM_SPECIES; ii++)
		specie[ii] = &(thermal->species_chara[ii * SPCH_Sz]);
	for (int k = 0; k < NUM_SPECIES; k++)
	{
		denominator = _DF(0.0);
		for (int i = 0; i < NUM_SPECIES; i++)
		{
			// real_t temp = PHI(specie[k], specie[i], fcv, T);
			denominator = denominator + X[i] * PHI(specie[k], specie[i], fcv, T);
		}
		// calculate viscosity_aver via equattion(5-49)//
		viscosity_aver = viscosity_aver + X[k] * Viscosity(fcv[int(specie[k][SID])], T) / denominator; // Pa.s=kg/(m.s)
#ifdef Heat
		// calculate thermal_conduct via Su Hongmin//
		thermal_conduct_aver = thermal_conduct_aver + X[k] * Thermal_conductivity(fct[int(specie[k][SID])], T) / denominator;
#endif // end Heat
	}
#ifdef Diffu
	// calculate diffusion coefficient specie_k to mixture via equation 5-45
	if (1 == NUM_SPECIES)
	{
		Dkm_aver_id[0] = GetDkj(specie[0], specie[0], Dkj, T, p); // trans_coeff.GetDkj(T, p, chemi.species[0], chemi.species[0], refstat);
		Dkm_aver_id[0] *= _DF(1.0e-1);							  // cm2/s==>m2/s
	}
	else
	{
		double temp1 = _DF(0.0), temp2 = _DF(0.0), temp3 = _DF(0.0);
		for (int k = 0; k < NUM_SPECIES; k++)
		{
			temp1 = _DF(0.0);
			temp2 = _DF(0.0);
			for (int i = 0; i < NUM_SPECIES; i++)
			{
				if (i != k)
				{
					temp1 += X[i] * thermal->Wi[i];
					temp2 += X[i] / GetDkj(specie[i], specie[k], Dkj, T, p); // trans_coeff.GetDkj(T, p, chemi.species[i], chemi.species[k], refstat);
				}
			}
			Dkm_aver_id[k] = temp1 / temp2 / (rho / C_total); // rho/C_total:the mole mass of mixture;
			Dkm_aver_id[k] *= _DF(1.0e-1);					  // cm2/s==>m2/s
		}
	}
#endif // end Diffu
}
#endif // end Visc
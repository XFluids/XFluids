#pragma once

#include "include/global_class.h"

using namespace std;
using namespace sycl;

/**
 *@brief calculate yi
 */
void get_yi(real_t *y, real_t yi[NUM_SPECIES], const int id)
{
	yi[0] = 1;
	for (size_t i = 1; i < NUM_SPECIES; i++)
	{
		yi[i] = y[NUM_COP * id + i - 1];
		yi[0] += -yi[i];
	}
	// printf("%lf,%lf\n", yi[0], yi[1]);
}

/**
 *@brief calculate R for every cell
 */
real_t get_CopR(real_t *species_chara, real_t yi[NUM_SPECIES])
{
	real_t R = 0;
	for (size_t n = 0; n < NUM_SPECIES; n++)
	{
		R += yi[n] * Ru / (species_chara[n * SPCH_Sz + 6]);
	}
	return R;
}

/**
 * @brief calculate Cpi of the specie at given point
 */
real_t get_Cpi(real_t *__restrict__ Hia, const real_t T0, const real_t Ri, const int n)
{
	real_t T = T0, Cpi = zero_float; // NOTE:注意由Hia和Hib计算得到的Cp单位是J/kg/K
	if (T < (200.0 / Tref))
		T = 200.0 / Tref;
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
		printf("T=%lf , Cpi=%lf , T > 15000 K,please check!!!NO Cpi[n] for T>15000 K \n", T, Cpi);
	}
#else
	// Cpi[n)/R = a1 + a2*T + a3*T^2 + a4*T^3 + a5*T^4
	if (T > (1000.0 / Tref))
		Cpi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 0] + Hia[n * 7 * 3 + 1 * 3 + 0] * T + Hia[n * 7 * 3 + 2 * 3 + 0] * T * T + Hia[n * 7 * 3 + 3 * 3 + 0] * T * T * T + Hia[n * 7 * 3 + 4 * 3 + 0] * T * T * T * T);
	else
		Cpi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 1] + Hia[n * 7 * 3 + 1 * 3 + 1] * T + Hia[n * 7 * 3 + 2 * 3 + 1] * T * T + Hia[n * 7 * 3 + 3 * 3 + 1] * T * T * T + Hia[n * 7 * 3 + 4 * 3 + 1] * T * T * T * T);
#endif
	return Cpi;
}

/**
 * @brief Compute the Cp of the mixture at given point unit:J/kg/K
 */
real_t get_CopCp(Thermal *material, real_t yi[NUM_SPECIES], const real_t T)
{
	real_t _CopCp = 0.0;
	for (size_t ii = 0; ii < NUM_SPECIES; ii++)
	{
		real_t Ri = Ru / material->species_chara[ii * SPCH_Sz + 6];
		_CopCp += yi[ii] * get_Cpi(material->Hia, T, Ri, ii); // real_t Cpi = get_Cpi(material->Hia, T, Ri, ii) ;
	}
	// printf("Cpi=%lf , %lf , yi=%lf , %lf , _CopCp=%lf \n", Cpi[0], Cpi[1], yi[0], yi[1], _CopCp);
	return _CopCp;
}

/**
 * @brief calculate W of the mixture at given point
 */
real_t get_CopW(Thermal *material, real_t yi[NUM_SPECIES])
{
	real_t _W = 0.0;
	for (size_t ii = 0; ii < NUM_SPECIES; ii++)
	{
		_W += yi[ii] / (material->species_chara[ii * SPCH_Sz + 6]); // Wi
	}
	// printf("W=%lf \n", 1.0 / _W);
	return one_float / _W;
}

/**
 * @brief calculate Gamma of the mixture at given point
 */
real_t get_CopGamma(Thermal *material, real_t yi[NUM_SPECIES], const real_t T)
{
	real_t Cp = get_CopCp(material, yi, T);
	real_t CopW = get_CopW(material, yi);
	real_t _CopGamma = Cp / (Cp - Ru / CopW);
	// printf("CopGamma=%lf,yi of qloc =%lf,%lf,Cp=%lf,CopW=%lf\n", _CopGamma, yi[0], yi[1], Cp, CopW);
	if (_CopGamma > 1)
	{
		return _CopGamma;
	}
	else
	{
		printf("CopGamma calculate error: CopGamma=%lf,Yi of qloc =%lf,%lf,Cp=%lf,CopW=%lf\n", _CopGamma, yi[0], yi[1], Cp, CopW);
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
	real_t hi = zero_float;
#if Thermo
	// NOTE：Non_dim of Hia && Hib*3+only for h&Cp not for S ATTENTATION
	//  200K~1000K
	Hia[n * 7 * 3 + 0 * 3 + 0] = Hia[n * 7 * 3 + 0 * 3 + 0] / pow(Tref, 2);
	Hia[n * 7 * 3 + 1 * 3 + 0] = Hia[n * 7 * 3 + 1 * 3 + 0] / Tref;

	Hia[n * 7 * 3 + 3 * 3 + 0] = Hia[n * 7 * 3 + 3 * 3 + 0] * Tref;
	Hia[n * 7 * 3 + 4 * 3 + 0] = Hia[n * 7 * 3 + 4 * 3 + 0] * pow(Tref, 2);
	Hia[n * 7 * 3 + 5 * 3 + 0] = Hia[n * 7 * 3 + 5 * 3 + 0] * pow(Tref, 3);
	Hia[n * 7 * 3 + 6 * 3 + 0] = Hia[n * 7 * 3 + 6 * 3 + 0] * pow(Tref, 4);
	Hib[n * 2 * 3 + 0 * 3 + 0] = Hib[n * 2 * 3 + 0 * 3 + 0] / Tref + Hia[n * 7 * 3 + 1 * 3 + 0] * log(Tref);
	// 1000K~6000K
	Hia[n * 7 * 3 + 0 * 3 + 1] = Hia[n * 7 * 3 + 0 * 3 + 1] / pow(Tref, 2);
	Hia[n * 7 * 3 + 1 * 3 + 1] = Hia[n * 7 * 3 + 1 * 3 + 1] / Tref;

	Hia[n * 7 * 3 + 3 * 3 + 1] = Hia[n * 7 * 3 + 3 * 3 + 1] * Tref;
	Hia[n * 7 * 3 + 4 * 3 + 1] = Hia[n * 7 * 3 + 4 * 3 + 1] * pow(Tref, 2);
	Hia[n * 7 * 3 + 5 * 3 + 1] = Hia[n * 7 * 3 + 5 * 3 + 1] * pow(Tref, 3);
	Hia[n * 7 * 3 + 6 * 3 + 1] = Hia[n * 7 * 3 + 6 * 3 + 1] * pow(Tref, 4);
	Hib[n * 2 * 3 + 0 * 3 + 1] = Hib[n * 2 * 3 + 0 * 3 + 1] / Tref + Hia[n * 7 * 3 + 1 * 3 + 1] * log(Tref);
	// 6000K~15000K
	Hia[n * 7 * 3 + 0 * 3 + 2] = Hia[n * 7 * 3 + 0 * 3 + 2] / pow(Tref, 2);
	Hia[n * 7 * 3 + 1 * 3 + 2] = Hia[n * 7 * 3 + 1 * 3 + 2] / Tref;

	Hia[n * 7 * 3 + 3 * 3 + 2] = Hia[n * 7 * 3 + 3 * 3 + 2] * Tref;
	Hia[n * 7 * 3 + 4 * 3 + 2] = Hia[n * 7 * 3 + 4 * 3 + 2] * pow(Tref, 2);
	Hia[n * 7 * 3 + 5 * 3 + 2] = Hia[n * 7 * 3 + 5 * 3 + 2] * pow(Tref, 3);
	Hia[n * 7 * 3 + 6 * 3 + 2] = Hia[n * 7 * 3 + 6 * 3 + 2] * pow(Tref, 4);
	Hib[n * 2 * 3 + 0 * 3 + 2] = Hib[n * 2 * 3 + 0 * 3 + 2] / Tref + Hia[n * 7 * 3 + 1 * 3 + 2] * log(Tref);
#else
	Hia[n * 7 * 3 + 1 * 3 + 0] = Hia[n * 7 * 3 + 1 * 3 + 0] * Tref;
	Hia[n * 7 * 3 + 2 * 3 + 0] = Hia[n * 7 * 3 + 2 * 3 + 0] * pow(Tref, 2);
	Hia[n * 7 * 3 + 3 * 3 + 0] = Hia[n * 7 * 3 + 3 * 3 + 0] * pow(Tref, 3);
	Hia[n * 7 * 3 + 4 * 3 + 0] = Hia[n * 7 * 3 + 4 * 3 + 0] * pow(Tref, 4);
	Hia[n * 7 * 3 + 5 * 3 + 0] = Hia[n * 7 * 3 + 5 * 3 + 0] / Tref;

	Hia[n * 7 * 3 + 1 * 3 + 1] = Hia[n * 7 * 3 + 1 * 3 + 1] * Tref;
	Hia[n * 7 * 3 + 2 * 3 + 1] = Hia[n * 7 * 3 + 2 * 3 + 1] * pow(Tref, 2);
	Hia[n * 7 * 3 + 3 * 3 + 1] = Hia[n * 7 * 3 + 3 * 3 + 1] * pow(Tref, 3);
	Hia[n * 7 * 3 + 4 * 3 + 1] = Hia[n * 7 * 3 + 4 * 3 + 1] * pow(Tref, 4);
	Hia[n * 7 * 3 + 5 * 3 + 1] = Hia[n * 7 * 3 + 5 * 3 + 1] / Tref;
#endif
	if (T < 200.0 / Tref)
	{
		TT = T;
		T = 200.0 / Tref;
	}
#if Thermo
	if (T >= (1000.0 / Tref) && T < (6000.0 / Tref))
		hi = Ri * (-Hia[n * 7 * 3 + 0 * 3 + 1] * 1.0 / T + Hia[n * 7 * 3 + 1 * 3 + 1] * log(T) + Hia[n * 7 * 3 + 2 * 3 + 1] * T + 0.5 * Hia[n * 7 * 3 + 3 * 3 + 1] * T * T + Hia[n * 7 * 3 + 4 * 3 + 1] * pow(T, 3) / 3.0 + Hia[n * 7 * 3 + 5 * 3 + 1] * pow(T, 4) / 4.0 + Hia[n * 7 * 3 + 6 * 3 + 1] * pow(T, 5) / 5.0 + Hib[n * 2 * 3 + 0 * 3 + 1]);
	else if (T < (1000.0 / Tref))
	{ // TODO: hi[n] caculate error
		hi = Ri * (-Hia[n * 7 * 3 + 0 * 3 + 0] * 1.0 / T + Hia[n * 7 * 3 + 1 * 3 + 0] * log(T) + Hia[n * 7 * 3 + 2 * 3 + 0] * T + 0.5 * Hia[n * 7 * 3 + 3 * 3 + 0] * T * T + Hia[n * 7 * 3 + 4 * 3 + 0] * pow(T, 3) / 3.0 + Hia[n * 7 * 3 + 5 * 3 + 0] * pow(T, 4) / 4.0 + Hia[n * 7 * 3 + 6 * 3 + 0] * pow(T, 5) / 5.0 + Hib[n * 2 * 3 + 0 * 3 + 0]);
	}
	else if (T >= (6000.0 / Tref) && T < (15000.0 / Tref))
	{
		hi = Ri * (-Hia[n * 7 * 3 + 0 * 3 + 2] * 1.0 / T + Hia[n * 7 * 3 + 1 * 3 + 2] * log(T) + Hia[n * 7 * 3 + 2 * 3 + 2] * T + 0.5 * Hia[n * 7 * 3 + 3 * 3 + 2] * T * T + Hia[n * 7 * 3 + 4 * 3 + 2] * pow(T, 3) / 3.0 + Hia[n * 7 * 3 + 5 * 3 + 2] * pow(T, 4) / 4.0 + Hia[n * 7 * 3 + 6 * 3 + 2] * pow(T, 5) / 5.0 + Hib[n * 2 * 3 + 0 * 3 + 2]);
	}
	else
	{
		printf("T=%lf,T > 15000 K,please check!!!NO h for T>15000 K. \n", T * Tref);
	}
#else
	// H/RT = a1 + a2/2*T + a3/3*T^2 + a4/4*T^3 + a5/5*T^4 + a6/T
	if (T > (1000.0 / Tref))
		hi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 0] * T + Hia[n * 7 * 3 + 1 * 3 + 0] * T * T / 2.0 + Hia[n * 7 * 3 + 2 * 3 + 0] * pow(T, 3) / 3.0 + Hia[n * 7 * 3 + 3 * 3 + 0] * pow(T, 4) / 4.0 + Hia[n * 7 * 3 + 4 * 3 + 0] * pow(T, 5) / 5.0 + Hia[n * 7 * 3 + 5 * 3 + 0]);
	else
		hi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 1] * T + Hia[n * 7 * 3 + 1 * 3 + 1] * T * T / 2.0 + Hia[n * 7 * 3 + 2 * 3 + 1] * pow(T, 3) / 3.0 + Hia[n * 7 * 3 + 3 * 3 + 1] * pow(T, 4) / 4.0 + Hia[n * 7 * 3 + 4 * 3 + 1] * pow(T, 5) / 5.0 + Hia[n * 7 * 3 + 5 * 3 + 1]);
#endif
	// printf("hi[n] of get_hi=%lf \n", hi[n]);
	// get_hi at T>200
	if (TT < 200.0 / Tref)
	{
		real_t Cpi = get_Cpi(Hia, 200.0 / Tref, Ri, n); // get_Cpi(real_t *__restrict__ Hia, const real_t T0, const real_t Ri, const int n)
		hi += Cpi * (TT - 200.0 / Tref);
		// printf("hi[%d] = %lf , Cpi[%d]=%lf \n", ii, hi[ii], ii, Cpi[ii]);
	}
	return hi;
}

/**
 * @brief calculate Hi of Mixture at given point	unit:J/kg/K
 */
real_t get_Coph(Thermal *material, real_t yi[NUM_SPECIES], const real_t T)
{
	real_t H = 0.0, hi[NUM_SPECIES];
	for (size_t i = 0; i < NUM_SPECIES; i++)
	{
		real_t Ri = Ru / material->species_chara[i * SPCH_Sz + 6];
		real_t hi = get_Enthalpy(material->Hia, material->Hib, T, Ri, i);
		H += hi * yi[i]; // hi[i] = get_Enthalpy(material->Hia, material->Hib, T, Ri, i);
						 // printf("yi=%lf,%lf,hi=%lf,%lf \n", yi[0], yi[1], hi[0], hi[1]);
	}
	return H;
}

/**
 *@brief sub_function_Steps of update T
 */
void sub_FuncT(real_t &func_T, real_t &dfunc_T, Thermal *thermal, real_t yi[NUM_SPECIES], const real_t e, const real_t T)
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
real_t get_T(Thermal *thermal, real_t yi[NUM_SPECIES], const real_t e, const real_t T0)
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
			printf("Temperature: Newton_Ramphson iteration failured, try Bisection Metho...d\n");
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
			   real_t &T, Thermal *thermal, real_t yi[NUM_SPECIES], real_t const Gamma0)
{
	rho = UI[0];
	real_t rho1 = one_float / rho;
	u = UI[1] * rho1;
	v = UI[2] * rho1;
	w = UI[3] * rho1;

// EOS was included
#ifdef COP
	real_t e = UI[4] * rho1 - half_float * (u * u + v * v + w * w);
	T = get_T(thermal, yi, e, T);
	real_t R = get_CopR(thermal->species_chara, yi);
	p = rho * R * T; // 对所有气体都适用
	real_t Gamma = get_CopGamma(thermal, yi, T);
	// printf("UI[0]=%lf,UI[4]=%lf,T2=%lf, e=%lf, yi=%lf, %lf, R=%lf,p=%lf,gamma=%lf\n", rho, UI[4], T, e, yi[0], yi[1], R, p, Gamma);
#else
	real_t Gamma = Gamma0;
	p = (Gamma - one_float) * (UI[4] - half_float * rho * (u * u + v * v + w * w));
#endif // COP
	H = (UI[4] + p) * rho1;
	c = sqrt(Gamma * p * rho1);
	// printf("rho=%lf , p=%lf , c=%lf \n", rho, p, c);
}

/**
 * @brief  Obtain fluxes at a grid point
 */
void GetPhysFlux(real_t UI[Emax], real_t yi[NUM_COP], real_t *FluxF, real_t *FluxG, real_t *FluxH, real_t const rho, real_t const u, real_t const v, real_t const w, real_t const p, real_t const H, real_t const c)
{
	FluxF[0] = UI[1];
	// *(FluxF+0) = UI[1];
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
	for (size_t ii = (Emax - NUM_COP); ii < Emax; ii++)
	{
		FluxF[ii] = UI[1] * yi[ii - (Emax - NUM_SPECIES)];
		FluxG[ii] = UI[2] * yi[ii - (Emax - NUM_SPECIES)];
		FluxH[ii] = UI[3] * yi[ii - (Emax - NUM_SPECIES)];
	}
#endif
}

inline void RoeAverage_x(real_t eigen_l[Emax][Emax], real_t eigen_r[Emax][Emax], real_t z[NUM_SPECIES], real_t yi[NUM_SPECIES], real_t const c2,
						 real_t const _rho, real_t const _u, real_t const _v, real_t const _w,
						 real_t const _H, real_t const D, real_t const D1, real_t Gamma)
{

	real_t _Gamma = Gamma - one_float;
	real_t q2 = _u * _u + _v * _v + _w * _w;
	// real_t c2 = _Gamma * (_H - half_float * q2);
	real_t _c = sqrt(c2);
	real_t c21_Gamma = _Gamma / c2;

#ifdef COP
	real_t b1 = c21_Gamma;
	real_t b2 = one_float + c21_Gamma * q2 - c21_Gamma * _H;
	real_t b3 = zero_float;
	for (size_t i = 1; i < NUM_SPECIES; i++)
	{
		b3 += b1 * yi[i] * z[i];
	}
	real_t _c1 = one_float / _c;
	// printf("b1=%lf,b2=%lf,b3=%lf,_c=%lf,yi[]=%lf,%lf,z[]=%lf,%lf\n", b1, b2, b3, _c, yi[0], yi[1], z[0], z[1]);
	eigen_l[0][0] = half_float * (b2 + _u / _c + b3);
	eigen_l[0][1] = -half_float * (b1 * _u + _c1);
	eigen_l[0][2] = -half_float * (b1 * _v);
	eigen_l[0][3] = -half_float * (b1 * _w);
	eigen_l[0][4] = half_float * b1;

	eigen_l[1][0] = (one_float - b2 - b3) / b1;
	eigen_l[1][1] = _u;
	eigen_l[1][2] = _v;
	eigen_l[1][3] = _w;
	eigen_l[1][4] = -one_float;

	eigen_l[2][0] = _v;
	eigen_l[2][1] = zero_float;
	eigen_l[2][2] = -one_float;
	eigen_l[2][3] = zero_float;
	eigen_l[2][4] = zero_float;

	eigen_l[3][0] = -_w;
	eigen_l[3][1] = zero_float;
	eigen_l[3][2] = zero_float;
	eigen_l[3][3] = one_float;
	eigen_l[3][4] = zero_float;

	eigen_l[Emax - 1][0] = half_float * (b2 - _u / _c + b3);
	eigen_l[Emax - 1][1] = half_float * (-b1 * _u + _c1);
	eigen_l[Emax - 1][2] = half_float * (-b1 * _v);
	eigen_l[Emax - 1][3] = half_float * (-b1 * _w);
	eigen_l[Emax - 1][4] = half_float * b1;

	// right eigen vectors
	eigen_r[0][0] = one_float;
	eigen_r[0][1] = b1;
	eigen_r[0][2] = zero_float;
	eigen_r[0][3] = zero_float;
	// for (int n = 0; n < NUM_COP; n++)
	// 	eigen_r[0][Emax - NUM_COP + n - 1] = zero_float;
	eigen_r[0][Emax - 1] = one_float;

	eigen_r[1][0] = _u - _c;
	eigen_r[1][1] = _u * b1;
	eigen_r[1][2] = zero_float;
	eigen_r[1][3] = zero_float;
	eigen_r[1][Emax - 1] = _u + _c;

	eigen_r[2][0] = _v;
	eigen_r[2][1] = _v * b1;
	eigen_r[2][2] = -one_float;
	eigen_r[2][3] = zero_float;
	eigen_r[2][Emax - 1] = _v;

	eigen_r[3][0] = _w;
	eigen_r[3][1] = _w * b1;
	eigen_r[3][2] = zero_float;
	eigen_r[3][3] = one_float;
	eigen_r[3][Emax - 1] = _w;

	eigen_r[4][0] = _H - _u * _c;
	eigen_r[4][1] = _H * b1 - one_float;
	eigen_r[4][2] = -_v;
	eigen_r[4][3] = _w;
	eigen_r[4][Emax - 1] = _H + _u * _c;

	// left eigen vectors
	for (int n = 0; n < NUM_COP; n++)
	{
		eigen_l[0][n + Emax - NUM_COP] = -half_float * b1 * z[n + 1];
		eigen_l[1][n + Emax - NUM_COP] = z[n + 1];
		eigen_l[2][n + Emax - NUM_COP] = 0;
		eigen_l[3][n + Emax - NUM_COP] = 0;
		eigen_l[Emax - 1][n + Emax - NUM_COP] = -half_float * b1 * z[n + 1];
	}
	for (int m = 0; m < NUM_COP; m++)
	{
		eigen_l[m + Emax - NUM_SPECIES][0] = -yi[m + 1];
		eigen_l[m + Emax - NUM_SPECIES][1] = zero_float;
		eigen_l[m + Emax - NUM_SPECIES][2] = zero_float;
		eigen_l[m + Emax - NUM_SPECIES][3] = zero_float;
		eigen_l[m + Emax - NUM_SPECIES][4] = zero_float;
		for (int n = 0; n < NUM_COP; n++)
			eigen_l[m + Emax - NUM_SPECIES][n + Emax - NUM_COP] = (m == n) ? one_float : zero_float;
	}
	// right eigen vectors
	for (int n = 0; n < NUM_COP; n++)
	{
		eigen_r[0][Emax - NUM_COP + n - 1] = zero_float;
		eigen_r[1][Emax - NUM_COP + n - 1] = zero_float;
		eigen_r[2][Emax - NUM_COP + n - 1] = zero_float;
		eigen_r[3][Emax - NUM_COP + n - 1] = zero_float;
		eigen_r[4][Emax - NUM_COP + n - 1] = z[n + 1];
	}
	for (int m = 0; m < NUM_COP; m++)
	{
		eigen_r[m + Emax - NUM_COP][0] = yi[m + 1];
		eigen_r[m + Emax - NUM_COP][1] = b1 * yi[m + 1];
		eigen_r[m + Emax - NUM_COP][2] = zero_float;
		eigen_r[m + Emax - NUM_COP][3] = zero_float;
		eigen_r[m + Emax - NUM_COP][Emax - 1] = yi[m + 1];
		for (int n = 0; n < NUM_COP; n++)
			eigen_r[m + Emax - NUM_COP][n + Emax - NUM_SPECIES] = (m == n) ? one_float : zero_float;
	}
#else
	real_t _rho1 = one_float / _rho;
	real_t _c1_rho = half_float * _rho / _c;
	real_t _c1_rho1_Gamma = _Gamma * _rho1 / _c;

	// left eigen vectors
	eigen_l[0][0] = one_float - half_float * c21_Gamma * q2;
	eigen_l[0][1] = c21_Gamma * _u;
	eigen_l[0][2] = c21_Gamma * _v;
	eigen_l[0][3] = c21_Gamma * _w;
	eigen_l[0][4] = -c21_Gamma;

	eigen_l[1][0] = -_w * _rho1;
	eigen_l[1][1] = zero_float;
	eigen_l[1][2] = zero_float;
	eigen_l[1][3] = _rho1;
	eigen_l[1][4] = zero_float;

	eigen_l[2][0] = _v * _rho1;
	eigen_l[2][1] = zero_float;
	eigen_l[2][2] = -_rho1;
	eigen_l[2][3] = zero_float;
	eigen_l[2][4] = zero_float;

	eigen_l[3][0] = half_float * _c1_rho1_Gamma * q2 - _u * _rho1;
	eigen_l[3][1] = -_c1_rho1_Gamma * _u + _rho1;
	eigen_l[3][2] = -_c1_rho1_Gamma * _v;
	eigen_l[3][3] = -_c1_rho1_Gamma * _w;
	eigen_l[3][4] = _c1_rho1_Gamma;

	eigen_l[4][0] = half_float * _c1_rho1_Gamma * q2 + _u * _rho1;
	eigen_l[4][1] = -_c1_rho1_Gamma * _u - _rho1;
	eigen_l[4][2] = -_c1_rho1_Gamma * _v;
	eigen_l[4][3] = -_c1_rho1_Gamma * _w;
	eigen_l[4][4] = _c1_rho1_Gamma;

	// right eigen vectors
	eigen_r[0][0] = one_float;
	eigen_r[0][1] = zero_float;
	eigen_r[0][2] = zero_float;
	eigen_r[0][3] = _c1_rho;
	eigen_r[0][4] = _c1_rho;

	eigen_r[1][0] = _u;
	eigen_r[1][1] = zero_float;
	eigen_r[1][2] = zero_float;
	eigen_r[1][3] = _c1_rho * (_u + _c);
	eigen_r[1][4] = _c1_rho * (_u - _c);

	eigen_r[2][0] = _v;
	eigen_r[2][1] = zero_float;
	eigen_r[2][2] = -_rho;
	eigen_r[2][3] = _c1_rho * _v;
	eigen_r[2][4] = _c1_rho * _v;

	eigen_r[3][0] = _w;
	eigen_r[3][1] = _rho;
	eigen_r[3][2] = zero_float;
	eigen_r[3][3] = _c1_rho * _w;
	eigen_r[3][4] = _c1_rho * _w;

	eigen_r[4][0] = half_float * q2;
	eigen_r[4][1] = _rho * _w;
	eigen_r[4][2] = -_rho * _v;
	eigen_r[4][3] = _c1_rho * (_H + _u * _c);
	eigen_r[4][4] = _c1_rho * (_H - _u * _c);

#endif // COP
}

inline void RoeAverage_y(real_t eigen_l[Emax][Emax], real_t eigen_r[Emax][Emax], real_t z[NUM_SPECIES], real_t yi[NUM_SPECIES], real_t const c2,
						 real_t const _rho, real_t const _u, real_t const _v, real_t const _w,
						 real_t const _H, real_t const D, real_t const D1, real_t const Gamma)
{

	real_t _Gamma = Gamma - one_float;
	real_t q2 = _u*_u + _v*_v + _w*_w;
	// real_t c2 = _Gamma*(_H - half_float*q2);
	real_t _c = sqrt(c2);
	real_t c21_Gamma = _Gamma / c2;

#ifdef COP
	real_t b1 = c21_Gamma;
	real_t b2 = one_float + c21_Gamma * q2 - c21_Gamma * _H;
	real_t b3 = zero_float;
	for (size_t i = 1; i < NUM_SPECIES; i++)
	{
		b3 += b1 * yi[i] * z[i];
	}
	real_t _c1 = one_float / _c;
	// printf("b1=%lf,b2=%lf,b3=%lf,_c=%lf,yi[]=%lf,%lf,z[]=%lf,%lf\n", b1, b2, b3, _c, yi[0], yi[1], z[0], z[1]);
	// left eigen vectors
	eigen_l[0][0] = half_float * (b2 + _v / _c + b3);
	eigen_l[0][1] = -half_float * (b1 * _u);
	eigen_l[0][2] = -half_float * (b1 * _v + _c1);
	eigen_l[0][3] = -half_float * (b1 * _w);
	eigen_l[0][4] = half_float * b1;

	eigen_l[1][0] = -_u;
	eigen_l[1][1] = one_float;
	eigen_l[1][2] = zero_float;
	eigen_l[1][3] = zero_float;
	eigen_l[1][4] = zero_float;

	eigen_l[2][0] = (one_float - b2 - b3) / b1;
	eigen_l[2][1] = _u;
	eigen_l[2][2] = _v;
	eigen_l[2][3] = _w;
	eigen_l[2][4] = -one_float;

	eigen_l[3][0] = _w;
	eigen_l[3][1] = zero_float;
	eigen_l[3][2] = zero_float;
	eigen_l[3][3] = -one_float;
	eigen_l[3][4] = zero_float;

	eigen_l[Emax - 1][0] = half_float * (b2 - _v / _c + b3);
	eigen_l[Emax - 1][1] = half_float * (-b1 * _u);
	eigen_l[Emax - 1][2] = half_float * (-b1 * _v + _c1);
	eigen_l[Emax - 1][3] = half_float * (-b1 * _w);
	eigen_l[Emax - 1][4] = half_float * b1;

	// right eigen vectors
	eigen_r[0][0] = one_float;
	eigen_r[0][1] = zero_float;
	eigen_r[0][2] = b1;
	eigen_r[0][3] = zero_float;
	// for (int n = 0; n < num_species - 1; n++)
	// 	eigen_r[0][Emax - NUM_COP + n - 1] = zero_float;
	eigen_r[0][Emax - 1] = one_float;

	eigen_r[1][0] = _u;
	eigen_r[1][1] = one_float;
	eigen_r[1][2] = _u * b1;
	eigen_r[1][3] = zero_float;
	eigen_r[1][Emax - 1] = _u;

	eigen_r[2][0] = _v - _c;
	eigen_r[2][1] = zero_float;
	eigen_r[2][2] = _v * b1;
	eigen_r[2][3] = zero_float;
	eigen_r[2][Emax - 1] = _v + _c;

	eigen_r[3][0] = _w;
	eigen_r[3][1] = zero_float;
	eigen_r[3][2] = _w * b1;
	eigen_r[3][3] = -one_float;
	eigen_r[3][Emax - 1] = _w;

	eigen_r[4][0] = _H - _v * _c;
	eigen_r[4][1] = _u;
	eigen_r[4][2] = _H * b1 - one_float;
	eigen_r[4][3] = -_w;
	eigen_r[4][Emax - 1] = _H + _v * _c;

	// left eigen vectors
	for (int n = 0; n < NUM_COP; n++)
	{
		eigen_l[0][n + Emax - NUM_COP] = -half_float * b1 * z[n + 1];
		eigen_l[1][n + Emax - NUM_COP] = zero_float;
		eigen_l[2][n + Emax - NUM_COP] = z[n + 1];
		eigen_l[3][n + Emax - NUM_COP] = zero_float;
		eigen_l[Emax - 1][n + Emax - NUM_COP] = -half_float * b1 * z[n + 1];
	}
	for (int m = 0; m < NUM_COP; m++)
	{
		eigen_l[m + Emax - NUM_SPECIES][0] = -yi[m + 1];
		eigen_l[m + Emax - NUM_SPECIES][1] = zero_float;
		eigen_l[m + Emax - NUM_SPECIES][2] = zero_float;
		eigen_l[m + Emax - NUM_SPECIES][3] = zero_float;
		eigen_l[m + Emax - NUM_SPECIES][4] = zero_float;
		for (int n = 0; n < NUM_COP; n++)
			eigen_l[m + Emax - NUM_SPECIES][n + Emax - NUM_COP] = (m == n) ? one_float : zero_float;
	}
	// right eigen vectors
	for (int n = 0; n < NUM_COP; n++)
	{
		eigen_r[0][Emax - NUM_COP + n - 1] = zero_float;
		eigen_r[1][Emax - NUM_COP + n - 1] = zero_float;
		eigen_r[2][Emax - NUM_COP + n - 1] = zero_float;
		eigen_r[3][Emax - NUM_COP + n - 1] = zero_float;
		eigen_r[4][Emax - NUM_COP + n - 1] = z[n + 1];
	}
	for (int m = 0; m < NUM_COP; m++)
	{
		eigen_r[m + Emax - NUM_COP][0] = yi[m + 1];
		eigen_r[m + Emax - NUM_COP][1] = zero_float;
		eigen_r[m + Emax - NUM_COP][2] = b1 * yi[m + 1];
		eigen_r[m + Emax - NUM_COP][3] = zero_float;
		eigen_r[m + Emax - NUM_COP][Emax - 1] = yi[m + 1];
		for (int n = 0; n < NUM_COP; n++)
			eigen_r[m + Emax - NUM_COP][n + Emax - NUM_SPECIES] = (m == n) ? one_float : zero_float;
	}
#else

	real_t _rho1 = one_float / _rho;
	real_t _c1_rho = half_float * _rho / _c;
	real_t _c1_rho1_Gamma = _Gamma * _rho1 / _c;
	// left eigen vectors 
	eigen_l[0][0] = _w*_rho1;
	eigen_l[0][1] = zero_float;
	eigen_l[0][2] = zero_float;
	eigen_l[0][3] = - _rho1;
	eigen_l[0][4] = zero_float;
	
	eigen_l[1][0] = one_float - half_float*c21_Gamma*q2;
	eigen_l[1][1] = c21_Gamma*_u;
	eigen_l[1][2] = c21_Gamma*_v;
	eigen_l[1][3] = c21_Gamma*_w;
	eigen_l[1][4] = - c21_Gamma;

	eigen_l[2][0] = - _u*_rho1;
	eigen_l[2][1] = _rho1;
	eigen_l[2][2] = zero_float;
	eigen_l[2][3] = zero_float;
	eigen_l[2][4] = zero_float;

	eigen_l[3][0] = half_float*_c1_rho1_Gamma*q2 - _v*_rho1;
	eigen_l[3][1] = -_c1_rho1_Gamma*_u;
	eigen_l[3][2] = -_c1_rho1_Gamma*_v + _rho1;
	eigen_l[3][3] = -_c1_rho1_Gamma*_w;
	eigen_l[3][4] = _c1_rho1_Gamma;

	eigen_l[4][0] = half_float*_c1_rho1_Gamma*q2 + _v*_rho1;
	eigen_l[4][1] = -_c1_rho1_Gamma*_u;
	eigen_l[4][2] = -_c1_rho1_Gamma*_v - _rho1;
	eigen_l[4][3] = -_c1_rho1_Gamma*_w;
	eigen_l[4][4] = _c1_rho1_Gamma;

	//right eigen vectors
	eigen_r[0][0] = zero_float;
	eigen_r[0][1] = one_float;
	eigen_r[0][2] = zero_float;
	eigen_r[0][3] = _c1_rho;
	eigen_r[0][4] = _c1_rho;
	
	eigen_r[1][0] = zero_float;
	eigen_r[1][1] = _u;
	eigen_r[1][2] = _rho;
	eigen_r[1][3] = _c1_rho*_u;
	eigen_r[1][4] = _c1_rho*_u;

	eigen_r[2][0] = zero_float;
	eigen_r[2][1] = _v;
	eigen_r[2][2] = zero_float;
	eigen_r[2][3] = _c1_rho*(_v + _c);
	eigen_r[2][4] = _c1_rho*(_v - _c);

	eigen_r[3][0] = - _rho;
	eigen_r[3][1] = _w;
	eigen_r[3][2] = zero_float;
	eigen_r[3][3] = _c1_rho*_w;
	eigen_r[3][4] = _c1_rho*_w;

	eigen_r[4][0] = - _rho*_w;
	eigen_r[4][1] = half_float*q2;
	eigen_r[4][2] = _rho*_u;
	eigen_r[4][3] = _c1_rho*(_H + _v*_c);
	eigen_r[4][4] = _c1_rho*(_H - _v*_c);

#endif // COP
}

inline void RoeAverage_z(real_t eigen_l[Emax][Emax], real_t eigen_r[Emax][Emax], real_t z[NUM_SPECIES], real_t yi[NUM_SPECIES], real_t const c2,
						 real_t const _rho, real_t const _u, real_t const _v, real_t const _w,
						 real_t const _H, real_t const D, real_t const D1, real_t const Gamma)
{
	// preparing some interval value

	real_t _Gamma = Gamma - one_float;
	real_t q2 = _u*_u + _v*_v + _w*_w;
	// real_t c2 = _Gamma*(_H - half_float*q2);
	real_t _c = sqrt(c2);
	real_t c21_Gamma = _Gamma / c2;

#ifdef COP
	real_t b1 = c21_Gamma;
	real_t b2 = one_float + c21_Gamma * q2 - c21_Gamma * _H;
	real_t b3 = zero_float;
	for (size_t i = 1; i < NUM_SPECIES; i++)
	{
		b3 += b1 * yi[i] * z[i];
	}
	real_t _c1 = one_float / _c;
	// printf("b1=%lf,b2=%lf,b3=%lf,_c=%lf,yi[]=%lf,%lf,z[]=%lf,%lf\n", b1, b2, b3, _c, yi[0], yi[1], z[0], z[1]);
	// // printf("b2=%lf , b3=%lf \n", b2, b3);
	// // printf("b2=%lf , b3=%lf , z1=%lf \n", b2, b3, z[1]);
	// eigen_l[2][0] += -b3; //在非组分时该项为1-b2,可以通过c2=Gamma*p/rho带入得到，在组分情况下某些文献中不能用前述c2式子
	// // printf("eigen_l[%d][0]=%lf , eigen_l[Emax - 1][0]=%lf \n", 0, eigen_l[Emax - 2][0], eigen_l[Emax - 1][0]);
	// eigen_l[Emax - 2][0] += b3 * _c * _rho1;
	// eigen_l[Emax - 1][0] += b3 * _c * _rho1;
	// // printf("eigen_l[Emax - 2][0]=%lf , eigen_l[Emax - 1][0]=%lf \n", eigen_l[Emax - 2][0], eigen_l[Emax - 1][0]);
	// for (size_t j = Emax - NUM_COP; j < Emax; j++)
	// { // COP相关列
	// 	eigen_l[2][j] = c21_Gamma * z[j + NUM_SPECIES - Emax];
	// 	eigen_l[0][j] = 0;
	// 	eigen_l[1][j] = 0;
	// 	eigen_l[Emax - 2][j] = -c21_Gamma * z[j + NUM_SPECIES - Emax] * _c * _rho1;
	// 	eigen_l[Emax - 1][j] = -c21_Gamma * z[j + NUM_SPECIES - Emax] * _c * _rho1;
	// }
	// // printf("eigen_l[Emax - 2][0]=%lf , eigen_l[Emax - 1][0]=%lf \n", eigen_l[Emax - 2][0], eigen_l[Emax - 1][0]);
	// //  COP相关行所有元素
	// for (size_t ii = 0; ii < NUM_COP; ii++)
	// {
	// 	eigen_l[Emax - 1 - NUM_SPECIES + ii][0] = -yi[ii + 1];
	// 	for (size_t jj = 1; jj < Emax; jj++)
	// 	{
	// 		eigen_l[Emax - 1 - NUM_SPECIES + ii][jj] = (ii + Emax - NUM_COP == jj) ? 1 : 0;
	// 	}
	// }
	// // printf("eigen_l[Emax - 2][0]=%lf , eigen_l[Emax - 1][0]=%lf \n", eigen_l[Emax - 2][0], eigen_l[Emax - 1][0]);
	// // R_Matrix
	// for (size_t m = Emax - NUM_COP; m < Emax; m++)
	// {
	// 	eigen_r[m][2] = yi[m + NUM_SPECIES - Emax];
	// 	eigen_r[m][0] = 0;
	// 	eigen_r[m][1] = 0;
	// 	eigen_r[m][Emax - 2] = yi[m + NUM_SPECIES - Emax] * _c1_rho;
	// 	eigen_r[m][Emax - 1] = yi[m + NUM_SPECIES - Emax] * _c1_rho;
	// 	for (size_t n = Emax - 1 - NUM_SPECIES; n < Emax - 2; n++)
	// 	{
	// 		eigen_r[m][n] = (m == n + 2) ? 1 : 0;
	// 	}
	// }

	// for (size_t nn = Emax - 1 - NUM_SPECIES; nn < Emax - 2; nn++)
	// {
	// 	for (size_t mm = 0; mm < Emax - NUM_SPECIES; mm++)
	// 	{
	// 		eigen_r[mm][nn] = 0;
	// 	}
	// 	eigen_r[Emax - NUM_SPECIES][nn] = z[nn - 2];
	// }
#else
	real_t _rho1 = one_float / _rho;
	real_t _c1_rho = half_float * _rho / _c;
	real_t _c1_rho1_Gamma = _Gamma * _rho1 / _c;
	// left eigen vectors 
	eigen_l[0][0] = - _v*_rho1;
	eigen_l[0][1] = zero_float;
	eigen_l[0][2] = _rho1;
	eigen_l[0][3] = zero_float;
	eigen_l[0][4] = zero_float;
	
	eigen_l[1][0] = _u*_rho1;
	eigen_l[1][1] = - _rho1;
	eigen_l[1][2] = zero_float;
	eigen_l[1][3] = zero_float;
	eigen_l[1][4] = zero_float;

	eigen_l[2][0] = one_float - half_float*c21_Gamma*q2; 
	eigen_l[2][1] = c21_Gamma*_u; 
	eigen_l[2][2] = c21_Gamma*_v; 
	eigen_l[2][3] = c21_Gamma*_w; 
	eigen_l[2][4] = - c21_Gamma;

	eigen_l[3][0] = half_float*_c1_rho1_Gamma*q2 - _w*_rho1;
	eigen_l[3][1] = -_c1_rho1_Gamma*_u;
	eigen_l[3][2] = -_c1_rho1_Gamma*_v;
	eigen_l[3][3] = -_c1_rho1_Gamma*_w + _rho1;
	eigen_l[3][4] = _c1_rho1_Gamma;

	eigen_l[4][0] = half_float*_c1_rho1_Gamma*q2 + _w*_rho1;
	eigen_l[4][1] = -_c1_rho1_Gamma*_u;
	eigen_l[4][2] = -_c1_rho1_Gamma*_v;
	eigen_l[4][3] = -_c1_rho1_Gamma*_w - _rho1;
	eigen_l[4][4] = _c1_rho1_Gamma;

	//right eigen vectors
	eigen_r[0][0] = zero_float;
	eigen_r[0][1] = zero_float;
	eigen_r[0][2] = one_float;
	eigen_r[0][3] = _c1_rho;
	eigen_r[0][4] = _c1_rho;
	
	eigen_r[1][0] = zero_float;
	eigen_r[1][1] = - _rho;
	eigen_r[1][2] = _u;
	eigen_r[1][3] = _c1_rho*_u;
	eigen_r[1][4] = _c1_rho*_u;

	eigen_r[2][0] = _rho;
	eigen_r[2][1] = zero_float;
	eigen_r[2][2] = _v;
	eigen_r[2][3] = _c1_rho*_v;
	eigen_r[2][4] = _c1_rho*_v;

	eigen_r[3][0] = zero_float;
	eigen_r[3][1] = zero_float;
	eigen_r[3][2] = _w;
	eigen_r[3][3] = _c1_rho*(_w + _c);
	eigen_r[3][4] = _c1_rho*(_w - _c);

	eigen_r[4][0] = _rho*_v;
	eigen_r[4][1] = -_rho*_u;
	eigen_r[4][2] = half_float*q2;
	eigen_r[4][3] = _c1_rho*(_H + _w*_c);
	eigen_r[4][4] = _c1_rho*(_H - _w*_c);
#endif // COP
}

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
//	real_t w1, w2, w3;

	//assign value to v1, v2,...
	k = 0;
	v1 = *(f + k - 2);
	v2 = *(f + k - 1);
	v3 = *(f + k);
	v4 = *(f + k + 1); 
	v5 = *(f + k + 2);

	//smoothness indicator
//	real_t s1 = 13.0*(v1 - 2.0*v2 + v3)*(v1 - 2.0*v2 + v3) 
//	   + 3.0*(v1 - 4.0*v2 + 3.0*v3)*(v1 - 4.0*v2 + 3.0*v3);
//	real_t s2 = 13.0*(v2 - 2.0*v3 + v4)*(v2 - 2.0*v3 + v4) 
//	   + 3.0*(v2 - v4)*(v2 - v4);
//	real_t s3 = 13.0*(v3 - 2.0*v4 + v5)*(v3 - 2.0*v4 + v5) 
//	   + 3.0*(3.0*v3 - 4.0*v4 + v5)*(3.0*v3 - 4.0*v4 + v5);
	
        //weights
//      a1 = 0.1/((1.0e-6 + s1)*(1.0e-6 + s1));
//      a2 = 0.6/((1.0e-6 + s2)*(1.0e-6 + s2));
//      a3 = 0.3/((1.0e-6 + s3)*(1.0e-6 + s3));
//      real_t tw1 = 1.0/(a1 + a2 +a3); 
//      w1 = a1*tw1;
//      w2 = a2*tw1;
//      w3 = a3*tw1;

//      a1 = w1*(2.0*v1 - 7.0*v2 + 11.0*v3);
//      a2 = w2*(-v2 + 5.0*v3 + 2.0*v4);
//      a3 = w3*(2.0*v3 + 5.0*v4 - v5);

//      return (a1+a2+a3)/6.0;

        //return weighted average
//      return  w1*(2.0*v1 - 7.0*v2 + 11.0*v3)/6.0
  //              + w2*(-v2 + 5.0*v3 + 2.0*v4)/6.0
    //            + w3*(2.0*v3 + 5.0*v4 - v5)/6.0;

#if USE_DP
	a1 = v1 - 2.0 * v2 + v3;
	real_t s1 = 13.0 * a1 * a1;
	a1 = v1 - 4.0 * v2 + 3.0 * v3;
	s1 += 3.0 * a1 * a1;
	a1 = v2 - 2.0 * v3 + v4;
	real_t s2 = 13.0 * a1 * a1;
	a1 = v2 - v4;
	s2 += 3.0 * a1 * a1;
	a1 = v3 - 2.0 * v4 + v5;
	real_t s3 = 13.0 * a1 * a1;
	a1 = 3.0 * v3 - 4.0 * v4 + v5;
	s3 += 3.0 * a1 * a1;
#else
	a1 = v1 - 2.0f * v2 + v3;
	real_t s1 = 13.0f * a1 * a1;
	a1 = v1 - 4.0f * v2 + 3.0f * v3;
	s1 += 3.0f * a1 * a1;
	a1 = v2 - 2.0f * v3 + v4;
	real_t s2 = 13.0f * a1 * a1;
	a1 = v2 - v4;
	s2 += 3.0f * a1 * a1;
	a1 = v3 - 2.0f * v4 + v5;
	real_t s3 = 13.0f * a1 * a1;
	a1 = 3.0f * v3 - 4.0f * v4 + v5;
	s3 += 3.0f * a1 * a1;
#endif

	// a1 = 0.1/((1.0e-6 + s1)*(1.0e-6 + s1));
	// a2 = 0.6/((1.0e-6 + s2)*(1.0e-6 + s2));
	// a3 = 0.3/((1.0e-6 + s3)*(1.0e-6 + s3));
	// real_t tw1 = 1.0/(a1 + a2 +a3);
	// a1 = a1*tw1;
	// a2 = a2*tw1;
	// a3 = a3*tw1;
	real_t tol = 1.0e-6;
#if USE_DP
	a1 = 0.1 * (tol + s2) * (tol + s2) * (tol + s3) * (tol + s3);
	a2 = 0.2 * (tol + s1) * (tol + s1) * (tol + s3) * (tol + s3);
	a3 = 0.3 * (tol + s1) * (tol + s1) * (tol + s2) * (tol + s2);
	real_t tw1 = 1.0 / (a1 + a2 + a3);
#else
	a1 = 0.1f * (tol + s2) * (tol + s2) * (tol + s3) * (tol + s3);
	a2 = 0.2f * (tol + s1) * (tol + s1) * (tol + s3) * (tol + s3);
	a3 = 0.3f * (tol + s1) * (tol + s1) * (tol + s2) * (tol + s2);
	real_t tw1 = 1.0f / (a1 + a2 + a3);
#endif

	a1 = a1 * tw1;
	a2 = a2 * tw1;
	a3 = a3 * tw1;

#if USE_DP
	s1 = a1 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3);
	s2 = a2 * (-v2 + 5.0 * v3 + 2.0 * v4);
	s3 = a3 * (2.0 * v3 + 5.0 * v4 - v5);
#else
	s1 = a1 * (2.0f * v1 - 7.0f * v2 + 11.0f * v3);
	s2 = a2 * (-v2 + 5.0f * v3 + 2.0f * v4);
	s3 = a3 * (2.0f * v3 + 5.0f * v4 - v5);
#endif

	// return (s1+s2+s3)/6.0;
	return (s1 + s2 + s3);

	// a1 = (1.0e-6 + s1)*(1.0e-6 + s1);
	// a2 = (1.0e-6 + s2)*(1.0e-6 + s2);
	// a3 = (1.0e-6 + s3)*(1.0e-6 + s3);
	// real_t tw1 = 6.0*(a1 + a2 + a3);
	// return (0.1*(2.0*v1 - 7.0*v2 + 11.0*v3) + 0.6*(-v2 + 5.0*v3 + 2.0*v4) + 0.2*(2.0*v3 + 5.0*v4 - v5))/tw1;
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
//	real_t w1, w2, w3;

	//assign value to v1, v2,...
	k = 1;
	v1 = *(f + k + 2);
	v2 = *(f + k + 1);
	v3 = *(f + k);
	v4 = *(f + k - 1); 
	v5 = *(f + k - 2);

	//smoothness indicator
//	real_t s1 = 13.0*(v1 - 2.0*v2 + v3)*(v1 - 2.0*v2 + v3) 
//	   + 3.0*(v1 - 4.0*v2 + 3.0*v3)*(v1 - 4.0*v2 + 3.0*v3);
//	real_t s2 = 13.0*(v2 - 2.0*v3 + v4)*(v2 - 2.0*v3 + v4) 
//	   + 3.0*(v2 - v4)*(v2 - v4);
//	real_t s3 = 13.0*(v3 - 2.0*v4 + v5)*(v3 - 2.0*v4 + v5) 
//	   + 3.0*(3.0*v3 - 4.0*v4 + v5)*(3.0*v3 - 4.0*v4 + v5);

        //weights
//      a1 = 0.1/((1.0e-6 + s1)*(1.0e-6 + s1));
//      a2 = 0.6/((1.0e-6 + s2)*(1.0e-6 + s2));
//      a3 = 0.3/((1.0e-6 + s3)*(1.0e-6 + s3));
//      real_t tw1 = 1.0/(a1 + a2 +a3); 
//      w1 = a1*tw1;
//      w2 = a2*tw1;
//      w3 = a3*tw1;

//      a1 = w1*(2.0*v1 - 7.0*v2 + 11.0*v3);
//      a2 = w2*(-v2 + 5.0*v3 + 2.0*v4);
//      a3 = w3*(2.0*v3 + 5.0*v4 - v5);

//      return (a1+a2+a3)/6.0;

        //return weighted average
	//      return  w1*(2.0*v1 - 7.0*v2 + 11.0*v3)/6.0
	//                + w2*(-v2 + 5.0*v3 + 2.0*v4)/6.0
	//                + w3*(2.0*v3 + 5.0*v4 - v5)/6.0;

#if USE_DP
	a1 = v1 - 2.0 * v2 + v3;
	real_t s1 = 13.0 * a1 * a1;
	a1 = v1 - 4.0 * v2 + 3.0 * v3;
	s1 += 3.0 * a1 * a1;
	a1 = v2 - 2.0 * v3 + v4;
	real_t s2 = 13.0 * a1 * a1;
	a1 = v2 - v4;
	s2 += 3.0 * a1 * a1;
	a1 = v3 - 2.0 * v4 + v5;
	real_t s3 = 13.0 * a1 * a1;
	a1 = 3.0 * v3 - 4.0 * v4 + v5;
	s3 += 3.0 * a1 * a1;
#else
	a1 = v1 - 2.0f * v2 + v3;
	real_t s1 = 13.0f * a1 * a1;
	a1 = v1 - 4.0f * v2 + 3.0f * v3;
	s1 += 3.0f * a1 * a1;
	a1 = v2 - 2.0f * v3 + v4;
	real_t s2 = 13.0f * a1 * a1;
	a1 = v2 - v4;
	s2 += 3.0f * a1 * a1;
	a1 = v3 - 2.0f * v4 + v5;
	real_t s3 = 13.0f * a1 * a1;
	a1 = 3.0f * v3 - 4.0f * v4 + v5;
	s3 += 3.0f * a1 * a1;
#endif

	//  a1 = 0.1/((1.0e-6 + s1)*(1.0e-6 + s1));
	//  a2 = 0.6/((1.0e-6 + s2)*(1.0e-6 + s2));
	//  a3 = 0.3/((1.0e-6 + s3)*(1.0e-6 + s3));
	//  real_t tw1 = 1.0/(a1 + a2 +a3);
	//  a1 = a1*tw1;
	//  a2 = a2*tw1;
	//  a3 = a3*tw1;
	real_t tol = 1.0e-6;
#if USE_DP
	a1 = 0.1 * (tol + s2) * (tol + s2) * (tol + s3) * (tol + s3);
	a2 = 0.2 * (tol + s1) * (tol + s1) * (tol + s3) * (tol + s3);
	a3 = 0.3 * (tol + s1) * (tol + s1) * (tol + s2) * (tol + s2);
	real_t tw1 = 1.0 / (a1 + a2 + a3);
#else
	a1 = 0.1f * (tol + s2) * (tol + s2) * (tol + s3) * (tol + s3);
	a2 = 0.2f * (tol + s1) * (tol + s1) * (tol + s3) * (tol + s3);
	a3 = 0.3f * (tol + s1) * (tol + s1) * (tol + s2) * (tol + s2);
	real_t tw1 = 1.0f / (a1 + a2 + a3);
#endif
	a1 = a1 * tw1;
	a2 = a2 * tw1;
	a3 = a3 * tw1;

#if USE_DP
	s1 = a1 * (2.0 * v1 - 7.0 * v2 + 11.0 * v3);
	s2 = a2 * (-v2 + 5.0 * v3 + 2.0 * v4);
	s3 = a3 * (2.0 * v3 + 5.0 * v4 - v5);
#else
	s1 = a1 * (2.0f * v1 - 7.0f * v2 + 11.0f * v3);
	s2 = a2 * (-v2 + 5.0f * v3 + 2.0f * v4);
	s3 = a3 * (2.0f * v3 + 5.0f * v4 - v5);
#endif

	//  return (s1+s2+s3)/6.0;
	return (s1 + s2 + s3);

	//  a1 = (1.0e-6 + s1)*(1.0e-6 + s1);
	//  a2 = (1.0e-6 + s2)*(1.0e-6 + s2);
	//  a3 = (1.0e-6 + s3)*(1.0e-6 + s3);
	//  real_t tw1 = 6.0*(a1 + a2 + a3);
	//  return (0.1*(2.0*v1 - 7.0*v2 + 11.0*v3) + 0.6*(-v2 + 5.0*v3 + 2.0*v4) + 0.2*(2.0*v3 + 5.0*v4 - v5))/tw1;
}
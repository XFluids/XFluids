#pragma once

#include "schemes_device.hpp"

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

// /**
//  *@brief calculate yi from y
//  */
// void get_yi(real_t *y, real_t yi[NUM_SPECIES], const int id)
// {
// #ifdef COP
// 	for (size_t i = 0; i < NUM_SPECIES; i++)
// 	{
// 		yi[i] = y[i][id];
// 	}
// #else
// 	yi[NUM_COP] = _DF(1.0);
// #endif
// }

/**
 *@brief calculate yi : mass fraction from xi : mole fraction.
 */
void get_yi(real_t xi[NUM_SPECIES], real_t const Wi[NUM_SPECIES])
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
real_t get_xi(real_t xi[NUM_SPECIES], real_t const yi[NUM_SPECIES], real_t const *_Wi, const real_t rho)
{
	real_t C[NUM_SPECIES] = {_DF(0.0)}, C_total = _DF(0.0);
	for (int i = 0; i < NUM_SPECIES; i++)
	{
		C[i] = yi[i] * _Wi[i] * _DF(1e-3) * rho;
		C_total = C_total + C[i];
	}
	// get mole fraction of each specie
	real_t _C_total = _DF(1.0) / C_total;
	for (int i = 0; i < NUM_SPECIES; i++)
		xi[i] = C[i] * _C_total;
	return C_total;
}

/**
 *@brief calculate R for every cell
 */
real_t get_CopR(const real_t *_Wi, const real_t yi[NUM_SPECIES])
{
	real_t R = _DF(0.0);
	for (size_t n = 0; n < NUM_SPECIES; n++)
		R += yi[n] * _Wi[n];

	return R * Ru;
}

/**
 * @brief calculate Cpi of the specie at given point
 * unit: J/kg/K
 */
real_t HeatCapacity(real_t *Hia, const real_t T0, const real_t Ri, const int n)
{
	// 	real_t T = T0; // sycl::max<real_t>(T0, _DF(200.0));
	// 	real_t Cpi = _DF(0.0), _T = _DF(1.0) / T;
	// #if Thermo
	// 	if (T >= (_DF(1000.0)) && T < (_DF(6000.0)))
	// 		Cpi = Ri * ((Hia[n * 7 * 3 + 0 * 3 + 1] * _T + Hia[n * 7 * 3 + 1 * 3 + 1]) * _T + Hia[n * 7 * 3 + 2 * 3 + 1] + (Hia[n * 7 * 3 + 3 * 3 + 1] + (Hia[n * 7 * 3 + 4 * 3 + 1] + (Hia[n * 7 * 3 + 5 * 3 + 1] + Hia[n * 7 * 3 + 6 * 3 + 1] * T) * T) * T) * T);
	// 	else if (T < (_DF(1000.0)))
	// 		Cpi = Ri * ((Hia[n * 7 * 3 + 0 * 3 + 0] * _T + Hia[n * 7 * 3 + 1 * 3 + 0]) * _T + Hia[n * 7 * 3 + 2 * 3 + 0] + (Hia[n * 7 * 3 + 3 * 3 + 0] + (Hia[n * 7 * 3 + 4 * 3 + 0] + (Hia[n * 7 * 3 + 5 * 3 + 0] + Hia[n * 7 * 3 + 6 * 3 + 0] * T) * T) * T) * T);
	// 	else if (T >= _DF(6000.0))
	// 		Cpi = Ri * ((Hia[n * 7 * 3 + 0 * 3 + 2] * _T + Hia[n * 7 * 3 + 1 * 3 + 2]) * _T + Hia[n * 7 * 3 + 2 * 3 + 2] + (Hia[n * 7 * 3 + 3 * 3 + 2] + (Hia[n * 7 * 3 + 4 * 3 + 2] + (Hia[n * 7 * 3 + 5 * 3 + 2] + Hia[n * 7 * 3 + 6 * 3 + 2] * T) * T) * T) * T);
	// #else
	// 	// if (T > _DF(1000.0))
	// 	// 	Cpi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 0] + (Hia[n * 7 * 3 + 1 * 3 + 0] + (Hia[n * 7 * 3 + 2 * 3 + 0] + (Hia[n * 7 * 3 + 3 * 3 + 0] + Hia[n * 7 * 3 + 4 * 3 + 0] * T) * T) * T) * T);
	// 	// else
	// 	// 	Cpi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 1] + (Hia[n * 7 * 3 + 1 * 3 + 1] + (Hia[n * 7 * 3 + 2 * 3 + 1] + (Hia[n * 7 * 3 + 3 * 3 + 1] + Hia[n * 7 * 3 + 4 * 3 + 1] * T) * T) * T) * T);
	// 	if (T > _DF(1000.0))
	// 		Cpi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 0] + Hia[n * 7 * 3 + 1 * 3 + 0] * T + Hia[n * 7 * 3 + 2 * 3 + 0] * T * T + Hia[n * 7 * 3 + 3 * 3 + 0] * T * T * T + Hia[n * 7 * 3 + 4 * 3 + 0] * T * T * T * T);
	// 	else
	// 		Cpi = Ri * (Hia[n * 7 * 3 + 0 * 3 + 1] + Hia[n * 7 * 3 + 1 * 3 + 1] * T + Hia[n * 7 * 3 + 2 * 3 + 1] * T * T + Hia[n * 7 * 3 + 3 * 3 + 1] * T * T * T + Hia[n * 7 * 3 + 4 * 3 + 1] * T * T * T * T);
	// #endif
#if Thermo
	MARCO_HeatCapacity_NASA();
#else
	MARCO_HeatCapacity_JANAF();
#endif // end Thermo

	return Cpi;
}

/**
 * @brief Compute the Cp of the mixture at given point unit:J/kg/K
 */
real_t get_CopCp(Thermal thermal, const real_t yi[NUM_SPECIES], const real_t T)
{
	real_t _CopCp = _DF(0.0);
	for (size_t ii = 0; ii < NUM_SPECIES; ii++)
		_CopCp += yi[ii] * HeatCapacity(thermal.Hia, T, thermal.Ri[ii], ii);

	return _CopCp;
}

/**
 * @brief calculate W of the mixture at given point
 */
real_t get_CopW(Thermal thermal, const real_t yi[NUM_SPECIES])
{
	real_t _W = _DF(0.0);
	for (size_t ii = 0; ii < NUM_SPECIES; ii++)
		_W += yi[ii] * thermal._Wi[ii]; // Wi
	// printf("W=%lf \n", 1.0 / _W);
	return _DF(1.0) / _W;
}

/**
 * @brief calculate Gamma of the mixture at given point
 */
real_t get_CopGamma(Thermal thermal, const real_t yi[NUM_SPECIES], const real_t T)
{
	real_t Cp = get_CopCp(thermal, yi, T);
	real_t CopW = get_CopW(thermal, yi);
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
	// 	real_t hi = _DF(0.0), TT = T0, T = TT; // sycl::max<real_t>(T0, _DF(200.0));
	// #if Thermo
	// 	if (T >= _DF(1000.0) && T < _DF(6000.0))
	// 		hi = Ri * (-Hia[n * 7 * 3 + 0 * 3 + 1] / T + Hia[n * 7 * 3 + 1 * 3 + 1] * sycl::log(T) + (Hia[n * 7 * 3 + 2 * 3 + 1] + (0.5 * Hia[n * 7 * 3 + 3 * 3 + 1] + (Hia[n * 7 * 3 + 4 * 3 + 1] / _DF(3.0) + (_DF(0.25) * Hia[n * 7 * 3 + 5 * 3 + 1] + _DF(0.2) * Hia[n * 7 * 3 + 6 * 3 + 1] * T) * T) * T) * T) * T + Hib[n * 2 * 3 + 0 * 3 + 1]);
	// 	else if (T < _DF(1000.0))
	// 		hi = Ri * (-Hia[n * 7 * 3 + 0 * 3 + 0] / T + Hia[n * 7 * 3 + 1 * 3 + 0] * sycl::log(T) + (Hia[n * 7 * 3 + 2 * 3 + 0] + (0.5 * Hia[n * 7 * 3 + 3 * 3 + 0] + (Hia[n * 7 * 3 + 4 * 3 + 0] / _DF(3.0) + (_DF(0.25) * Hia[n * 7 * 3 + 5 * 3 + 0] + _DF(0.2) * Hia[n * 7 * 3 + 6 * 3 + 0] * T) * T) * T) * T) * T + Hib[n * 2 * 3 + 0 * 3 + 0]);
	// 	else if (T >= _DF(6000.0))
	// 		hi = Ri * (-Hia[n * 7 * 3 + 0 * 3 + 2] / T + Hia[n * 7 * 3 + 1 * 3 + 2] * sycl::log(T) + (Hia[n * 7 * 3 + 2 * 3 + 2] + (0.5 * Hia[n * 7 * 3 + 3 * 3 + 2] + (Hia[n * 7 * 3 + 4 * 3 + 2] / _DF(3.0) + (_DF(0.25) * Hia[n * 7 * 3 + 5 * 3 + 2] + _DF(0.2) * Hia[n * 7 * 3 + 6 * 3 + 2] * T) * T) * T) * T) * T + Hib[n * 2 * 3 + 0 * 3 + 2]);
	// #else
	// 	// H/RT = a1 + a2/2*T + a3/3*T^2 + a4/4*T^3 + a5/5*T^4 + a6/T
	// 	// if (T > _DF(1000.0))
	// 	// 	hi = Ri * (T * (Hia[n * 7 * 3 + 0 * 3 + 0] + T * (Hia[n * 7 * 3 + 1 * 3 + 0] * _DF(0.5) + T * (Hia[n * 7 * 3 + 2 * 3 + 0] / _DF(3.0) + T * (Hia[n * 7 * 3 + 3 * 3 + 0] * _DF(0.25) + Hia[n * 7 * 3 + 4 * 3 + 0] * T * _DF(0.2))))) + Hia[n * 7 * 3 + 5 * 3 + 0]);
	// 	// else
	// 	// 	hi = Ri * (T * (Hia[n * 7 * 3 + 0 * 3 + 1] + T * (Hia[n * 7 * 3 + 1 * 3 + 1] * _DF(0.5) + T * (Hia[n * 7 * 3 + 2 * 3 + 1] / _DF(3.0) + T * (Hia[n * 7 * 3 + 3 * 3 + 1] * _DF(0.25) + Hia[n * 7 * 3 + 4 * 3 + 1] * T * _DF(0.2))))) + Hia[n * 7 * 3 + 5 * 3 + 1]);
	// 	if (T > _DF(1000.0))
	// 		hi = Ri * (T * (Hia[n * 7 * 3 + 0 * 3 + 0] + T * (Hia[n * 7 * 3 + 1 * 3 + 0] / _DF(2.0) + T * (Hia[n * 7 * 3 + 2 * 3 + 0] / _DF(3.0) + T * (Hia[n * 7 * 3 + 3 * 3 + 0] / _DF(4.0) + Hia[n * 7 * 3 + 4 * 3 + 0] * T / _DF(5.0))))) + Hia[n * 7 * 3 + 5 * 3 + 0]);
	// 	else
	// 		hi = Ri * (T * (Hia[n * 7 * 3 + 0 * 3 + 1] + T * (Hia[n * 7 * 3 + 1 * 3 + 1] / _DF(2.0) + T * (Hia[n * 7 * 3 + 2 * 3 + 1] / _DF(3.0) + T * (Hia[n * 7 * 3 + 3 * 3 + 1] / _DF(4.0) + Hia[n * 7 * 3 + 4 * 3 + 1] * T / _DF(5.0))))) + Hia[n * 7 * 3 + 5 * 3 + 1]);
	// #endif
	// 	// if (TT < _DF(200.0))							  // take low tempreture into consideration
	// 	// {												  // get_hi at T>200
	// 	// 	real_t Cpi = HeatCapacity(Hia, _DF(200.0), Ri, n);
	// 	// 	hi += Cpi * (TT - _DF(200.0));
	// 	// }

#if Thermo
	MARCO_Enthalpy_NASA();
#else
	MARCO_Enthalpy_JANAF();
#endif

	return hi;
}

/**
 * @brief calculate Hi of Mixture at given point	unit:J/kg/K
 */
real_t get_Coph(Thermal thermal, const real_t yi[NUM_SPECIES], const real_t T)
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
void sub_FuncT(real_t &func_T, real_t &dfunc_T, Thermal thermal, const real_t yi[NUM_SPECIES], const real_t e, const real_t T)
{
	real_t h = get_Coph(thermal, yi, T);			 // J/kg/K
	real_t R = get_CopR(thermal._Wi, yi);			 // J/kg/K
	real_t Cp = get_CopCp(thermal, yi, T);			 // J/kg/K
	func_T = h - R * T - e;							 // unit:J/kg/K
	dfunc_T = Cp - R;								 // unit:J/kg/K
}

/**
 *@brief update T through Newtonian dynasty
 */
real_t get_T(Thermal thermal, const real_t yi[NUM_SPECIES], const real_t e, const real_t T0)
{
	real_t T = T0;
	real_t tol = 1.0e-6, T_dBdr = 100.0, T_uBdr = 1.0e4, x_eps = 1.0e-3;
	// tol /= Tref, T_dBdr /= Tref, T_uBdr /= Tref, x_eps /= Tref;
	real_t rt_bis, f, f_mid;
	real_t func_T = 0, dfunc_T = 0;

	for (int i = 1; i <= 150; i++)
	{
		sub_FuncT(func_T, dfunc_T, thermal, yi, e, T);
		real_t df = func_T / dfunc_T;
		T = T - df;
		if (sycl::abs<real_t>(df) <= tol)
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
				if (sycl::abs<real_t>(df) <= x_eps || f_mid == 0.0)
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
			   real_t &gamma, real_t &T, Thermal thermal, real_t yi[NUM_SPECIES])
{
	rho = UI[0];
	real_t rho1 = _DF(1.0) / rho;
	u = UI[1] * rho1;
	v = UI[2] * rho1;
	w = UI[3] * rho1;

	yi[NUM_COP] = _DF(1.0);

#ifdef COP
	for (size_t ii = 5; ii < Emax; ii++)
	{ // calculate yi
		yi[ii - 5] = UI[ii] * rho1;
		// yi[ii - 5] = sycl::max(_DF(0.0), UI[ii] * rho1);
		yi[NUM_COP] += -yi[ii - 5];
		// UI[ii] = yi[ii - 5] * rho;
	}
#endif // end COP

#ifdef COP
	real_t e = UI[4] * rho1 - _DF(0.5) * (u * u + v * v + w * w);
	real_t R = get_CopR(thermal._Wi, yi);
	T = get_T(thermal, yi, e, T);
	p = rho * R * T; // 对所有气体都适用
	gamma = get_CopGamma(thermal, yi, T);
#else
	gamma = NCOP_Gamma;
	p = (NCOP_Gamma - _DF(1.0)) * (UI[4] - _DF(0.5) * rho * (u * u + v * v + w * w));
#endif // end COP
	H = (UI[4] + p) * rho1;
	c = sycl::sqrt(gamma * p * rho1);
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

/**
 * @brief Roe average of value u_l and u_r
 * @brief  _u = (u[id_l] + D * u[id_r]) * D1
 * @param ul left value
 * @param ur right value
 * @param D
 * @return real_t
 */
real_t get_RoeAverage(const real_t left, const real_t right, const real_t D, const real_t D1)
{
	return (left + D * right) * D1;
}

/**
 * @brief \frac{\partial p}{\partial \rho}
 * @param hiN: hi[NUM_COP]
 * @param RiN: Ru/thermal.specie_chara[NUM_COP*SPCH_Sz+6]
 * @param q2: u*u+v*v+w*w
 * @param
 * @return real_t
 */
real_t get_DpDrho(const real_t hN, const real_t RN, const real_t q2, const real_t Cp, const real_t R, const real_t T, const real_t e, const real_t gamma)
{
#if CJ
	return (Gamma0 - 1.0) * e; // p/rho;//
#else
	double RNT = RN * T; // unit: J/kg
	return (gamma - _DF(1.0)) * (_DF(0.5) * q2 - hN + Cp * RNT / R);
	// NOTE:return gamma * RNT + (gamma - _DF(1.0)) * (e - hN); // is not right
#endif
}

/**
 * @brief \frac{\partial p}{\partial \rho_i}
 * @param hin: hi of the n-th species
 * @param hiN: hi of the N-th species
 * @param Cp: get_CopCp for mixture
 * @param R: get_CopR for mixture
 * @return real_t
 */
real_t get_DpDrhoi(const real_t hin, const real_t Rin, const real_t hiN, const real_t RiN, const real_t T, const real_t Cp, const real_t R, const real_t gamma)
{
#if CJ
	return 0; //(Gamma0-1.0)*(-heat_release[n]);
#else
	real_t hN_minus_hi = -hin + hiN;  // unit: J/kg
	real_t Ri_minus_RN = (Rin - RiN); // unit: J/kg/K
	real_t temp = (gamma - _DF(1.0)) * (hN_minus_hi + Cp * Ri_minus_RN * T / R);
	return temp;
#endif
}

/**
 * @brief compute Roe-averaged sound speed of multicomponent flows
 * @param zi: for eigen matrix
 * @return real_t c*c
 */
real_t SoundSpeedMultiSpecies(real_t *zi, real_t &b1, real_t &b3, real_t *_Yi, real_t *_dpdrhoi, real_t *drhoi, const real_t _dpdrho, const real_t _dpde,
							  const real_t _dpdE, const real_t _prho, const real_t dp, const real_t drho, const real_t de, const real_t _rho)
{
	// sum
	real_t _dpdrhoi_new[NUM_COP], Sum_dpdrhoi = _DF(0.0), Sum_drhoi = _DF(0.0), Sum_dpdrhoi2 = _DF(0.0), Sum_Yidpdrhoi = _DF(0.0);
	for (int n = 0; n < NUM_COP; n++)
	{
		Sum_dpdrhoi += _dpdrhoi[n] * drhoi[n];
		Sum_dpdrhoi2 += _dpdrhoi[n] * drhoi[n] * _dpdrhoi[n] * drhoi[n];
	}
	// method 1
	real_t temp1 = dp - (_dpdrho * drho + _dpde * de + Sum_dpdrhoi);
	real_t temp = temp1 / (_dpdrho * _dpdrho * drho * drho + _dpde * de * _dpde * de + Sum_dpdrhoi2 + 1e-19);

	for (int n = 0; n < NUM_COP; n++)
	{
		Sum_drhoi += drhoi[n] * drhoi[n];
		Sum_Yidpdrhoi += _Yi[n] * _dpdrhoi[n];
	}

	real_t _dpdE_new = _dpdE + _dpdE * _dpdE * de * _rho * temp;
	real_t _dpdrho_new = _dpdrho + _dpdrho * _dpdrho * drho * temp;

	// sound speed
	for (int n = 0; n < NUM_COP; n++)
		_dpdrhoi_new[n] = _dpdrhoi[n] + _dpdrhoi[n] * _dpdrhoi[n] * drhoi[n] * temp;

	real_t csqr = _dpdrho_new + _dpdE_new * _prho + Sum_Yidpdrhoi;
	b1 = _dpdE_new / csqr;
	for (int n = 0; n < NUM_COP; n++)
	{
		zi[n] = -_dpdrhoi_new[n] / _dpdE_new;
		b3 += _Yi[n] * zi[n];
	}
	b3 *= b1;

	return csqr;
}

/**
 * @brief calculate c^2 of the mixture at given point
 * // NOTE: realted with yn=yi[0] or yi[N] : hi[] Ri[]
 */
real_t get_CopC2(real_t z[NUM_SPECIES], real_t const Ri[NUM_SPECIES], real_t const yi[NUM_SPECIES], real_t const hi[NUM_SPECIES], const real_t gamma, const real_t h, const real_t T)
{
	real_t Sum_dpdrhoi = _DF(0.0);				   // Sum_dpdrhoi:first of c2,存在累加项
	real_t _dpdrhoi[NUM_SPECIES];
	for (size_t n = 0; n < NUM_SPECIES; n++)
	{
		_dpdrhoi[n] = (gamma - _DF(1.0)) * (hi[NUM_COP] - hi[n]) + gamma * (Ri[n] - Ri[NUM_COP]) * T; // related with yi
		z[n] = -_DF(1.0) * _dpdrhoi[n] / (gamma - _DF(1.0));
		if (NUM_COP != n) // related with yi
			Sum_dpdrhoi += yi[n] * _dpdrhoi[n];
	}
	real_t _CopC2 = Sum_dpdrhoi + (gamma - _DF(1.0)) * (h - hi[NUM_COP]) + gamma * Ri[NUM_COP] * T; // related with yi
	return _CopC2;
}

#if 1 == EIGEN_ALLOC || 2 == EIGEN_ALLOC

#if 1 == EIGEN_ALLOC
inline void RoeAverage_x(real_t eigen_l[Emax][Emax], real_t eigen_r[Emax][Emax], real_t eigen_value[Emax], real_t *z, const real_t *yi, real_t const c2,
						 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H, real_t const b1, real_t const b3, real_t Gamma)
#elif 2 == EIGEN_ALLOC
inline void RoeAverage_x(real_t *eigen_l[Emax], real_t *eigen_r[Emax], real_t eigen_value[Emax], real_t *z, const real_t *yi, real_t const c2,
						 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H, real_t const b1, real_t const b3, real_t Gamma)
#endif
{

	MARCO_PREEIGEN();

	real_t _u_c = _u * _c1;

	eigen_l[0][0] = _DF(0.5) * (b2 + _u_c + b3);
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

	eigen_l[Emax - 1][0] = _DF(0.5) * (b2 - _u_c + b3);
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

	eigen_value[0] = sycl::fabs<real_t>(_u - _c);
	eigen_value[1] = sycl::fabs<real_t>(_u);
	eigen_value[2] = eigen_value[1];
	eigen_value[3] = eigen_value[1];
	eigen_value[Emax - 1] = sycl::fabs<real_t>(_u + _c);

#ifdef COP
	for (int n = 0; n < NUM_COP; n++)
		eigen_value[n + Emax - NUM_SPECIES] = eigen_value[1];

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

#if 1 == EIGEN_ALLOC
inline void RoeAverage_y(real_t eigen_l[Emax][Emax], real_t eigen_r[Emax][Emax], real_t eigen_value[Emax], real_t *z, const real_t *yi, real_t const c2,
						 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H, real_t const b1, real_t const b3, real_t Gamma)
#elif 2 == EIGEN_ALLOC
inline void RoeAverage_y(real_t *eigen_l[Emax], real_t *eigen_r[Emax], real_t eigen_value[Emax], real_t *z, const real_t *yi, real_t const c2,
						 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H, real_t const b1, real_t const b3, real_t Gamma)
#endif
{
	MARCO_PREEIGEN();

	real_t _v_c = _v * _c1;

	// left eigen vectors
	eigen_l[0][0] = _DF(0.5) * (b2 + _v_c + b3);
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

	eigen_l[Emax - 1][0] = _DF(0.5) * (b2 - _v_c + b3);
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

	eigen_value[0] = sycl::fabs<real_t>(_v - _c);
	eigen_value[1] = sycl::fabs<real_t>(_v);
	eigen_value[2] = eigen_value[1];
	eigen_value[3] = eigen_value[1];
	eigen_value[Emax - 1] = sycl::fabs<real_t>(_v + _c);

#ifdef COP
	for (int n = 0; n < NUM_COP; n++)
		eigen_value[n + Emax - NUM_SPECIES] = eigen_value[1];

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

#if 1 == EIGEN_ALLOC
inline void RoeAverage_z(real_t eigen_l[Emax][Emax], real_t eigen_r[Emax][Emax], real_t eigen_value[Emax], real_t *z, const real_t *yi, real_t const c2,
						 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H, real_t const b1, real_t const b3, real_t Gamma)
#elif 2 == EIGEN_ALLOC
inline void RoeAverage_z(real_t *eigen_l[Emax], real_t *eigen_r[Emax], real_t eigen_value[Emax], real_t *z, const real_t *yi, real_t const c2,
						 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H, real_t const b1, real_t const b3, real_t Gamma)
#endif
{
	MARCO_PREEIGEN();

	real_t _w_c = _w * _c1;

	// left eigen vectors
	eigen_l[0][0] = _DF(0.5) * (b2 + _w_c + b3);
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

	eigen_l[Emax - 1][0] = _DF(0.5) * (b2 - _w_c + b3);
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

	eigen_value[0] = sycl::fabs<real_t>(_w - _c);
	eigen_value[1] = sycl::fabs<real_t>(_w);
	eigen_value[2] = eigen_value[1];
	eigen_value[3] = eigen_value[1];
	eigen_value[Emax - 1] = sycl::fabs<real_t>(_w + _c);

#ifdef COP
	for (int n = 0; n < NUM_COP; n++)
		eigen_value[n + Emax - NUM_SPECIES] = eigen_value[1];

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
#endif // end EIGEN_ALLOC

inline void RoeAverageLeft_x(int n, real_t *eigen_l, real_t &eigen_value, real_t *z, const real_t *yi, real_t const c2,
							 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
							 real_t const b1, real_t const b3, real_t Gamma)
{

	MARCO_PREEIGEN();

	real_t _u_c = _u * _c1;

	switch (n)
	{
	case 0:
		eigen_l[0] = _DF(0.5) * (b2 + _u_c + b3);
		eigen_l[1] = -_DF(0.5) * (b1 * _u + _c1);
		eigen_l[2] = -_DF(0.5) * (b1 * _v);
		eigen_l[3] = -_DF(0.5) * (b1 * _w);
		eigen_l[4] = _DF(0.5) * b1;
		eigen_value = sycl::fabs<real_t>(_u - _c);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = -_DF(0.5) * b1 * z[m];
		break;

	case 1:
		eigen_l[0] = (_DF(1.0) - b2 - b3) / b1;
		eigen_l[1] = _u;
		eigen_l[2] = _v;
		eigen_l[3] = _w;
		eigen_l[4] = -_DF(1.0);
		eigen_value = sycl::fabs<real_t>(_u);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = z[m];
		break;

	case 2:
		eigen_l[0] = _v;
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = -_DF(1.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs<real_t>(_u);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case 3:
		eigen_l[0] = -_w;
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = _DF(1.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs<real_t>(_u);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case Emax - 1:
		eigen_l[0] = _DF(0.5) * (b2 - _u_c + b3);
		eigen_l[1] = _DF(0.5) * (-b1 * _u + _c1);
		eigen_l[2] = _DF(0.5) * (-b1 * _v);
		eigen_l[3] = _DF(0.5) * (-b1 * _w);
		eigen_l[4] = _DF(0.5) * b1;
		eigen_value = sycl::fabs<real_t>(_u + _c);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = -_DF(0.5) * b1 * z[m];
		break;

#ifdef COP
	default: // For COP eigen
		eigen_l[0] = -yi[n + NUM_SPECIES - Emax];
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs<real_t>(_u);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = (n + NUM_SPECIES - Emax == m) ? _DF(1.0) : _DF(0.0);
		break;
#endif // COP
	}
}

inline void RoeAverageRight_x(int n, real_t *eigen_r, real_t *z, const real_t *yi, real_t const c2,
							  real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
							  real_t const b1, real_t const b3, real_t Gamma)
{

	MARCO_PREEIGEN();

	switch (n)
	{
	case 0:
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = _u - _c;
		eigen_r[2] = _v;
		eigen_r[3] = _w;
		eigen_r[4] = _H - _u * _c;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = yi[m];
		break;

	case 1:
		eigen_r[0] = b1;
		eigen_r[1] = _u * b1;
		eigen_r[2] = _v * b1;
		eigen_r[3] = _w * b1;
		eigen_r[4] = _H * b1 - _DF(1.0);
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = b1 * yi[m];
		break;

	case 2:
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = -_DF(1.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = -_v;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case 3:
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = _DF(1.0);
		eigen_r[4] = _w;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case Emax - 1:
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = _u + _c;
		eigen_r[2] = _v;
		eigen_r[3] = _w;
		eigen_r[4] = _H + _u * _c;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = yi[m];
		break;

#ifdef COP
	default: // For COP eigen
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = z[n + NUM_SPECIES - Emax];
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = (m == n + NUM_SPECIES - Emax) ? _DF(1.0) : _DF(0.0);
		break;
#endif // COP
	}
}

inline void RoeAverageLeft_y(int n, real_t *eigen_l, real_t &eigen_value, real_t *z, const real_t *yi, real_t const c2,
							 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
							 real_t const b1, real_t const b3, real_t Gamma)
{

	MARCO_PREEIGEN();

	real_t _v_c = _v * _c1;

	switch (n)
	{
	case 0:
		eigen_l[0] = _DF(0.5) * (b2 + _v_c + b3);
		eigen_l[1] = -_DF(0.5) * (b1 * _u);
		eigen_l[2] = -_DF(0.5) * (b1 * _v + _c1);
		eigen_l[3] = -_DF(0.5) * (b1 * _w);
		eigen_l[4] = _DF(0.5) * b1;
		eigen_value = sycl::fabs<real_t>(_v - _c);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = -_DF(0.5) * b1 * z[m];
		break;

	case 1:
		eigen_l[0] = -_u;
		eigen_l[1] = _DF(1.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs<real_t>(_v);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case 2:
		eigen_l[0] = (_DF(1.0) - b2 - b3) / b1;
		eigen_l[1] = _u;
		eigen_l[2] = _v;
		eigen_l[3] = _w;
		eigen_l[4] = -_DF(1.0);
		eigen_value = sycl::fabs<real_t>(_v);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = z[m];
		break;

	case 3:
		eigen_l[0] = _w;
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = -_DF(1.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs<real_t>(_v);
		for (int n = 0; n < NUM_COP; n++)
			eigen_l[n + Emax - NUM_COP] = _DF(0.0);
		break;

	case Emax - 1:
		eigen_l[0] = _DF(0.5) * (b2 - _v_c + b3);
		eigen_l[1] = _DF(0.5) * (-b1 * _u);
		eigen_l[2] = _DF(0.5) * (-b1 * _v + _c1);
		eigen_l[3] = _DF(0.5) * (-b1 * _w);
		eigen_l[4] = _DF(0.5) * b1;
		eigen_value = sycl::fabs<real_t>(_v + _c);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = -_DF(0.5) * b1 * z[m];
		break;

#ifdef COP
	default: // For COP eigen
		eigen_l[0] = -yi[n + NUM_SPECIES - Emax];
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs<real_t>(_v);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = (n + NUM_SPECIES - Emax == m) ? _DF(1.0) : _DF(0.0);
		break;
#endif // COP
	}
}

inline void RoeAverageRight_y(int n, real_t *eigen_r, real_t *z, const real_t *yi, real_t const c2,
							  real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
							  real_t const b1, real_t const b3, real_t Gamma)
{

	MARCO_PREEIGEN();

	switch (n)
	{
	case 0:
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = _u;
		eigen_r[2] = _v - _c;
		eigen_r[3] = _w;
		eigen_r[4] = _H - _v * _c;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = yi[m];
		break;

	case 1:
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(1.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = _u;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case 2:
		eigen_r[0] = b1;
		eigen_r[1] = _u * b1;
		eigen_r[2] = _v * b1;
		eigen_r[3] = _w * b1;
		eigen_r[4] = _H * b1 - _DF(1.0);
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = b1 * yi[m];
		break;

	case 3:
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = -_DF(1.0);
		eigen_r[4] = -_w;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case Emax - 1:
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = _u;
		eigen_r[2] = _v + _c;
		eigen_r[3] = _w;
		eigen_r[4] = _H + _v * _c;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = yi[m];
		break;

#ifdef COP
	default: // For COP eigen
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = z[n + NUM_SPECIES - Emax];
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = (m == n + NUM_SPECIES - Emax) ? _DF(1.0) : _DF(0.0);
		break;
#endif // COP
	}
}

inline void RoeAverageLeft_z(int n, real_t *eigen_l, real_t &eigen_value, real_t *z, const real_t *yi, real_t const c2,
							 real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
							 real_t const b1, real_t const b3, real_t Gamma)
{

	MARCO_PREEIGEN();

	real_t _w_c = _w * _c1;

	switch (n)
	{
	case 0:
		eigen_l[0] = _DF(0.5) * (b2 + _w_c + b3);
		eigen_l[1] = -_DF(0.5) * (b1 * _u);
		eigen_l[2] = -_DF(0.5) * (b1 * _v);
		eigen_l[3] = -_DF(0.5) * (b1 * _w + _c1);
		eigen_l[4] = _DF(0.5) * b1;
		eigen_value = sycl::fabs<real_t>(_w - _c);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = -_DF(0.5) * b1 * z[m];
		break;

	case 1:
		eigen_l[0] = _u;
		eigen_l[1] = -_DF(1.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs<real_t>(_w);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case 2:
		eigen_l[0] = -_v;
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = _DF(1.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs<real_t>(_w);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case 3:
		eigen_l[0] = (_DF(1.0) - b2 - b3) / b1; //-q2 + _H;
		eigen_l[1] = _u;
		eigen_l[2] = _v;
		eigen_l[3] = _w;
		eigen_l[4] = -_DF(1.0);
		eigen_value = sycl::fabs<real_t>(_w);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = z[m];
		break;

	case Emax - 1:
		eigen_l[0] = _DF(0.5) * (b2 - _w_c + b3);
		eigen_l[1] = _DF(0.5) * (-b1 * _u);
		eigen_l[2] = _DF(0.5) * (-b1 * _v);
		eigen_l[3] = _DF(0.5) * (-b1 * _w + _c1);
		eigen_l[4] = _DF(0.5) * b1;
		eigen_value = sycl::fabs<real_t>(_w + _c);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = -_DF(0.5) * b1 * z[m];
		break;

#ifdef COP
	default: // For COP eigen
		eigen_l[0] = -yi[n + NUM_SPECIES - Emax];
		eigen_l[1] = _DF(0.0);
		eigen_l[2] = _DF(0.0);
		eigen_l[3] = _DF(0.0);
		eigen_l[4] = _DF(0.0);
		eigen_value = sycl::fabs<real_t>(_w);
		for (int m = 0; m < NUM_COP; m++)
			eigen_l[m + Emax - NUM_COP] = (n + NUM_SPECIES - Emax == m) ? _DF(1.0) : _DF(0.0);
		break;
#endif // COP
	}
}

inline void RoeAverageRight_z(int n, real_t *eigen_r, real_t *z, const real_t *yi, real_t const c2,
							  real_t const _rho, real_t const _u, real_t const _v, real_t const _w, real_t const _H,
							  real_t const b1, real_t const b3, real_t Gamma)
{

	MARCO_PREEIGEN();

	switch (n)
	{
	case 0:
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = _u;
		eigen_r[2] = _v;
		eigen_r[3] = _w - _c;
		eigen_r[4] = _H - _w * _c;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = yi[m];
		break;

	case 1:
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = -_DF(1.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = -_u;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case 2:
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(1.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = _v;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = _DF(0.0);
		break;

	case 3:
		eigen_r[0] = b1;
		eigen_r[1] = b1 * _u;
		eigen_r[2] = b1 * _v;
		eigen_r[3] = b1 * _w;
		eigen_r[4] = _H * b1 - _DF(1.0);
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = b1 * yi[m];
		break;

	case Emax - 1:
		eigen_r[0] = _DF(1.0);
		eigen_r[1] = _u;
		eigen_r[2] = _v;
		eigen_r[3] = _w + _c;
		eigen_r[4] = _H + _w * _c;
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = yi[m];
		break;

#ifdef COP
	default: // For COP eigen
		eigen_r[0] = _DF(0.0);
		eigen_r[1] = _DF(0.0);
		eigen_r[2] = _DF(0.0);
		eigen_r[3] = _DF(0.0);
		eigen_r[4] = z[n + NUM_SPECIES - Emax];
		for (int m = 0; m < NUM_COP; m++)
			eigen_r[m + Emax - NUM_COP] = (m == n + NUM_SPECIES - Emax) ? _DF(1.0) : _DF(0.0);
		break;
#endif // COP
	}
}

#ifdef COP_CHEME
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
real_t get_Entropy(real_t *__restrict__ Hia, real_t *__restrict__ Hib, const real_t Ri, const real_t T0, const int n)
{
	real_t T = sycl::max<real_t>(T0, _DF(200.0));
	real_t S = _DF(0.0), _T = _DF(1.0) / T;
#if Thermo
	if (T >= _DF(1000.0) && T < _DF(6000.0))
		S = Ri * ((-_DF(0.5) * Hia[n * 7 * 3 + 0 * 3 + 1] * _T - Hia[n * 7 * 3 + 1 * 3 + 1]) * _T + Hia[n * 7 * 3 + 2 * 3 + 1] * sycl::log(T) + (Hia[n * 7 * 3 + 3 * 3 + 1] + (_DF(0.5) * Hia[n * 7 * 3 + 4 * 3 + 1] + (Hia[n * 7 * 3 + 5 * 3 + 1] * _OT + Hia[n * 7 * 3 + 6 * 3 + 1] * _DF(0.25) * T) * T) * T) * T + Hib[n * 2 * 3 + 1 * 3 + 1]);
	else if (T < _DF(1000.0))
		S = Ri * ((-_DF(0.5) * Hia[n * 7 * 3 + 0 * 3 + 0] * _T - Hia[n * 7 * 3 + 1 * 3 + 0]) * _T + Hia[n * 7 * 3 + 2 * 3 + 0] * sycl::log(T) + (Hia[n * 7 * 3 + 3 * 3 + 0] + (_DF(0.5) * Hia[n * 7 * 3 + 4 * 3 + 0] + (Hia[n * 7 * 3 + 5 * 3 + 0] * _OT + Hia[n * 7 * 3 + 6 * 3 + 0] * _DF(0.25) * T) * T) * T) * T + Hib[n * 2 * 3 + 1 * 3 + 0]);
	else if (T >= _DF(6000.0))
		S = Ri * ((-_DF(0.5) * Hia[n * 7 * 3 + 0 * 3 + 2] * _T - Hia[n * 7 * 3 + 1 * 3 + 2]) * _T + Hia[n * 7 * 3 + 2 * 3 + 2] * sycl::log(T) + (Hia[n * 7 * 3 + 3 * 3 + 2] + (_DF(0.5) * Hia[n * 7 * 3 + 4 * 3 + 2] + (Hia[n * 7 * 3 + 5 * 3 + 2] * _OT + Hia[n * 7 * 3 + 6 * 3 + 2] * _DF(0.25) * T) * T) * T) * T + Hib[n * 2 * 3 + 1 * 3 + 2]);
#else
	if (T > 1000)
		S = Ri * (Hia[n * 7 * 3 + 0 * 3 + 0] * sycl::log(T) + (Hia[n * 7 * 3 + 1 * 3 + 0] + (_DF(0.5) * Hia[n * 7 * 3 + 2 * 3 + 0] + (Hia[n * 7 * 3 + 3 * 3 + 0] * _OT + Hia[n * 7 * 3 + 4 * 3 + 0] * _DF(0.25) * T) * T) * T) * T + Hia[n * 7 * 3 + 6 * 3 + 0]);
	else
		S = Ri * (Hia[n * 7 * 3 + 0 * 3 + 1] * sycl::log(T) + (Hia[n * 7 * 3 + 1 * 3 + 1] + (_DF(0.5) * Hia[n * 7 * 3 + 2 * 3 + 1] + (Hia[n * 7 * 3 + 3 * 3 + 1] * _OT + Hia[n * 7 * 3 + 4 * 3 + 1] * _DF(0.25) * T) * T) * T) * T + Hia[n * 7 * 3 + 6 * 3 + 1]);
		// if (T > 1000)
		// 	S = Ri * (Hia[n * 7 * 3 + 0 * 3 + 0] * sycl::log(T) + Hia[n * 7 * 3 + 1 * 3 + 0] * T + _DF(0.5) * Hia[n * 7 * 3 + 2 * 3 + 0] * T * T + Hia[n * 7 * 3 + 3 * 3 + 0] * _OT * T * T * T + Hia[n * 7 * 3 + 4 * 3 + 0] * _DF(0.25) * T * T * T * T + Hia[n * 7 * 3 + 6 * 3 + 0]);
		// else
		// 	S = Ri * (Hia[n * 7 * 3 + 0 * 3 + 1] * sycl::log(T) + Hia[n * 7 * 3 + 1 * 3 + 1] * T + _DF(0.5) * Hia[n * 7 * 3 + 2 * 3 + 1] * T * T + Hia[n * 7 * 3 + 3 * 3 + 1] * _OT * T * T * T + Hia[n * 7 * 3 + 4 * 3 + 1] * _DF(0.25) * T * T * T * T + Hia[n * 7 * 3 + 6 * 3 + 1]);
#endif

	// 	real_t S = _DF(0.0), T = T0;
	// #if Thermo
	// 	if (T > 1000) // Hia[n * 7 * 3 + 0 * 3 + 1]//Hib[n * 2 * 3 + 0 * 3 + 1]
	// 		S = Ri * (-_DF(0.5) * Hia[n * 7 * 3 + 0 * 3 + 1] / T / T - Hia[n * 7 * 3 + 1 * 3 + 1] / T + Hia[n * 7 * 3 + 2 * 3 + 1] * sycl::log(T) + Hia[n * 7 * 3 + 3 * 3 + 1] * T + _DF(0.5) * Hia[n * 7 * 3 + 4 * 3 + 1] * T * T + Hia[n * 7 * 3 + 5 * 3 + 1] * sycl::pow<real_t>(T, 3) / real_t(3.0) + Hia[n * 7 * 3 + 6 * 3 + 1] * sycl::pow<real_t>(T, 4) / real_t(4.0) + Hib[n * 2 * 3 + 1 * 3 + 1]);
	// 	else
	// 		S = Ri * (-_DF(0.5) * Hia[n * 7 * 3 + 0 * 3 + 0] / T / T - Hia[n * 7 * 3 + 1 * 3 + 0] / T + Hia[n * 7 * 3 + 2 * 3 + 0] * sycl::log(T) + Hia[n * 7 * 3 + 3 * 3 + 0] * T + _DF(0.5) * Hia[n * 7 * 3 + 4 * 3 + 0] * T * T + Hia[n * 7 * 3 + 5 * 3 + 0] * sycl::pow<real_t>(T, 3) / real_t(3.0) + Hia[n * 7 * 3 + 6 * 3 + 0] * sycl::pow<real_t>(T, 4) / real_t(4.0) + Hib[n * 2 * 3 + 1 * 3 + 0]);
	// #else
	// 	if (T > 1000)
	// 		S = Ri * (Hia[n * 7 * 3 + 0 * 3 + 0] * sycl::log(T) + Hia[n * 7 * 3 + 1 * 3 + 0] * T + _DF(0.5) * Hia[n * 7 * 3 + 2 * 3 + 0] * T * T + Hia[n * 7 * 3 + 3 * 3 + 0] * _OT * T * T * T + Hia[n * 7 * 3 + 4 * 3 + 0] / _DF(4.0) * T * T * T * T + Hia[n * 7 * 3 + 6 * 3 + 0]);
	// 	else
	// 		S = Ri * (Hia[n * 7 * 3 + 0 * 3 + 1] * sycl::log(T) + Hia[n * 7 * 3 + 1 * 3 + 1] * T + _DF(0.5) * Hia[n * 7 * 3 + 2 * 3 + 1] * T * T + Hia[n * 7 * 3 + 3 * 3 + 1] * _OT * T * T * T + Hia[n * 7 * 3 + 4 * 3 + 1] / _DF(4.0) * T * T * T * T + Hia[n * 7 * 3 + 6 * 3 + 1]);
	// #endif

	return S;
}

/**
 * @brief get_Gibson
 */
real_t get_Gibson(real_t *__restrict__ Hia, real_t *__restrict__ Hib, const real_t T, const real_t Ri, const int n)
{
	return (get_Entropy(Hia, Hib, Ri, T, n) - get_Enthalpy(Hia, Hib, T, Ri, n) / T) / Ri;
}

/**
 * @brief get_Kc
 */
real_t get_Kc(const real_t *_Wi, real_t *__restrict__ Hia, real_t *__restrict__ Hib, int *__restrict__ Nu_d_, const real_t T, const int m)
{
	real_t Kck = _DF(0.0), Nu_sum = _DF(0.0);
	for (size_t n = 0; n < NUM_SPECIES; n++)
	{
		real_t Ri = Ru * _Wi[n];
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
void get_KbKf(real_t *Kf, real_t *Kb, real_t *Rargus, real_t *_Wi, real_t *Hia, real_t *Hib, int *Nu_d_, const real_t T)
{
	for (size_t m = 0; m < NUM_REA; m++)
	{
		real_t A = Rargus[m * 6 + 0], B = Rargus[m * 6 + 1], E = Rargus[m * 6 + 2];
#if CJ
		Kf[m] = sycl::min<real_t>((20 * _DF(1.0)), A * sycl::pow<real_t>(T, B) * sycl::exp(-E / T));
		Kb[m] = _DF(0.0);
#else
		Kf[m] = get_Kf_ArrheniusLaw(A, B, E, T);
		real_t Kck = get_Kc(_Wi, Hia, Hib, Nu_d_, T, m);
		Kb[m] = Kf[m] / Kck;
#endif
	}
}

/**
 * @brief QSSAFun
 */
void QSSAFun(real_t *q, real_t *d, real_t *Kf, real_t *Kb, const real_t yi[NUM_SPECIES], Thermal thermal, real_t *React_ThirdCoef,
			 int **reaction_list, int **reactant_list, int **product_list, int *rns, int *rts, int *pls,
			 int *Nu_b_, int *Nu_f_, int *third_ind, const real_t rho)
{
	real_t C[NUM_SPECIES] = {_DF(0.0)}, _rho = _DF(1.0) / rho;
	for (int n = 0; n < NUM_SPECIES; n++)
		C[n] = rho * yi[n] * thermal._Wi[n] * _DF(1e-6);

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
		q[n] *= thermal.Wi[n] * _rho * _DF(1.0e6);
		d[n] *= thermal.Wi[n] * _rho * _DF(1.0e6);
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
	return sign(b) * sycl::abs(a);
}

/**
 * @brief Chemeq2
 */
void Chemeq2(Thermal thermal, real_t *Kf, real_t *Kb, real_t *React_ThirdCoef, real_t *Rargus, int *Nu_b_, int *Nu_f_, int *Nu_d_,
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
	real_t *species_chara = thermal.species_chara, *Hia = thermal.Hia, *Hib = thermal.Hib;
	//=========================================================
	// to initilize the first 'dt', q, d
	get_KbKf(Kf, Kb, Rargus, thermal._Wi, Hia, Hib, Nu_d_, TTn);
	QSSAFun(q, d, Kf, Kb, y, thermal, React_ThirdCoef, reaction_list, reactant_list, product_list, rns, rts, pls, Nu_b_, Nu_f_, third_ind, rho);
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
		TTn = get_T(thermal, y, e, TTs); // UpdateTemperature(-1, rhoi, rho, e, TTs); // predicted T
		// GetKfKb(TTn);
		if (iter == 1)
		{
			tn = ts + dt;
			for (int i = 0; i < NUM_SPECIES; i++)
				y1[i] = y[i]; // prediction results stored by y1
		}
		QSSAFun(q, d, Kf, Kb, y, thermal, React_ThirdCoef, reaction_list, reactant_list, product_list, rns, rts, pls, Nu_b_, Nu_f_, third_ind, rho);
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
			TT = get_T(thermal, y, e, TTs); //  UpdateTemperature(-1, rhoi, rho, e, TTs); // final T
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
	TTn = get_T(thermal, y, e, TTs); // UpdateTemperature(-1, rhoi, rho, e, TTs); // new T
	// GetKfKb(TTn);
	QSSAFun(q, d, Kf, Kb, y, thermal, React_ThirdCoef, reaction_list, reactant_list, product_list, rns, rts, pls, Nu_b_, Nu_f_, third_ind, rho);
	gcount++;
	goto flag1;
}
#endif // end COP_CHEME
#ifdef Visc
/**
 * @brief get viscosity at temperature T(unit:K)(fit)
 * @return double,unit: Pa.s=kg/(m.s)
 */
real_t Viscosity(real_t fitted_coefficients_visc[order_polynominal_fitted], const double T0)
{
	// real_t Tref = Reference_params[3], visref = Reference_params[5];
	real_t T = T0; //* Tref; // nondimension==>dimension
	real_t viscosity = fitted_coefficients_visc[order_polynominal_fitted - 1];
	for (int i = order_polynominal_fitted - 2; i >= 0; i--)
		viscosity = viscosity * sycl::log(T) + fitted_coefficients_visc[i];
	real_t temp = sycl::exp(viscosity); // / visref;
	return temp; // dimension==>nondimension

	// real_t Tref = Reference_params[3], visref = Reference_params[5];
	// real_t T = T0 * Tref; // nondimension==>dimension
	// real_t viscosity = fitted_coefficients_visc[order_polynominal_fitted - 1];
	// for (int i = order_polynominal_fitted - 2; i >= 0; i--)
	// 	viscosity = viscosity * sycl::log(T) + fitted_coefficients_visc[i];
	// real_t temp = sycl::exp(viscosity) / visref;
	// return temp; // dimension==>nondimension
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
	// real_t rhoref = Reference_params[1], pref = Reference_params[2];
	// real_t Tref = Reference_params[3], W0ref = Reference_params[6], visref = Reference_params[7];
	// real_t kref = visref * (pref / rhoref);
	real_t T = T0; //* Tref; // nondimension==>dimension
	real_t thermal_conductivity = fitted_coefficients_therm[order_polynominal_fitted - 1];
	for (int i = order_polynominal_fitted - 2; i >= 0; i--)
		thermal_conductivity = thermal_conductivity * sycl::log(T) + fitted_coefficients_therm[i];
	real_t temp = sycl::exp(thermal_conductivity); // / kref;
	return temp; // dimension==>nondimension

	// real_t rhoref = Reference_params[1], pref = Reference_params[2];
	// real_t Tref = Reference_params[3], W0ref = Reference_params[6], visref = Reference_params[7];
	// real_t kref = visref * (pref / rhoref);
	// real_t T = T0 * Tref; // nondimension==>dimension
	// real_t thermal_conductivity = fitted_coefficients_therm[order_polynominal_fitted - 1];
	// for (int i = order_polynominal_fitted - 2; i >= 0; i--)
	// 	thermal_conductivity = thermal_conductivity * sycl::log(T) + fitted_coefficients_therm[i];
	// real_t temp = sycl::exp(thermal_conductivity) / kref;
	// return temp; // dimension==>nondimension
}

/**
 * @brief get Dkj:the binary difffusion coefficient of specie-k to specie-j via equation 5-37
 * @para TT temperature unit:K
 * @para PP pressure unit:Pa
 */
real_t GetDkj(real_t *specie_k, real_t *specie_j, real_t **Dkj_matrix, const real_t T0, const real_t P0)
{
	// real_t rhoref = Reference_params[1], pref = Reference_params[2];
	// real_t Tref = Reference_params[3], W0ref = Reference_params[6], visref = Reference_params[7];
	// real_t Dref = visref / rhoref;
	real_t TT = T0; // * Tref; // nondimension==>dimension
	real_t PP = P0; // * pref; // nondimension==>dimension
	real_t Dkj = Dkj_matrix[int(specie_k[SID]) * NUM_SPECIES + int(specie_j[SID])][order_polynominal_fitted - 1];
	for (int i = order_polynominal_fitted - 2; i >= 0; i--)
		Dkj = Dkj * sycl::log(TT) + Dkj_matrix[int(specie_k[SID]) * NUM_SPECIES + int(specie_j[SID])][i];
	real_t temp = (sycl::exp(Dkj) / PP); // / Dref;
	return temp; // unit:cm*cm/s　//dimension==>nondimension

	// real_t rhoref = Reference_params[1], pref = Reference_params[2];
	// real_t Tref = Reference_params[3], W0ref = Reference_params[6], visref = Reference_params[7];
	// real_t Dref = visref / rhoref;
	// real_t TT = T0 * Tref; // nondimension==>dimension
	// real_t PP = P0 * pref; // nondimension==>dimension
	// real_t Dkj = Dkj_matrix[int(specie_k[SID]) * NUM_SPECIES + int(specie_j[SID])][order_polynominal_fitted - 1];
	// for (int i = order_polynominal_fitted - 2; i >= 0; i--)
	// 	Dkj = Dkj * sycl::log(TT) + Dkj_matrix[int(specie_k[SID]) * NUM_SPECIES + int(specie_j[SID])][i];
	// real_t temp = (sycl::exp(Dkj) / PP) / Dref;
	// return temp; // unit:cm*cm/s　//dimension==>nondimension
}

/**
 * @brief get average transport coefficient
 * @param chemi is set to get species information
 */
void Get_transport_coeff_aver(const int i_id, const int j_id, const int k_id, Thermal thermal, real_t *Dkm_aver_id, real_t &viscosity_aver, real_t &thermal_conduct_aver, real_t const X[NUM_SPECIES],
							  const real_t rho, const real_t p, const real_t T, const real_t C_total)
{
	real_t **fcv = thermal.fitted_coefficients_visc;
	real_t **fct = thermal.fitted_coefficients_therm;
	real_t **Dkj = thermal.Dkj_matrix;
	viscosity_aver = _DF(0.0);
#ifdef Heat
	thermal_conduct_aver = _DF(0.0);
#endif
	real_t denominator = _DF(0.0);
	real_t *specie[NUM_SPECIES];
	for (size_t ii = 0; ii < NUM_SPECIES; ii++)
		specie[ii] = &(thermal.species_chara[ii * SPCH_Sz]);
	for (int k = 0; k < NUM_SPECIES; k++)
	{
		denominator = _DF(0.0);
		for (int i = 0; i < NUM_SPECIES; i++)
		{
			// real_t temp = PHI(specie[k], specie[i], fcv, T);
			denominator = denominator + X[i] * PHI(specie[k], specie[i], fcv, T);
		}
		// calculate viscosity_aver via equattion(5-49)//
		real_t _denominator = _DF(1.0) / denominator;
		viscosity_aver = viscosity_aver + X[k] * Viscosity(fcv[int(specie[k][SID])], T) * _denominator; // Pa.s=kg/(m.s)
#ifdef Heat
		// calculate thermal_conduct via Su Hongmin//
		thermal_conduct_aver = thermal_conduct_aver + X[k] * Thermal_conductivity(fct[int(specie[k][SID])], T) * _denominator;
#endif // end Heat
	}
#ifdef Diffu
	// calculate diffusion coefficient specie_k to mixture via equation 5-45
#if 1 < NUM_SPECIES
	{
		double temp1, temp2;
		for (int k = 0; k < NUM_SPECIES; k++)
		{
			temp1 = _DF(0.0);
			temp2 = _DF(0.0);
			for (int i = 0; i < NUM_SPECIES; i++)
			{
				if (i != k)
				{
					temp1 += X[i] * thermal.Wi[i];
					temp2 += X[i] / GetDkj(specie[i], specie[k], Dkj, T, p); // trans_coeff.GetDkj(T, p, chemi.species[i], chemi.species[k], refstat);
				}
			}											 // cause nan error while only one yi of the mixture given(temp1/temp2=0/0).
			if (sycl::step(sycl::ceil(temp1), _DF(0.0))) // =1 while temp1==0.0;
				Dkm_aver_id[k] = GetDkj(specie[k], specie[k], Dkj, T, p);
			else
				Dkm_aver_id[k] = temp1 / temp2 / rho * C_total; // rho/C_total:the mole mass of mixture;
			Dkm_aver_id[k] *= _DF(1.0e-1);						// cm2/s==>m2/s
		}
	}
#else
	{															  // NUM_SPECIES==1
		Dkm_aver_id[0] = GetDkj(specie[0], specie[0], Dkj, T, p); // trans_coeff.GetDkj(T, p, chemi.species[0], chemi.species[0], refstat);
		Dkm_aver_id[0] *= _DF(1.0e-1);							  // cm2/s==>m2/s
	}
#endif // end NUM_SPECIES>1
#endif // end Diffu
}
#endif // end Visc

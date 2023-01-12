#pragma once

#include "schemes_device.hpp"

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